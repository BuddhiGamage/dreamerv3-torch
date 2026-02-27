#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wm_progress_monitor_per_task_threshold.py

World Model as TASK PROGRESSION MONITOR (no counterfactual action prediction).

Uses RSSM features extracted from Dreamer WorldModel:
  feat[t] = dynamics.get_feat(post)[t]  (T,D)

Per-task setup (WHAT YOU ASKED FOR):
- Fit progress-binned success reference (mu_b, var_b) using THAT TASK'S success episodes only.
- Build success window-score distribution using THAT TASK'S success episodes only.
- Compute ONE conformal threshold thr_task (single scalar) for that task.
- Test that task using ONLY that thr_task.

So if you pass 5 task roots, you get 5 thresholds (one per task),
and each task is evaluated with its own threshold.

Inputs per task root:
  calib_success_feats.npy  (object array, each element: (T_i, D))
  test_feats.npy           (object array, each element: (T_i, D))
  test_labels.npy          (int array, 1=failure, 0=success)
"""

import os, json, argparse
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

EPS = 1e-8


# -------------------------
# Conformal
# -------------------------
def conformal_upper_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Upper-tail conformal quantile with finite-sample correction:
      r = ceil((n+1)*(1-alpha)), threshold = sorted(scores)[r-1]
    """
    s = np.sort(np.asarray(scores, dtype=float))
    n = s.shape[0]
    r = int(np.ceil((n + 1) * (1 - alpha)))
    r = min(max(1, r), n)
    return float(s[r - 1])


def safe_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


# -------------------------
# Helpers
# -------------------------
def topk_mean(x: np.ndarray, k: int) -> float:
    if len(x) == 0:
        return 0.0
    kk = min(int(k), len(x))
    return float(np.mean(np.sort(x)[-kk:]))


def agg_score(x: np.ndarray, mode: str, topk: int) -> float:
    if len(x) == 0:
        return 0.0
    if mode == "max":
        return float(np.max(x))
    if mode == "mean":
        return float(np.mean(x))
    if mode == "topk_mean":
        return topk_mean(x, topk)
    raise ValueError(f"Unknown score_agg: {mode}")


def normalized_time_bins(T: int, B: int) -> np.ndarray:
    """
    Bin each timestep by tau=t/(T-1) into [0..B-1]
    """
    if T <= 1:
        return np.zeros((T,), dtype=np.int64)
    tau = np.arange(T, dtype=np.float64) / float(T - 1)
    b = np.floor(tau * B).astype(np.int64)
    b = np.clip(b, 0, B - 1)
    return b


# -------------------------
# Per-bin reference (diag Mahalanobis)
# -------------------------
@dataclass
class BinRef:
    mu: np.ndarray       # (D,)
    inv_var: np.ndarray  # (D,)


def fit_bin_refs(success_feats: List[np.ndarray], B: int, var_floor: float = 1e-6) -> List[BinRef]:
    """
    Fit per-bin mean/diag-var on THIS TASK's success episodes only.
    Empty bins fall back to global stats (within the same task).
    """
    per_bin: List[List[np.ndarray]] = [[] for _ in range(B)]
    D = None

    for feat in tqdm(success_feats, desc="FIT refs (success)", leave=False):
        feat = np.asarray(feat, dtype=np.float64)
        if feat.ndim != 2 or feat.shape[0] < 1:
            continue
        if D is None:
            D = feat.shape[1]
        elif feat.shape[1] != D:
            raise ValueError(f"Feature dim mismatch: expected {D}, got {feat.shape[1]}")

        bins = normalized_time_bins(feat.shape[0], B)
        for t in range(feat.shape[0]):
            per_bin[bins[t]].append(feat[t])

    if D is None:
        raise RuntimeError("No valid success features to fit refs.")

    # global fallback (within-task)
    all_pts = []
    for b in range(B):
        if len(per_bin[b]) > 0:
            all_pts.append(np.stack(per_bin[b], axis=0))
    Xg = np.concatenate(all_pts, axis=0) if all_pts else None
    if Xg is None:
        raise RuntimeError("No success points after binning.")
    mu_g = np.mean(Xg, axis=0)
    var_g = np.maximum(np.var(Xg, axis=0), var_floor)
    inv_var_g = 1.0 / var_g

    refs: List[BinRef] = []
    for b in range(B):
        if len(per_bin[b]) == 0:
            refs.append(BinRef(mu=mu_g, inv_var=inv_var_g))
            continue
        Xb = np.stack(per_bin[b], axis=0)
        mu = np.mean(Xb, axis=0)
        var = np.maximum(np.var(Xb, axis=0), var_floor)
        inv_var = 1.0 / var
        refs.append(BinRef(mu=mu, inv_var=inv_var))

    return refs


def per_timestep_diag_mahal(feat: np.ndarray, refs: List[BinRef]) -> np.ndarray:
    feat = np.asarray(feat, dtype=np.float64)
    T, D = feat.shape
    B = len(refs)
    bins = normalized_time_bins(T, B)

    d = np.zeros((T,), dtype=np.float64)
    for t in range(T):
        r = refs[bins[t]]
        diff = feat[t] - r.mu
        d[t] = float(np.sum((diff * diff) * r.inv_var))
    return d


# -------------------------
# Window + persistence
# -------------------------
def window_len(T: int, mode: str, fixed: int, frac: float, min_w: int) -> int:
    if mode == "fixed":
        return max(1, int(fixed))
    if mode == "adaptive":
        return max(int(min_w), int(round(frac * T)))
    raise ValueError(mode)


def persist_len(T: int, mode: str, fixed: int, frac: float, max_p: int, min_p: int = 1) -> int:
    if mode == "fixed":
        return max(min_p, int(fixed))
    if mode == "adaptive":
        p = max(min_p, int(round(frac * T)))
        return int(min(max_p, p))
    raise ValueError(mode)


def sliding_window_scores(d: np.ndarray, W: int, agg: str, topk: int) -> np.ndarray:
    T = len(d)
    s = np.zeros((T,), dtype=np.float64)
    for t in range(T):
        a = max(0, t - W + 1)
        s[t] = agg_score(d[a:t+1], mode=agg, topk=topk)
    return s


def first_persist_crossing(s: np.ndarray, thr: float, P: int) -> Optional[int]:
    run = 0
    P = max(1, int(P))
    for t in range(len(s)):
        if s[t] > thr:
            run += 1
        else:
            run = 0
        if run >= P:
            return int(t - P + 1)
    return None


# -------------------------
# IO
# -------------------------
def load_task_root(root: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    calib = np.load(os.path.join(root, "calib_success_feats.npy"), allow_pickle=True)
    test  = np.load(os.path.join(root, "test_feats.npy"), allow_pickle=True)
    y     = np.load(os.path.join(root, "test_labels.npy")).astype(np.int64)
    return calib, test, y


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_roots", nargs="+", required=True,
                    help="Each root contains calib_success_feats.npy, test_feats.npy, test_labels.npy")

    ap.add_argument("--alpha", type=float, default=0.10,
                    help="Per-task conformal alpha. One threshold will be computed per task root.")
    ap.add_argument("--bins", type=int, default=20)
    ap.add_argument("--var_floor", type=float, default=1e-6)

    ap.add_argument("--window_mode", type=str, default="adaptive", choices=["adaptive", "fixed"])
    ap.add_argument("--window_fixed", type=int, default=5)
    ap.add_argument("--window_frac", type=float, default=0.2)
    ap.add_argument("--min_window", type=int, default=3)

    ap.add_argument("--score_agg", type=str, default="topk_mean", choices=["max", "mean", "topk_mean"])
    ap.add_argument("--topk", type=int, default=7)

    ap.add_argument("--persist_mode", type=str, default="adaptive", choices=["adaptive", "fixed"])
    ap.add_argument("--persist_fixed", type=int, default=3)
    ap.add_argument("--persist_frac", type=float, default=0.2)
    ap.add_argument("--persist_max", type=int, default=8)

    ap.add_argument("--out_json", type=str, default="wm_progress_monitor_per_task_threshold.json")
    return ap.parse_args()


def main():
    args = parse_args()
    roots = [os.path.abspath(r) for r in args.task_roots]
    B = int(args.bins)

    print("====================================")
    print("[CFG] per-task thresholds (NOT pooled across tasks)")
    print(f"[CFG] alpha={args.alpha} bins={B} var_floor={args.var_floor}")
    print(f"[CFG] window_mode={args.window_mode} fixed={args.window_fixed} frac={args.window_frac} min={args.min_window}")
    print(f"[CFG] score_agg={args.score_agg} topk={args.topk}")
    print(f"[CFG] persist_mode={args.persist_mode} fixed={args.persist_fixed} frac={args.persist_frac} max={args.persist_max}")
    print("[CFG] roots:")
    for r in roots:
        print("  -", r)

    per_task_results: List[Dict[str, Any]] = []

    pooled_y, pooled_pred, pooled_rank = [], [], []
    pooled_det, pooled_det_norm = [], []

    for root in roots:
        name = os.path.basename(root)

        calib, test, y = load_task_root(root)
        success_feats = list(calib)

        print("\n====================================")
        print(f"[TASK] {name}")
        print(f"[LOAD] calib_success_eps={len(calib)} test_eps={len(test)} failures={(y==1).sum()} successes={(y==0).sum()}")

        # 1) Fit per-bin refs on THIS TASK's successes
        refs = fit_bin_refs(success_feats, B=B, var_floor=float(args.var_floor))

        # 2) Build THIS TASK's success window-score distribution
        calib_window_scores = []
        for feat in tqdm(success_feats, desc=f"{name} CALIB window-scores", leave=False):
            feat = np.asarray(feat)
            if feat.ndim != 2 or feat.shape[0] < 2:
                continue
            T = feat.shape[0]
            W = window_len(T, args.window_mode, args.window_fixed, args.window_frac, args.min_window)
            d = per_timestep_diag_mahal(feat, refs)
            s = sliding_window_scores(d, W=W, agg=args.score_agg, topk=args.topk)
            calib_window_scores.append(s.astype(np.float64))

        if len(calib_window_scores) == 0:
            raise RuntimeError(f"[{name}] No calibration windows produced.")

        calib_window_scores = np.concatenate(calib_window_scores, axis=0).astype(np.float64)

        # 3) ONE threshold for THIS TASK
        thr = conformal_upper_quantile(calib_window_scores, float(args.alpha))
        print(f"[CALIB] success windows N={calib_window_scores.shape[0]}")
        print(f"[CALIB] thr_task={thr:.6f} (single threshold for {name}, alpha={args.alpha})")
        print(f"[CALIB] score stats: mean={calib_window_scores.mean():.4f} std={calib_window_scores.std():.4f} "
              f"min={calib_window_scores.min():.4f} max={calib_window_scores.max():.4f}")

        # 4) Test
        y_pred = []
        rank_scores = []
        det_times, det_times_norm = [], []

        for feat, yi in tqdm(list(zip(test, y)), desc=f"{name} TEST", leave=False):
            feat = np.asarray(feat)
            if feat.ndim != 2 or feat.shape[0] < 2:
                y_pred.append(0)
                rank_scores.append(0.0)
                continue

            T = feat.shape[0]
            W = window_len(T, args.window_mode, args.window_fixed, args.window_frac, args.min_window)
            P = persist_len(T, args.persist_mode, args.persist_fixed, args.persist_frac, args.persist_max)

            d = per_timestep_diag_mahal(feat, refs)
            s = sliding_window_scores(d, W=W, agg=args.score_agg, topk=args.topk)

            # rank score for AUROC
            rank_ep = agg_score(s, mode="topk_mean", topk=min(args.topk, len(s)))
            rank_scores.append(float(rank_ep))

            dt = first_persist_crossing(s, thr=thr, P=P)
            pred = 1 if dt is not None else 0
            y_pred.append(pred)

            if int(yi) == 1 and pred == 1 and dt is not None:
                det_times.append(int(dt))
                det_times_norm.append(float(dt / max(T - 1, 1)))

        y_pred = np.asarray(y_pred, dtype=np.int64)
        rank_scores = np.asarray(rank_scores, dtype=np.float64)

        acc = accuracy_score(y, y_pred)
        tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
        tpr = tp / (tp + fn + EPS)
        tnr = tn / (tn + fp + EPS)
        bal = 0.5 * (tpr + tnr)
        auroc = safe_auroc(y, rank_scores)
        mean_dt = float(np.mean(det_times)) if det_times else float("nan")
        mean_dt_norm = float(np.mean(det_times_norm)) if det_times_norm else float("nan")

        print(f"[TEST] Acc={acc:.4f} TPR={tpr:.4f} TNR={tnr:.4f} BalAcc={bal:.4f} AUROC(rank)={auroc:.4f}")
        print(f"[TEST] Confusion: TN={tn} FP={fp} FN={fn} TP={tp}")
        print(f"[TEST] Detected failures: {len(det_times)} / {(y==1).sum()}  mean_dt={mean_dt:.2f}  mean_dt_norm={mean_dt_norm:.4f}")

        per_task_results.append({
            "task": name,
            "root": root,
            "thr_task": float(thr),
            "alpha": float(args.alpha),
            "bins": int(B),
            "calib_success_eps": int(len(calib)),
            "calib_success_windows_n": int(calib_window_scores.shape[0]),
            "metrics": {
                "accuracy": float(acc),
                "TPR": float(tpr),
                "TNR": float(tnr),
                "bal_acc": float(bal),
                "AUROC_rank": float(auroc) if not np.isnan(auroc) else None,
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                "mean_detection_time": float(mean_dt),
                "mean_detection_time_norm": float(mean_dt_norm),
                "num_failures": int((y==1).sum()),
                "num_detected_failures": int(len(det_times)),
            },
        })

        pooled_y.append(y)
        pooled_pred.append(y_pred)
        pooled_rank.append(rank_scores)
        pooled_det.extend(det_times)
        pooled_det_norm.extend(det_times_norm)

    # Optional pooled summary (note: this is NOT using a pooled threshold; it’s just reporting)
    pooled_y = np.concatenate(pooled_y, axis=0)
    pooled_pred = np.concatenate(pooled_pred, axis=0)
    pooled_rank = np.concatenate(pooled_rank, axis=0)

    acc = accuracy_score(pooled_y, pooled_pred)
    tn, fp, fn, tp = confusion_matrix(pooled_y, pooled_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn + EPS)
    tnr = tn / (tn + fp + EPS)
    bal = 0.5 * (tpr + tnr)
    auroc = safe_auroc(pooled_y, pooled_rank)
    mean_dt = float(np.mean(pooled_det)) if pooled_det else float("nan")
    mean_dt_norm = float(np.mean(pooled_det_norm)) if pooled_det_norm else float("nan")

    print("\n====================================")
    print("[POOLED REPORT] (each task used its OWN threshold)")
    print(f"  Accuracy      : {acc:.4f}")
    print(f"  TPR (failures): {tpr:.4f}")
    print(f"  TNR (success) : {tnr:.4f}")
    print(f"  BalAcc        : {bal:.4f}")
    print(f"  AUROC (rank)  : {auroc:.4f}")
    print(f"  Confusion     : TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"  Mean detection time (steps): {mean_dt:.2f}")
    print(f"  Mean detection time (normalized): {mean_dt_norm:.4f}")

    out: Dict[str, Any] = {
        "method": "wm_progress_monitor_per_task_threshold_single_scalar",
        "config": {
            "alpha": float(args.alpha),
            "bins": int(B),
            "var_floor": float(args.var_floor),
            "window_mode": args.window_mode,
            "window_fixed": int(args.window_fixed),
            "window_frac": float(args.window_frac),
            "min_window": int(args.min_window),
            "score_agg": args.score_agg,
            "topk": int(args.topk),
            "persist_mode": args.persist_mode,
            "persist_fixed": int(args.persist_fixed),
            "persist_frac": float(args.persist_frac),
            "persist_max": int(args.persist_max),
        },
        "per_task": per_task_results,
        "pooled_report": {
            "accuracy": float(acc),
            "TPR": float(tpr),
            "TNR": float(tnr),
            "bal_acc": float(bal),
            "AUROC_rank": float(auroc) if not np.isnan(auroc) else None,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "mean_detection_time": float(mean_dt),
            "mean_detection_time_norm": float(mean_dt_norm),
        }
    }

    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved:", args.out_json)


if __name__ == "__main__":
    main()