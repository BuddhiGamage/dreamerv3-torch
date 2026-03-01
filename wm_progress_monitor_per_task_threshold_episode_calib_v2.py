#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
wm_progress_monitor_per_task_threshold_episode_calib_v2.py

Per-task single-threshold progress monitor using Dreamer WM RSSM features.

Key upgrades:
- EPISODE-LEVEL conformal calibration (thr calibrated on per-episode score, not pooled windows).
- Optional success pool source:
    * calib_only (strict)
    * calib_plus_test_success (practical, stabilizes when calib successes are tiny / shifted)
- Extra diagnostics:
    * episode length stats for calib/test successes/failures
    * per-bin point counts for success pool
    * fraction of TEST successes whose episode score exceeds thr
- Adds robust 'median' score mode.
- NEW: computes TWA (timestep-weighted accuracy) in the same spirit as FIPER:
      TN contributes 1, TP contributes (1 - DT_norm), FP/FN contribute 0.

Input files expected under each task folder:
  calib_success_feats.npy   (object array of (T_i, D) success-only)
  test_feats.npy            (object array of (T_i, D))
  test_labels.npy           (int array: 1=failure, 0=success)

  python wm_progress_monitor_per_task_threshold_episode_calib_v2.py \
  --data_root /home/s447658/projects/dreamer_fiper_feats_all5 \
  --out_root  /home/s447658/projects/dreamer_fiper_offline/all5_tasks \
  --tasks push_t push_chair stacking sorting pretzel\
  --calib_success_source calib_only \
  --alpha 0.10 --bins 10 --var_floor 1e-4 \
  --window_mode adaptive --window_frac 0.2 --min_window 5 \
  --score_agg topk_mean --topk 7 \
  --persist_mode adaptive --persist_frac 0.25 --persist_max 5 \
  --calib_score_mode topk_mean \
  --debug 1

"""

import os
import json
import argparse
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

EPS = 1e-8


# ============================================================
# Conformal + metrics
# ============================================================
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


def compute_twa_from_counts_and_dts(tn: int, fp: int, fn: int, tp: int, det_times_norm: List[float]) -> float:
    """
    FIPER-style TWA (episode-wise):
      - TN contributes 1
      - TP contributes (1 - DT_norm)
      - FP/FN contribute 0
    Then normalized by total episodes N.

    Requires det_times_norm for detected failures (TP episodes) only.
    """
    N = int(tn + fp + fn + tp)
    if N <= 0:
        return float("nan")
    # Safety: if det_times_norm length mismatches tp (shouldn't), clip to min
    k = min(int(tp), len(det_times_norm))
    tp_contrib = float(np.sum([1.0 - float(dt) for dt in det_times_norm[:k]]))
    return float((float(tn) + tp_contrib) / float(N))


# ============================================================
# Aggregations
# ============================================================
def topk_mean(x: np.ndarray, k: int) -> float:
    if len(x) == 0:
        return 0.0
    kk = min(int(k), len(x))
    return float(np.mean(np.sort(x)[-kk:]))


def agg_score(x: np.ndarray, mode: str, topk: int) -> float:
    """
    Aggregate a 1D array into a scalar.
    """
    if len(x) == 0:
        return 0.0
    mode = str(mode).lower()
    if mode == "max":
        return float(np.max(x))
    if mode == "mean":
        return float(np.mean(x))
    if mode == "median":
        return float(np.median(x))
    if mode == "topk_mean":
        return topk_mean(x, topk)
    raise ValueError(f"Unknown score mode: {mode}")


# ============================================================
# Progress binning + diag Mahalanobis refs
# ============================================================
def normalized_time_bins(T: int, B: int) -> np.ndarray:
    """
    Assign each timestep to a bin based on normalized time tau=t/(T-1).
    """
    if T <= 1:
        return np.zeros((T,), dtype=np.int64)
    tau = np.arange(T, dtype=np.float64) / float(T - 1)
    b = np.floor(tau * B).astype(np.int64)
    return np.clip(b, 0, B - 1)


@dataclass
class BinRef:
    mu: np.ndarray       # (D,)
    inv_var: np.ndarray  # (D,)


def fit_bin_refs(success_feats: List[np.ndarray], B: int, var_floor: float) -> Tuple[List[BinRef], np.ndarray]:
    """
    Fit per-bin mean and diag variance from success features.
    Returns:
      refs: list of BinRef length B
      bin_counts: (B,) counts of success points per bin
    """
    per_bin: List[List[np.ndarray]] = [[] for _ in range(B)]
    D = None

    for feat in tqdm(success_feats, desc="FIT refs (success pool)", leave=False):
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

    bin_counts = np.array([len(per_bin[b]) for b in range(B)], dtype=np.int64)

    # Global fallback within this task/pool
    all_pts = [np.stack(per_bin[b], axis=0) for b in range(B) if len(per_bin[b]) > 0]
    if len(all_pts) == 0:
        raise RuntimeError("No success points after binning.")
    Xg = np.concatenate(all_pts, axis=0)

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
        refs.append(BinRef(mu=mu, inv_var=1.0 / var))

    return refs, bin_counts


def per_timestep_diag_mahal(feat: np.ndarray, refs: List[BinRef]) -> np.ndarray:
    feat = np.asarray(feat, dtype=np.float64)
    T, _ = feat.shape
    B = len(refs)
    bins = normalized_time_bins(T, B)
    d = np.zeros((T,), dtype=np.float64)
    for t in range(T):
        r = refs[bins[t]]
        diff = feat[t] - r.mu
        d[t] = float(np.sum((diff * diff) * r.inv_var))
    return d


# ============================================================
# Window scores + persistence detection
# ============================================================
def window_len(T: int, mode: str, fixed: int, frac: float, min_w: int) -> int:
    mode = str(mode).lower()
    if mode == "fixed":
        return max(1, int(fixed))
    if mode == "adaptive":
        return max(int(min_w), int(round(frac * T)))
    raise ValueError(mode)


def persist_len(T: int, mode: str, fixed: int, frac: float, max_p: int, min_p: int = 1) -> int:
    mode = str(mode).lower()
    if mode == "fixed":
        return max(min_p, int(fixed))
    if mode == "adaptive":
        p = max(min_p, int(round(frac * T)))
        return int(min(max_p, p))
    raise ValueError(mode)


def sliding_window_scores(d: np.ndarray, W: int, agg: str, topk: int) -> np.ndarray:
    """
    s[t] = aggregate(d[t-W+1 : t]) with truncation at start.
    """
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


# ============================================================
# IO + diagnostics
# ============================================================
def load_task_root(root: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    calib = np.load(os.path.join(root, "calib_success_feats.npy"), allow_pickle=True)
    test  = np.load(os.path.join(root, "test_feats.npy"), allow_pickle=True)
    y     = np.load(os.path.join(root, "test_labels.npy")).astype(np.int64)
    return calib, test, y


def _length_stats(arrs: List[np.ndarray]) -> Dict[str, Any]:
    if len(arrs) == 0:
        return {"n": 0}
    Ts = np.array([int(np.asarray(a).shape[0]) for a in arrs if np.asarray(a).ndim == 2], dtype=np.int64)
    if Ts.size == 0:
        return {"n": 0}
    return {
        "n": int(Ts.size),
        "min": int(Ts.min()),
        "median": int(np.median(Ts)),
        "max": int(Ts.max()),
        "mean": float(Ts.mean()),
    }


# ============================================================
# Args
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_root", type=str, default="/home/s447658/projects/dreamer_fiper_feats_all5")
    ap.add_argument("--out_root", type=str, default="/home/s447658/projects/dreamer_fiper_offline/all5_tasks")
    ap.add_argument("--tasks", type=str, nargs="*", default=["sorting", "push_chair"])

    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--bins", type=int, default=20)
    ap.add_argument("--var_floor", type=float, default=1e-6)

    ap.add_argument("--window_mode", type=str, default="adaptive", choices=["adaptive", "fixed"])
    ap.add_argument("--window_fixed", type=int, default=5)
    ap.add_argument("--window_frac", type=float, default=0.2)
    ap.add_argument("--min_window", type=int, default=3)

    ap.add_argument("--score_agg", type=str, default="topk_mean",
                    choices=["max", "mean", "median", "topk_mean"])
    ap.add_argument("--topk", type=int, default=7)

    ap.add_argument("--persist_mode", type=str, default="adaptive", choices=["adaptive", "fixed"])
    ap.add_argument("--persist_fixed", type=int, default=3)
    ap.add_argument("--persist_frac", type=float, default=0.2)
    ap.add_argument("--persist_max", type=int, default=8)

    ap.add_argument("--calib_score_mode", type=str, default="topk_mean",
                    choices=["max", "mean", "median", "topk_mean"],
                    help="Episode statistic used to calibrate thr_task from success episodes.")

    ap.add_argument("--calib_success_source", type=str, default="calib_only",
                    choices=["calib_only", "calib_plus_test_success"],
                    help="Success pool used to fit refs and calibrate threshold. "
                         "'calib_plus_test_success' stabilizes when calib is tiny/shifted.")

    ap.add_argument("--out_json", type=str, default="",
                    help="If empty, saves to --out_root/wm_progress_monitor_episode_calib_v2.json")

    ap.add_argument("--debug", type=int, default=1,
                    help="If 1, prints extra diagnostics.")

    return ap.parse_args()


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()

    print("====================================")
    print("[CFG] per-task thresholds (EPISODE-LEVEL CALIBRATION) + optional success-pool expansion")
    print("[CFG] tasks:", args.tasks)
    print(f"[CFG] alpha={args.alpha} bins={args.bins} var_floor={args.var_floor}")
    print(f"[CFG] window_mode={args.window_mode} fixed={args.window_fixed} frac={args.window_frac} min={args.min_window}")
    print(f"[CFG] score_agg={args.score_agg} topk={args.topk}")
    print(f"[CFG] persist_mode={args.persist_mode} fixed={args.persist_fixed} frac={args.persist_frac} max={args.persist_max}")
    print(f"[CFG] calib_score_mode={args.calib_score_mode}")
    print(f"[CFG] calib_success_source={args.calib_success_source}")

    per_task_results: List[Dict[str, Any]] = []

    pooled_y, pooled_pred, pooled_rank = [], [], []
    pooled_det, pooled_det_norm = [], []
    pooled_tn = pooled_fp = pooled_fn = pooled_tp = 0

    for task in args.tasks:
        root = os.path.join(args.data_root, task)

        print("\n====================================")
        print(f"[TASK] {task}")
        print(f"[ROOT] {root}")

        calib, test, y = load_task_root(root)
        n_fail = int((y == 1).sum())
        n_succ = int((y == 0).sum())
        print(f"[LOAD] calib_success_eps={len(calib)} test_eps={len(test)} failures={n_fail} successes={n_succ}")

        # Build success pool
        if args.calib_success_source == "calib_only":
            success_pool = list(calib)
            success_pool_note = "calib_only"
        else:
            test_succ = [feat for feat, yi in zip(test, y) if int(yi) == 0]
            success_pool = list(calib) + list(test_succ)
            success_pool_note = f"calib_plus_test_success (added {len(test_succ)} test successes)"

        if args.debug:
            print(f"[POOL] success_pool_eps={len(success_pool)} source={success_pool_note}")
            print(f"[LEN] calib_success: {_length_stats(list(calib))}")
            test_succ = [feat for feat, yi in zip(test, y) if int(yi) == 0]
            test_fail = [feat for feat, yi in zip(test, y) if int(yi) == 1]
            print(f"[LEN] test_success : {_length_stats(test_succ)}")
            print(f"[LEN] test_failure : {_length_stats(test_fail)}")

        # Fit refs on success pool
        refs, bin_counts = fit_bin_refs(success_pool, B=int(args.bins), var_floor=float(args.var_floor))
        if args.debug:
            print(f"[REF] bin_counts (success points per progress bin): {bin_counts.tolist()}")

        # Calibrate threshold on episode scores of success pool
        calib_ep_scores = []
        for feat in tqdm(success_pool, desc=f"{task} CALIB (episode scores)", leave=False):
            feat = np.asarray(feat)
            if feat.ndim != 2 or feat.shape[0] < 2:
                continue
            T = feat.shape[0]
            W = window_len(T, args.window_mode, args.window_fixed, args.window_frac, args.min_window)
            d = per_timestep_diag_mahal(feat, refs)
            s = sliding_window_scores(d, W=W, agg=args.score_agg, topk=args.topk)
            ep_score = agg_score(s, mode=args.calib_score_mode, topk=args.topk)
            calib_ep_scores.append(float(ep_score))

        if len(calib_ep_scores) == 0:
            raise RuntimeError(f"[{task}] No calibration episode scores produced.")

        calib_ep_scores = np.asarray(calib_ep_scores, dtype=np.float64)
        thr = conformal_upper_quantile(calib_ep_scores, float(args.alpha))

        print(f"[CALIB] thr_task={thr:.6f} (single threshold for {task}, alpha={args.alpha})")
        print(f"[CALIB] ep_score stats ({args.calib_score_mode} over s[t]): "
              f"mean={calib_ep_scores.mean():.4f} std={calib_ep_scores.std():.4f} "
              f"min={calib_ep_scores.min():.4f} max={calib_ep_scores.max():.4f}")

        # Test
        y_pred = []
        rank_scores = []
        det_times, det_times_norm = [], []
        test_success_ep_scores = []

        for feat, yi in tqdm(list(zip(test, y)), desc=f"{task} TEST", leave=False):
            feat = np.asarray(feat)
            if feat.ndim != 2 or feat.shape[0] < 2:
                # conservative default: no alarm
                y_pred.append(0)
                rank_scores.append(0.0)
                continue

            T = feat.shape[0]
            W = window_len(T, args.window_mode, args.window_fixed, args.window_frac, args.min_window)
            P = persist_len(T, args.persist_mode, args.persist_fixed, args.persist_frac, args.persist_max)

            d = per_timestep_diag_mahal(feat, refs)
            s = sliding_window_scores(d, W=W, agg=args.score_agg, topk=args.topk)

            # Ranking score for AUROC
            rank_ep = agg_score(s, mode="topk_mean", topk=min(args.topk, len(s)))
            rank_scores.append(float(rank_ep))

            # Online detection
            dt = first_persist_crossing(s, thr=thr, P=P)
            pred = 1 if dt is not None else 0
            y_pred.append(pred)

            # Record detection time only for TRUE POSITIVES (failures detected)
            if int(yi) == 1 and pred == 1 and dt is not None:
                det_times.append(int(dt))
                det_times_norm.append(float(dt / max(T - 1, 1)))

            if args.debug and int(yi) == 0:
                test_success_ep_scores.append(float(agg_score(s, mode=args.calib_score_mode, topk=args.topk)))

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

        # NEW: exact TWA for this task
        twa = compute_twa_from_counts_and_dts(tn, fp, fn, tp, det_times_norm)

        print(f"[TEST] Acc={acc:.4f} TPR={tpr:.4f} TNR={tnr:.4f} BalAcc={bal:.4f} AUROC(rank)={auroc:.4f} TWA={twa:.4f}")
        print(f"[TEST] Confusion: TN={tn} FP={fp} FN={fn} TP={tp}")
        print(f"[TEST] Detected failures: {len(det_times)} / {n_fail}  mean_dt={mean_dt:.2f}  mean_dt_norm={mean_dt_norm:.4f}")

        if args.debug and len(test_success_ep_scores) > 0:
            test_success_ep_scores = np.asarray(test_success_ep_scores, dtype=np.float64)
            frac_exceed = float(np.mean(test_success_ep_scores > thr))
            print(f"[DEBUG] fraction of TEST successes with ep_score({args.calib_score_mode}) > thr: {frac_exceed:.4f}")
            print(f"[DEBUG] TEST success ep_score stats: min={test_success_ep_scores.min():.4f} "
                  f"median={np.median(test_success_ep_scores):.4f} max={test_success_ep_scores.max():.4f}")

        per_task_results.append({
            "task": task,
            "root": root,
            "thr_task": float(thr),
            "alpha": float(args.alpha),
            "bins": int(args.bins),
            "var_floor": float(args.var_floor),
            "window": {
                "mode": args.window_mode,
                "fixed": int(args.window_fixed),
                "frac": float(args.window_frac),
                "min_window": int(args.min_window),
                "score_agg": args.score_agg,
                "topk": int(args.topk),
            },
            "persist": {
                "mode": args.persist_mode,
                "fixed": int(args.persist_fixed),
                "frac": float(args.persist_frac),
                "max": int(args.persist_max),
            },
            "calib_score_mode": args.calib_score_mode,
            "calib_success_source": args.calib_success_source,
            "bin_counts": bin_counts.tolist(),
            "calib_scores_stats": {
                "mean": float(calib_ep_scores.mean()),
                "std": float(calib_ep_scores.std()),
                "min": float(calib_ep_scores.min()),
                "max": float(calib_ep_scores.max()),
                "n": int(len(calib_ep_scores)),
            },
            "metrics": {
                "accuracy": float(acc),
                "TPR": float(tpr),
                "TNR": float(tnr),
                "bal_acc": float(bal),
                "AUROC_rank": float(auroc) if not np.isnan(auroc) else None,
                "TWA": float(twa) if not np.isnan(twa) else None,
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                "mean_detection_time": float(mean_dt),
                "mean_detection_time_norm": float(mean_dt_norm),
                "num_failures": int(n_fail),
                "num_detected_failures": int(len(det_times)),
                "tp_det_times_norm": det_times_norm,  # optional: keep for exact reproducibility
            },
        })

        pooled_y.append(y)
        pooled_pred.append(y_pred)
        pooled_rank.append(rank_scores)
        pooled_det.extend(det_times)
        pooled_det_norm.extend(det_times_norm)

        pooled_tn += int(tn)
        pooled_fp += int(fp)
        pooled_fn += int(fn)
        pooled_tp += int(tp)

    # pooled report
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

    # NEW: pooled exact TWA
    twa = compute_twa_from_counts_and_dts(int(tn), int(fp), int(fn), int(tp), pooled_det_norm)

    print("\n====================================")
    print("[POOLED REPORT] (each task used its OWN threshold)")
    print(f"  Accuracy      : {acc:.4f}")
    print(f"  TPR (failures): {tpr:.4f}")
    print(f"  TNR (success) : {tnr:.4f}")
    print(f"  BalAcc        : {bal:.4f}")
    print(f"  AUROC (rank)  : {auroc:.4f}")
    print(f"  TWA           : {twa:.4f}")
    print(f"  Confusion     : TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"  Mean detection time (steps): {mean_dt:.2f}")
    print(f"  Mean detection time (normalized): {mean_dt_norm:.4f}")

    out: Dict[str, Any] = {
        "method": "wm_progress_monitor_per_task_single_threshold_episode_calib_v2",
        "config": vars(args),
        "per_task": per_task_results,
        "pooled_report": {
            "accuracy": float(acc),
            "TPR": float(tpr),
            "TNR": float(tnr),
            "bal_acc": float(bal),
            "AUROC_rank": float(auroc) if not np.isnan(auroc) else None,
            "TWA": float(twa) if not np.isnan(twa) else None,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "mean_detection_time": float(mean_dt),
            "mean_detection_time_norm": float(mean_dt_norm),
        }
    }

    os.makedirs(args.out_root, exist_ok=True)
    out_json = args.out_json.strip()
    if out_json == "":
        out_json = os.path.join(args.out_root, "wm_progress_monitor_episode_calib_v2.json")

    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)

    print("\nSaved:", out_json)


if __name__ == "__main__":
    main()