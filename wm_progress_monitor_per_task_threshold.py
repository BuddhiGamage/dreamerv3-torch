#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
World Model Progress Monitor — per-task threshold version (ARGPARSE FRIENDLY)

✔ One threshold per task
✔ No pooled threshold across tasks
✔ Compatible with your training-style args:
      --data_root
      --out_root
      --tasks

Expected structure:

{data_root}/{task}/
    calib_success_feats.npy
    test_feats.npy
    test_labels.npy

python wm_progress_monitor_per_task_threshold.py \
  --data_root /home/s447658/projects/dreamer_fiper_feats_all5 \
  --tasks pretzel push_t stacking sorting push_chair\
  --alpha 0.10 --bins 20 \
  --window_mode adaptive --window_frac 0.2 --min_window 3 \
  --score_agg topk_mean --topk 7 \
  --persist_mode adaptive --persist_frac 0.2 --persist_max 8
"""

import os
import json
import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

EPS = 1e-8


# ============================================================
# Conformal
# ============================================================
def conformal_upper_quantile(scores: np.ndarray, alpha: float) -> float:
    s = np.sort(np.asarray(scores, dtype=float))
    n = s.shape[0]
    r = int(np.ceil((n + 1) * (1 - alpha)))
    r = min(max(1, r), n)
    return float(s[r - 1])


def safe_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


# ============================================================
# Utils
# ============================================================
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
    raise ValueError(mode)


def normalized_time_bins(T: int, B: int) -> np.ndarray:
    if T <= 1:
        return np.zeros((T,), dtype=np.int64)
    tau = np.arange(T, dtype=np.float64) / float(T - 1)
    b = np.floor(tau * B).astype(np.int64)
    return np.clip(b, 0, B - 1)


# ============================================================
# Bin reference (diag Mahalanobis)
# ============================================================
@dataclass
class BinRef:
    mu: np.ndarray
    inv_var: np.ndarray


def fit_bin_refs(success_feats: List[np.ndarray], B: int, var_floor: float) -> List[BinRef]:
    per_bin: List[List[np.ndarray]] = [[] for _ in range(B)]
    D = None

    for feat in tqdm(success_feats, desc="FIT refs (success)", leave=False):
        feat = np.asarray(feat, dtype=np.float64)
        if feat.ndim != 2:
            continue

        if D is None:
            D = feat.shape[1]

        bins = normalized_time_bins(feat.shape[0], B)
        for t in range(feat.shape[0]):
            per_bin[bins[t]].append(feat[t])

    if D is None:
        raise RuntimeError("No success features found.")

    # global fallback (within task)
    all_pts = [np.stack(b) for b in per_bin if len(b) > 0]
    Xg = np.concatenate(all_pts, axis=0)

    mu_g = np.mean(Xg, axis=0)
    var_g = np.maximum(np.var(Xg, axis=0), var_floor)
    inv_var_g = 1.0 / var_g

    refs: List[BinRef] = []
    for b in range(B):
        if len(per_bin[b]) == 0:
            refs.append(BinRef(mu_g, inv_var_g))
            continue

        Xb = np.stack(per_bin[b], axis=0)
        mu = np.mean(Xb, axis=0)
        var = np.maximum(np.var(Xb, axis=0), var_floor)
        refs.append(BinRef(mu, 1.0 / var))

    return refs


def per_timestep_diag_mahal(feat: np.ndarray, refs: List[BinRef]) -> np.ndarray:
    T, _ = feat.shape
    B = len(refs)
    bins = normalized_time_bins(T, B)

    d = np.zeros((T,), dtype=np.float64)
    for t in range(T):
        r = refs[bins[t]]
        diff = feat[t] - r.mu
        d[t] = np.sum((diff * diff) * r.inv_var)
    return d


# ============================================================
# Window & persistence
# ============================================================
def window_len(T: int, mode: str, fixed: int, frac: float, min_w: int) -> int:
    if mode == "fixed":
        return max(1, fixed)
    return max(min_w, int(round(frac * T)))


def persist_len(T: int, mode: str, fixed: int, frac: float, max_p: int) -> int:
    if mode == "fixed":
        return max(1, fixed)
    return min(max_p, max(1, int(round(frac * T))))


def sliding_window_scores(d: np.ndarray, W: int, agg: str, topk: int) -> np.ndarray:
    T = len(d)
    s = np.zeros((T,), dtype=np.float64)
    for t in range(T):
        a = max(0, t - W + 1)
        s[t] = agg_score(d[a:t+1], agg, topk)
    return s


def first_persist_crossing(s: np.ndarray, thr: float, P: int) -> Optional[int]:
    run = 0
    for t in range(len(s)):
        run = run + 1 if s[t] > thr else 0
        if run >= P:
            return int(t - P + 1)
    return None


# ============================================================
# Data loading
# ============================================================
def load_task_root(root: str):
    calib = np.load(os.path.join(root, "calib_success_feats.npy"), allow_pickle=True)
    test = np.load(os.path.join(root, "test_feats.npy"), allow_pickle=True)
    labels = np.load(os.path.join(root, "test_labels.npy")).astype(np.int64)
    return calib, test, labels


# ============================================================
# Args
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_root", type=str,
                    default="/home/s447658/projects/dreamer_fiper_feats_all5")
    ap.add_argument("--out_root", type=str,
                    default="/home/s447658/projects/dreamer_fiper_offline/all5_tasks")

    ap.add_argument("--tasks", type=str, nargs="*",
                    default=["sorting", "push_chair"])

    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--bins", type=int, default=20)
    ap.add_argument("--var_floor", type=float, default=1e-6)

    ap.add_argument("--window_mode", type=str, default="adaptive")
    ap.add_argument("--window_frac", type=float, default=0.2)
    ap.add_argument("--min_window", type=int, default=3)

    ap.add_argument("--score_agg", type=str, default="topk_mean")
    ap.add_argument("--topk", type=int, default=7)

    ap.add_argument("--persist_mode", type=str, default="adaptive")
    ap.add_argument("--persist_frac", type=float, default=0.2)
    ap.add_argument("--persist_max", type=int, default=8)

    return ap.parse_args()


# ============================================================
# MAIN
# ============================================================
def main():
    args = parse_args()

    print("====================================")
    print("[CFG] per-task thresholds via task list")
    print("[CFG] tasks:", args.tasks)

    results = []

    for task in args.tasks:
        root = os.path.join(args.data_root, task)

        print("\n====================================")
        print(f"[TASK] {task}")
        print(f"[ROOT] {root}")

        calib, test, y = load_task_root(root)

        # ---------- fit refs ----------
        refs = fit_bin_refs(list(calib), args.bins, args.var_floor)

        # ---------- build success window scores ----------
        # calib_scores = []
        # for feat in tqdm(calib, desc=f"{task} CALIB", leave=False):
        #     T = feat.shape[0]
        #     W = window_len(T, args.window_mode, 5, args.window_frac, args.min_window)
        #     d = per_timestep_diag_mahal(feat, refs)
        #     s = sliding_window_scores(d, W, args.score_agg, args.topk)
        #     calib_scores.append(s)

        # calib_scores = np.concatenate(calib_scores)

        # thr = conformal_upper_quantile(calib_scores, args.alpha)
        # print(f"[CALIB] thr_task={thr:.6f}")

        calib_ep_scores = []
        for feat in calib:
            T = feat.shape[0]
            W = window_len(T, args.window_mode, 5, args.window_frac, args.min_window)

            d = per_timestep_diag_mahal(feat, refs)
            s = sliding_window_scores(d, W=W, agg=args.score_agg, topk=args.topk)

            # IMPORTANT: calibrate on SAME statistic you use to judge episodes
            ep_score = float(np.max(s))          # option A (matches your debug)
            # ep_score = topk_mean(s, topk)      # option B (less spiky, often better)
            calib_ep_scores.append(ep_score)

        calib_ep_scores = np.asarray(calib_ep_scores, dtype=np.float64)
        thr = conformal_upper_quantile(calib_ep_scores, args.alpha)

        print(f"[CALIB] thr_task={thr:.6f} (episode-level)")
        print(f"[CALIB] ep_score stats: mean={calib_ep_scores.mean():.4f} std={calib_ep_scores.std():.4f} "
            f"min={calib_ep_scores.min():.4f} max={calib_ep_scores.max():.4f}")

        # ---------- test ----------
        y_pred = []
        rank_scores = []
        det_times = []

        for feat, yi in tqdm(list(zip(test, y)), desc=f"{task} TEST", leave=False):
            T = feat.shape[0]
            W = window_len(T, args.window_mode, 5, args.window_frac, args.min_window)
            P = persist_len(T, args.persist_mode, 3, args.persist_frac, args.persist_max)

            d = per_timestep_diag_mahal(feat, refs)
            s = sliding_window_scores(d, W, args.score_agg, args.topk)

            rank_scores.append(agg_score(s, "topk_mean", args.topk))

            dt = first_persist_crossing(s, thr, P)
            pred = 1 if dt is not None else 0
            y_pred.append(pred)

            if yi == 1 and pred == 1:
                det_times.append(dt)

        y_pred = np.asarray(y_pred)
        rank_scores = np.asarray(rank_scores)

        acc = accuracy_score(y, y_pred)
        tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
        tpr = tp / (tp + fn + EPS)
        tnr = tn / (tn + fp + EPS)
        bal = 0.5 * (tpr + tnr)
        auroc = safe_auroc(y, rank_scores)

        print(f"[TEST] Acc={acc:.4f} TPR={tpr:.4f} TNR={tnr:.4f} BalAcc={bal:.4f} AUROC={auroc:.4f}")

        results.append({
            "task": task,
            "threshold": float(thr),
            "metrics": {
                "accuracy": float(acc),
                "TPR": float(tpr),
                "TNR": float(tnr),
                "bal_acc": float(bal),
                "AUROC": float(auroc),
            }
        })

    # ---------- save ----------
    os.makedirs(args.out_root, exist_ok=True)
    out_path = os.path.join(args.out_root, "wm_progress_monitor_per_task_v2.json")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved:", out_path)

    # DEBUG: success episode scores
    succ_scores = []
    for feat, yi in zip(test, y):
        if yi == 0:
            T = feat.shape[0]
            W = window_len(T, args.window_mode, 5, args.window_frac, args.min_window)
            d = per_timestep_diag_mahal(feat, refs)
            s = sliding_window_scores(d, W, args.score_agg, args.topk)
            succ_scores.append(np.max(s))

    succ_scores = np.array(succ_scores)

    print("[DEBUG] push_chair success episode scores:")
    print("  min :", succ_scores.min())
    print("  median :", np.median(succ_scores))
    print("  max :", succ_scores.max())
    print("  thr :", thr)


if __name__ == "__main__":
    main()