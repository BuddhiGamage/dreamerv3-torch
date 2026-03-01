#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
wm_progress_monitor_per_task_threshold_episode_calib_v3_seeds.py

Runs your v3 progress monitor for multiple seeds (end-to-end randomness),
then reports mean ± std across seeds (like FIPER).

Expected feature layout:
  feats_root/<task>/seed_<s>/calib_success_feats.npy
  feats_root/<task>/seed_<s>/test_feats.npy
  feats_root/<task>/seed_<s>/test_labels.npy

This script:
  - Executes the exact v3 per-task logic for each seed separately
  - Aggregates mean±std for:
      accuracy, TPR, TNR, bal_acc, AUROC_rank, TWA,
      mean_detection_time, mean_detection_time_norm
  - Also aggregates pooled (across all tasks) per seed, then mean±std.

Protocol note:
  Use --calib_success_source calib_only for protocol-safe results.
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
# Conformal + metrics  (copied from v3)
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


def compute_twa_from_counts_and_dts(tn: int, fp: int, fn: int, tp: int, det_times_norm: List[float]) -> float:
    N = int(tn + fp + fn + tp)
    if N <= 0:
        return float("nan")
    k = min(int(tp), len(det_times_norm))
    tp_contrib = float(np.sum([1.0 - float(dt) for dt in det_times_norm[:k]]))
    return float((float(tn) + tp_contrib) / float(N))


# ============================================================
# Aggregations (v3)
# ============================================================
def topk_mean(x: np.ndarray, k: int) -> float:
    if len(x) == 0:
        return 0.0
    kk = min(int(k), len(x))
    return float(np.mean(np.sort(x)[-kk:]))


def agg_score(x: np.ndarray, mode: str, topk: int) -> float:
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
# Progress binning + diag Mahalanobis refs (v3)
# ============================================================
def normalized_time_bins(T: int, B: int) -> np.ndarray:
    if T <= 1:
        return np.zeros((T,), dtype=np.int64)
    tau = np.arange(T, dtype=np.float64) / float(T - 1)
    b = np.floor(tau * B).astype(np.int64)
    return np.clip(b, 0, B - 1)


@dataclass
class BinRef:
    mu: np.ndarray
    inv_var: np.ndarray


def compute_bin_counts_only(success_feats: List[np.ndarray], B: int) -> np.ndarray:
    counts = np.zeros((B,), dtype=np.int64)
    for feat in success_feats:
        feat = np.asarray(feat)
        if feat.ndim != 2 or feat.shape[0] < 1:
            continue
        bins = normalized_time_bins(int(feat.shape[0]), int(B))
        for b in bins:
            counts[int(b)] += 1
    return counts


def fit_bin_refs(success_feats: List[np.ndarray], B: int, var_floor: float) -> Tuple[List[BinRef], np.ndarray]:
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
# Window scores + persistence detection (v3)
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
    T = len(d)
    s = np.zeros((T,), dtype=np.float64)
    for t in range(T):
        a = max(0, t - W + 1)
        s[t] = agg_score(d[a:t + 1], mode=agg, topk=topk)
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
# Diagnostics helpers
# ============================================================
def _length_stats(arrs: List[np.ndarray]) -> Dict[str, Any]:
    if len(arrs) == 0:
        return {"n": 0}
    Ts = np.array([int(np.asarray(a).shape[0]) for a in arrs if np.asarray(a).ndim == 2], dtype=np.int64)
    if Ts.size == 0:
        return {"n": 0}
    return {"n": int(Ts.size), "min": int(Ts.min()), "median": int(np.median(Ts)),
            "max": int(Ts.max()), "mean": float(Ts.mean())}


# ============================================================
# Bins selection helpers (v3)
# ============================================================
def parse_bins_per_task(s: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    s = (s or "").strip()
    if not s:
        return out
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if "=" not in p:
            raise ValueError(f"Bad --bins_per_task entry '{p}'. Expected key=value.")
        k, v = p.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            raise ValueError(f"Bad --bins_per_task entry '{p}'. Empty task name.")
        out[k] = int(v)
    return out


def choose_bins_auto_from_calib_success(
    calib_success: List[np.ndarray],
    bins_min: int,
    bins_max: int,
    min_pts_per_bin: int,
) -> Tuple[int, np.ndarray]:
    bins_min = int(max(1, bins_min))
    bins_max = int(max(bins_min, bins_max))
    min_pts_per_bin = int(max(1, min_pts_per_bin))

    for B in range(bins_max, bins_min - 1, -1):
        counts = compute_bin_counts_only(calib_success, B=B)
        if counts.size == 0:
            continue
        if int(counts.min()) >= min_pts_per_bin:
            return int(B), counts

    counts = compute_bin_counts_only(calib_success, B=bins_min)
    return int(bins_min), counts


def resolve_bins_for_task(
    task: str,
    calib_success_list: List[np.ndarray],
    bins_arg: str,
    bins_per_task: Dict[str, int],
    bins_min: int,
    bins_max: int,
    min_pts_per_bin: int,
) -> Tuple[int, str, np.ndarray]:
    if task in bins_per_task:
        B = int(bins_per_task[task])
        counts = compute_bin_counts_only(calib_success_list, B=B)
        return B, f"override:{task}={B}", counts

    b = str(bins_arg).strip().lower()
    if b == "auto":
        B, counts = choose_bins_auto_from_calib_success(
            calib_success=calib_success_list,
            bins_min=bins_min,
            bins_max=bins_max,
            min_pts_per_bin=min_pts_per_bin,
        )
        return B, f"auto(min={bins_min},max={bins_max},min_pts_per_bin={min_pts_per_bin})", counts

    try:
        B = int(bins_arg)
    except Exception as e:
        raise ValueError(f"--bins must be an int or 'auto', got: {bins_arg}") from e
    counts = compute_bin_counts_only(calib_success_list, B=B)
    return B, f"fixed:{B}", counts


# ============================================================
# I/O
# ============================================================
def load_task_seed_root(feats_root: str, task: str, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    root = os.path.join(feats_root, task, f"seed_{seed}")
    calib = np.load(os.path.join(root, "calib_success_feats.npy"), allow_pickle=True)
    test = np.load(os.path.join(root, "test_feats.npy"), allow_pickle=True)
    y = np.load(os.path.join(root, "test_labels.npy")).astype(np.int64)
    return calib, test, y, root


# ============================================================
# Core eval: one task, one seed (mirrors v3)
# ============================================================
def eval_task_seed(args, task: str, seed: int) -> Dict[str, Any]:
    calib, test, y, root = load_task_seed_root(args.data_root, task, seed)

    n_fail = int((y == 1).sum())
    n_succ = int((y == 0).sum())

    # success pool
    if args.calib_success_source == "calib_only":
        success_pool = list(calib)
        success_pool_note = "calib_only"
    else:
        test_succ = [feat for feat, yi in zip(test, y) if int(yi) == 0]
        success_pool = list(calib) + list(test_succ)
        success_pool_note = f"calib_plus_test_success (added {len(test_succ)} test successes)"

    # bins (AUTO uses calib successes ONLY)
    B_task, bins_note, bin_counts_est = resolve_bins_for_task(
        task=task,
        calib_success_list=list(calib),
        bins_arg=args.bins,
        bins_per_task=args._bins_per_task_dict,
        bins_min=args.bins_min,
        bins_max=args.bins_max,
        min_pts_per_bin=args.min_pts_per_bin,
    )

    # refs
    refs, bin_counts = fit_bin_refs(success_pool, B=int(B_task), var_floor=float(args.var_floor))

    # calibrate threshold on episode scores of success pool
    calib_ep_scores = []
    for feat in success_pool:
        feat = np.asarray(feat)
        if feat.ndim != 2 or feat.shape[0] < 2:
            continue
        T = feat.shape[0]
        W = window_len(T, args.window_mode, args.window_fixed, args.window_frac, args.min_window)
        d = per_timestep_diag_mahal(feat, refs)
        s = sliding_window_scores(d, W=W, agg=args.score_agg, topk=args.topk)
        calib_ep_scores.append(float(agg_score(s, mode=args.calib_score_mode, topk=args.topk)))

    if len(calib_ep_scores) == 0:
        raise RuntimeError(f"[{task}][seed={seed}] No calibration episode scores produced.")

    calib_ep_scores = np.asarray(calib_ep_scores, dtype=np.float64)
    thr = conformal_upper_quantile(calib_ep_scores, float(args.alpha))

    # test
    y_pred = []
    rank_scores = []
    det_times, det_times_norm = [], []

    test_success_ep_scores = []

    for feat, yi in zip(test, y):
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

        # AUROC ranking score
        rank_ep = agg_score(s, mode="topk_mean", topk=min(args.topk, len(s)))
        rank_scores.append(float(rank_ep))

        dt = first_persist_crossing(s, thr=thr, P=P)
        pred = 1 if dt is not None else 0
        y_pred.append(pred)

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
    twa = compute_twa_from_counts_and_dts(int(tn), int(fp), int(fn), int(tp), det_times_norm)

    out = {
        "task": task,
        "seed": int(seed),
        "root": root,
        "load": {"calib_success_eps": int(len(calib)), "test_eps": int(len(test)), "failures": n_fail, "successes": n_succ},
        "pool": {"success_pool_eps": int(len(success_pool)), "success_pool_note": success_pool_note},
        "bins": {"bins_used": int(B_task), "bins_note": bins_note,
                 "calib_success_bin_counts_est": bin_counts_est.tolist(),
                 "bin_counts_fit": bin_counts.tolist()},
        "calib": {"thr_task": float(thr),
                  "calib_scores_stats": {"mean": float(calib_ep_scores.mean()),
                                        "std": float(calib_ep_scores.std()),
                                        "min": float(calib_ep_scores.min()),
                                        "max": float(calib_ep_scores.max()),
                                        "n": int(len(calib_ep_scores))}},
        "metrics": {"accuracy": float(acc), "TPR": float(tpr), "TNR": float(tnr), "bal_acc": float(bal),
                    "AUROC_rank": float(auroc) if not np.isnan(auroc) else None,
                    "TWA": float(twa) if not np.isnan(twa) else None,
                    "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                    "mean_detection_time": float(mean_dt),
                    "mean_detection_time_norm": float(mean_dt_norm),
                    "num_failures": int(n_fail),
                    "num_detected_failures": int(len(det_times)),
                    "tp_det_times_norm": det_times_norm},
    }

    if args.debug:
        out["debug"] = {
            "len_stats": {
                "calib_success": _length_stats(list(calib)),
                "test_success": _length_stats([feat for feat, yi in zip(test, y) if int(yi) == 0]),
                "test_failure": _length_stats([feat for feat, yi in zip(test, y) if int(yi) == 1]),
            },
            "fraction_test_success_exceed_thr": (
                float(np.mean(np.asarray(test_success_ep_scores) > thr)) if len(test_success_ep_scores) > 0 else None
            ),
        }

    return out


# ============================================================
# Aggregate helpers
# ============================================================
def mean_std(vals: List[float]) -> Dict[str, Any]:
    a = np.asarray(vals, dtype=np.float64)
    return {"mean": float(np.nanmean(a)), "std": float(np.nanstd(a, ddof=0)), "n": int(a.size)}


def get_metric(seed_task_rows: List[Dict[str, Any]], name: str) -> List[float]:
    out = []
    for r in seed_task_rows:
        v = r["metrics"].get(name, None)
        out.append(float(v) if v is not None else float("nan"))
    return out


# ============================================================
# Args
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser()

    # NOTE: data_root now points to feats_root, not the old single-seed root
    ap.add_argument("--data_root", type=str, required=True,
                    help="Features root: contains <task>/seed_<s>/{calib_success_feats.npy,test_feats.npy,test_labels.npy}")
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--tasks", type=str, nargs="*", default=["pretzel", "push_chair", "push_t", "sorting", "stacking"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])

    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--bins", type=str, default="10")
    ap.add_argument("--bins_min", type=int, default=4)
    ap.add_argument("--bins_max", type=int, default=10)
    ap.add_argument("--min_pts_per_bin", type=int, default=20)
    ap.add_argument("--bins_per_task", type=str, default="")

    ap.add_argument("--var_floor", type=float, default=1e-6)

    ap.add_argument("--window_mode", type=str, default="adaptive", choices=["adaptive", "fixed"])
    ap.add_argument("--window_fixed", type=int, default=5)
    ap.add_argument("--window_frac", type=float, default=0.2)
    ap.add_argument("--min_window", type=int, default=3)

    ap.add_argument("--score_agg", type=str, default="topk_mean", choices=["max", "mean", "median", "topk_mean"])
    ap.add_argument("--topk", type=int, default=7)

    ap.add_argument("--persist_mode", type=str, default="adaptive", choices=["adaptive", "fixed"])
    ap.add_argument("--persist_fixed", type=int, default=3)
    ap.add_argument("--persist_frac", type=float, default=0.25)
    ap.add_argument("--persist_max", type=int, default=8)

    ap.add_argument("--calib_score_mode", type=str, default="topk_mean", choices=["max", "mean", "median", "topk_mean"])
    ap.add_argument("--calib_success_source", type=str, default="calib_only",
                    choices=["calib_only", "calib_plus_test_success"])

    ap.add_argument("--out_json", type=str, default="wm_progress_monitor_episode_calib_v3_seeds.json")
    ap.add_argument("--debug", type=int, default=1)

    return ap.parse_args()


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()
    args._bins_per_task_dict = parse_bins_per_task(args.bins_per_task)

    print("====================================")
    print("[CFG] v3 monitor aggregated across seeds")
    print("  tasks:", args.tasks)
    print("  seeds:", args.seeds)
    print(f"  alpha={args.alpha} bins={args.bins} var_floor={args.var_floor}")
    if str(args.bins).strip().lower() == "auto" or args._bins_per_task_dict:
        print(f"  bins_auto: min={args.bins_min} max={args.bins_max} min_pts_per_bin={args.min_pts_per_bin}")
        if args._bins_per_task_dict:
            print("  bins_per_task:", args._bins_per_task_dict)
    print(f"  window: {args.window_mode} fixed={args.window_fixed} frac={args.window_frac} min={args.min_window}")
    print(f"  score_agg={args.score_agg} topk={args.topk}")
    print(f"  persist: {args.persist_mode} fixed={args.persist_fixed} frac={args.persist_frac} max={args.persist_max}")
    print(f"  calib_score_mode={args.calib_score_mode} calib_success_source={args.calib_success_source}")

    all_seed_runs: List[Dict[str, Any]] = []
    per_task_agg: Dict[str, Any] = {}
    pooled_per_seed: List[Dict[str, Any]] = []

    os.makedirs(args.out_root, exist_ok=True)

    # Run per seed
    for seed in args.seeds:
        print("\n====================================")
        print(f"[SEED] {seed}")

        seed_task_rows = []
        pooled_y, pooled_pred, pooled_rank = [], [], []
        pooled_det_norm = []

        for task in args.tasks:
            print(f"  [TASK] {task}")
            r = eval_task_seed(args, task=task, seed=int(seed))
            seed_task_rows.append(r)

            # gather pooled
            # we need y, y_pred, rank_scores, but v3 stores only aggregate;
            # so we compute pooled from confusion + det_norm: use episode-wise pooling compatible with TWA definition.
            pooled_det_norm.extend(r["metrics"]["tp_det_times_norm"])

        # pooled across tasks for this seed (episode-wise):
        # We approximate pooled using summed confusion counts (exact), and det_norm list (exact for TP episodes).
        tn = sum(rr["metrics"]["tn"] for rr in seed_task_rows)
        fp = sum(rr["metrics"]["fp"] for rr in seed_task_rows)
        fn = sum(rr["metrics"]["fn"] for rr in seed_task_rows)
        tp = sum(rr["metrics"]["tp"] for rr in seed_task_rows)

        # For pooled acc/bal, we need pooled_y/pred; confusion counts suffice:
        N = tn + fp + fn + tp
        acc = (tn + tp) / max(N, 1)
        tpr = tp / max(tp + fn, 1)
        tnr = tn / max(tn + fp, 1)
        bal = 0.5 * (tpr + tnr)

        twa = compute_twa_from_counts_and_dts(int(tn), int(fp), int(fn), int(tp), pooled_det_norm)

        pooled_seed = {
            "seed": int(seed),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "metrics": {
                "accuracy": float(acc),
                "TPR": float(tpr),
                "TNR": float(tnr),
                "bal_acc": float(bal),
                "TWA": float(twa) if not np.isnan(twa) else None,
            }
        }

        pooled_per_seed.append(pooled_seed)
        all_seed_runs.append({"seed": int(seed), "per_task": seed_task_rows, "pooled": pooled_seed})

    # Aggregate per task across seeds
    for task in args.tasks:
        rows = []
        for sr in all_seed_runs:
            # find that task
            for tr in sr["per_task"]:
                if tr["task"] == task:
                    rows.append(tr)
                    break

        summary = {
            "accuracy": mean_std(get_metric(rows, "accuracy")),
            "TPR": mean_std(get_metric(rows, "TPR")),
            "TNR": mean_std(get_metric(rows, "TNR")),
            "bal_acc": mean_std(get_metric(rows, "bal_acc")),
            "AUROC_rank": mean_std([float(r["metrics"]["AUROC_rank"]) if r["metrics"]["AUROC_rank"] is not None else float("nan") for r in rows]),
            "TWA": mean_std([float(r["metrics"]["TWA"]) if r["metrics"]["TWA"] is not None else float("nan") for r in rows]),
            "mean_detection_time": mean_std(get_metric(rows, "mean_detection_time")),
            "mean_detection_time_norm": mean_std(get_metric(rows, "mean_detection_time_norm")),
        }

        per_task_agg[task] = {"summary_mean_std": summary}

    # Aggregate pooled across seeds
    pooled_summary = {
        "accuracy": mean_std([p["metrics"]["accuracy"] for p in pooled_per_seed]),
        "TPR": mean_std([p["metrics"]["TPR"] for p in pooled_per_seed]),
        "TNR": mean_std([p["metrics"]["TNR"] for p in pooled_per_seed]),
        "bal_acc": mean_std([p["metrics"]["bal_acc"] for p in pooled_per_seed]),
        "TWA": mean_std([float(p["metrics"]["TWA"]) if p["metrics"]["TWA"] is not None else float("nan") for p in pooled_per_seed]),
    }

    out = {
        "method": "wm_progress_monitor_per_task_single_threshold_episode_calib_v3_seeds",
        "config": vars(args),
        "bins_per_task_overrides": args._bins_per_task_dict,
        "per_task_aggregate": per_task_agg,
        "pooled_aggregate": pooled_summary,
        "per_seed_runs": all_seed_runs,
    }

    out_json = args.out_json.strip()
    if out_json == "":
        out_json = "wm_progress_monitor_episode_calib_v3_seeds.json"
    out_path = os.path.join(args.out_root, out_json)

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("\n====================================")
    print("[POOLED mean±std across seeds]")
    for k, v in pooled_summary.items():
        print(f"  {k:>10s}: {v['mean']:.4f} ± {v['std']:.4f} (n={v['n']})")

    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()