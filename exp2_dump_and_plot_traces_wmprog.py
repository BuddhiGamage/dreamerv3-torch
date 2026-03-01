#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
exp2_dump_and_plot_traces_wmprog.py

Single-script Experiment 2: Qualitative score traces + alarms over time.

What it does:
  - Loads per-task features:
      calib_success_feats.npy (success-only episodes, object array of (T_i, D))
      test_feats.npy          (object array of (T_i, D))
      test_labels.npy         (1=failure, 0=success)
  - Fits progress-conditioned diagonal Gaussian refs from success pool
  - Calibrates a single per-task threshold via EPISODE-LEVEL conformal prediction (success-only)
  - Selects up to K success episodes and K failure episodes from TEST
  - Computes per-timestep Mahalanobis deviations d[t], window scores s[t], detection time (persistence)
  - Saves traces as .npz and plots as .png (one file per episode)

This script is independent of your main v3 monitor, but it matches its logic closely.
Use this to generate Figure X traces for the paper.

Example:
  python exp2_dump_and_plot_traces_wmprog.py \
    --data_root /home/s447658/projects/dreamer_fiper_feats_all5 \
    --out_dir   /home/s447658/projects/dreamer_fiper_results/exp2_traces \
    --tasks pretzel push_t sorting stacking push_chair\
    --calib_success_source calib_only \
    --bins auto --bins_min 4 --bins_max 10 --min_pts_per_bin 20 \
    --alpha 0.10 --var_floor 1e-4 \
    --window_mode adaptive --window_frac 0.2 --min_window 5 \
    --score_agg topk_mean --topk 7 \
    --persist_mode adaptive --persist_frac 0.25 --persist_max 8 \
    --calib_score_mode topk_mean \
    --k_succ 5 --k_fail 5

Outputs:
  out_dir/
    pretzel/
      traces/*.npz
      plots/*.png
      summary.json
    push_t/...
"""

import os
import json
import argparse
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
from tqdm import tqdm

# matplotlib only needed for plotting
import matplotlib.pyplot as plt

EPS = 1e-8


# ============================================================
# Conformal
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


# ============================================================
# Aggregations
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
# Progress binning + diag Mahalanobis refs
# ============================================================
def normalized_time_bins(T: int, B: int) -> np.ndarray:
    """Assign each timestep to a bin based on normalized time tau=t/(T-1)."""
    if T <= 1:
        return np.zeros((T,), dtype=np.int64)
    tau = np.arange(T, dtype=np.float64) / float(T - 1)
    b = np.floor(tau * B).astype(np.int64)
    return np.clip(b, 0, B - 1)


@dataclass
class BinRef:
    mu: np.ndarray       # (D,)
    inv_var: np.ndarray  # (D,)


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

    # global fallback
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
# IO
# ============================================================
def load_task_root(root: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    calib = np.load(os.path.join(root, "calib_success_feats.npy"), allow_pickle=True)
    test = np.load(os.path.join(root, "test_feats.npy"), allow_pickle=True)
    y = np.load(os.path.join(root, "test_labels.npy")).astype(np.int64)
    return calib, test, y


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def plot_trace(task: str, out_png: str, s: np.ndarray, thr: float, dt: Optional[int], label: str):
    plt.figure()
    x = np.arange(len(s))
    plt.plot(x, s, label="score s[t]")
    plt.axhline(thr, linestyle="--", label="threshold")
    if dt is not None:
        plt.axvline(dt, linestyle=":", label=f"detect @ {dt}")
    plt.xlabel("timestep")
    plt.ylabel("monitor score")
    plt.title(f"{task} | {label}")
    plt.legend()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


# ============================================================
# Args
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_root", type=str, required=True,
                    help="Root containing task folders with calib_success_feats.npy/test_feats.npy/test_labels.npy")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Output dir for traces+plots (will create subfolders per task)")
    ap.add_argument("--tasks", type=str, nargs="*", default=["pretzel", "push_t", "sorting"])

    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--bins", type=str, default="auto",
                    help="Either an integer (fixed bins) or 'auto' (chosen per task from calib successes only).")
    ap.add_argument("--bins_min", type=int, default=4)
    ap.add_argument("--bins_max", type=int, default=10)
    ap.add_argument("--min_pts_per_bin", type=int, default=20)

    ap.add_argument("--var_floor", type=float, default=1e-4)

    ap.add_argument("--window_mode", type=str, default="adaptive", choices=["adaptive", "fixed"])
    ap.add_argument("--window_fixed", type=int, default=5)
    ap.add_argument("--window_frac", type=float, default=0.2)
    ap.add_argument("--min_window", type=int, default=5)

    ap.add_argument("--score_agg", type=str, default="topk_mean",
                    choices=["max", "mean", "median", "topk_mean"])
    ap.add_argument("--topk", type=int, default=7)

    ap.add_argument("--persist_mode", type=str, default="adaptive", choices=["adaptive", "fixed"])
    ap.add_argument("--persist_fixed", type=int, default=3)
    ap.add_argument("--persist_frac", type=float, default=0.25)
    ap.add_argument("--persist_max", type=int, default=8)

    ap.add_argument("--calib_score_mode", type=str, default="topk_mean",
                    choices=["max", "mean", "median", "topk_mean"],
                    help="Episode statistic used to calibrate thr_task from success episodes.")

    ap.add_argument("--calib_success_source", type=str, default="calib_only",
                    choices=["calib_only", "calib_plus_test_success"],
                    help="Success pool used to fit refs and calibrate threshold. "
                         "NOTE: 'calib_plus_test_success' uses test successes -> not protocol-safe.")

    ap.add_argument("--k_succ", type=int, default=1, help="#test success episodes to plot per task")
    ap.add_argument("--k_fail", type=int, default=1, help="#test failure episodes to plot per task")

    ap.add_argument("--debug", type=int, default=1)

    return ap.parse_args()


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    all_summary: Dict[str, Any] = {
        "config": vars(args),
        "tasks": {},
    }

    for task in args.tasks:
        task_root = os.path.join(args.data_root, task)
        print("\n====================================")
        print("[TASK]", task)
        print("[ROOT]", task_root)

        calib, test, y = load_task_root(task_root)
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

        # Choose bins
        b = str(args.bins).strip().lower()
        if b == "auto":
            B, bin_counts_est = choose_bins_auto_from_calib_success(
                calib_success=list(calib),
                bins_min=args.bins_min,
                bins_max=args.bins_max,
                min_pts_per_bin=args.min_pts_per_bin,
            )
            bins_note = f"auto(min={args.bins_min},max={args.bins_max},min_pts_per_bin={args.min_pts_per_bin})"
        else:
            B = int(args.bins)
            bin_counts_est = compute_bin_counts_only(list(calib), B=B)
            bins_note = f"fixed:{B}"

        print(f"[BINS] bins_task={B} ({bins_note}) calib_success_bin_counts_est={bin_counts_est.tolist()}")
        if args.debug:
            print(f"[POOL] success_pool_eps={len(success_pool)} source={success_pool_note}")

        # Fit refs + bin counts from success pool
        refs, bin_counts = fit_bin_refs(success_pool, B=int(B), var_floor=float(args.var_floor))
        if args.debug:
            print(f"[REF] bin_counts (success points per progress bin): {bin_counts.tolist()}")

        # Calibrate threshold on success episode scores
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

        print(f"[CALIB] thr_task={thr:.6f} (alpha={args.alpha})")
        print(f"[CALIB] ep_score stats ({args.calib_score_mode} over s[t]): "
              f"mean={calib_ep_scores.mean():.4f} std={calib_ep_scores.std():.4f} "
              f"min={calib_ep_scores.min():.4f} max={calib_ep_scores.max():.4f}")

        # Pick episodes from TEST to plot
        idx_succ = [i for i, yi in enumerate(y) if int(yi) == 0][:max(0, int(args.k_succ))]
        idx_fail = [i for i, yi in enumerate(y) if int(yi) == 1][:max(0, int(args.k_fail))]

        out_task = os.path.join(args.out_dir, task)
        out_traces = os.path.join(out_task, "traces")
        out_plots = os.path.join(out_task, "plots")
        ensure_dir(out_traces)
        ensure_dir(out_plots)

        plotted = []

        def process_and_save(i: int, label: str):
            feat = np.asarray(test[i])
            if feat.ndim != 2 or feat.shape[0] < 2:
                return None
            yi = int(y[i])
            T = feat.shape[0]
            W = window_len(T, args.window_mode, args.window_fixed, args.window_frac, args.min_window)
            P = persist_len(T, args.persist_mode, args.persist_fixed, args.persist_frac, args.persist_max)

            d = per_timestep_diag_mahal(feat, refs)
            s = sliding_window_scores(d, W=W, agg=args.score_agg, topk=args.topk)

            dt = first_persist_crossing(s, thr=thr, P=P)
            pred = 1 if dt is not None else 0

            # save trace
            tag = f"{label}_{i}"
            npz_path = os.path.join(out_traces, f"{task}_{tag}.npz")
            np.savez_compressed(
                npz_path,
                task=task,
                index=int(i),
                y=int(yi),
                pred=int(pred),
                T=int(T),
                W=int(W),
                P=int(P),
                thr=float(thr),
                dt=(-1 if dt is None else int(dt)),
                d=d.astype(np.float32),
                s=s.astype(np.float32),
            )

            # plot
            png_path = os.path.join(out_plots, f"{task}_{tag}.png")
            plot_trace(
                task=task,
                out_png=png_path,
                s=s,
                thr=float(thr),
                dt=dt,
                label=("FAIL" if yi == 1 else "SUCCESS"),
            )
            return {"npz": npz_path, "png": png_path, "i": int(i), "y": int(yi), "dt": (None if dt is None else int(dt))}

        for i in idx_succ:
            res = process_and_save(i, label="succ")
            if res:
                plotted.append(res)

        for i in idx_fail:
            res = process_and_save(i, label="fail")
            if res:
                plotted.append(res)

        # Save per-task summary
        summary = {
            "task": task,
            "thr_task": float(thr),
            "alpha": float(args.alpha),
            "bins_used": int(B),
            "bins_note": bins_note,
            "calib_success_source": args.calib_success_source,
            "calib_scores_stats": {
                "mean": float(calib_ep_scores.mean()),
                "std": float(calib_ep_scores.std()),
                "min": float(calib_ep_scores.min()),
                "max": float(calib_ep_scores.max()),
                "n": int(len(calib_ep_scores)),
            },
            "bin_counts_success_pool": bin_counts.tolist(),
            "bin_counts_est_calib_only": bin_counts_est.tolist(),
            "plotted": plotted,
        }

        with open(os.path.join(out_task, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        all_summary["tasks"][task] = summary
        print(f"[SAVE] {task} -> {out_task}")
        for p in plotted:
            print("  plot:", p["png"])

    with open(os.path.join(args.out_dir, "summary_all_tasks.json"), "w") as f:
        json.dump(all_summary, f, indent=2)

    print("\nAll done.")
    print("Saved:", os.path.join(args.out_dir, "summary_all_tasks.json"))


if __name__ == "__main__":
    main()