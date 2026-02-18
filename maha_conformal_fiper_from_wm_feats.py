# maha_conformal_fiper_from_wm_feats_per_timestep.py
# - NO running-max
# - Fits global Gaussian (scaler + LedoitWolf) on ALL success timesteps
# - Builds per-timestep conformal thresholds q_t from success-only calib distances at each t
# - Evaluates on test:
#     Predict failure if exists t where d_t > q_t (only where q_t exists)
#   AUROC uses episode score = max_t (d_t - q_t) over valid timesteps
# - Also (optional) evaluates a global threshold q_global from pooled distances
#
# Expects you already ran feature extraction and have:
#   ROOT/calib_success_feats.npy  (object array of (T_i, D))
#   ROOT/test_feats.npy           (object array of (T_i, D))
#   ROOT/test_labels.npy          (int array: 1=failure, 0=success)

import os, json
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

ROOT  = "/data/home/buddhig/projects/dreamer_fiper_offline"
ALPHA = 0.10  # set as you like (e.g., 0.31)

CALIB_FEATS_PATH = os.path.join(ROOT, "calib_success_feats.npy")
TEST_FEATS_PATH  = os.path.join(ROOT, "test_feats.npy")
TEST_LABELS_PATH = os.path.join(ROOT, "test_labels.npy")

# For per-timestep thresholds: require at least this many calib episodes to define q_t
# If too strict, q_t will be NaN for late timesteps. If too loose, q_t may be noisy.
MIN_CALIB_AT_T = 5

# Also compute global-q baseline
COMPUTE_GLOBAL_Q = True


@dataclass
class MahaModel:
    scaler: StandardScaler
    mean_: np.ndarray
    inv_cov_: np.ndarray
    dim_: int


def maha_dist(Z: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray) -> np.ndarray:
    D = Z - mean[None, :]
    return np.sqrt(np.einsum("ij,jk,ik->i", D, inv_cov, D))


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    s = np.sort(np.asarray(scores, dtype=float))
    n = s.shape[0]
    r = int(np.ceil((n + 1) * (1 - alpha)))
    r = min(max(1, r), n)
    return float(s[r - 1])


def fit_maha_global(calib_feats: np.ndarray) -> MahaModel:
    feats_list = list(calib_feats)
    if len(feats_list) == 0:
        raise RuntimeError("calib_success_feats is empty.")

    X = np.concatenate(feats_list, axis=0)  # (sumT, D)
    print(f"[CALIB] Success timesteps total: {X.shape[0]}, dim={X.shape[1]}")

    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)

    lw = LedoitWolf().fit(Z)
    mean = lw.location_
    cov  = lw.covariance_
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov)

    return MahaModel(scaler=scaler, mean_=mean, inv_cov_=inv_cov, dim_=Z.shape[1])


def dists_for_episode(feat: np.ndarray, model: MahaModel) -> np.ndarray:
    Z = model.scaler.transform(feat)
    return maha_dist(Z, model.mean_, model.inv_cov_)


def build_q_per_timestep(calib_feats: np.ndarray, model: MahaModel, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      q: (Tmax,) thresholds, possibly NaN where count < MIN_CALIB_AT_T
      counts: (Tmax,) number of calib episodes contributing at timestep t
    """
    dists_list = [dists_for_episode(feat, model) for feat in calib_feats]
    Tmax = max(len(d) for d in dists_list)
    q = np.full((Tmax,), np.nan, dtype=np.float64)
    counts = np.zeros((Tmax,), dtype=np.int64)

    for t in range(Tmax):
        vals = [d[t] for d in dists_list if len(d) > t]
        counts[t] = len(vals)
        if len(vals) >= MIN_CALIB_AT_T:
            q[t] = conformal_quantile(np.asarray(vals, dtype=float), alpha)

    n_defined = int(np.sum(~np.isnan(q)))
    print(f"[CALIB] Built per-timestep thresholds up to Tmax={Tmax}")
    print(f"[CALIB] MIN_CALIB_AT_T={MIN_CALIB_AT_T} -> defined q_t for {n_defined}/{Tmax} timesteps")
    print(f"[CALIB] counts[t]: min={counts.min()} max={counts.max()}")

    # helpful: show last defined timestep
    if n_defined > 0:
        last_t = int(np.max(np.where(~np.isnan(q))[0]))
        print(f"[CALIB] last defined q_t at t={last_t} (count={counts[last_t]})")

    return q, counts


def build_global_q(calib_feats: np.ndarray, model: MahaModel, alpha: float) -> float:
    pooled = np.concatenate([dists_for_episode(feat, model) for feat in calib_feats], axis=0)
    q = conformal_quantile(pooled, alpha)
    print(f"[CALIB] Global q (alpha={alpha}) from pooled success distances: {q:.4f}")
    return q


def evaluate_per_timestep(
    test_feats: np.ndarray,
    y_true: np.ndarray,      # 1=failure, 0=success
    model: MahaModel,
    q_per_t: np.ndarray,
):
    print(f"[TEST] Found {len(test_feats)} rollouts in test_feats.npy")

    y_pred = []
    scores = []        # AUROC score: max_t (d_t - q_t) over valid timesteps
    det_times = []     # earliest t where d_t > q_t (only for true failures)

    for feat, is_fail in zip(test_feats, y_true):
        d = dists_for_episode(feat, model)  # (T,)
        Tuse = min(len(d), len(q_per_t))    # ✅ avoid broadcast mismatch
        du = d[:Tuse]
        qt = q_per_t[:Tuse]

        valid = ~np.isnan(qt)
        if not np.any(valid):
            # no calibrated thresholds available for this episode length
            y_pred.append(0)
            scores.append(float(np.max(du)) if len(du) else 0.0)
            continue

        exceed = np.where(valid & (du > qt))[0]
        y_pred.append(1 if exceed.size > 0 else 0)

        # score: worst margin above threshold
        scores.append(float(np.max(du[valid] - qt[valid])))

        if int(is_fail) == 1 and exceed.size > 0:
            det_times.append(int(exceed[0]))

    y_pred = np.asarray(y_pred, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)

    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn + 1e-8)
    tnr = tn / (tn + fp + 1e-8)
    try:
        auroc = roc_auc_score(y_true, scores)
    except ValueError:
        auroc = float("nan")

    mean_dt = float(np.mean(det_times)) if det_times else float("nan")

    print("\n[TEST] Per-timestep q_t results (no running-max):")
    print(f"  Accuracy      : {acc:.4f}")
    print(f"  TPR (failures): {tpr:.4f}")
    print(f"  TNR (success) : {tnr:.4f}")
    print(f"  AUROC (max(d_t-q_t)) : {auroc:.4f}")
    print(f"  Detected failures: {len(det_times)} / {int((y_true==1).sum())}")
    print(f"  Mean detection time (steps) on detected failures: {mean_dt:.2f}")

    return {
        "accuracy": float(acc),
        "TPR": float(tpr),
        "TNR": float(tnr),
        "AUROC": float(auroc) if not np.isnan(auroc) else None,
        "num_failures": int((y_true == 1).sum()),
        "num_detected_failures": int(len(det_times)),
        "mean_detection_time": float(mean_dt),
    }


def evaluate_global_q(
    test_feats: np.ndarray,
    y_true: np.ndarray,      # 1=failure, 0=success
    model: MahaModel,
    q_global: float,
):
    y_pred = []
    scores = []       # AUROC score: max_t d_t
    det_times = []

    for feat, is_fail in zip(test_feats, y_true):
        d = dists_for_episode(feat, model)
        exceed = np.where(d > q_global)[0]
        y_pred.append(1 if exceed.size > 0 else 0)
        scores.append(float(np.max(d)) if len(d) else 0.0)
        if int(is_fail) == 1 and exceed.size > 0:
            det_times.append(int(exceed[0]))

    y_pred = np.asarray(y_pred, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)

    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn + 1e-8)
    tnr = tn / (tn + fp + 1e-8)
    try:
        auroc = roc_auc_score(y_true, scores)
    except ValueError:
        auroc = float("nan")
    mean_dt = float(np.mean(det_times)) if det_times else float("nan")

    print("\n[TEST] Global q results (no running-max):")
    print(f"  Accuracy      : {acc:.4f}")
    print(f"  TPR (failures): {tpr:.4f}")
    print(f"  TNR (success) : {tnr:.4f}")
    print(f"  AUROC (max(d_t)) : {auroc:.4f}")
    print(f"  Detected failures: {len(det_times)} / {int((y_true==1).sum())}")
    print(f"  Mean detection time (steps) on detected failures: {mean_dt:.2f}")

    return {
        "accuracy": float(acc),
        "TPR": float(tpr),
        "TNR": float(tnr),
        "AUROC": float(auroc) if not np.isnan(auroc) else None,
        "num_failures": int((y_true == 1).sum()),
        "num_detected_failures": int(len(det_times)),
        "mean_detection_time": float(mean_dt),
        "q_global": float(q_global),
    }


def main():
    calib_feats = np.load(CALIB_FEATS_PATH, allow_pickle=True)
    test_feats  = np.load(TEST_FEATS_PATH, allow_pickle=True)
    y_true      = np.load(TEST_LABELS_PATH).astype(np.int64)

    print(f"[LOAD] calib_success_feats: {len(calib_feats)} episodes")
    print(f"[LOAD] test_feats        : {len(test_feats)} episodes")

    # 1) Fit global Gaussian reference on success timesteps
    model = fit_maha_global(calib_feats)

    # 2) Per-timestep conformal thresholds q_t
    q_per_t, counts = build_q_per_timestep(calib_feats, model, alpha=ALPHA)

    # Save thresholds
    q_path = os.path.join(ROOT, f"q_per_t_alpha{ALPHA:.2f}_min{MIN_CALIB_AT_T}.npy")
    np.save(q_path, q_per_t)
    np.save(os.path.join(ROOT, f"q_per_t_counts.npy"), counts)
    print("Saved:", q_path)

    # 3) Evaluate (per-timestep)
    results = {
        "alpha": float(ALPHA),
        "min_calib_at_t": int(MIN_CALIB_AT_T),
        "per_timestep": evaluate_per_timestep(test_feats, y_true, model, q_per_t),
    }

    # Optional: global q baseline
    if COMPUTE_GLOBAL_Q:
        q_global = build_global_q(calib_feats, model, alpha=ALPHA)
        results["global"] = evaluate_global_q(test_feats, y_true, model, q_global)

        with open(os.path.join(ROOT, f"q_global_alpha{ALPHA:.2f}.txt"), "w") as f:
            f.write(f"{q_global}\n")
        print("Saved:", os.path.join(ROOT, f"q_global_alpha{ALPHA:.2f}.txt"))

    # Save results JSON
    out_path = os.path.join(ROOT, f"maha_conformal_per_timestep_alpha{ALPHA:.2f}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()
