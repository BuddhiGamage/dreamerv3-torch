# maha_conformal_dreamer_feats.py
import os
import json
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

ROOT  = "/data/home/buddhig/projects/dreamer_fiper_offline"
ALPHA = 0.10

# Choose how to turn per-timestep distances into an episode score
# "max"  : strong OOD detector, often best for failure detection
# "mean" : smoother but can miss short spikes
EP_SCORE_MODE = "max"   # "max" or "mean"

# If True, also compute a simple early-detection time: first t where dist>thr
COMPUTE_EARLY_DET_TIME = True


def fit_maha(Z_flat: np.ndarray):
    mu = Z_flat.mean(axis=0)
    cov = LedoitWolf().fit(Z_flat).covariance_
    inv = np.linalg.inv(cov)
    return mu, inv


def maha_dist(Z: np.ndarray, mu: np.ndarray, inv: np.ndarray):
    # Z: (T,D) or (N,D)
    X = Z - mu
    return np.sqrt(np.einsum("nd,dd,nd->n", X, inv, X))


def conformal_threshold(calib_scores: np.ndarray, alpha=0.10):
    # Split-conformal quantile with finite-sample correction
    n = len(calib_scores)
    q = np.ceil((n + 1) * (1 - alpha)) / n
    return np.quantile(calib_scores, q, method="higher")


def episode_score(dists_1d: np.ndarray, mode: str):
    if mode == "max":
        return float(np.max(dists_1d))
    if mode == "mean":
        return float(np.mean(dists_1d))
    raise ValueError(f"Unknown EP_SCORE_MODE={mode}")


def first_crossing_time(dists_1d: np.ndarray, thr: float):
    idx = np.where(dists_1d > thr)[0]
    return int(idx[0]) if len(idx) else None


def safe_auroc(y_true: np.ndarray, scores: np.ndarray):
    # AUROC undefined if only one class present
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, scores)


def main():
    calib_feats = np.load(os.path.join(ROOT, "calib_success_feats.npy"), allow_pickle=True)
    test_feats  = np.load(os.path.join(ROOT, "test_feats.npy"), allow_pickle=True)
    y           = np.load(os.path.join(ROOT, "test_labels.npy"))  # 1=failure, 0=success

    # -------------------------
    # Fit Gaussian reference on ALL success calibration timesteps
    # -------------------------
    calib_list = list(calib_feats)
    test_list  = list(test_feats)

    calib_flat = np.concatenate(calib_list, axis=0)  # (sumT, D)
    mu, inv = fit_maha(calib_flat)

    # -------------------------
    # Calibration scores (episode-level)
    # -------------------------
    calib_dists = [maha_dist(feat, mu, inv) for feat in calib_list]  # list of (T,)
    calib_scores = np.array([episode_score(d, EP_SCORE_MODE) for d in calib_dists], dtype=np.float64)

    thr = conformal_threshold(calib_scores, ALPHA)
    print(f"Conformal threshold (alpha={ALPHA}, mode={EP_SCORE_MODE}): {thr:.6f}")
    print(f"Calib scores: mean={calib_scores.mean():.4f} std={calib_scores.std():.4f} "
          f"min={calib_scores.min():.4f} max={calib_scores.max():.4f} n={len(calib_scores)}")

    # -------------------------
    # Test scores + predictions
    # -------------------------
    test_dists = [maha_dist(feat, mu, inv) for feat in test_list]  # list of (T,)
    test_scores = np.array([episode_score(d, EP_SCORE_MODE) for d in test_dists], dtype=np.float64)

    # Predict failure if score > threshold
    yhat = (test_scores > thr).astype(np.int64)

    # -------------------------
    # Metrics
    # -------------------------
    acc = accuracy_score(y, yhat)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn + 1e-9)  # recall on failures
    tnr = tn / (tn + fp + 1e-9)  # specificity on successes
    auroc = safe_auroc(y, test_scores)

    print("\n[TEST] Results")
    print(f"  Accuracy      : {acc:.4f}")
    print(f"  TPR (failures): {tpr:.4f}")
    print(f"  TNR (success) : {tnr:.4f}")
    print(f"  AUROC (score) : {auroc:.4f}")
    print(f"  Confusion     : TN={tn} FP={fp} FN={fn} TP={tp}")

    # -------------------------
    # Early detection time (optional)
    # -------------------------
    det_times = None
    if COMPUTE_EARLY_DET_TIME:
        # detection time defined only for predicted failures (or true failures; choose what you like)
        # Here: for TRUE failures, compute first timestep crossing threshold, else None
        times = []
        for yi, d in zip(y, test_dists):
            if yi == 1:  # true failure
                times.append(first_crossing_time(d, thr))
        # mean over failures where crossing happens
        valid = [t for t in times if t is not None]
        det_times = {
            "n_failures": int(np.sum(y == 1)),
            "n_detected_failures": int(len(valid)),
            "mean_det_time": float(np.mean(valid)) if valid else None,
            "median_det_time": float(np.median(valid)) if valid else None,
        }
        print("\n[Early detection on TRUE failures]")
        print(f"  failures total        : {det_times['n_failures']}")
        print(f"  failures detected     : {det_times['n_detected_failures']}")
        print(f"  mean det time (steps) : {det_times['mean_det_time']}")
        print(f"  median det time       : {det_times['median_det_time']}")

    # -------------------------
    # Save outputs for later plotting
    # -------------------------
    out = dict(
        alpha=ALPHA,
        ep_score_mode=EP_SCORE_MODE,
        threshold=float(thr),
        test_scores=test_scores.tolist(),
        test_labels=y.astype(int).tolist(),
        test_pred=yhat.astype(int).tolist(),
        metrics=dict(
            accuracy=float(acc),
            tpr=float(tpr),
            tnr=float(tnr),
            auroc=float(auroc) if not np.isnan(auroc) else None,
            tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        ),
        early_detection=det_times,
    )
    out_path = os.path.join(ROOT, f"maha_conformal_results_alpha{ALPHA:.2f}_{EP_SCORE_MODE}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()
