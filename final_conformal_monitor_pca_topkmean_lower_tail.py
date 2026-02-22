# final_conformal_monitor_pca_topkmean_lower_tail.py
import os, json
import numpy as np
from dataclasses import dataclass
from typing import Optional

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

# =========================
# CONFIG
# =========================
ROOT  = "/data/home/buddhig/projects/dreamer_fiper_offline/sorting/feats_both6ch64"
ALPHA = 0.10

CALIB_FEATS_PATH = os.path.join(ROOT, "calib_success_feats.npy")  # object array: each (T,D), success-only
TEST_FEATS_PATH  = os.path.join(ROOT, "test_feats.npy")           # object array: each (T,D)
TEST_LABELS_PATH = os.path.join(ROOT, "test_labels.npy")          # int: 1=failure, 0=success

# ---- hyperparams (tune if needed) ----
PCA_DIM  = 32
BURN_IN  = 4
TOPK     = 7
EPS      = 1e-8


# =========================
# Conformal quantiles
# =========================
def conformal_upper_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Upper-tail conformal quantile with finite-sample correction:
      r = ceil((n+1)*(1-alpha)), threshold = sorted(scores)[r-1]
    Use when LARGE scores are anomalous.
    """
    s = np.sort(np.asarray(scores, dtype=float))
    n = s.shape[0]
    r = int(np.ceil((n + 1) * (1 - alpha)))
    r = min(max(1, r), n)
    return float(s[r - 1])


def conformal_lower_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Lower-tail conformal quantile with finite-sample correction:
      r = ceil((n+1)*alpha), threshold = sorted(scores)[r-1]
    Use when SMALL scores are anomalous.
    """
    s = np.sort(np.asarray(scores, dtype=float))
    n = s.shape[0]
    r = int(np.ceil((n + 1) * alpha))
    r = min(max(1, r), n)
    return float(s[r - 1])


# =========================
# PCA ref
# =========================
@dataclass
class PCARef:
    scaler: StandardScaler
    pca: PCA
    pca_dim: int


def fit_pca_ref_success(calib_feats: np.ndarray, pca_dim: int) -> PCARef:
    X = np.concatenate(list(calib_feats), axis=0)  # (sumT, D)
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)

    d = Z.shape[1]
    k = int(min(pca_dim, d))
    pca = PCA(n_components=k, whiten=True, random_state=0).fit(Z)

    print(f"[CALIB] Fit StandardScaler + PCA(whiten=True): dim={d} -> pca_dim={k}")
    return PCARef(scaler=scaler, pca=pca, pca_dim=k)


def pca_l2_dists(feat: np.ndarray, ref: PCARef) -> np.ndarray:
    """
    feat: (T,D) -> distances d: (T,)
    """
    Z = ref.scaler.transform(feat)
    Y = ref.pca.transform(Z)          # (T,k), whitened
    d = np.linalg.norm(Y, axis=1)     # (T,)
    return d


# =========================
# Scoring helpers
# =========================
def apply_burn_in(d: np.ndarray, burn_in: int) -> np.ndarray:
    if burn_in <= 0:
        return d
    if len(d) <= burn_in:
        return d[-1:]  # keep at least 1 element
    return d[burn_in:]


def topk_mean(d: np.ndarray, k: int) -> float:
    if len(d) == 0:
        return 0.0
    kk = min(int(k), len(d))
    return float(np.mean(np.sort(d)[-kk:]))


def first_crossing_time_lower(d: np.ndarray, thr: float) -> Optional[int]:
    """
    For LOWER-tail rule (failure if score < thr),
    return the first index where d < thr.
    """
    idx = np.where(d < thr)[0]
    return int(idx[0]) if idx.size else None


def safe_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


# =========================
# MAIN
# =========================
def main():
    calib_feats = np.load(CALIB_FEATS_PATH, allow_pickle=True)
    test_feats  = np.load(TEST_FEATS_PATH, allow_pickle=True)
    y_true      = np.load(TEST_LABELS_PATH).astype(np.int64)  # 1=failure, 0=success

    print(f"[LOAD] calib_success_feats: {len(calib_feats)} episodes (success-only)")
    print(f"[LOAD] test_feats        : {len(test_feats)} episodes")
    print(f"[LOAD] test_labels       : failures={(y_true==1).sum()} successes={(y_true==0).sum()}")

    # 1) Fit ref on ALL success timesteps
    ref = fit_pca_ref_success(calib_feats, PCA_DIM)

    # 2) Calib episode scores (success-only)
    calib_scores = []
    for feat in calib_feats:
        d = pca_l2_dists(feat, ref)
        d = apply_burn_in(d, BURN_IN)
        calib_scores.append(topk_mean(d, TOPK))
    calib_scores = np.asarray(calib_scores, dtype=np.float64)

    # ✅ LOWER-tail threshold (because AUROC(-score) > 0.5)
    thr = conformal_lower_quantile(calib_scores, ALPHA)
    print(f"[CALIB] Lower-tail threshold q (alpha={ALPHA}) from {len(calib_scores)} success rollouts: {thr:.4f}")
    print(f"[CALIB] Scores stats: mean={calib_scores.mean():.4f} std={calib_scores.std():.4f} "
          f"min={calib_scores.min():.4f} max={calib_scores.max():.4f}")

    # 3) Test
    test_scores = []
    y_pred = []
    det_times = []
    per_episode_dists = []  # optional, for later plotting

    for feat, yi in zip(test_feats, y_true):
        d0 = pca_l2_dists(feat, ref)
        d  = apply_burn_in(d0, BURN_IN)

        score = topk_mean(d, TOPK)

        # ✅ LOWER-tail decision: failure if score < thr
        pred = 1 if score < thr else 0

        test_scores.append(score)
        y_pred.append(pred)

        if int(yi) == 1 and pred == 1:
            dt_rel = first_crossing_time_lower(d, thr)  # in burned sequence
            if dt_rel is not None:
                det_times.append(int(dt_rel + BURN_IN))

        per_episode_dists.append(d0.astype(np.float32))

    test_scores = np.asarray(test_scores, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    # 4) Metrics
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn + EPS)
    tnr = tn / (tn + fp + EPS)
    bal = 0.5 * (tpr + tnr)

    # Note: AUROC of raw score remains the same; the "good" direction is -score
    auroc_score = safe_auroc(y_true, test_scores)
    auroc_neg   = safe_auroc(y_true, -test_scores)

    mean_dt = float(np.mean(det_times)) if det_times else float("nan")

    print("\n[TEST] Results (PCA-whiten L2 + topk_mean)  [LOWER-TAIL CONFORMAL]")
    print(f"  Accuracy         : {acc:.4f}")
    print(f"  TPR (failures)   : {tpr:.4f}")
    print(f"  TNR (success)    : {tnr:.4f}")
    print(f"  BalAcc           : {bal:.4f}")
    print(f"  AUROC (score)    : {auroc_score:.4f}")
    print(f"  AUROC (-score)   : {auroc_neg:.4f}   (expected > 0.5 if direction is inverted)")
    print(f"  Confusion        : TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"  Detected failures: {len(det_times)} / {int((y_true==1).sum())}")
    print(f"  Mean detection time (steps) on detected failures: {mean_dt:.2f}")

    # 5) Save everything needed to reproduce + plot
    out = {
        "root": ROOT,
        "alpha": float(ALPHA),
        "method": "pca_whiten_l2",
        "tail": "lower",
        "pca_dim": int(ref.pca_dim),
        "burn_in": int(BURN_IN),
        "topk": int(TOPK),
        "threshold": float(thr),
        "calib_scores_stats": {
            "mean": float(calib_scores.mean()),
            "std": float(calib_scores.std()),
            "min": float(calib_scores.min()),
            "max": float(calib_scores.max()),
            "n": int(len(calib_scores)),
        },
        "metrics": {
            "accuracy": float(acc),
            "TPR": float(tpr),
            "TNR": float(tnr),
            "bal_acc": float(bal),
            "AUROC_score": float(auroc_score) if not np.isnan(auroc_score) else None,
            "AUROC_neg_score": float(auroc_neg) if not np.isnan(auroc_neg) else None,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "mean_detection_time": float(mean_dt),
            "num_failures": int((y_true==1).sum()),
            "num_detected_failures": int(len(det_times)),
        },
    }

    out_json = os.path.join(
        ROOT, f"final_monitor_lower_alpha{ALPHA:.2f}_pca{ref.pca_dim}_burn{BURN_IN}_topk{TOPK}.json"
    )
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved:", out_json)

    out_npz = os.path.join(
        ROOT, f"final_monitor_lower_arrays_alpha{ALPHA:.2f}_pca{ref.pca_dim}_burn{BURN_IN}_topk{TOPK}.npz"
    )
    np.savez_compressed(
        out_npz,
        threshold=np.array([thr], dtype=np.float64),
        test_scores=test_scores.astype(np.float64),
        test_labels=y_true.astype(np.int64),
        test_pred=y_pred.astype(np.int64),
        calib_scores=calib_scores.astype(np.float64),
        det_times=np.asarray(det_times, dtype=np.int64),
    )
    print("Saved:", out_npz)

    dists_path = os.path.join(ROOT, f"per_episode_dists_lower_alpha{ALPHA:.2f}_pca{ref.pca_dim}.npy")
    np.save(dists_path, np.array(per_episode_dists, dtype=object), allow_pickle=True)
    print("Saved:", dists_path)


if __name__ == "__main__":
    main()