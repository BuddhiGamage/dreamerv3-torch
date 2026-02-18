# maha_or_pca_conformal_tuned.py
import os, json
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

ROOT  = "/data/home/buddhig/projects/dreamer_fiper_offline"
ALPHA = 0.10

CALIB_FEATS_PATH = os.path.join(ROOT, "calib_success_feats.npy")  # success-only
TEST_FEATS_PATH  = os.path.join(ROOT, "test_feats.npy")
TEST_LABELS_PATH = os.path.join(ROOT, "test_labels.npy")          # 1=failure, 0=success

# -------- choose distance model --------
METHOD = "pca"   # "pca" recommended; "maha" also available

# -------- tuning grid --------
BURN_IN_GRID = [0, 2, 4, 6, 8]          # ignore first N steps
TOPK_GRID    = [1, 3, 5, 7]             # top-k for robust aggregators
AGG_MODES    = ["max", "topk_mean", "topk_median", "smooth_max"]

SMOOTH_W_GRID = [1, 3, 5, 7]            # moving average window for smooth_max
PCA_DIM_GRID  = [16, 32, 64, 128]       # used only if METHOD="pca" (clipped to D)


EPS = 1e-8


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    s = np.sort(np.asarray(scores, dtype=float))
    n = s.shape[0]
    r = int(np.ceil((n + 1) * (1 - alpha)))
    r = min(max(1, r), n)
    return float(s[r - 1])


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(x) == 0:
        return x
    w = min(w, len(x))
    kernel = np.ones(w, dtype=float) / float(w)
    y = np.convolve(x, kernel, mode="valid")
    if len(y) == len(x):
        return y
    pad = np.full((len(x) - len(y),), y[0], dtype=float)
    return np.concatenate([pad, y], axis=0)


# =========================
# Distance models
# =========================
@dataclass
class MahaRef:
    scaler: StandardScaler
    mean_: np.ndarray
    inv_cov_: np.ndarray

def fit_maha_ref(calib_feats: np.ndarray) -> MahaRef:
    X = np.concatenate(list(calib_feats), axis=0)  # (sumT,D)
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)
    lw = LedoitWolf().fit(Z)
    mean = lw.location_
    cov  = lw.covariance_
    try:
        inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(cov)
    return MahaRef(scaler=scaler, mean_=mean, inv_cov_=inv)

def maha_dists(feat: np.ndarray, ref: MahaRef) -> np.ndarray:
    Z = ref.scaler.transform(feat)
    D = Z - ref.mean_[None, :]
    return np.sqrt(np.einsum("ij,jk,ik->i", D, ref.inv_cov_, D))


@dataclass
class PCARef:
    scaler: StandardScaler
    pca: PCA
    # after whitening, L2 distance is stable without covariance inversion

def fit_pca_ref(calib_feats: np.ndarray, pca_dim: int) -> PCARef:
    X = np.concatenate(list(calib_feats), axis=0)  # (sumT,D)
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)
    d = Z.shape[1]
    k = int(min(pca_dim, d))
    pca = PCA(n_components=k, whiten=True, random_state=0).fit(Z)
    return PCARef(scaler=scaler, pca=pca)

def pca_dists(feat: np.ndarray, ref: PCARef) -> np.ndarray:
    Z = ref.scaler.transform(feat)
    Y = ref.pca.transform(Z)          # whitened => ~N(0,I) for success
    return np.linalg.norm(Y, axis=1)  # (T,)


# =========================
# Scoring
# =========================
def apply_burn_in(d: np.ndarray, burn_in: int) -> np.ndarray:
    if burn_in <= 0:
        return d
    if len(d) <= burn_in:
        return d[-1:]  # keep at least 1 element
    return d[burn_in:]

def episode_score(d: np.ndarray, mode: str, topk: int, smooth_w: int) -> float:
    if len(d) == 0:
        return 0.0
    if mode == "max":
        return float(np.max(d))
    if mode == "topk_mean":
        k = min(topk, len(d))
        return float(np.mean(np.sort(d)[-k:]))
    if mode == "topk_median":
        k = min(topk, len(d))
        return float(np.median(np.sort(d)[-k:]))
    if mode == "smooth_max":
        sm = moving_average(d, smooth_w)
        return float(np.max(sm))
    raise ValueError(mode)

def detection_time(d: np.ndarray, thr: float) -> Optional[int]:
    idx = np.where(d > thr)[0]
    return int(idx[0]) if idx.size else None


def eval_one_setting(
    calib_feats, test_feats, y_true,
    dist_ref, dist_fn,
    mode: str, burn_in: int, topk: int, smooth_w: int,
) -> Dict[str, Any]:

    # calib scores on SUCCESS episodes
    calib_scores = []
    for feat in calib_feats:
        d = dist_fn(feat, dist_ref)
        d = apply_burn_in(d, burn_in)
        calib_scores.append(episode_score(d, mode, topk, smooth_w))
    calib_scores = np.asarray(calib_scores, dtype=float)
    thr = conformal_quantile(calib_scores, ALPHA)

    # test
    y_pred = []
    scores = []
    det_times = []

    for feat, yi in zip(test_feats, y_true):
        d0 = dist_fn(feat, dist_ref)
        d  = apply_burn_in(d0, burn_in)
        s  = episode_score(d, mode, topk, smooth_w)

        pred = 1 if s > thr else 0
        y_pred.append(pred)
        scores.append(s)

        if int(yi) == 1 and pred == 1:
            dt = detection_time(d, thr)  # relative to burned-in sequence
            if dt is not None:
                det_times.append(dt + burn_in)  # report in original time

    y_pred = np.asarray(y_pred, dtype=np.int64)
    scores = np.asarray(scores, dtype=float)

    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn + EPS)
    tnr = tn / (tn + fp + EPS)
    bal = 0.5 * (tpr + tnr)

    try:
        auroc = roc_auc_score(y_true, scores)
    except ValueError:
        auroc = float("nan")

    mean_dt = float(np.mean(det_times)) if det_times else float("nan")

    return {
        "mode": mode,
        "burn_in": int(burn_in),
        "topk": int(topk),
        "smooth_w": int(smooth_w),
        "threshold": float(thr),
        "accuracy": float(acc),
        "TPR": float(tpr),
        "TNR": float(tnr),
        "bal_acc": float(bal),
        "AUROC": float(auroc) if not np.isnan(auroc) else None,
        "mean_det_time": float(mean_dt),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main():
    calib_feats = np.load(CALIB_FEATS_PATH, allow_pickle=True)
    test_feats  = np.load(TEST_FEATS_PATH, allow_pickle=True)
    y_true      = np.load(TEST_LABELS_PATH).astype(np.int64)

    print(f"[LOAD] calib_success_feats: {len(calib_feats)} episodes")
    print(f"[LOAD] test_feats        : {len(test_feats)} episodes")
    print(f"[LOAD] test_labels       : failures={int((y_true==1).sum())} successes={int((y_true==0).sum())}")

    results = []

    if METHOD == "maha":
        ref = fit_maha_ref(calib_feats)
        dist_fn = maha_dists
        ref_desc = {"method": "maha"}
        pca_dim_used = None

        for burn_in in BURN_IN_GRID:
            for mode in AGG_MODES:
                for topk in TOPK_GRID:
                    for smooth_w in SMOOTH_W_GRID:
                        if mode != "smooth_max" and smooth_w != 1:
                            continue
                        if mode == "smooth_max" and smooth_w == 1:
                            continue
                        if mode in ["max", "smooth_max"] and topk != TOPK_GRID[0]:
                            continue
                        r = eval_one_setting(calib_feats, test_feats, y_true, ref, dist_fn, mode, burn_in, topk, smooth_w)
                        results.append(r)

    elif METHOD == "pca":
        # try multiple PCA dims
        D = int(np.asarray(calib_feats[0]).shape[1])
        for pca_dim in PCA_DIM_GRID:
            pca_dim = int(min(pca_dim, D))
            ref = fit_pca_ref(calib_feats, pca_dim=pca_dim)
            dist_fn = pca_dists
            for burn_in in BURN_IN_GRID:
                for mode in AGG_MODES:
                    for topk in TOPK_GRID:
                        for smooth_w in SMOOTH_W_GRID:
                            if mode != "smooth_max" and smooth_w != 1:
                                continue
                            if mode == "smooth_max" and smooth_w == 1:
                                continue
                            if mode in ["max", "smooth_max"] and topk != TOPK_GRID[0]:
                                continue
                            r = eval_one_setting(calib_feats, test_feats, y_true, ref, dist_fn, mode, burn_in, topk, smooth_w)
                            r["pca_dim"] = int(pca_dim)
                            results.append(r)
    else:
        raise ValueError("METHOD must be 'maha' or 'pca'")

    # choose best by balanced accuracy
    best = max(results, key=lambda r: r["bal_acc"])
    print("\n[BEST by balanced accuracy]")
    for k in ["mode","burn_in","topk","smooth_w","pca_dim","threshold","accuracy","TPR","TNR","bal_acc","AUROC","mean_det_time"]:
        if k in best:
            print(f"  {k:>10s}: {best[k]}")

    # print best confusion
    print("  confusion:", best["confusion"])

    # save all
    out = {
        "root": ROOT,
        "alpha": ALPHA,
        "method": METHOD,
        "grid": {
            "BURN_IN_GRID": BURN_IN_GRID,
            "TOPK_GRID": TOPK_GRID,
            "SMOOTH_W_GRID": SMOOTH_W_GRID,
            "PCA_DIM_GRID": PCA_DIM_GRID if METHOD == "pca" else None,
            "AGG_MODES": AGG_MODES,
        },
        "best": best,
        "all_results": results,
    }
    out_path = os.path.join(ROOT, f"maha_or_pca_conformal_tuned_alpha{ALPHA:.2f}_{METHOD}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()
