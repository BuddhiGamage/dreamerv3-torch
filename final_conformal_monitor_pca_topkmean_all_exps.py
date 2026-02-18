# final_conformal_monitor_pca_topkmean_all_exps.py
# - Runs your PCA-whiten L2 + topk_mean conformal monitor PER EXPERIMENT
# - Assumes you extracted per-exp feats using the "all_exps_resize" extractor:
#     <FEATS_ROOT>/<exp_id>/calib_success_feats.npy
#     <FEATS_ROOT>/<exp_id>/test_feats.npy
#     <FEATS_ROOT>/<exp_id>/test_labels.npy
# - Writes per-exp results to:
#     <OUT_ROOT>/<exp_id>/final_monitor_...json
#     <OUT_ROOT>/<exp_id>/final_monitor_arrays_...npz
#     <OUT_ROOT>/<exp_id>/per_episode_dists_...npy (optional)
#
# IMPORTANT:
#   This does NOT do resizing itself. Resizing was already done during WM feature extraction.
#   So just point FEATS_ROOT to the folder that contains per-exp feats.

import os, glob, json, argparse, hashlib
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


# ---- frozen best hyperparams ----
ALPHA   = 0.10
PCA_DIM = 32
BURN_IN = 4
TOPK    = 7
EPS     = 1e-8


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    s = np.sort(np.asarray(scores, dtype=float))
    n = s.shape[0]
    r = int(np.ceil((n + 1) * (1 - alpha)))
    r = min(max(1, r), n)
    return float(s[r - 1])


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
    Z = ref.scaler.transform(feat)
    Y = ref.pca.transform(Z)
    d = np.linalg.norm(Y, axis=1)
    return d


def apply_burn_in(d: np.ndarray, burn_in: int) -> np.ndarray:
    if burn_in <= 0:
        return d
    if len(d) <= burn_in:
        return d[-1:]
    return d[burn_in:]


def topk_mean(d: np.ndarray, k: int) -> float:
    if len(d) == 0:
        return 0.0
    kk = min(int(k), len(d))
    return float(np.mean(np.sort(d)[-kk:]))


def first_crossing_time(d: np.ndarray, thr: float) -> Optional[int]:
    idx = np.where(d > thr)[0]
    return int(idx[0]) if idx.size else None


def safe_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def discover_exp_ids(feats_root: str) -> List[str]:
    """
    Experiments are subfolders that contain calib_success_feats.npy
    """
    paths = glob.glob(os.path.join(feats_root, "*", "calib_success_feats.npy"))
    exp_ids = sorted({os.path.basename(os.path.dirname(p)) for p in paths})
    return exp_ids


def run_one_experiment(
    exp_id: str,
    feats_root: str,
    out_root: str,
    save_dists: bool = True,
):
    exp_dir = os.path.join(feats_root, exp_id)
    calib_path = os.path.join(exp_dir, "calib_success_feats.npy")
    test_path  = os.path.join(exp_dir, "test_feats.npy")
    labels_path= os.path.join(exp_dir, "test_labels.npy")

    if not (os.path.exists(calib_path) and os.path.exists(test_path) and os.path.exists(labels_path)):
        print(f"[SKIP] {exp_id}: missing required feat files")
        return

    calib_feats = np.load(calib_path, allow_pickle=True)
    test_feats  = np.load(test_path, allow_pickle=True)
    y_true      = np.load(labels_path).astype(np.int64)

    print(f"\n=== EXP: {exp_id} ===")
    print(f"[LOAD] calib_success_feats: {len(calib_feats)} episodes (success-only)")
    print(f"[LOAD] test_feats        : {len(test_feats)} episodes")
    print(f"[LOAD] test_labels       : failures={(y_true==1).sum()} successes={(y_true==0).sum()}")

    if len(calib_feats) == 0:
        print(f"[WARN] {exp_id}: calib_success_feats is empty, skipping.")
        return

    # 1) Fit ref on ALL success timesteps
    ref = fit_pca_ref_success(calib_feats, PCA_DIM)

    # 2) Calib episode scores (success-only)
    calib_scores = []
    for feat in calib_feats:
        d = pca_l2_dists(feat, ref)
        d = apply_burn_in(d, BURN_IN)
        calib_scores.append(topk_mean(d, TOPK))
    calib_scores = np.asarray(calib_scores, dtype=np.float64)

    thr = conformal_quantile(calib_scores, ALPHA)
    print(f"[CALIB] Threshold q (alpha={ALPHA}) from {len(calib_scores)} success rollouts: {thr:.4f}")

    # 3) Test
    test_scores = []
    y_pred = []
    det_times = []
    per_episode_dists = []

    for feat, yi in zip(test_feats, y_true):
        d0 = pca_l2_dists(feat, ref)
        d  = apply_burn_in(d0, BURN_IN)

        score = topk_mean(d, TOPK)
        pred = 1 if score > thr else 0

        test_scores.append(score)
        y_pred.append(pred)

        if int(yi) == 1 and pred == 1:
            dt_rel = first_crossing_time(d, thr)
            if dt_rel is not None:
                det_times.append(int(dt_rel + BURN_IN))

        if save_dists:
            per_episode_dists.append(d0.astype(np.float32))

    test_scores = np.asarray(test_scores, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    # 4) Metrics
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn + EPS)
    tnr = tn / (tn + fp + EPS)
    bal = 0.5 * (tpr + tnr)
    auroc = safe_auroc(y_true, test_scores)
    mean_dt = float(np.mean(det_times)) if det_times else float("nan")

    print("\n[TEST] Results (PCA-whiten L2 + topk_mean)")
    print(f"  Accuracy      : {acc:.4f}")
    print(f"  TPR (failures): {tpr:.4f}")
    print(f"  TNR (success) : {tnr:.4f}")
    print(f"  BalAcc        : {bal:.4f}")
    print(f"  AUROC (score) : {auroc:.4f}")
    print(f"  Confusion     : TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"  Detected failures: {len(det_times)} / {int((y_true==1).sum())}")
    print(f"  Mean detection time (steps) on detected failures: {mean_dt:.2f}")

    # 5) Save
    exp_out = os.path.join(out_root, exp_id)
    os.makedirs(exp_out, exist_ok=True)

    out = {
        "exp_id": exp_id,
        "feats_root": feats_root,
        "alpha": float(ALPHA),
        "method": "pca_whiten_l2",
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
            "AUROC": float(auroc) if not np.isnan(auroc) else None,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "mean_detection_time": float(mean_dt),
            "num_failures": int((y_true==1).sum()),
            "num_detected_failures": int(len(det_times)),
        },
    }

    out_json = os.path.join(
        exp_out,
        f"final_monitor_alpha{ALPHA:.2f}_pca{ref.pca_dim}_burn{BURN_IN}_topk{TOPK}.json"
    )
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print("Saved:", out_json)

    out_npz = os.path.join(
        exp_out,
        f"final_monitor_arrays_alpha{ALPHA:.2f}_pca{ref.pca_dim}_burn{BURN_IN}_topk{TOPK}.npz"
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

    if save_dists:
        dists_path = os.path.join(
            exp_out,
            f"per_episode_dists_alpha{ALPHA:.2f}_pca{ref.pca_dim}.npy"
        )
        np.save(dists_path, np.array(per_episode_dists, dtype=object), allow_pickle=True)
        print("Saved:", dists_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats_root", type=str, default="/data/home/buddhig/projects/dreamer_fiper_feats_all",
                    help="Folder containing per-exp feat subfolders")
    ap.add_argument("--out_root", type=str, default="/data/home/buddhig/projects/dreamer_fiper_monitor_all",
                    help="Where to write per-exp monitor outputs")
    ap.add_argument("--save_dists", action="store_true",
                    help="Also save per-episode full distance sequences (object .npy)")
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)

    exp_ids = discover_exp_ids(args.feats_root)
    print(f"Discovered experiments: {len(exp_ids)}")
    if not exp_ids:
        raise RuntimeError(
            f"No experiments found under {args.feats_root}. "
            f"Expected <feats_root>/<exp_id>/calib_success_feats.npy"
        )

    for exp_id in exp_ids:
        run_one_experiment(
            exp_id=exp_id,
            feats_root=args.feats_root,
            out_root=args.out_root,
            save_dists=args.save_dists,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
