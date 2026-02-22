import os, glob, pickle, json
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from models import WorldModel

# =========================
# PATHS
# =========================
CALIB_GLOB = "/data/home/buddhig/data_all/sorting/rollouts/calibration/*.pkl"
TEST_GLOB  = "/data/home/buddhig/data_all/sorting/rollouts/test/*.pkl"
CKPT_PATH  = "/data/home/buddhig/projects/dreamer_fiper_offline/sorting/wm_success_only_rgb_both_views_6ch_64.pt"

OUT_DIR = "/data/home/buddhig/projects/dreamer_fiper_offline/sorting/monitor_rssm_nll_residual_best"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# PREPROCESS (match training)
# =========================
TARGET_HW = (64, 64)
VIEW_MODE = "both6"   # "left64" | "right64" | "both6"

# =========================
# WINDOWS / SMOOTHING
# =========================
BURN_IN = 4
BASE_W = 20
IGNORE_AFTER_BASE = 10
SMOOTH_W = 7

# =========================
# MONITOR (chosen from sweep)
# =========================
ALPHA_RES_STEP = 0.02
PERSIST = 8

# AUROC ranking score (not used for decision)
SCORE_MODE = "topk_mean"   # "max" | "topk_mean"
TOPK = 15

EPS = 1e-8
DEBUG_SHAPES_ONCE = True


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


def _resize_uint8_hwc(img: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    pil = Image.fromarray(img.astype(np.uint8))
    pil = pil.resize((hw[1], hw[0]), resample=Image.BILINEAR)
    out = np.asarray(pil, dtype=np.uint8)
    if out.ndim == 2:
        out = np.stack([out] * 3, axis=-1)
    return out


def _split_lr(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    H, W, C = rgb.shape
    if C != 3:
        raise ValueError(f"Expected RGB, got {rgb.shape}")
    if W % 2 != 0:
        raise ValueError(f"Expected even width for concat views, got W={W}")
    half = W // 2
    return rgb[:, :half, :], rgb[:, half:, :]


def preprocess_frame(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.uint8)
    left, right = _split_lr(rgb)
    left = _resize_uint8_hwc(left, TARGET_HW)
    right = _resize_uint8_hwc(right, TARGET_HW)

    if VIEW_MODE == "left64":
        return left
    if VIEW_MODE == "right64":
        return right
    if VIEW_MODE == "both6":
        return np.concatenate([left, right], axis=-1)  # 6ch
    raise ValueError(VIEW_MODE)


def load_fiper_pkl(path: str):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["metadata"], d["rollout"]


def build_episode(steps, action_dim: int) -> Dict[str, np.ndarray]:
    T = len(steps)
    images = np.stack([preprocess_frame(s["rgb"]) for s in steps], axis=0).astype(np.uint8)

    actions = np.zeros((T, action_dim), dtype=np.float32)
    reward  = np.zeros((T, 1), dtype=np.float32)

    is_first = np.zeros((T,), dtype=np.float32); is_first[0] = 1.0
    is_terminal = np.zeros((T,), dtype=np.float32); is_terminal[-1] = 1.0
    discount = np.ones((T,), dtype=np.float32); discount[-1] = 0.0

    return dict(
        image=images,
        action=actions,
        reward=reward,
        discount=discount,
        is_first=is_first,
        is_terminal=is_terminal,
    )


def build_wm_from_ckpt(ckpt: dict) -> WorldModel:
    cfgd = ckpt["config"]
    device = cfgd["device"]

    class DummySpace:
        def __init__(self, shape): self.shape = shape

    class DummyDictSpace:
        def __init__(self, spaces): self.spaces = spaces

    image_shape = tuple(ckpt["image_shape"])
    action_dim  = int(ckpt["action_dim"])

    obs_space = DummyDictSpace({"image": DummySpace(image_shape)})
    act_space = DummySpace((action_dim,))
    step = torch.tensor(0, device=device)

    class Cfg: pass
    cfg = Cfg()
    for k, v in cfgd.items():
        setattr(cfg, k, v)

    wm = WorldModel(obs_space, act_space, step, cfg).to(device).eval()
    wm.load_state_dict(ckpt["wm_state"], strict=True)
    return wm


def _get_decoder_head(wm: WorldModel):
    def pick(container):
        try:
            if "decoder" in container:
                return container["decoder"]
        except Exception:
            pass
        for k in ["dec", "decoder", "obs", "image_decoder"]:
            try:
                if k in container:
                    return container[k]
            except Exception:
                continue
        return None

    for attr in ["heads", "_heads", "head", "_head"]:
        if hasattr(wm, attr):
            h = getattr(wm, attr)
            if isinstance(h, (dict, nn.ModuleDict)) or hasattr(h, "__contains__"):
                dec = pick(h)
                if dec is not None:
                    return dec

    for mid in ["model", "_model", "wm", "_wm", "world_model", "_world_model"]:
        if hasattr(wm, mid):
            m = getattr(wm, mid)
            for attr in ["heads", "_heads", "head", "_head"]:
                if hasattr(m, attr):
                    h = getattr(m, attr)
                    if isinstance(h, (dict, nn.ModuleDict)) or hasattr(h, "__contains__"):
                        dec = pick(h)
                        if dec is not None:
                            return dec
            for attr in ["decoder", "_decoder", "dec", "_dec"]:
                if hasattr(m, attr):
                    return getattr(m, attr)

    for attr in ["decoder", "_decoder", "dec", "_dec"]:
        if hasattr(wm, attr):
            return getattr(wm, attr)

    for name, module in wm.named_modules():
        lname = name.lower()
        if "decoder" in lname or lname.endswith("dec") or ".dec" in lname:
            return module

    raise AttributeError("Could not find decoder head.")


@torch.no_grad()
def per_timestep_nll_pp(wm: WorldModel, ep: Dict[str, np.ndarray]) -> np.ndarray:
    global DEBUG_SHAPES_ONCE

    data = wm.preprocess(ep)
    for k in data:
        data[k] = data[k].unsqueeze(0)

    embed = wm.encoder(data)
    post, prior = wm.dynamics.observe(embed, data["action"], data["is_first"])
    feat = wm.dynamics.get_feat(post)

    dec = _get_decoder_head(wm)
    pred = dec(feat)
    dist = pred["image"] if isinstance(pred, dict) and "image" in pred else pred
    target = data["image"]

    logp = dist.log_prob(target)

    if DEBUG_SHAPES_ONCE:
        print(f"[debug] target shape: {tuple(target.shape)}")
        print(f"[debug] logp   shape: {tuple(logp.shape)}")
        DEBUG_SHAPES_ONCE = False

    if logp.ndim == 5:
        logp_bt = logp.sum(dim=(-1, -2, -3))
    elif logp.ndim == 4:
        logp_bt = logp.sum(dim=(-1, -2))
    elif logp.ndim == 3:
        logp_bt = logp.sum(dim=-1)
    elif logp.ndim == 2:
        logp_bt = logp
    else:
        raise RuntimeError(f"Unexpected log_prob shape {tuple(logp.shape)}")

    nll = (-logp_bt).squeeze(0)  # (T,)
    H, W, C = int(target.shape[-3]), int(target.shape[-2]), int(target.shape[-1])
    return (nll / float(H * W * C)).detach().cpu().numpy().astype(np.float64)


def smooth_ma(x: np.ndarray, w: int) -> np.ndarray:
    if w is None or int(w) <= 1:
        return x
    w = int(w)
    if len(x) < w:
        return x
    kernel = np.ones((w,), dtype=np.float64) / float(w)
    return np.convolve(x, kernel, mode="same")


def apply_burn_in(x: np.ndarray, burn: int) -> np.ndarray:
    if burn <= 0:
        return x
    if len(x) <= burn:
        return x[-1:]
    return x[burn:]


def topk_mean(x: np.ndarray, k: int) -> float:
    if len(x) == 0:
        return 0.0
    kk = min(int(k), len(x))
    return float(np.mean(np.sort(x)[-kk:]))


def start_detect() -> int:
    return int(BURN_IN + BASE_W + IGNORE_AFTER_BASE)


def residual_series(nll_pp: np.ndarray) -> Tuple[np.ndarray, float]:
    start = min(BURN_IN, len(nll_pp) - 1)
    end = min(start + BASE_W, len(nll_pp))
    base_window = nll_pp[start:end]
    baseline = float(np.median(base_window)) if len(base_window) else float(np.median(nll_pp))
    return (nll_pp - baseline).astype(np.float64), baseline


def episode_score(r: np.ndarray) -> float:
    x = apply_burn_in(r, start_detect())
    if len(x) == 0:
        return 0.0
    if SCORE_MODE == "max":
        return float(np.max(x))
    if SCORE_MODE == "topk_mean":
        return topk_mean(x, TOPK)
    raise ValueError(SCORE_MODE)


def first_crossing_persist(r: np.ndarray, thr: float, persist: int) -> Optional[int]:
    x = apply_burn_in(r, start_detect())
    if len(x) == 0:
        return None
    above = (x > thr).astype(np.int32)
    run = 0
    mm = max(1, int(persist))
    for i, a in enumerate(above):
        run = run + 1 if a else 0
        if run >= mm:
            start_rel = i - mm + 1
            return int(start_rel + start_detect())
    return None


def main():
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    action_dim = int(ckpt["action_dim"])

    print(f"[CKPT] action_dim={action_dim} image_shape={tuple(ckpt['image_shape'])} device={ckpt['config']['device']}")
    print(f"[CFG ] VIEW_MODE={VIEW_MODE} TARGET_HW={TARGET_HW}")
    print(f"[CFG ] BURN_IN={BURN_IN} BASE_W={BASE_W} IGNORE_AFTER_BASE={IGNORE_AFTER_BASE} SMOOTH_W={SMOOTH_W}")
    print(f"[CFG ] ALPHA_RES_STEP={ALPHA_RES_STEP} PERSIST={PERSIST} SCORE_MODE={SCORE_MODE} TOPK={TOPK}")

    wm = build_wm_from_ckpt(ckpt)
    wm.eval()

    # ---------- CALIB ----------
    calib_paths = sorted(glob.glob(CALIB_GLOB))
    resid_steps = []
    n_success = 0

    for p in tqdm(calib_paths, desc="CALIB (success only)"):
        meta, steps = load_fiper_pkl(p)
        if not bool(meta.get("successful", False)):
            continue
        n_success += 1
        ep = build_episode(steps, action_dim)
        nll = smooth_ma(per_timestep_nll_pp(wm, ep), SMOOTH_W)
        r, _ = residual_series(nll)
        x = apply_burn_in(r, start_detect())
        if len(x) > 0:
            resid_steps.append(x)

    if n_success == 0:
        raise RuntimeError("No successful calibration episodes found.")

    resid_steps = np.concatenate(resid_steps, axis=0).astype(np.float64)
    thr = conformal_upper_quantile(resid_steps, ALPHA_RES_STEP)

    print(f"[CALIB] n_success_eps={n_success} thr={thr:.8f}")
    print(f"[CALIB] residual steps used: {resid_steps.shape[0]}")

    # ---------- TEST ----------
    test_paths = sorted(glob.glob(TEST_GLOB))
    y_true, y_pred = [], []
    scores = []
    det_times, det_times_norm = [], []

    for p in tqdm(test_paths, desc="TEST"):
        meta, steps = load_fiper_pkl(p)
        ep = build_episode(steps, action_dim)
        nll = smooth_ma(per_timestep_nll_pp(wm, ep), SMOOTH_W)
        r, _ = residual_series(nll)

        scores.append(episode_score(r))

        succ = bool(meta.get("successful", False))
        yi = 0 if succ else 1
        y_true.append(yi)

        dt = first_crossing_persist(r, thr, PERSIST)
        pred = 1 if dt is not None else 0
        y_pred.append(pred)

        if yi == 1 and pred == 1 and dt is not None:
            det_times.append(int(dt))
            denom = max(int(len(r)) - 1, 1)
            det_times_norm.append(float(dt / denom))

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)

    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn + EPS)
    tnr = tn / (tn + fp + EPS)
    bal = 0.5 * (tpr + tnr)
    auroc = safe_auroc(y_true, scores)
    mean_dt = float(np.mean(det_times)) if det_times else float("nan")
    mean_dt_norm = float(np.mean(det_times_norm)) if det_times_norm else float("nan")

    print("\n[TEST] Results (RSSM NLL residual + conformal + persistence)  [BEST]")
    print(f"  Accuracy      : {acc:.4f}")
    print(f"  TPR (failures): {tpr:.4f}")
    print(f"  TNR (success) : {tnr:.4f}")
    print(f"  BalAcc        : {bal:.4f}")
    print(f"  AUROC (residual score): {auroc:.4f}")
    print(f"  Confusion     : TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"  Detected failures: {len(det_times)} / {int((y_true==1).sum())}")
    print(f"  Mean detection time (steps): {mean_dt:.2f}")
    print(f"  Mean detection time (normalized): {mean_dt_norm:.4f}")

    out = {
        "thr_resid_step": float(thr),
        "alpha_res_step": float(ALPHA_RES_STEP),
        "persist": int(PERSIST),
        "burn_in": int(BURN_IN),
        "base_w": int(BASE_W),
        "ignore_after_base": int(IGNORE_AFTER_BASE),
        "smooth_w": int(SMOOTH_W),
        "view_mode": VIEW_MODE,
        "target_hw": list(TARGET_HW),
        "score_mode": SCORE_MODE,
        "topk": int(TOPK),
        "metrics": {
            "accuracy": float(acc),
            "TPR": float(tpr),
            "TNR": float(tnr),
            "bal_acc": float(bal),
            "AUROC_residual_score": float(auroc) if not np.isnan(auroc) else None,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "mean_detection_time": float(mean_dt),
            "mean_detection_time_norm": float(mean_dt_norm),
        },
    }

    out_json = os.path.join(
        OUT_DIR,
        f"rssm_nll_resid_best_alpha{ALPHA_RES_STEP:.2f}_{VIEW_MODE}_p{PERSIST}_burn{BURN_IN}_base{BASE_W}_ign{IGNORE_AFTER_BASE}_w{SMOOTH_W}.json"
    )
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved:", out_json)


if __name__ == "__main__":
    main()