# final_conformal_monitor_rssm_nll_residual_all5_FIXED.py
#
# Runs YOUR RSSM likelihood-residual step-conformal + persistence monitor
# across ALL 5 tasks: pretzel, push_chair, push_t, sorting, stacking.
#
# Fix included:
# - Robust image conversion to uint8 HWC (handles shapes like (1,1,320), CHW, batched, grayscale, etc.)
#   so PIL resize never crashes.
#
# Usage:
#   python final_conformal_monitor_rssm_nll_residual_all5_FIXED.py
#   python final_conformal_monitor_rssm_nll_residual_all5_FIXED.py --tasks push_t sorting
#
# ------------------------------------------------------------

import os, glob, pickle, json
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from models import WorldModel


# =========================
# TASK CONFIG (EDIT PATHS / SIZES HERE)
# =========================
DATA_ROOT = "/data/home/buddhig/data_all"
WM_ROOT   = "/data/home/buddhig/projects/dreamer_fiper_offline/all5_tasks"

# NOTE:
# - sorting/stacking: VIEW_MODE in {"left64","right64","both6"} and TARGET_HW must match training.
# - single-cam tasks: VIEW_MODE="single" and TARGET_HW must match training.
# - pad_to_square must match training. (If you trained by letterboxing to square then resizing, set True.)
TASKS_CONFIG = {
    "sorting": dict(
        calib_glob=f"{DATA_ROOT}/sorting/rollouts/calibration/*.pkl",
        test_glob=f"{DATA_ROOT}/sorting/rollouts/test/*.pkl",
        ckpt_path=f"{WM_ROOT}/sorting/wm_success_only_rgb_both_views_6ch_64.pt",
        out_dir=f"{WM_ROOT}/sorting/monitor_rssm_nll_residual_best_all5",
        target_hw=(64, 64),
        view_mode="both6",          # "left64" | "right64" | "both6"
        pad_to_square=False,
    ),
    "stacking": dict(
        calib_glob=f"{DATA_ROOT}/stacking/rollouts/calibration/*.pkl",
        test_glob=f"{DATA_ROOT}/stacking/rollouts/test/*.pkl",
        ckpt_path=f"{WM_ROOT}/stacking/wm_success_only_rgb.pt",
        out_dir=f"{WM_ROOT}/stacking/monitor_rssm_nll_residual_best_all5",
        target_hw=(64, 64),
        view_mode="both6",
        pad_to_square=False,
    ),
    "push_t": dict(
        calib_glob=f"{DATA_ROOT}/push_t/rollouts/calibration/*.pkl",
        test_glob=f"{DATA_ROOT}/push_t/rollouts/test/*.pkl",
        ckpt_path=f"{WM_ROOT}/push_t/wm_success_only_rgb.pt",
        out_dir=f"{WM_ROOT}/push_t/monitor_rssm_nll_residual_best_all5",
        target_hw=(512, 512),       # MUST match training
        view_mode="single",
        pad_to_square=True,
    ),
    "pretzel": dict(
        calib_glob=f"{DATA_ROOT}/pretzel/rollouts/calibration/*.pkl",
        test_glob=f"{DATA_ROOT}/pretzel/rollouts/test/*.pkl",
        ckpt_path=f"{WM_ROOT}/pretzel/wm_success_only_rgb.pt",
        out_dir=f"{WM_ROOT}/pretzel/monitor_rssm_nll_residual_best_all5",
        target_hw=(256, 256),       # MUST match training (edit if different)
        view_mode="single",
        pad_to_square=True,
    ),
    "push_chair": dict(
        calib_glob=f"{DATA_ROOT}/push_chair/rollouts/calibration/*.pkl",
        test_glob=f"{DATA_ROOT}/push_chair/rollouts/test/*.pkl",
        ckpt_path=f"{WM_ROOT}/push_chair/wm_success_only_rgb.pt",
        out_dir=f"{WM_ROOT}/push_chair/monitor_rssm_nll_residual_best_all5",
        target_hw=(256, 256),       # MUST match training (edit if different)
        view_mode="single",
        pad_to_square=True,
    ),
}

# =========================
# WINDOWS / SMOOTHING (defaults)
# (override per-task by adding keys in TASKS_CONFIG if needed)
# =========================
BURN_IN = 4
BASE_W = 20
IGNORE_AFTER_BASE = 10
SMOOTH_W = 7

# =========================
# MONITOR (defaults)
# =========================
ALPHA_RES_STEP = 0.02
PERSIST = 8

# AUROC ranking score (not used for decision)
SCORE_MODE = "topk_mean"  # "max" | "topk_mean"
TOPK = 15

EPS = 1e-8
DEBUG_SHAPES_ONCE = True


# =========================
# Conformal helpers
# =========================
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


# =========================
# Robust image conversion (FIX)
# =========================
def to_uint8_hwc(x: np.ndarray) -> np.ndarray:
    """
    Convert various image formats to uint8 HWC.
    Handles:
      - HWC, CHW
      - grayscale HW / HW1 / 1HW
      - batched 1xHWC
      - weird flattened like (1,1,N) or (N,)
    """
    x = np.asarray(x)

    # squeeze trivial dims: (1,H,W,C)->(H,W,C) , (1,1,N)->(N,)
    x = np.squeeze(x)

    # flattened vector -> try reshape to HWC with 3 channels
    if x.ndim == 1:
        n = int(x.shape[0])
        if n % 3 == 0:
            pix = n // 3
            h = int(np.floor(np.sqrt(pix)))
            if h > 0 and pix % h == 0:
                w = pix // h
                x = x.reshape(h, w, 3)
            else:
                # fallback (rare): treat as 1xN grayscale-ish
                x = x.reshape(1, n, 1)

    # grayscale HW -> HWC
    if x.ndim == 2:
        x = np.stack([x] * 3, axis=-1)

    # HxW×1 -> replicate to 3ch
    if x.ndim == 3 and x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)

    # CHW -> HWC (common torch)
    if x.ndim == 3 and x.shape[0] in (1, 3, 6) and x.shape[-1] not in (1, 3, 6):
        x = np.transpose(x, (1, 2, 0))

    if x.ndim != 3:
        raise TypeError(f"to_uint8_hwc: expected 3D image after conversion, got shape={x.shape}")

    # dtype convert
    if x.dtype != np.uint8:
        mx = float(np.max(x)) if x.size else 1.0
        if mx <= 1.0:
            x = (x * 255.0).clip(0, 255).astype(np.uint8)
        else:
            x = x.clip(0, 255).astype(np.uint8)

    return x


def _resize_uint8_hwc(img: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    img = to_uint8_hwc(img)
    pil = Image.fromarray(img)
    pil = pil.resize((hw[1], hw[0]), resample=Image.BILINEAR)
    out = np.asarray(pil, dtype=np.uint8)
    if out.ndim == 2:
        out = np.stack([out] * 3, axis=-1)
    return out


def _letterbox_to_square(img: np.ndarray) -> np.ndarray:
    img = to_uint8_hwc(img)
    H, W, C = img.shape
    S = max(H, W)
    out = np.zeros((S, S, C), dtype=np.uint8)
    y0 = (S - H) // 2
    x0 = (S - W) // 2
    out[y0:y0+H, x0:x0+W] = img
    return out


def _split_lr(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rgb = to_uint8_hwc(rgb)
    H, W, C = rgb.shape
    if C != 3:
        raise ValueError(f"Expected RGB 3ch before split, got {rgb.shape}")
    if W % 2 != 0:
        raise ValueError(f"Expected even width for concat views, got W={W}")
    half = W // 2
    return rgb[:, :half, :], rgb[:, half:, :]


def preprocess_frame(rgb: np.ndarray, target_hw: Tuple[int, int], view_mode: str, pad_to_square: bool) -> np.ndarray:
    rgb = to_uint8_hwc(rgb)

    if view_mode == "single":
        img = rgb
        if pad_to_square:
            img = _letterbox_to_square(img)
        img = _resize_uint8_hwc(img, target_hw)
        return img

    left, right = _split_lr(rgb)
    left = _resize_uint8_hwc(left, target_hw)
    right = _resize_uint8_hwc(right, target_hw)

    if view_mode == "left64":
        return left
    if view_mode == "right64":
        return right
    if view_mode == "both6":
        return np.concatenate([left, right], axis=-1)
    raise ValueError(view_mode)


# =========================
# FIPER loader
# =========================
def load_fiper_pkl(path: str):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["metadata"], d["rollout"]


def build_episode(steps, action_dim: int, target_hw: Tuple[int, int], view_mode: str, pad_to_square: bool) -> Dict[str, np.ndarray]:
    T = len(steps)
    images = np.stack(
        [preprocess_frame(s["rgb"], target_hw, view_mode, pad_to_square) for s in steps],
        axis=0
    ).astype(np.uint8)

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


# =========================
# Build WM from ckpt
# =========================
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


# =========================
# Decoder head finder
# =========================
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


# =========================
# RSSM likelihood per timestep (NLL per pixel)
# =========================
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


# =========================
# Smoothing + residual + detection
# =========================
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


def start_detect(burn_in: int, base_w: int, ignore_after: int) -> int:
    return int(burn_in + base_w + ignore_after)


def residual_series(nll_pp: np.ndarray, burn_in: int, base_w: int) -> Tuple[np.ndarray, float]:
    start = min(burn_in, len(nll_pp) - 1)
    end = min(start + base_w, len(nll_pp))
    base_window = nll_pp[start:end]
    baseline = float(np.median(base_window)) if len(base_window) else float(np.median(nll_pp))
    return (nll_pp - baseline).astype(np.float64), baseline


def episode_score(r: np.ndarray, burn_in: int, base_w: int, ignore_after: int, score_mode: str, topk: int) -> float:
    x = apply_burn_in(r, start_detect(burn_in, base_w, ignore_after))
    if len(x) == 0:
        return 0.0
    if score_mode == "max":
        return float(np.max(x))
    if score_mode == "topk_mean":
        return topk_mean(x, topk)
    raise ValueError(score_mode)


def first_crossing_persist(
    r: np.ndarray, thr: float, persist: int, burn_in: int, base_w: int, ignore_after: int
) -> Optional[int]:
    x = apply_burn_in(r, start_detect(burn_in, base_w, ignore_after))
    if len(x) == 0:
        return None
    above = (x > thr).astype(np.int32)
    run = 0
    mm = max(1, int(persist))
    for i, a in enumerate(above):
        run = run + 1 if a else 0
        if run >= mm:
            start_rel = i - mm + 1
            return int(start_rel + start_detect(burn_in, base_w, ignore_after))
    return None


# =========================
# Per-task runner
# =========================
def run_task(task: str, cfg: dict):
    global DEBUG_SHAPES_ONCE
    DEBUG_SHAPES_ONCE = True  # print shapes once per task

    calib_glob = cfg["calib_glob"]
    test_glob  = cfg["test_glob"]
    ckpt_path  = cfg["ckpt_path"]
    out_dir    = cfg["out_dir"]
    target_hw  = tuple(cfg["target_hw"])
    view_mode  = str(cfg["view_mode"])
    pad_sq     = bool(cfg.get("pad_to_square", False))

    burn_in = int(cfg.get("burn_in", BURN_IN))
    base_w  = int(cfg.get("base_w", BASE_W))
    ign     = int(cfg.get("ignore_after_base", IGNORE_AFTER_BASE))
    smooth_w= int(cfg.get("smooth_w", SMOOTH_W))

    alpha_step = float(cfg.get("alpha_res_step", ALPHA_RES_STEP))
    persist    = int(cfg.get("persist", PERSIST))

    score_mode = str(cfg.get("score_mode", SCORE_MODE))
    topk       = int(cfg.get("topk", TOPK))

    os.makedirs(out_dir, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    action_dim = int(ckpt["action_dim"])

    print("\n====================")
    print(f"[TASK] {task}")
    print(f"[CKPT] {ckpt_path}")
    print(f"[CKPT] action_dim={action_dim} image_shape={tuple(ckpt['image_shape'])} device={ckpt['config']['device']}")
    print(f"[CFG ] VIEW_MODE={view_mode} TARGET_HW={target_hw} PAD_TO_SQUARE={pad_sq}")
    print(f"[CFG ] BURN_IN={burn_in} BASE_W={base_w} IGNORE_AFTER_BASE={ign} SMOOTH_W={smooth_w}")
    print(f"[CFG ] ALPHA_RES_STEP={alpha_step} PERSIST={persist} SCORE_MODE={score_mode} TOPK={topk}")

    wm = build_wm_from_ckpt(ckpt)
    wm.eval()

    # ---------- CALIB ----------
    calib_paths = sorted(glob.glob(calib_glob))
    resid_steps = []
    n_success = 0

    for p in tqdm(calib_paths, desc=f"{task} CALIB (success only)"):
        meta, steps = load_fiper_pkl(p)
        if not bool(meta.get("successful", False)):
            continue
        n_success += 1
        ep = build_episode(steps, action_dim, target_hw, view_mode, pad_sq)
        nll = smooth_ma(per_timestep_nll_pp(wm, ep), smooth_w)
        r, _ = residual_series(nll, burn_in, base_w)
        x = apply_burn_in(r, start_detect(burn_in, base_w, ign))
        if len(x) > 0:
            resid_steps.append(x)

    if n_success == 0:
        raise RuntimeError(f"[{task}] No successful calibration episodes found.")

    resid_steps = np.concatenate(resid_steps, axis=0).astype(np.float64)
    thr = conformal_upper_quantile(resid_steps, alpha_step)

    print(f"[CALIB] n_success_eps={n_success} thr={thr:.8f}")
    print(f"[CALIB] residual steps used: {resid_steps.shape[0]}")

    # ---------- TEST ----------
    test_paths = sorted(glob.glob(test_glob))
    y_true, y_pred = [], []
    scores = []
    det_times, det_times_norm = [], []

    for p in tqdm(test_paths, desc=f"{task} TEST"):
        meta, steps = load_fiper_pkl(p)
        ep = build_episode(steps, action_dim, target_hw, view_mode, pad_sq)
        nll = smooth_ma(per_timestep_nll_pp(wm, ep), smooth_w)
        r, _ = residual_series(nll, burn_in, base_w)

        scores.append(episode_score(r, burn_in, base_w, ign, score_mode, topk))

        succ = bool(meta.get("successful", False))
        yi = 0 if succ else 1
        y_true.append(yi)

        dt = first_crossing_persist(r, thr, persist, burn_in, base_w, ign)
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

    print(f"\n[TEST] Results ({task}) (RSSM NLL residual + step-conformal + persistence)")
    print(f"  Accuracy      : {acc:.4f}")
    print(f"  TPR (failures): {tpr:.4f}")
    print(f"  TNR (success) : {tnr:.4f}")
    print(f"  BalAcc        : {bal:.4f}")
    print(f"  AUROC (episode score): {auroc:.4f}  (score used only for ranking)")
    print(f"  Confusion     : TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"  Detected failures: {len(det_times)} / {int((y_true==1).sum())}")
    print(f"  Mean detection time (steps): {mean_dt:.2f}")
    print(f"  Mean detection time (normalized): {mean_dt_norm:.4f}")

    out = {
        "task": task,
        "thr_resid_step": float(thr),
        "alpha_res_step": float(alpha_step),
        "persist": int(persist),
        "burn_in": int(burn_in),
        "base_w": int(base_w),
        "ignore_after_base": int(ign),
        "smooth_w": int(smooth_w),
        "view_mode": view_mode,
        "pad_to_square": bool(pad_sq),
        "target_hw": list(target_hw),
        "score_mode": score_mode,
        "topk": int(topk),
        "metrics": {
            "accuracy": float(acc),
            "TPR": float(tpr),
            "TNR": float(tnr),
            "bal_acc": float(bal),
            "AUROC_episode_score": float(auroc) if not np.isnan(auroc) else None,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "mean_detection_time": float(mean_dt),
            "mean_detection_time_norm": float(mean_dt_norm),
            "num_failures": int((y_true==1).sum()),
            "num_detected_failures": int(len(det_times)),
        },
        "paths": {
            "calib_glob": calib_glob,
            "test_glob": test_glob,
            "ckpt_path": ckpt_path,
        },
    }

    out_json = os.path.join(
        out_dir,
        f"rssm_nll_resid_alpha{alpha_step:.2f}_{view_mode}_p{persist}_burn{burn_in}_base{base_w}_ign{ign}_w{smooth_w}.json"
    )
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved:", out_json)

    del wm
    torch.cuda.empty_cache()


# =========================
# Main
# =========================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=str, nargs="*", default=list(TASKS_CONFIG.keys()))
    args = ap.parse_args()

    for t in args.tasks:
        if t not in TASKS_CONFIG:
            print(f"[SKIP] unknown task: {t}")
            continue
        run_task(t, TASKS_CONFIG[t])