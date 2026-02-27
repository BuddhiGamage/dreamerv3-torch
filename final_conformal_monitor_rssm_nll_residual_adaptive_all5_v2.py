#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
final_conformal_monitor_rssm_nll_residual_adaptive_all5_v2.py

Fixes:
  ✅ robust RGB handling (HWC/CHW/grayscale/singletons) -> avoids PIL TypeError (pretzel)
  ✅ supports two decision modes:
        - episode_score (recommended; mirrors your PCA monitor; strong on push_t)
        - step_persist (your original per-step threshold + persistence)
  ✅ episode-length-aware detection start + adaptive persistence for short episodes

Usage examples:

# Pretzel (episode_score decision recommended)
python final_conformal_monitor_rssm_nll_residual_adaptive_all5_v2.py \
  --task pretzel \
  --calib_glob "/home/s447658/project/fiper/data/pretzel/rollouts/calibration/*.pkl" \
  --test_glob  "/home/s447658/project/fiper/data/pretzel/rollouts/test/*.pkl" \
  --ckpt_path  "/home/s447658/projects/dreamer_fiper_offline/all5_tasks/pretzel/wm_success_only_rgb.pt" \
  --out_dir    "/home/s447658/projects/dreamer_fiper_offline/all5_tasks/pretzel/monitor_rssm_nll_residual_v2" \
  --view_mode single --target_hw 256 256 --pad_to_square 1 \
  --burn_in 4 --base_w 12 --ignore_after_base 2 --smooth_w 5 \
  --alpha 0.05 --use_robust_z 1 \
  --decision_mode episode_score --topk 15

# Push-T (episode_score decision recommended; like PCA)
python final_conformal_monitor_rssm_nll_residual_adaptive_all5_v2.py \
  --task push_t \
  --calib_glob "/home/s447658/project/fiper/data/push_t/rollouts/calibration/*.pkl" \
  --test_glob  "/home/s447658/project/fiper/data/push_t/rollouts/test/*.pkl" \
  --ckpt_path  "/home/s447658/projects/dreamer_fiper_offline/all5_tasks/push_t/wm_success_only_rgb.pt" \
  --out_dir    "/home/s447658/projects/dreamer_fiper_offline/all5_tasks/push_t/monitor_rssm_nll_residual_v2" \
  --view_mode single --target_hw 512 512 --pad_to_square 1 \
  --burn_in 2 --base_w 6 --ignore_after_base 0 --smooth_w 3 \
  --alpha 0.10 --use_robust_z 1 \
  --decision_mode episode_score --topk 7

Notes:
- Requires: models.py providing WorldModel
- Assumes FIPER PKL: d["metadata"], d["rollout"], each step has step["rgb"]
"""

import os, glob, pickle, json, argparse
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from models import WorldModel

EPS = 1e-8


# =========================
# Utilities
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

def smooth_ma(x: np.ndarray, w: int) -> np.ndarray:
    if w is None or int(w) <= 1:
        return x
    w = int(w)
    if len(x) < w:
        return x
    kernel = np.ones((w,), dtype=np.float64) / float(w)
    return np.convolve(x, kernel, mode="same")

def apply_burn_in_empty(x: np.ndarray, burn: int) -> np.ndarray:
    """Return EMPTY if burn exceeds length (do NOT keep last element)."""
    if burn <= 0:
        return x
    if len(x) <= burn:
        return np.asarray([], dtype=x.dtype)
    return x[burn:]

def apply_burn_in_keep1(x: np.ndarray, burn: int) -> np.ndarray:
    """Keep at least 1 element (useful for episode_score to avoid empty)."""
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

def pad_to_square_uint8_hwc(img: np.ndarray) -> np.ndarray:
    H, W, C = img.shape
    if H == W:
        return img
    m = max(H, W)
    pad_h = m - H
    pad_w = m - W
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return np.pad(img, ((top, bottom), (left, right), (0, 0)), mode="edge")

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


# =========================
# Robust image conversion
# =========================
def to_uint8_hwc(x: Any) -> np.ndarray:
    """
    Convert various layouts into HWC uint8.
    Handles:
      - HxW (grayscale)
      - HxWxC (C in {1,3,4,6})
      - CxHxW (C in {1,3,4,6})
      - extra singleton dims
      - float images in [0,1]
    """
    arr = np.asarray(x)

    # squeeze singleton dims (common cause of (1,1,320)-style junk)
    # but keep 2D/3D structure if possible
    while arr.ndim > 3 and 1 in arr.shape:
        arr = np.squeeze(arr)

    # If still weird 3D like (1,1,320), squeeze again (be aggressive)
    if arr.ndim == 3 and (arr.shape[0] == 1 or arr.shape[1] == 1) and (arr.shape[2] not in (1,3,4,6)):
        arr = np.squeeze(arr)

    # Now handle 2D -> grayscale -> RGB
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)

    # Handle CHW -> HWC
    if arr.ndim == 3:
        # If last dim is channel-like, assume HWC
        if arr.shape[-1] in (1, 3, 4, 6):
            pass
        # If first dim is channel-like, assume CHW
        elif arr.shape[0] in (1, 3, 4, 6) and arr.shape[-1] not in (1, 3, 4, 6):
            arr = np.transpose(arr, (1, 2, 0))
        else:
            # last-resort heuristic: if one dim is 3 and others look like H,W
            if 3 in arr.shape:
                cidx = int(np.where(np.array(arr.shape) == 3)[0][0])
                if cidx == 0:
                    arr = np.transpose(arr, (1, 2, 0))
                elif cidx == 1:
                    arr = np.transpose(arr, (0, 2, 1))
                # if cidx==2 already HWC
            else:
                raise TypeError(f"Unrecognized image shape for rgb: {arr.shape}")

    else:
        raise TypeError(f"Unrecognized rgb ndim: {arr.ndim} shape={arr.shape}")

    # Convert dtype to uint8
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        maxv = float(np.nanmax(arr)) if arr.size else 0.0
        if maxv <= 1.0 + 1e-6:
            arr = np.clip(arr, 0.0, 1.0) * 255.0
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)

    # If 1-channel, expand to 3
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    return arr


# =========================
# Config
# =========================
@dataclass
class MonitorCfg:
    # preprocessing
    view_mode: str = "single"       # "single" | "left64" | "right64" | "both6"
    target_hw: Tuple[int, int] = (256, 256)
    pad_to_square: bool = True

    # windows
    burn_in: int = 4
    base_w: int = 20
    ignore_after_base: int = 10
    smooth_w: int = 7

    # conformal alpha (used differently depending on decision_mode)
    alpha: float = 0.05

    # decision
    decision_mode: str = "episode_score"   # "episode_score" | "step_persist"
    persist: int = 8
    adaptive_persist: bool = True
    persist_frac: float = 0.2

    # scoring
    score_mode: str = "topk_mean"  # "max" | "topk_mean"
    topk: int = 15

    # residual type
    use_robust_z: bool = True
    robust_z_eps: float = 1e-8

    # debug
    debug_shapes_once: bool = True


# =========================
# FIPER PKL helpers
# =========================
def load_fiper_pkl(path: str):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["metadata"], d["rollout"]


# =========================
# Episode builder
# =========================
def preprocess_frame(rgb_in: Any, cfg: MonitorCfg) -> np.ndarray:
    rgb = to_uint8_hwc(rgb_in)  # <-- PRETZEL FIX: always normalize layout/dtype first

    if cfg.view_mode in ("left64", "right64", "both6"):
        left, right = _split_lr(rgb)
        if cfg.pad_to_square:
            left = pad_to_square_uint8_hwc(left)
            right = pad_to_square_uint8_hwc(right)
        left = _resize_uint8_hwc(left, cfg.target_hw)
        right = _resize_uint8_hwc(right, cfg.target_hw)

        if cfg.view_mode == "left64":
            return left
        if cfg.view_mode == "right64":
            return right
        if cfg.view_mode == "both6":
            return np.concatenate([left, right], axis=-1)  # 6ch

    # single-view
    if cfg.pad_to_square:
        rgb = pad_to_square_uint8_hwc(rgb)
    rgb = _resize_uint8_hwc(rgb, cfg.target_hw)
    return rgb


def build_episode(steps, action_dim: int, cfg: MonitorCfg) -> Dict[str, np.ndarray]:
    T = len(steps)
    images = np.stack([preprocess_frame(s["rgb"], cfg) for s in steps], axis=0).astype(np.uint8)

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
# World model loader
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
# NLL per timestep
# =========================
@torch.no_grad()
def per_timestep_nll_pp(wm: WorldModel, ep: Dict[str, np.ndarray], cfg: MonitorCfg) -> np.ndarray:
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

    if cfg.debug_shapes_once:
        print(f"[debug] target shape: {tuple(target.shape)}")
        print(f"[debug] logp   shape: {tuple(logp.shape)}")
        cfg.debug_shapes_once = False

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
# Residual series + window logic
# =========================
def start_detect_T(T: int, cfg: MonitorCfg) -> int:
    nominal = int(cfg.burn_in + cfg.base_w + cfg.ignore_after_base)
    return int(min(nominal, max(0, T - 1)))

def persist_T(T: int, cfg: MonitorCfg) -> int:
    if not cfg.adaptive_persist:
        return int(cfg.persist)
    p = max(2, int(round(cfg.persist_frac * T)))
    return int(min(p, max(2, int(cfg.persist))))

def residual_series_raw(nll_pp: np.ndarray, cfg: MonitorCfg) -> Tuple[np.ndarray, Dict[str, float]]:
    start = min(cfg.burn_in, len(nll_pp) - 1) if len(nll_pp) else 0
    end = min(start + cfg.base_w, len(nll_pp))
    base = nll_pp[start:end] if end > start else nll_pp
    baseline = float(np.median(base)) if len(base) else (float(np.median(nll_pp)) if len(nll_pp) else 0.0)
    r = (nll_pp - baseline).astype(np.float64)
    return r, {"baseline_median": baseline}

def residual_series_robust_z(nll_pp: np.ndarray, cfg: MonitorCfg) -> Tuple[np.ndarray, Dict[str, float]]:
    start = min(cfg.burn_in, len(nll_pp) - 1) if len(nll_pp) else 0
    end = min(start + cfg.base_w, len(nll_pp))
    base = nll_pp[start:end] if end > start else nll_pp

    med = float(np.median(base)) if len(base) else 0.0
    mad = float(np.median(np.abs(base - med))) if len(base) else 0.0
    denom = (1.4826 * mad) + float(cfg.robust_z_eps)
    z = (nll_pp - med) / denom
    return z.astype(np.float64), {"baseline_median": med, "baseline_mad": mad, "z_denom": denom}

def residual_series(nll_pp: np.ndarray, cfg: MonitorCfg) -> Tuple[np.ndarray, Dict[str, float]]:
    return residual_series_robust_z(nll_pp, cfg) if cfg.use_robust_z else residual_series_raw(nll_pp, cfg)

def episode_score_from_r(r: np.ndarray, cfg: MonitorCfg) -> float:
    sd = start_detect_T(len(r), cfg)
    x = apply_burn_in_keep1(r, sd)  # keep 1 so we can always score
    if len(x) == 0:
        return 0.0
    if cfg.score_mode == "max":
        return float(np.max(x))
    if cfg.score_mode == "topk_mean":
        return topk_mean(x, cfg.topk)
    raise ValueError(cfg.score_mode)

def first_crossing_persist_stepthr(r: np.ndarray, thr_step: float, cfg: MonitorCfg) -> Optional[int]:
    T = len(r)
    sd = start_detect_T(T, cfg)
    x = apply_burn_in_empty(r, sd)
    if len(x) == 0:
        return None

    p = persist_T(T, cfg)
    if len(x) < p:
        p = 1  # relax for very short remainder

    above = (x > thr_step).astype(np.int32)
    run = 0
    for i, a in enumerate(above):
        run = run + 1 if a else 0
        if run >= p:
            start_rel = i - p + 1
            return int(start_rel + sd)
    return None

def first_crossing_episode_score(r: np.ndarray, thr_score: float, cfg: MonitorCfg) -> Optional[int]:
    """
    Runtime-ish detection time for episode_score decision:
    find first t where running score (over r[sd:t]) exceeds thr_score.
    For topk_mean: compute over the prefix after sd.
    """
    T = len(r)
    sd = start_detect_T(T, cfg)
    if sd >= T:
        return None

    # prefix after sd
    x = r[sd:].astype(np.float64)
    if len(x) == 0:
        return None

    if cfg.score_mode == "max":
        cur = -np.inf
        for i in range(len(x)):
            cur = max(cur, float(x[i]))
            if cur > thr_score:
                return int(sd + i)
        return None

    if cfg.score_mode == "topk_mean":
        k = max(1, int(cfg.topk))
        # maintain a small buffer; simplest O(T log T) is fine for short episodes
        for i in range(len(x)):
            prefix = x[: i + 1]
            score_i = topk_mean(prefix, k)
            if score_i > thr_score:
                return int(sd + i)
        return None

    raise ValueError(cfg.score_mode)


# =========================
# Main
# =========================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, default="task")

    ap.add_argument("--calib_glob", type=str, required=True)
    ap.add_argument("--test_glob", type=str, required=True)
    ap.add_argument("--ckpt_path", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    # preprocessing
    ap.add_argument("--view_mode", type=str, default="single", choices=["single", "left64", "right64", "both6"])
    ap.add_argument("--target_hw", type=int, nargs=2, default=[256, 256])
    ap.add_argument("--pad_to_square", type=int, default=1)

    # windows
    ap.add_argument("--burn_in", type=int, default=4)
    ap.add_argument("--base_w", type=int, default=20)
    ap.add_argument("--ignore_after_base", type=int, default=10)
    ap.add_argument("--smooth_w", type=int, default=7)

    # alpha
    ap.add_argument("--alpha", type=float, default=0.05)

    # decision
    ap.add_argument("--decision_mode", type=str, default="episode_score",
                    choices=["episode_score", "step_persist"])
    ap.add_argument("--persist", type=int, default=8)
    ap.add_argument("--adaptive_persist", type=int, default=1)
    ap.add_argument("--persist_frac", type=float, default=0.2)

    # score
    ap.add_argument("--score_mode", type=str, default="topk_mean", choices=["max", "topk_mean"])
    ap.add_argument("--topk", type=int, default=15)

    # residual
    ap.add_argument("--use_robust_z", type=int, default=1)
    ap.add_argument("--robust_z_eps", type=float, default=1e-8)

    return ap.parse_args()


def main():
    args = parse_args()

    cfg = MonitorCfg(
        view_mode=args.view_mode,
        target_hw=(int(args.target_hw[0]), int(args.target_hw[1])),
        pad_to_square=bool(args.pad_to_square),

        burn_in=int(args.burn_in),
        base_w=int(args.base_w),
        ignore_after_base=int(args.ignore_after_base),
        smooth_w=int(args.smooth_w),

        alpha=float(args.alpha),

        decision_mode=str(args.decision_mode),
        persist=int(args.persist),
        adaptive_persist=bool(args.adaptive_persist),
        persist_frac=float(args.persist_frac),

        score_mode=str(args.score_mode),
        topk=int(args.topk),

        use_robust_z=bool(args.use_robust_z),
        robust_z_eps=float(args.robust_z_eps),

        debug_shapes_once=True,
    )

    os.makedirs(args.out_dir, exist_ok=True)

    print("====================================")
    print(f"[TASK] {args.task}")
    print(f"[CKPT] {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    action_dim = int(ckpt["action_dim"])
    print(f"[CKPT] action_dim={action_dim} image_shape={tuple(ckpt['image_shape'])} device={ckpt['config']['device']}")

    print(f"[CFG ] view_mode={cfg.view_mode} target_hw={cfg.target_hw} pad_to_square={cfg.pad_to_square}")
    print(f"[CFG ] burn_in={cfg.burn_in} base_w={cfg.base_w} ignore_after_base={cfg.ignore_after_base} smooth_w={cfg.smooth_w}")
    print(f"[CFG ] alpha={cfg.alpha} decision_mode={cfg.decision_mode}")
    print(f"[CFG ] score_mode={cfg.score_mode} topk={cfg.topk}")
    print(f"[CFG ] persist(max)={cfg.persist} adaptive_persist={cfg.adaptive_persist} persist_frac={cfg.persist_frac}")
    print(f"[CFG ] use_robust_z={cfg.use_robust_z}")

    wm = build_wm_from_ckpt(ckpt)
    wm.eval()

    calib_paths = sorted(glob.glob(args.calib_glob))
    test_paths = sorted(glob.glob(args.test_glob))

    # -------------------------
    # CALIBRATION
    # -------------------------
    n_success = 0

    if cfg.decision_mode == "episode_score":
        # Calibrate on SUCCESS ROLLOUT SCORES (like your PCA monitor)
        calib_scores: List[float] = []

        for p in tqdm(calib_paths, desc=f"{args.task} CALIB (success only)"):
            meta, steps = load_fiper_pkl(p)
            if not bool(meta.get("successful", False)):
                continue
            n_success += 1

            ep = build_episode(steps, action_dim, cfg)
            nll = smooth_ma(per_timestep_nll_pp(wm, ep, cfg), cfg.smooth_w)
            r, _ = residual_series(nll, cfg)

            calib_scores.append(episode_score_from_r(r, cfg))

        if n_success == 0:
            raise RuntimeError("No successful calibration episodes found.")

        calib_scores_np = np.asarray(calib_scores, dtype=np.float64)
        thr = conformal_upper_quantile(calib_scores_np, cfg.alpha)

        print(f"[CALIB] n_success_eps={n_success}")
        print(f"[CALIB] rollout scores used: {len(calib_scores_np)}")
        print(f"[CALIB] thr_score={thr:.10f}")
        print(f"[CALIB] score stats: mean={calib_scores_np.mean():.4f} std={calib_scores_np.std():.4f} "
              f"min={calib_scores_np.min():.4f} max={calib_scores_np.max():.4f}")

    elif cfg.decision_mode == "step_persist":
        # Calibrate on SUCCESS TIMESTEPS (your original idea)
        resid_steps: List[np.ndarray] = []
        skipped_empty = 0

        for p in tqdm(calib_paths, desc=f"{args.task} CALIB (success only)"):
            meta, steps = load_fiper_pkl(p)
            if not bool(meta.get("successful", False)):
                continue
            n_success += 1

            ep = build_episode(steps, action_dim, cfg)
            nll = smooth_ma(per_timestep_nll_pp(wm, ep, cfg), cfg.smooth_w)
            r, _ = residual_series(nll, cfg)

            sd = start_detect_T(len(r), cfg)
            x = apply_burn_in_empty(r, sd)
            if len(x) == 0:
                skipped_empty += 1
                continue
            resid_steps.append(x)

        if n_success == 0:
            raise RuntimeError("No successful calibration episodes found.")
        if len(resid_steps) == 0:
            raise RuntimeError("All successful calibration episodes produced empty detection regions.")

        resid_steps_all = np.concatenate(resid_steps, axis=0).astype(np.float64)
        thr = conformal_upper_quantile(resid_steps_all, cfg.alpha)

        print(f"[CALIB] n_success_eps={n_success} skipped_empty={skipped_empty}")
        print(f"[CALIB] residual steps used: {resid_steps_all.shape[0]}")
        print(f"[CALIB] thr_step={thr:.10f}")

    else:
        raise ValueError(cfg.decision_mode)

    # -------------------------
    # TEST
    # -------------------------
    y_true: List[int] = []
    y_pred: List[int] = []
    scores: List[float] = []
    det_times: List[int] = []
    det_times_norm: List[float] = []

    for p in tqdm(test_paths, desc=f"{args.task} TEST"):
        meta, steps = load_fiper_pkl(p)
        ep = build_episode(steps, action_dim, cfg)
        nll = smooth_ma(per_timestep_nll_pp(wm, ep, cfg), cfg.smooth_w)
        r, _ = residual_series(nll, cfg)

        succ = bool(meta.get("successful", False))
        yi = 0 if succ else 1
        y_true.append(yi)

        if cfg.decision_mode == "episode_score":
            score = episode_score_from_r(r, cfg)
            pred = 1 if score > thr else 0
            scores.append(score)

            dt = first_crossing_episode_score(r, thr, cfg)
        else:
            # step_persist
            score = episode_score_from_r(r, cfg)  # still useful for AUROC ranking
            scores.append(score)

            dt = first_crossing_persist_stepthr(r, thr, cfg)
            pred = 1 if dt is not None else 0

        y_pred.append(pred)

        if yi == 1 and pred == 1 and dt is not None:
            det_times.append(int(dt))
            denom = max(int(len(r)) - 1, 1)
            det_times_norm.append(float(dt / denom))

    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)
    scores_np = np.asarray(scores, dtype=np.float64)

    acc = accuracy_score(y_true_np, y_pred_np)
    tn, fp, fn, tp = confusion_matrix(y_true_np, y_pred_np, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn + EPS)
    tnr = tn / (tn + fp + EPS)
    bal = 0.5 * (tpr + tnr)
    auroc = safe_auroc(y_true_np, scores_np)
    mean_dt = float(np.mean(det_times)) if det_times else float("nan")
    mean_dt_norm = float(np.mean(det_times_norm)) if det_times_norm else float("nan")

    print("\n[TEST] Results (RSSM NLL residual monitor)  [V2]")
    print(f"  decision_mode : {cfg.decision_mode}")
    if cfg.decision_mode == "episode_score":
        print(f"  thr_score     : {thr:.6f}")
    else:
        print(f"  thr_step      : {thr:.6f}")
    print(f"  Accuracy      : {acc:.4f}")
    print(f"  TPR (failures): {tpr:.4f}")
    print(f"  TNR (success) : {tnr:.4f}")
    print(f"  BalAcc        : {bal:.4f}")
    print(f"  AUROC (score) : {auroc:.4f}  (ranking score = episode_score)")
    print(f"  Confusion     : TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"  Detected failures: {len(det_times)} / {int((y_true_np==1).sum())}")
    print(f"  Mean detection time (steps): {mean_dt:.2f}")
    print(f"  Mean detection time (normalized): {mean_dt_norm:.4f}")

    out: Dict[str, Any] = {
        "task": args.task,
        "ckpt_path": args.ckpt_path,
        "decision_mode": cfg.decision_mode,
        "alpha": float(cfg.alpha),
        "threshold": float(thr),
        "burn_in": int(cfg.burn_in),
        "base_w": int(cfg.base_w),
        "ignore_after_base": int(cfg.ignore_after_base),
        "smooth_w": int(cfg.smooth_w),
        "view_mode": cfg.view_mode,
        "target_hw": list(cfg.target_hw),
        "pad_to_square": bool(cfg.pad_to_square),
        "score_mode": cfg.score_mode,
        "topk": int(cfg.topk),
        "persist_max": int(cfg.persist),
        "adaptive_persist": bool(cfg.adaptive_persist),
        "persist_frac": float(cfg.persist_frac),
        "use_robust_z": bool(cfg.use_robust_z),
        "metrics": {
            "accuracy": float(acc),
            "TPR": float(tpr),
            "TNR": float(tnr),
            "bal_acc": float(bal),
            "AUROC": float(auroc) if not np.isnan(auroc) else None,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "mean_detection_time": float(mean_dt),
            "mean_detection_time_norm": float(mean_dt_norm),
        },
    }

    out_json = os.path.join(
        args.out_dir,
        f"rssm_nll_resid_v2_{cfg.decision_mode}_alpha{cfg.alpha:.3f}_{cfg.view_mode}"
        f"_burn{cfg.burn_in}_base{cfg.base_w}_ign{cfg.ignore_after_base}_w{cfg.smooth_w}"
        f"_topk{cfg.topk}_z{int(cfg.use_robust_z)}.json"
    )
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved:", out_json)


if __name__ == "__main__":
    main()