#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
extract_resnet_feats_fiper_all5_tasks.py

Extract ResNet18 embeddings from FIPER rollouts to match the feature format expected by:
  wm_progress_monitor_per_task_threshold_episode_calib_v3.py

Outputs per task folder:
  calib_success_feats.npy   (object array of (T, D) success-only)
  test_feats.npy            (object array of (T, D))
  test_labels.npy           (int array: 1=failure, 0=success)

Uses your same preprocessing style:
- sorting/stacking: split concat L/R, letterbox to square, resize, optionally both views
- other tasks: letterbox to square, resize

By default, uses target_hw stored in the WM meta json (so preprocessing matches your WM training),
or falls back to heuristics if meta is unavailable.

Usage:
  python extract_resnet_feats_fiper_all5_tasks.py \
    --data_root /home/s447658/project/fiper/data \
    --out_root  /home/s447658/projects/resnet_fiper_feats_all5 \
    --tasks pretzel push_chair push_t sorting stacking \
    --view_mode_sort both6 \
    --view_mode_stack both6 \
    --device cpu
"""

import os, glob, json, argparse, pickle
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from tqdm import tqdm
from PIL import Image

# -------------------------
# FIPER loader
# -------------------------
def load_fiper_pkl(path: str):
    with open(path, "rb") as f:
        d = pickle.load(f)
    meta = d["metadata"]
    steps = d["rollout"]
    return meta, steps

# -------------------------
# Image utils (copied from your training script, lightly adapted)
# -------------------------
def _to_uint8_hwc(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        mx = float(np.max(img)) if img.size else 1.0
        if mx <= 1.0:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = img.clip(0, 255).astype(np.uint8)
    return img

def _letterbox_to_square(img: np.ndarray) -> np.ndarray:
    img = _to_uint8_hwc(img)
    H, W, C = img.shape
    S = max(H, W)
    out = np.zeros((S, S, C), dtype=np.uint8)
    y0 = (S - H) // 2
    x0 = (S - W) // 2
    out[y0:y0+H, x0:x0+W] = img
    return out

def _resize(img: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    img = _to_uint8_hwc(img)
    pil = Image.fromarray(img)
    pil = pil.resize((hw[1], hw[0]), resample=Image.BILINEAR)
    out = np.asarray(pil, dtype=np.uint8)
    if out.ndim == 2:
        out = np.stack([out] * 3, axis=-1)
    return out

def _split_lr_concat(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rgb = _to_uint8_hwc(rgb)
    H, W, C = rgb.shape
    if C != 3:
        raise ValueError(f"Expected RGB 3ch, got {rgb.shape}")
    if W % 2 != 0:
        raise ValueError(f"Concat view width must be even; got W={W}")
    half = W // 2
    left = rgb[:, :half, :]
    right = rgb[:, half:, :]
    return left, right

def choose_target_hw_from_raw(raw_h: int, raw_w: int, task_name: str) -> Tuple[int, int]:
    if task_name in ("sorting", "stacking"):
        return (64, 64)
    m = max(raw_h, raw_w)
    if m >= 480:
        return (512, 512)
    if m >= 256:
        return (256, 256)
    return (128, 128)

def preprocess_rgb_for_task(
    rgb: np.ndarray,
    task_name: str,
    target_hw: Tuple[int, int],
    view_mode: str,
    pad_to_square: bool = True,
) -> np.ndarray:
    """
    Returns uint8 HWC:
      - single-cam: (H,W,3)
      - both6:      (H,W,6) when view_mode == 'both6' for sorting/stacking
    """
    rgb = _to_uint8_hwc(rgb)

    if task_name in ("sorting", "stacking"):
        left, right = _split_lr_concat(rgb)
        if pad_to_square:
            left = _letterbox_to_square(left)
            right = _letterbox_to_square(right)
        left = _resize(left, target_hw)
        right = _resize(right, target_hw)

        if view_mode == "left":
            return left
        if view_mode == "right":
            return right
        if view_mode == "both6":
            return np.concatenate([left, right], axis=-1)
        raise ValueError(f"Unknown view_mode={view_mode} for {task_name}")

    img = rgb
    if pad_to_square:
        img = _letterbox_to_square(img)
    img = _resize(img, target_hw)
    return img

# -------------------------
# ResNet18 featurizer
# -------------------------
class ResNet18Featurizer(nn.Module):
    def __init__(self, device: str = "cuda"):
        super().__init__()
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(m.children())[:-1])  # to avgpool => (B,512,1,1)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.device = device
        self.backbone.to(device)

        self.tf = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def encode_rgb_uint8(self, img_uint8_hwc: np.ndarray) -> np.ndarray:
        # img_uint8_hwc: (H,W,3)
        pil = Image.fromarray(img_uint8_hwc)
        x = self.tf(pil).unsqueeze(0).to(self.device)  # (1,3,224,224)
        feat = self.backbone(x).squeeze(-1).squeeze(-1)  # (1,512)
        return feat.squeeze(0).detach().cpu().numpy().astype(np.float32)  # (512,)

def extract_episode_feats(
    fe: ResNet18Featurizer,
    steps: List[Dict],
    task: str,
    target_hw: Tuple[int, int],
    view_mode: str,
    pad_to_square: bool,
) -> np.ndarray:
    """
    Returns (T, D) float32.
    For sorting/stacking:
      - view_mode left/right => D=512
      - view_mode both6      => D=1024 (concat left,right)
    """
    feats = []
    for s in steps:
        proc = preprocess_rgb_for_task(
            s["rgb"], task_name=task, target_hw=target_hw,
            view_mode=view_mode, pad_to_square=pad_to_square
        )
        if proc.shape[-1] == 3:
            f = fe.encode_rgb_uint8(proc)
        elif proc.shape[-1] == 6:
            left = proc[..., :3]
            right = proc[..., 3:]
            f = np.concatenate([fe.encode_rgb_uint8(left), fe.encode_rgb_uint8(right)], axis=0)
        else:
            raise ValueError(f"Unexpected channels after preprocess: {proc.shape}")
        feats.append(f)
    return np.stack(feats, axis=0)  # (T,D)

def find_task_target_hw_from_wm_meta(wm_root: str, task: str) -> Optional[Tuple[int,int]]:
    """
    If you trained WMs, their meta json has target_hw; use it to match preprocessing.
    """
    meta_path = os.path.join(wm_root, task, "wm_success_only_rgb_meta.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r") as f:
        d = json.load(f)
    hw = d.get("target_hw", None)
    if isinstance(hw, list) and len(hw) == 2:
        return (int(hw[0]), int(hw[1]))
    return None

# -------------------------
# Main
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True,
                    help="FIPER data root, contains task/rollouts/calibration and task/rollouts/test")
    ap.add_argument("--out_root", type=str, required=True,
                    help="Output root for ResNet features (one folder per task)")
    ap.add_argument("--tasks", type=str, nargs="*", default=["pretzel","push_chair","push_t","sorting","stacking"])
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--pad_to_square", action="store_true", default=True)
    ap.add_argument("--wm_meta_root", type=str, default="",
                    help="If set, reads target_hw from wm_success_only_rgb_meta.json to match preprocessing.")
    ap.add_argument("--view_mode_sort", type=str, default="both6", choices=["left","right","both6"])
    ap.add_argument("--view_mode_stack", type=str, default="both6", choices=["left","right","both6"])
    ap.add_argument("--max_calib_success_eps", type=int, default=None)
    ap.add_argument("--max_test_eps", type=int, default=None)
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_root, exist_ok=True)

    fe = ResNet18Featurizer(device=args.device)

    for task in args.tasks:
        print("\n==============================")
        print("[TASK]", task)

        calib_glob = os.path.join(args.data_root, task, "rollouts", "calibration", "*.pkl")
        test_glob  = os.path.join(args.data_root, task, "rollouts", "test", "*.pkl")
        calib_paths = sorted(glob.glob(calib_glob))
        test_paths  = sorted(glob.glob(test_glob))
        if len(calib_paths) == 0 or len(test_paths) == 0:
            print("[SKIP] missing rollouts for", task)
            continue

        # Infer raw size from first calib success
        first_success_steps = None
        raw_h = raw_w = None
        for p in calib_paths:
            meta, steps = load_fiper_pkl(p)
            if bool(meta.get("successful", False)):
                first_success_steps = steps
                rgb0 = _to_uint8_hwc(steps[0]["rgb"])
                raw_h, raw_w = int(rgb0.shape[0]), int(rgb0.shape[1])
                break
        if first_success_steps is None:
            print("[SKIP] no calib successes for", task)
            continue

        # Choose target_hw
        target_hw = None
        if args.wm_meta_root.strip():
            target_hw = find_task_target_hw_from_wm_meta(args.wm_meta_root.strip(), task)
        if target_hw is None:
            target_hw = choose_target_hw_from_raw(raw_h, raw_w, task)

        # View mode
        if task == "sorting":
            view_mode = args.view_mode_sort
        elif task == "stacking":
            view_mode = args.view_mode_stack
        else:
            view_mode = "left"  # ignored for single cam

        print(f"[CFG] raw=({raw_h},{raw_w}) target_hw={target_hw} view_mode={view_mode} pad_sq={args.pad_to_square}")

        out_task = os.path.join(args.out_root, task)
        os.makedirs(out_task, exist_ok=True)

        # --- calib successes (success-only) ---
        calib_feats: List[np.ndarray] = []
        for p in tqdm(calib_paths, desc=f"{task} calib", leave=False):
            meta, steps = load_fiper_pkl(p)
            if not bool(meta.get("successful", False)):
                continue
            f = extract_episode_feats(fe, steps, task, target_hw, view_mode, args.pad_to_square)
            calib_feats.append(f)
            if args.max_calib_success_eps is not None and len(calib_feats) >= args.max_calib_success_eps:
                break

        if len(calib_feats) == 0:
            print("[SKIP] no calib success feats for", task)
            continue

        # --- test (all eps) ---
        test_feats: List[np.ndarray] = []
        test_labels: List[int] = []
        for p in tqdm(test_paths, desc=f"{task} test", leave=False):
            meta, steps = load_fiper_pkl(p)
            y = 0 if bool(meta.get("successful", False)) else 1
            f = extract_episode_feats(fe, steps, task, target_hw, view_mode, args.pad_to_square)
            test_feats.append(f)
            test_labels.append(int(y))
            if args.max_test_eps is not None and len(test_feats) >= args.max_test_eps:
                break

        np.save(os.path.join(out_task, "calib_success_feats.npy"), np.array(calib_feats, dtype=object), allow_pickle=True)
        np.save(os.path.join(out_task, "test_feats.npy"), np.array(test_feats, dtype=object), allow_pickle=True)
        np.save(os.path.join(out_task, "test_labels.npy"), np.array(test_labels, dtype=np.int64))

        print(f"[SAVE] {out_task}")
        print(f"  calib_success_eps={len(calib_feats)}  test_eps={len(test_feats)}  D={calib_feats[0].shape[1]}")

    print("\nDone.")

if __name__ == "__main__":
    main()