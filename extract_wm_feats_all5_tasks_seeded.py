#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, pickle, json, argparse
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

from models import WorldModel


def set_global_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Image utils (must match training)
# -----------------------------
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
    return rgb[:, :half, :], rgb[:, half:, :]


def preprocess_rgb_for_task(
    rgb: np.ndarray,
    task_name: str,
    target_hw: Tuple[int, int],
    view_mode: str,
    pad_to_square: bool,
) -> np.ndarray:
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
        raise ValueError(f"Unknown view_mode={view_mode}")

    img = rgb
    if pad_to_square:
        img = _letterbox_to_square(img)
    img = _resize(img, target_hw)
    return img


# -----------------------------
# FIPER loader
# -----------------------------
def load_fiper_pkl(path: str):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["metadata"], d["rollout"]


# -----------------------------
# WM build from ckpt
# -----------------------------
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


def make_episode_from_steps(
    steps,
    task_name: str,
    action_dim: int,
    target_hw: Tuple[int, int],
    view_mode: str,
    pad_to_square: bool,
) -> Dict[str, np.ndarray]:
    T = len(steps)
    images = np.stack(
        [
            preprocess_rgb_for_task(
                s["rgb"],
                task_name=task_name,
                target_hw=target_hw,
                view_mode=view_mode,
                pad_to_square=pad_to_square,
            )
            for s in steps
        ],
        axis=0,
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


@torch.no_grad()
def embed_episode_feat(wm: WorldModel, ep: dict) -> np.ndarray:
    data = wm.preprocess(ep)
    for k in data:
        data[k] = data[k].unsqueeze(0)

    embed = wm.encoder(data)
    post, prior = wm.dynamics.observe(embed, data["action"], data["is_first"])
    feat = wm.dynamics.get_feat(post)
    return feat.squeeze(0).detach().cpu().numpy()


def parse_hw(s: str) -> Tuple[int, int]:
    a, b = s.split(",")
    return (int(a), int(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="/home/s447658/project/fiper/data")
    ap.add_argument("--wm_root", type=str, default="/home/s447658/projects/dreamer_fiper_offline/wm_all5_seeds")
    ap.add_argument("--out_root", type=str, default="/home/s447658/projects/dreamer_fiper_feats_all5/feats_all5_seeds")

    ap.add_argument("--tasks", type=str, nargs="*", default=["pretzel", "push_chair", "push_t", "sorting", "stacking"])
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--pad_to_square", action="store_true", default=True)
    ap.add_argument("--no_pad_to_square", action="store_true", default=False)

    ap.add_argument("--view_mode_sort", type=str, default="both6", choices=["left", "right", "both6"])
    ap.add_argument("--view_mode_stack", type=str, default="both6", choices=["left", "right", "both6"])

    ap.add_argument("--target_hw_override", type=str, default=None)  # apply to ALL tasks if provided, e.g. "64,64"
    args = ap.parse_args()

    set_global_seed(int(args.seed))

    pad_to_square = False if args.no_pad_to_square else bool(args.pad_to_square)
    hw_override = parse_hw(args.target_hw_override) if args.target_hw_override else None

    os.makedirs(args.out_root, exist_ok=True)

    for task in args.tasks:
        task_wm_dir = os.path.join(args.wm_root, task, f"seed_{args.seed}")
        ckpt_path = os.path.join(task_wm_dir, "wm_success_only_rgb.pt")
        if not os.path.exists(ckpt_path):
            print(f"[SKIP] missing ckpt for {task}: {ckpt_path}")
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu")
        wm = build_wm_from_ckpt(ckpt).eval()

        action_dim = int(ckpt["action_dim"])
        image_shape = tuple(ckpt["image_shape"])
        target_hw = tuple(ckpt.get("target_hw", (image_shape[0], image_shape[1])))
        view_mode = ckpt.get("view_mode", "single")
        pad_sq = bool(ckpt.get("pad_to_square", pad_to_square))

        if task == "sorting":
            view_mode = args.view_mode_sort
        if task == "stacking":
            view_mode = args.view_mode_stack
        if hw_override is not None:
            target_hw = hw_override

        print(f"\n[TASK] {task} | seed={args.seed}")
        print(f"  ckpt: {ckpt_path}")
        print(f"  image_shape (ckpt): {image_shape}")
        print(f"  target_hw (used)  : {target_hw}")
        print(f"  view_mode (used)  : {view_mode}")
        print(f"  pad_to_square     : {pad_sq}")
        print(f"  action_dim        : {action_dim}")

        calib_glob = os.path.join(args.data_root, task, "rollouts", "calibration", "*.pkl")
        test_glob  = os.path.join(args.data_root, task, "rollouts", "test", "*.pkl")
        calib_paths = sorted(glob.glob(calib_glob))
        test_paths  = sorted(glob.glob(test_glob))

        out_task_dir = os.path.join(args.out_root, task, f"seed_{args.seed}")
        os.makedirs(out_task_dir, exist_ok=True)

        # -------- CALIB success-only --------
        calib_feats = []
        calib_info = []
        for p in tqdm(calib_paths, desc=f"{task} CALIB seed={args.seed} (success only)"):
            meta, steps = load_fiper_pkl(p)
            if not bool(meta.get("successful", False)):
                continue
            ep = make_episode_from_steps(steps, task, action_dim, target_hw, view_mode, pad_sq)
            feat = embed_episode_feat(wm, ep)
            calib_feats.append(feat)
            calib_info.append({"path": p, "episode": meta.get("episode", None), "T": int(feat.shape[0]), "successful": True})

        np.save(os.path.join(out_task_dir, "calib_success_feats.npy"),
                np.array(calib_feats, dtype=object), allow_pickle=True)
        with open(os.path.join(out_task_dir, "calib_success_feats_meta.json"), "w") as f:
            json.dump(calib_info, f, indent=2)

        # -------- TEST --------
        test_feats = []
        test_labels = []
        test_info = []
        for p in tqdm(test_paths, desc=f"{task} TEST seed={args.seed}"):
            meta, steps = load_fiper_pkl(p)
            ep = make_episode_from_steps(steps, task, action_dim, target_hw, view_mode, pad_sq)
            feat = embed_episode_feat(wm, ep)
            test_feats.append(feat)

            succ = bool(meta.get("successful", False))
            test_labels.append(0 if succ else 1)

            test_info.append({"path": p, "episode": meta.get("episode", None), "T": int(feat.shape[0]), "successful": succ})

        np.save(os.path.join(out_task_dir, "test_feats.npy"),
                np.array(test_feats, dtype=object), allow_pickle=True)
        np.save(os.path.join(out_task_dir, "test_labels.npy"),
                np.array(test_labels, dtype=np.int64))
        with open(os.path.join(out_task_dir, "test_feats_meta.json"), "w") as f:
            json.dump(test_info, f, indent=2)

        print(f"[DONE] saved feats to {out_task_dir}")

        del wm
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()