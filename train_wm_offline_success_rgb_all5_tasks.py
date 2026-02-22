# train_wm_offline_success_rgb_all5_tasks.py
#
# Trains 5 DreamerV3-torch WorldModels (SUCCESS-ONLY) for:
#   /data/home/buddhig/data_all/{pretzel,push_chair,push_t,sorting,stacking}
#
# Key features:
# ✅ One script trains ALL tasks (one checkpoint per task)
# ✅ sorting + stacking: concatenated two-view RGB → split L/R → resize → optionally stack to 6ch
# ✅ push_t: large single camera → auto-resize (defaults to 512x512 if large)
# ✅ pretzel + push_chair: single camera → auto-infer raw size + robust letterbox-to-square + resize
# ✅ Does NOT rely on hardcoded image sizes; it infers from first successful episode and selects a target
# ✅ Uses ZERO actions + ZERO rewards (like your earlier success-only setup)
#
# Usage:
#   python train_wm_offline_success_rgb_all5_tasks.py
#
# Optional:
#   python train_wm_offline_success_rgb_all5_tasks.py --tasks push_t sorting
#   python train_wm_offline_success_rgb_all5_tasks.py --train_steps 30000 --seq_len 32 --batch_size 8
#   python train_wm_offline_success_rgb_all5_tasks.py --view_mode_sort both6 --view_mode_stack both6
#
# ------------------------------------------------------------

import os, glob, pickle, json, random, time, argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from tqdm import trange
from PIL import Image

from models import WorldModel


# =========================
# ENV / GPU
# =========================
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.backends.cudnn.benchmark = True


# =========================
# Minimal config object
# =========================
@dataclass
class WMConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    precision: int = 16
    discount: float = 0.99

    encoder: Dict = None
    decoder: Dict = None

    dyn_stoch: int = 32
    dyn_deter: int = 256
    dyn_hidden: int = 256
    dyn_rec_depth: int = 1
    dyn_discrete: int = 0
    act: str = "SiLU"
    norm: str = "LayerNorm"
    dyn_mean_act: str = "none"
    dyn_std_act: str = "sigmoid2"
    dyn_min_std: float = 0.1
    units: int = 512
    unimix_ratio: float = 0.01
    initial: str = "learned"
    num_actions: int = -1

    reward_head: Dict = None
    cont_head: Dict = None
    grad_heads: Tuple[str, ...] = ("decoder", "reward", "cont")

    model_lr: float = 1e-4
    opt_eps: float = 1e-8
    grad_clip: float = 100.0
    weight_decay: float = 0.0
    opt: str = "adam"

    kl_free: float = 1.0
    dyn_scale: float = 1.0
    rep_scale: float = 0.1


def build_default_config(action_dim: int) -> WMConfig:
    cfg = WMConfig()
    cfg.num_actions = action_dim

    cfg.encoder = dict(
        mlp_keys="^$",
        cnn_keys="image",
        act=cfg.act,
        norm=cfg.norm,
        cnn_depth=48,
        kernel_size=4,
        minres=4,
        mlp_layers=2,
        mlp_units=256,
        symlog_inputs=False,
    )

    cfg.decoder = dict(
        mlp_keys="^$",
        cnn_keys="image",
        act=cfg.act,
        norm=cfg.norm,
        cnn_depth=48,
        kernel_size=4,
        minres=4,
        mlp_layers=2,
        mlp_units=256,
        cnn_sigmoid=False,
        image_dist="normal",
        vector_dist="normal",
        outscale=1.0,
    )

    cfg.reward_head = dict(
        dist="normal",
        layers=2,
        outscale=1.0,
        loss_scale=0.0,   # disable reward learning
    )

    cfg.cont_head = dict(
        layers=2,
        outscale=1.0,
        loss_scale=1.0,   # keep cont head on (you can set to 0.0 if you want)
    )

    return cfg


# =========================
# FIPER loader
# =========================
def load_fiper_pkl(path: str):
    with open(path, "rb") as f:
        d = pickle.load(f)
    meta = d["metadata"]
    steps = d["rollout"]
    return meta, steps


def infer_action_dim_from_step(step) -> int:
    a = np.asarray(step["action"])
    if a.ndim == 1:
        return int(a.shape[-1])
    if a.ndim >= 2:
        return int(a.shape[-1])
    raise ValueError(f"Unsupported action shape: {a.shape}")


# =========================
# Image utilities
# =========================
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
    """Pad to square with black borders."""
    img = _to_uint8_hwc(img)
    H, W, C = img.shape
    S = max(H, W)
    out = np.zeros((S, S, C), dtype=np.uint8)
    y0 = (S - H) // 2
    x0 = (S - W) // 2
    out[y0:y0+H, x0:x0+W] = img
    return out


def _resize(img: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    """Resize uint8 HWC to (H,W, C)."""
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


# =========================
# Task-specific preprocessing
# =========================
def choose_target_hw_from_raw(raw_h: int, raw_w: int, task_name: str) -> Tuple[int, int]:
    """
    Auto-pick a training resolution per task.
    You can override via CLI if needed.
    """
    # For sorting/stacking we almost always want small (your working setup)
    if task_name in ("sorting", "stacking"):
        return (64, 64)

    # Heuristics for single-cam tasks
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
      - both6:      (H,W,6)
    """
    rgb = _to_uint8_hwc(rgb)

    if task_name in ("sorting", "stacking"):
        # these are concatenated (left|right) in width
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

    # single-cam tasks
    img = rgb
    if pad_to_square:
        img = _letterbox_to_square(img)
    img = _resize(img, target_hw)
    return img


# =========================
# Episode stacker (success-only, RGB + zero actions)
# =========================
def stack_episode_rgb_only(
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

    actions = np.zeros((T, action_dim), dtype=np.float32)  # ignore dataset actions
    rewards = np.zeros((T, 1), dtype=np.float32)

    is_first = np.zeros((T,), dtype=np.float32); is_first[0] = 1.0
    is_terminal = np.zeros((T,), dtype=np.float32); is_terminal[-1] = 1.0
    discount = np.ones((T,), dtype=np.float32); discount[-1] = 0.0

    return dict(
        image=images,
        action=actions,
        reward=rewards,
        discount=discount,
        is_first=is_first,
        is_terminal=is_terminal,
    )


# =========================
# Offline sampler: random windows
# =========================
class OfflineEpisodeDataset:
    def __init__(self, episodes: List[Dict[str, np.ndarray]]):
        self.episodes = episodes

    def sample_batch(self, batch_size: int, seq_len: int) -> Dict[str, np.ndarray]:
        keys = ["image", "action", "reward", "discount", "is_first", "is_terminal"]
        out = {k: [] for k in keys}

        for _ in range(batch_size):
            ep = self.episodes[random.randrange(len(self.episodes))]
            T = int(ep["image"].shape[0])

            if T >= seq_len:
                start = random.randrange(0, T - seq_len + 1)
                sl = slice(start, start + seq_len)
                for k in keys:
                    out[k].append(ep[k][sl])
            else:
                pad = seq_len - T
                for k in keys:
                    arr = ep[k]
                    pad_arr = np.repeat(arr[-1][None], pad, axis=0)
                    out[k].append(np.concatenate([arr, pad_arr], axis=0))

                out["is_first"][-1] = np.concatenate([ep["is_first"], np.zeros((pad,), np.float32)], axis=0)
                out["is_terminal"][-1] = np.concatenate([ep["is_terminal"], np.zeros((pad,), np.float32)], axis=0)
                out["discount"][-1] = np.concatenate([ep["discount"], np.zeros((pad,), np.float32)], axis=0)

        batch = {k: np.stack(out[k], axis=0) for k in keys}

        assert batch["action"].ndim == 3, f"action must be (B,T,A), got {batch['action'].shape}"
        return batch


# =========================
# Train ONE task
# =========================
def train_one_task(
    task_name: str,
    data_root: str,
    out_root: str,
    batch_size: int,
    seq_len: int,
    train_steps: int,
    log_every: int,
    max_success_eps: Optional[int],
    view_mode: str,
    target_hw_override: Optional[Tuple[int, int]],
    pad_to_square: bool,
    cuda_visible_devices: Optional[str],
):
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    calib_glob = os.path.join(data_root, task_name, "rollouts", "calibration", "*.pkl")
    paths = sorted(glob.glob(calib_glob))
    print(f"\n====================")
    print(f"[TASK] {task_name}")
    print(f"[CALIB_GLOB] {calib_glob}")
    print(f"Found calibration PKLs: {len(paths)}")

    if not paths:
        print(f"[SKIP] No calibration PKLs found for {task_name}")
        return

    # Find first successful episode to infer dims + action_dim + raw RGB size
    action_dim = None
    raw_h = raw_w = None
    first_success_steps = None

    for p in paths:
        meta, steps = load_fiper_pkl(p)
        if bool(meta.get("successful", False)):
            first_success_steps = steps
            action_dim = infer_action_dim_from_step(steps[0])
            rgb0 = _to_uint8_hwc(steps[0]["rgb"])
            raw_h, raw_w = int(rgb0.shape[0]), int(rgb0.shape[1])
            break

    if first_success_steps is None:
        print(f"[SKIP] No SUCCESS episodes in calibration for {task_name}")
        return

    # target hw
    if target_hw_override is None:
        target_hw = choose_target_hw_from_raw(raw_h, raw_w, task_name)
    else:
        target_hw = target_hw_override

    # view mode sanity
    if task_name in ("sorting", "stacking"):
        assert view_mode in ("left", "right", "both6")
    else:
        view_mode = "single"  # ignored in preprocess for single cam

    # Determine final image shape (H,W,C)
    sample_img = preprocess_rgb_for_task(
        first_success_steps[0]["rgb"],
        task_name=task_name,
        target_hw=target_hw,
        view_mode=view_mode if task_name in ("sorting", "stacking") else "left",
        pad_to_square=pad_to_square,
    )
    image_shape = tuple(sample_img.shape)  # (H,W,C)

    print(f"[RAW] first success rgb0 shape: ({raw_h},{raw_w},3)")
    print(f"[PROC] target_hw={target_hw} pad_to_square={pad_to_square} view_mode={view_mode}")
    print(f"[PROC] image_shape={image_shape} action_dim={action_dim}")

    # Load success episodes
    episodes: List[Dict[str, np.ndarray]] = []
    for p in paths:
        meta, steps = load_fiper_pkl(p)
        if not bool(meta.get("successful", False)):
            continue

        ep = stack_episode_rgb_only(
            steps,
            task_name=task_name,
            action_dim=action_dim,
            target_hw=target_hw,
            view_mode=view_mode if task_name in ("sorting", "stacking") else "left",
            pad_to_square=pad_to_square,
        )
        episodes.append(ep)

        if max_success_eps is not None and len(episodes) >= max_success_eps:
            break

    if not episodes:
        print(f"[SKIP] No SUCCESS episodes loaded for {task_name}")
        return

    print(f"Success-only episodes loaded: {len(episodes)}")

    # Dummy spaces
    class DummySpace:
        def __init__(self, shape): self.shape = shape

    class DummyDictSpace:
        def __init__(self, spaces): self.spaces = spaces

    obs_space = DummyDictSpace({"image": DummySpace(image_shape)})
    act_space = DummySpace((action_dim,))

    cfg = build_default_config(action_dim)
    step = torch.tensor(0, device=cfg.device)

    wm = WorldModel(obs_space, act_space, step, cfg).to(cfg.device).train()
    ds = OfflineEpisodeDataset(episodes)

    # Train
    t0 = time.time()
    for it in trange(1, train_steps + 1, desc=f"TRAIN {task_name}"):
        batch = ds.sample_batch(batch_size, seq_len)

        # Print once
        if it == 1:
            print("Batch shapes:",
                  "image", batch["image"].shape,
                  "action", batch["action"].shape,
                  "reward", batch["reward"].shape,
                  "is_first", batch["is_first"].shape,
                  "is_terminal", batch["is_terminal"].shape)

        post, context, metrics = wm._train(batch)

        if it % log_every == 0:
            dt = time.time() - t0

            def to_scalar(x):
                if x is None:
                    return 0.0
                if isinstance(x, (float, int)):
                    return float(x)
                try:
                    x = np.asarray(x)
                    return float(x.mean())
                except Exception:
                    return 0.0

            kl       = to_scalar(metrics.get("kl", 0.0))
            dec_loss = to_scalar(metrics.get("decoder_loss", 0.0))
            r_loss   = to_scalar(metrics.get("reward_loss", 0.0))
            c_loss   = to_scalar(metrics.get("cont_loss", 0.0))

            print(
                f"[{task_name}][{it:6d}] decoder={dec_loss:.4f} reward={r_loss:.4f} cont={c_loss:.4f} kl={kl:.4f}  ({dt:.1f}s)"
            )
            t0 = time.time()

    # Save
    out_dir = os.path.join(out_root, task_name)
    os.makedirs(out_dir, exist_ok=True)

    ckpt = {
        "wm_state": wm.state_dict(),
        "config": asdict(cfg),
        "action_dim": int(action_dim),
        "image_shape": tuple(image_shape),
        "n_success_eps": int(len(episodes)),
        "task_name": task_name,
        "raw_rgb_shape_first_success": (int(raw_h), int(raw_w), 3),
        "target_hw": tuple(target_hw),
        "view_mode": view_mode if task_name in ("sorting", "stacking") else "single",
        "pad_to_square": bool(pad_to_square),
    }
    ckpt_path = os.path.join(out_dir, "wm_success_only_rgb.pt")
    torch.save(ckpt, ckpt_path)
    print("Saved:", ckpt_path)

    meta_path = os.path.join(out_dir, "wm_success_only_rgb_meta.json")
    with open(meta_path, "w") as f:
        json.dump(
            {
                "task_name": task_name,
                "n_success_eps": int(len(episodes)),
                "action_dim": int(action_dim),
                "image_shape": list(image_shape),
                "raw_rgb_shape_first_success": [int(raw_h), int(raw_w), 3],
                "target_hw": list(target_hw),
                "view_mode": view_mode if task_name in ("sorting", "stacking") else "single",
                "pad_to_square": bool(pad_to_square),
            },
            f,
            indent=2,
        )
    print("Saved:", meta_path)

    # Cleanup GPU memory between tasks
    del wm
    torch.cuda.empty_cache()


# =========================
# Main
# =========================
def parse_hw(s: str) -> Tuple[int, int]:
    # "64,64" -> (64,64)
    a, b = s.split(",")
    return (int(a), int(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="/data/home/buddhig/data_all")
    ap.add_argument("--out_root", type=str, default="/data/home/buddhig/projects/dreamer_fiper_offline/all5_tasks")
    # ap.add_argument("--tasks", type=str, nargs="*", default=["pretzel", "push_chair", "push_t", "sorting", "stacking"])
    ap.add_argument("--tasks", type=str, nargs="*", default=[ "stacking"])

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seq_len", type=int, default=16)
    ap.add_argument("--train_steps", type=int, default=20000)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--max_success_eps", type=int, default=None)

    ap.add_argument("--pad_to_square", action="store_true", default=True)
    ap.add_argument("--no_pad_to_square", action="store_true", default=False)

    # view modes only matter for sorting/stacking
    ap.add_argument("--view_mode_sort", type=str, default="both6", choices=["left", "right", "both6"])
    ap.add_argument("--view_mode_stack", type=str, default="both6", choices=["left", "right", "both6"])

    # optional overrides for target HW per task
    ap.add_argument("--target_hw_pretzel", type=str, default=None)    # e.g. "256,256"
    ap.add_argument("--target_hw_push_chair", type=str, default=None)
    ap.add_argument("--target_hw_push_t", type=str, default=None)
    ap.add_argument("--target_hw_sorting", type=str, default=None)
    ap.add_argument("--target_hw_stacking", type=str, default=None)

    ap.add_argument("--cuda_visible_devices", type=str, default=None)  # e.g. "0" or "0,1"
    args = ap.parse_args()

    if args.no_pad_to_square:
        pad_to_square = False
    else:
        pad_to_square = True if args.pad_to_square else False

    os.makedirs(args.out_root, exist_ok=True)

    # map overrides
    overrides = {
        "pretzel": parse_hw(args.target_hw_pretzel) if args.target_hw_pretzel else None,
        "push_chair": parse_hw(args.target_hw_push_chair) if args.target_hw_push_chair else None,
        "push_t": parse_hw(args.target_hw_push_t) if args.target_hw_push_t else None,
        "sorting": parse_hw(args.target_hw_sorting) if args.target_hw_sorting else None,
        "stacking": parse_hw(args.target_hw_stacking) if args.target_hw_stacking else None,
    }

    view_modes = {
        "sorting": args.view_mode_sort,
        "stacking": args.view_mode_stack,
    }

    # Train tasks sequentially
    for task in args.tasks:
        vm = view_modes.get(task, "single")
        train_one_task(
            task_name=task,
            data_root=args.data_root,
            out_root=args.out_root,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            train_steps=args.train_steps,
            log_every=args.log_every,
            max_success_eps=args.max_success_eps,
            view_mode=vm,
            target_hw_override=overrides.get(task, None),
            pad_to_square=pad_to_square,
            cuda_visible_devices=args.cuda_visible_devices,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()