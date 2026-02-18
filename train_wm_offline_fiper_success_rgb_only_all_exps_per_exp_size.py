# train_wm_offline_fiper_success_rgb_only_all_exps_force_size.py
# - Trains ONE DreamerV3-torch WorldModel per experiment directory (rollouts/calibration/*.pkl)
# - SUCCESS-only episodes (meta["successful"] == True)
# - Handles RGB stored as CHW (3,H,W) OR HWC (H,W,3) + dtype float/uint8
# - IMPORTANT: DreamerV3-torch CNN encoder/decoder expects "friendly" resolutions (typically powers of 2):
#     64, 128, 256, 512 ...
#   Non-friendly sizes like 320x320 can crash in decoder reshape.
#
# New features:
#   * --force_h/--force_w : FORCE ALL experiments to exactly that size (recommended: 256 or 512)
#   * --resize_mode {pad,stretch,none} : how to convert frames (pad preserves aspect)
#   * --max_h/--max_w : cap size (used only if force is not set and resize_mode!=none)
#   * --snap_allowed : snap final target to nearest allowed square size (default on)

import os, glob, pickle, json, random, time, argparse, hashlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from tqdm import trange
from PIL import Image

# --- DreamerV3-torch imports (repo-local) ---
from models import WorldModel


# =========================
# ENV / GPU
# =========================
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
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
        loss_scale=0.0,
    )

    cfg.cont_head = dict(
        layers=2,
        outscale=1.0,
        loss_scale=1.0,
    )

    return cfg


# =========================
# Loader
# =========================
def load_fiper_pkl(path: str):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["metadata"], d["rollout"]


def infer_action_dim_from_step(step) -> int:
    a = np.asarray(step["action"])
    if a.ndim == 1:
        return int(a.shape[-1])
    if a.ndim >= 2:
        return int(a.shape[-1])
    raise ValueError(f"Unsupported action shape: {a.shape}")


# =========================
# Image helpers
# =========================
def to_hwc_uint8(x: np.ndarray) -> np.ndarray:
    """
    Accepts:
      - HWC uint8 or float
      - CHW (3,H,W) uint8 or float
      - HW gray
    Returns:
      - HWC uint8 contiguous, C in {1,3,4}
    """
    arr = np.asarray(x)

    # dtype -> uint8
    if arr.dtype in (np.float32, np.float64):
        mx = float(np.nanmax(arr)) if arr.size else 0.0
        if mx <= 1.0 + 1e-6:
            arr = (arr * 255.0).round()
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    # HW -> HWC
    if arr.ndim == 2:
        arr = arr[:, :, None]

    # CHW -> HWC
    if arr.ndim == 3:
        if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))

    if arr.ndim != 3:
        raise ValueError(f"RGB must be 2D/3D, got shape {arr.shape}")

    H, W, C = arr.shape
    if C not in (1, 3, 4):
        raise ValueError(f"Unexpected channels C={C} for shape {arr.shape}")

    return np.ascontiguousarray(arr)


def resize_stretch_hwc_uint8(img_hwc: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """Stretch resize HWC uint8 using PIL. target_hw=(H,W)."""
    th, tw = int(target_hw[0]), int(target_hw[1])
    if img_hwc.shape[0] == th and img_hwc.shape[1] == tw:
        return img_hwc

    if img_hwc.shape[2] == 1:
        pil = Image.fromarray(img_hwc[:, :, 0], mode="L")
        pil = pil.resize((tw, th), resample=Image.BILINEAR)
        out = np.asarray(pil, dtype=np.uint8)[:, :, None]
        return np.ascontiguousarray(out)

    pil = Image.fromarray(img_hwc)
    pil = pil.resize((tw, th), resample=Image.BILINEAR)
    out = np.asarray(pil, dtype=np.uint8)
    return np.ascontiguousarray(out)


def resize_pad_hwc_uint8(img_hwc: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """Aspect-preserving resize + center pad to target_hw. Pads with black."""
    th, tw = int(target_hw[0]), int(target_hw[1])
    H, W, C = img_hwc.shape
    if (H, W) == (th, tw):
        return img_hwc

    scale = min(th / float(H), tw / float(W))
    nh, nw = max(1, int(round(H * scale))), max(1, int(round(W * scale)))
    resized = resize_stretch_hwc_uint8(img_hwc, (nh, nw))

    out = np.zeros((th, tw, C), dtype=np.uint8)
    top = (th - nh) // 2
    left = (tw - nw) // 2
    out[top:top + nh, left:left + nw, :] = resized
    return np.ascontiguousarray(out)


def snap_square_to_allowed(hw: Tuple[int, int], allowed=(64, 128, 256, 512)) -> Tuple[int, int]:
    """Make square then snap to smallest allowed >= max(h,w), else largest allowed."""
    h, w = int(hw[0]), int(hw[1])
    m = max(h, w)
    for a in allowed:
        if m <= a:
            return (a, a)
    return (allowed[-1], allowed[-1])


def pick_target_hw_per_experiment(
    sample_hwc: np.ndarray,
    resize_mode: str,
    force_hw: Optional[Tuple[int, int]],
    max_hw: Optional[Tuple[int, int]],
    snap_allowed: bool,
    allowed_sizes: Tuple[int, ...],
) -> Tuple[int, int]:
    """
    Decide final (H,W) used for this experiment.
    Priority:
      1) force_hw if provided
      2) if resize_mode == 'none': use native H,W (but may break Dreamer if not compatible)
      3) else: use native, optionally cap by max_hw (fit within), then snap to allowed square if enabled
    """
    native_hw = (int(sample_hwc.shape[0]), int(sample_hwc.shape[1]))

    if force_hw is not None:
        hw = (int(force_hw[0]), int(force_hw[1]))
        if snap_allowed:
            hw = snap_square_to_allowed(hw, allowed=allowed_sizes)
        return hw

    if resize_mode == "none":
        hw = native_hw
        if snap_allowed:
            hw = snap_square_to_allowed(hw, allowed=allowed_sizes)
        return hw

    # resize_mode in {'pad','stretch'}
    hw = native_hw
    if max_hw is not None:
        mh, mw = int(max_hw[0]), int(max_hw[1])
        H, W = hw
        if H > mh or W > mw:
            scale = min(mh / float(H), mw / float(W))
            hw = (max(1, int(round(H * scale))), max(1, int(round(W * scale))))

    if snap_allowed:
        hw = snap_square_to_allowed(hw, allowed=allowed_sizes)

    return hw


def stack_episode_rgb_only(
    steps,
    resize_mode: str,
    target_hw: Tuple[int, int],
) -> Dict[str, np.ndarray]:
    """
    Builds one episode dict for DreamerV3-torch:
      image: (T,H,W,C) uint8
      action: (T,A) float32 (zeros here)
      reward: (T,1) float32 (zeros)
      discount/is_first/is_terminal
    """
    T = len(steps)
    frames = []
    for s in steps:
        im = to_hwc_uint8(s["rgb"])
        if resize_mode == "pad":
            im = resize_pad_hwc_uint8(im, target_hw)
        elif resize_mode == "stretch":
            im = resize_stretch_hwc_uint8(im, target_hw)
        elif resize_mode == "none":
            # leave native size (danger unless snapped/forced to dreamer-friendly)
            pass
        else:
            raise ValueError(f"Unknown resize_mode: {resize_mode}")
        frames.append(im)

    images = np.stack(frames, axis=0).astype(np.uint8)

    action_dim = infer_action_dim_from_step(steps[0])
    actions = np.zeros((T, action_dim), dtype=np.float32)
    rewards = np.zeros((T, 1), dtype=np.float32)

    is_first = np.zeros((T,), dtype=np.float32)
    is_first[0] = 1.0

    is_terminal = np.zeros((T,), dtype=np.float32)
    is_terminal[-1] = 1.0

    discount = np.ones((T,), dtype=np.float32)
    discount[-1] = 0.0

    return dict(
        image=images,
        action=actions,
        reward=rewards,
        discount=discount,
        is_first=is_first,
        is_terminal=is_terminal,
    )


# =========================
# Offline sampler
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

                out["is_first"][-1] = np.concatenate(
                    [ep["is_first"], np.zeros((pad,), dtype=np.float32)], axis=0
                )
                out["is_terminal"][-1] = np.concatenate(
                    [ep["is_terminal"], np.zeros((pad,), dtype=np.float32)], axis=0
                )
                out["discount"][-1] = np.concatenate(
                    [ep["discount"], np.zeros((pad,), dtype=np.float32)], axis=0
                )

        batch = {k: np.stack(out[k], axis=0) for k in keys}

        if batch["image"].ndim != 5:
            raise ValueError(f"Expected image (B,T,H,W,C), got {batch['image'].shape}")
        if batch["action"].ndim != 3:
            raise ValueError(f"action must be (B,T,A), got {batch['action'].shape}")

        return batch


# =========================
# Experiment discovery
# =========================
def discover_calibration_dirs(data_root: str) -> List[str]:
    pattern = os.path.join(data_root, "**", "rollouts", "calibration", "*.pkl")
    pkls = glob.glob(pattern, recursive=True)
    return sorted({os.path.dirname(p) for p in pkls})


def exp_id_from_calib_dir(calib_dir: str, data_root: str) -> str:
    rel = os.path.relpath(calib_dir, data_root).replace(os.sep, "__")
    rel = rel.replace("__rollouts__calibration", "").strip("_")
    h = hashlib.md5(calib_dir.encode("utf-8")).hexdigest()[:8]
    return f"{rel}__{h}"


# =========================
# Train one experiment
# =========================
def train_one_experiment(
    calib_dir: str,
    out_root: str,
    data_root: str,
    resize_mode: str,
    force_hw: Optional[Tuple[int, int]],
    max_hw: Optional[Tuple[int, int]],
    snap_allowed: bool,
    allowed_sizes: Tuple[int, ...],
    batch_size: int,
    seq_len: int,
    train_steps: int,
    log_every: int,
    max_success_eps: Optional[int],
    seed: int,
    skip_if_exists: bool,
):
    exp_id = exp_id_from_calib_dir(calib_dir, data_root)
    exp_out = os.path.join(out_root, exp_id)
    os.makedirs(exp_out, exist_ok=True)

    ckpt_path = os.path.join(exp_out, "wm_success_only_rgb_only.pt")
    meta_path = os.path.join(exp_out, "wm_success_only_rgb_only_meta.json")

    if skip_if_exists and os.path.exists(ckpt_path):
        print(f"[SKIP] {exp_id} (checkpoint exists)")
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    paths = sorted(glob.glob(os.path.join(calib_dir, "*.pkl")))
    print(f"\n=== EXP: {exp_id} ===")
    print(f"Calibration PKLs: {len(paths)}")
    if not paths:
        print("[WARN] No PKLs found, skipping.")
        return

    # Find first SUCCESS episode for size inference
    sample_native_hw = None
    sample_layout = None
    sample_channels = None
    target_hw = None

    for p in paths:
        meta, steps = load_fiper_pkl(p)
        if not bool(meta.get("successful", False)):
            continue

        raw = np.asarray(steps[0]["rgb"])
        sample_layout = "CHW" if (raw.ndim == 3 and raw.shape[0] in (1, 3, 4) and raw.shape[-1] not in (1, 3, 4)) else "HWC/other"
        hwc = to_hwc_uint8(raw)
        sample_native_hw = (int(hwc.shape[0]), int(hwc.shape[1]))
        sample_channels = int(hwc.shape[2])

        target_hw = pick_target_hw_per_experiment(
            sample_hwc=hwc,
            resize_mode=resize_mode,
            force_hw=force_hw,
            max_hw=max_hw,
            snap_allowed=snap_allowed,
            allowed_sizes=allowed_sizes,
        )
        break

    if target_hw is None:
        print("[WARN] No SUCCESSFUL episodes found to infer image size, skipping.")
        return

    print(f"[IMG] native (after HWC conversion): {sample_native_hw} C={sample_channels} (raw layout={sample_layout})")
    print(f"[IMG] resize_mode={resize_mode} force_hw={force_hw} max_hw={max_hw} snap_allowed={snap_allowed} allowed={allowed_sizes}")
    print(f"[IMG] FINAL target_hw for this exp: {target_hw}")

    episodes: List[Dict[str, np.ndarray]] = []
    action_dim = None
    image_shape_hwc = None

    for p in paths:
        meta, steps = load_fiper_pkl(p)
        if not bool(meta.get("successful", False)):
            continue

        ep = stack_episode_rgb_only(
            steps,
            resize_mode=resize_mode,
            target_hw=target_hw,
        )

        if action_dim is None:
            action_dim = int(ep["action"].shape[-1])
            image_shape_hwc = tuple(ep["image"].shape[1:])  # (H,W,C)

        if int(ep["action"].shape[-1]) != int(action_dim):
            raise RuntimeError(
                f"action_dim mismatch within exp {exp_id}: got {ep['action'].shape[-1]} "
                f"vs expected {action_dim} (file={p})"
            )

        if tuple(ep["image"].shape[1:]) != tuple(image_shape_hwc):
            raise RuntimeError(
                f"image_shape mismatch within exp {exp_id}: got {ep['image'].shape[1:]} "
                f"vs expected {image_shape_hwc} (file={p})."
            )

        episodes.append(ep)

        if max_success_eps is not None and len(episodes) >= max_success_eps:
            break

    if not episodes:
        print("[WARN] No SUCCESSFUL episodes in calibration set, skipping.")
        return

    print(f"Success-only episodes: {len(episodes)}")
    print(f"Image shape used (HWC): {image_shape_hwc}, action_dim (ignored but required): {action_dim}")

    # Dummy spaces for DreamerV3-torch
    class DummySpace:
        def __init__(self, shape): self.shape = shape

    class DummyDictSpace:
        def __init__(self, spaces): self.spaces = spaces

    obs_space = DummyDictSpace({"image": DummySpace(image_shape_hwc)})
    act_space = DummySpace((action_dim,))

    cfg = build_default_config(action_dim)
    step = torch.tensor(0, device=cfg.device)

    wm = WorldModel(obs_space, act_space, step, cfg).to(cfg.device).train()
    ds = OfflineEpisodeDataset(episodes)

    t0 = time.time()
    for it in trange(1, train_steps + 1, desc=f"train[{exp_id[:32]}]"):
        batch = ds.sample_batch(batch_size, seq_len)

        if it == 1:
            print("Batch shapes:",
                  "image", batch["image"].shape,
                  "action", batch["action"].shape,
                  "reward", batch["reward"].shape,
                  "is_first", batch["is_first"].shape,
                  "is_terminal", batch["is_terminal"].shape)

        _, _, metrics = wm._train(batch)

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

            print(f"[{it:6d}] decoder={dec_loss:.4f} reward={r_loss:.4f} cont={c_loss:.4f} kl={kl:.4f}  ({dt:.1f}s)")
            t0 = time.time()

    ckpt = {
        "wm_state": wm.state_dict(),
        "config": asdict(cfg),
        "action_dim": action_dim,
        "image_shape_hwc": image_shape_hwc,
        "n_success_eps": len(episodes),
        "calib_dir": calib_dir,
        "resize_mode": resize_mode,
        "force_hw": list(force_hw) if force_hw is not None else None,
        "max_hw": list(max_hw) if max_hw is not None else None,
        "snap_allowed": bool(snap_allowed),
        "allowed_sizes": list(allowed_sizes),
        "target_hw": list(target_hw),
    }
    torch.save(ckpt, ckpt_path)
    print("Saved:", ckpt_path)

    meta_out = {
        "exp_id": exp_id,
        "calib_dir": calib_dir,
        "n_calib_pkls": len(paths),
        "n_success_eps": len(episodes),
        "action_dim": action_dim,
        "image_shape_hwc": image_shape_hwc,
        "resize_mode": resize_mode,
        "force_hw": list(force_hw) if force_hw is not None else None,
        "max_hw": list(max_hw) if max_hw is not None else None,
        "snap_allowed": bool(snap_allowed),
        "allowed_sizes": list(allowed_sizes),
        "target_hw": list(target_hw),
        "batch_size": batch_size,
        "seq_len": seq_len,
        "train_steps": train_steps,
        "log_every": log_every,
        "max_success_eps": max_success_eps,
        "seed": seed,
    }
    with open(meta_path, "w") as f:
        json.dump(meta_out, f, indent=2)
    print("Saved:", meta_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="/data/home/buddhig/data_all")
    ap.add_argument("--out_root", type=str, default="/data/home/buddhig/projects/dreamer_fiper_offline_all")

    # Resize controls
    ap.add_argument("--resize_mode", type=str, default="pad", choices=["pad", "stretch", "none"])
    ap.add_argument("--force_h", type=int, default=0, help="Force target height for ALL experiments (e.g., 256 or 512)")
    ap.add_argument("--force_w", type=int, default=0, help="Force target width for ALL experiments (e.g., 256 or 512)")
    ap.add_argument("--max_h", type=int, default=0, help="Cap target height if not forcing (used when resize_mode!=none)")
    ap.add_argument("--max_w", type=int, default=0, help="Cap target width if not forcing (used when resize_mode!=none)")

    # Snap to Dreamer-friendly sizes
    ap.add_argument("--no_snap_allowed", action="store_true",
                    help="Disable snapping to allowed square sizes (NOT recommended for Dreamer).")
    ap.add_argument("--allowed_sizes", type=str, default="64,128,256,512",
                    help="Comma-separated allowed square sizes, e.g. '64,128,256,512'.")

    # Training
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seq_len", type=int, default=16)
    ap.add_argument("--train_steps", type=int, default=20000)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--max_success_eps", type=int, default=0, help="0 => no limit")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_if_exists", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)

    max_success_eps = None if args.max_success_eps == 0 else int(args.max_success_eps)

    # Parse allowed sizes
    allowed_sizes = tuple(int(x.strip()) for x in args.allowed_sizes.split(",") if x.strip())
    if not allowed_sizes:
        raise ValueError("allowed_sizes parsed empty; provide e.g. --allowed_sizes 64,128,256,512")

    snap_allowed = not args.no_snap_allowed

    force_hw = None
    if args.force_h > 0 or args.force_w > 0:
        if args.force_h <= 0 or args.force_w <= 0:
            raise ValueError("Set BOTH --force_h and --force_w (e.g., --force_h 256 --force_w 256)")
        force_hw = (int(args.force_h), int(args.force_w))

    max_hw = None
    if args.max_h > 0 or args.max_w > 0:
        if args.max_h <= 0 or args.max_w <= 0:
            raise ValueError("Set BOTH --max_h and --max_w (e.g., --max_h 256 --max_w 256)")
        max_hw = (int(args.max_h), int(args.max_w))

    calib_dirs = discover_calibration_dirs(args.data_root)
    print(f"Discovered calibration dirs: {len(calib_dirs)}")
    if not calib_dirs:
        raise RuntimeError(
            f"No experiments found under {args.data_root}. "
            f"Expected **/rollouts/calibration/*.pkl"
        )

    for calib_dir in calib_dirs:
        train_one_experiment(
            calib_dir=calib_dir,
            out_root=args.out_root,
            data_root=args.data_root,
            resize_mode=args.resize_mode,
            force_hw=force_hw,
            max_hw=max_hw,
            snap_allowed=snap_allowed,
            allowed_sizes=allowed_sizes,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            train_steps=args.train_steps,
            log_every=args.log_every,
            max_success_eps=max_success_eps,
            seed=args.seed,
            skip_if_exists=args.skip_if_exists,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
