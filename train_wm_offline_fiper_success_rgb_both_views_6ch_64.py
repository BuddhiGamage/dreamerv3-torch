# train_wm_offline_fiper_success_rgb_both_views_6ch_64.py
import os, glob, pickle, json, random, time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from tqdm import trange
from PIL import Image

from models import WorldModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cudnn.benchmark = True

CALIB_GLOB = "/data/home/buddhig/data_all/sorting/rollouts/calibration/*.pkl"
OUT_DIR    = "/data/home/buddhig/projects/dreamer_fiper_offline/sorting"
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_HW = (64, 64)  # (H,W)

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

    cfg.reward_head = dict(dist="normal", layers=2, outscale=1.0, loss_scale=0.0)
    cfg.cont_head   = dict(layers=2, outscale=1.0, loss_scale=1.0)
    return cfg


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


def _resize_uint8_hwc(img: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    pil = Image.fromarray(img.astype(np.uint8))
    pil = pil.resize((hw[1], hw[0]), resample=Image.BILINEAR)
    out = np.asarray(pil, dtype=np.uint8)
    if out.ndim == 2:
        out = np.stack([out] * 3, axis=-1)
    return out


def both_views_to_6ch(rgb: np.ndarray) -> np.ndarray:
    """
    rgb: (96,192,3) = left+right concat
    -> split to two (96,96,3), resize each to (64,64,3)
    -> concat channels -> (64,64,6)
    """
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB, got {rgb.shape}")
    H, W, _ = rgb.shape
    if W % 2 != 0:
        raise ValueError(f"Expected even width, got W={W}")
    half = W // 2
    left  = rgb[:, :half, :]
    right = rgb[:, half:, :]
    left  = _resize_uint8_hwc(left, TARGET_HW)
    right = _resize_uint8_hwc(right, TARGET_HW)
    img6 = np.concatenate([left, right], axis=-1)  # (64,64,6)
    return img6


def stack_episode_rgb_both_views_6ch(steps) -> Dict[str, np.ndarray]:
    T = len(steps)
    imgs = []
    for s in steps:
        rgb = np.asarray(s["rgb"], dtype=np.uint8)
        imgs.append(both_views_to_6ch(rgb))
    images = np.stack(imgs, axis=0).astype(np.uint8)  # (T,64,64,6)

    action_dim = infer_action_dim_from_step(steps[0])
    actions = np.zeros((T, action_dim), dtype=np.float32)
    rewards = np.zeros((T, 1), dtype=np.float32)

    is_first = np.zeros((T,), dtype=np.float32); is_first[0] = 1.0
    is_terminal = np.zeros((T,), dtype=np.float32); is_terminal[-1] = 1.0
    discount = np.ones((T,), dtype=np.float32); discount[-1] = 0.0

    return dict(
        image=images,      # ✅ preprocess expects "image"
        action=actions,
        reward=rewards,
        discount=discount,
        is_first=is_first,
        is_terminal=is_terminal,
    )


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
        assert batch["action"].ndim == 3
        return batch


def main(
    batch_size: int = 8,
    seq_len: int = 16,
    train_steps: int = 20000,
    log_every: int = 200,
    max_success_eps: Optional[int] = None,
):
    paths = sorted(glob.glob(CALIB_GLOB))
    print(f"Found calibration PKLs: {len(paths)}")

    episodes: List[Dict[str, np.ndarray]] = []
    action_dim = None
    image_shape = (TARGET_HW[0], TARGET_HW[1], 6)  # ✅ 6 channels

    for p in paths:
        meta, steps = load_fiper_pkl(p)
        if not bool(meta.get("successful", False)):
            continue

        ep = stack_episode_rgb_both_views_6ch(steps)

        if action_dim is None:
            action_dim = int(ep["action"].shape[-1])

        episodes.append(ep)
        if max_success_eps is not None and len(episodes) >= max_success_eps:
            break

    if not episodes:
        raise RuntimeError("No SUCCESSFUL episodes found in calibration PKLs.")

    print(f"Success-only episodes: {len(episodes)}")
    print(f"Image shape: {image_shape}, action_dim (ignored but required): {action_dim}")

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

    t0 = time.time()
    for it in trange(1, train_steps + 1):
        batch = ds.sample_batch(batch_size, seq_len)

        if it == 1:
            print("Batch shapes:",
                  "image", batch["image"].shape,
                  "action", batch["action"].shape,
                  "reward", batch["reward"].shape)

        post, context, metrics = wm._train(batch)

        if it % log_every == 0:
            dt = time.time() - t0

            def to_scalar(x):
                if x is None: return 0.0
                if isinstance(x, (float, int)): return float(x)
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
        "image_shape": image_shape,
        "target_hw": TARGET_HW,
        "both_views_6ch": True,
        "n_success_eps": len(episodes),
    }
    ckpt_path = os.path.join(OUT_DIR, "wm_success_only_rgb_both_views_6ch_64.pt")
    torch.save(ckpt, ckpt_path)
    print("Saved:", ckpt_path)

    with open(os.path.join(OUT_DIR, "wm_success_only_rgb_both_views_6ch_64_meta.json"), "w") as f:
        json.dump(
            {"n_success_eps": len(episodes), "action_dim": action_dim, "image_shape": image_shape, "both_views_6ch": True},
            f,
            indent=2,
        )

if __name__ == "__main__":
    main()