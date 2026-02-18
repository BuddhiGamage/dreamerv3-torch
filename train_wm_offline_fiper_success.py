# train_wm_offline_fiper_success_rgb_only.py
import os, glob, pickle, json, random, time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from tqdm import trange

# --- DreamerV3-torch imports (repo-local) ---
from models import WorldModel


# =========================
# ENV / GPU
# =========================
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"   # start with 1 GPU while debugging
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cudnn.benchmark = True


# =========================
# USER PATHS
# =========================
CALIB_GLOB = "/data/home/buddhig/data_all/push_t/rollouts/calibration/*.pkl"
OUT_DIR    = "/data/home/buddhig/projects/dreamer_fiper_offline/push_t"
os.makedirs(OUT_DIR, exist_ok=True)


# =========================
# Minimal config object
# (only fields accessed in models.py you pasted)
# =========================
@dataclass
class WMConfig:
    # device / precision
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    precision: int = 16          # AMP autocast enabled when ==16
    discount: float = 0.99

    # encoder/decoder kwargs
    encoder: Dict = None
    decoder: Dict = None

    # RSSM params
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
    num_actions: int = -1  # must be set

    # heads
    reward_head: Dict = None
    cont_head: Dict = None
    grad_heads: Tuple[str, ...] = ("decoder", "reward", "cont")

    # optimization
    model_lr: float = 1e-4
    opt_eps: float = 1e-8
    grad_clip: float = 100.0
    weight_decay: float = 0.0
    opt: str = "adam"

    # KL loss scales
    kl_free: float = 1.0
    dyn_scale: float = 1.0
    rep_scale: float = 0.1


def build_default_config(action_dim: int) -> WMConfig:
    cfg = WMConfig()
    cfg.num_actions = action_dim

    # MultiEncoder / MultiDecoder usually auto-detects "image" keys as CNN.
    # Keep configs conservative.
    cfg.encoder = dict(
        mlp_keys="^$",        # no MLP keys (RGB-only)
        cnn_keys="image",     # use CNN for image
        act=cfg.act,
        norm=cfg.norm,
        cnn_depth=48,
        kernel_size=4,
        minres=4,
        mlp_layers=2,
        mlp_units=256,
        symlog_inputs=False,
    )

    # IMPORTANT:
    # Your earlier error was about MLP dist "mse". Decoder uses MultiDecoder,
    # which has its own image_dist handling. To be safe, use "normal".
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
        image_dist="normal",   # safer default than "mse" in this repo
        vector_dist="normal",
        outscale=1.0,
    )

    # Reward head: MUST be a supported distribution string, but we disable its loss.
    cfg.reward_head = dict(
        dist="normal",      # "mse" is NOT implemented in networks.MLP.dist()
        layers=2,
        outscale=1.0,
        loss_scale=0.0,     # ✅ disables reward learning completely
    )

    cfg.cont_head = dict(
        layers=2,
        outscale=1.0,
        loss_scale=1.0,     # you can set 0.0 if you want to disable cont too
    )

    return cfg


# =========================
# FIPER loader (success-only)
# =========================
def load_fiper_pkl(path: str):
    with open(path, "rb") as f:
        d = pickle.load(f)
    meta = d["metadata"]
    steps = d["rollout"]
    return meta, steps


def infer_action_dim_from_step(step) -> int:
    """
    FIPER can store chunked actions like (H,A). We ignore actions anyway,
    but we still need action_dim for RSSM wiring.
    """
    a = np.asarray(step["action"])
    if a.ndim == 1:
        return int(a.shape[-1])
    if a.ndim >= 2:
        return int(a.shape[-1])
    raise ValueError(f"Unsupported action shape: {a.shape}")


def stack_episode_rgb_only(steps) -> Dict[str, np.ndarray]:
    """
    Returns episode dict with keys required by WorldModel.preprocess():
      image (T,H,W,C) uint8
      action (T,A) float32   -> set to zeros (ignore dataset actions)
      reward (T,) float32    -> zeros
      discount (T,) float32  -> 1 except last=0
      is_first (T,) float32  -> 1 at t=0
      is_terminal (T,) float32 -> 1 at last step
    """
    T = len(steps)
    images = np.stack([s["rgb"] for s in steps], axis=0).astype(np.uint8)  # (T,H,W,C)

    action_dim = infer_action_dim_from_step(steps[0])
    actions = np.zeros((T, action_dim), dtype=np.float32)  # (T,A)  ✅ action-free

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
# Offline sampler: random windows
# =========================
class OfflineEpisodeDataset:
    def __init__(self, episodes: List[Dict[str, np.ndarray]]):
        self.episodes = episodes
        self.lengths = [int(ep["image"].shape[0]) for ep in episodes]

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

                # Fix is_first/is_terminal semantics after padding
                out["is_first"][-1] = np.concatenate(
                    [ep["is_first"], np.zeros((pad,), dtype=np.float32)], axis=0
                )
                out["is_terminal"][-1] = np.concatenate(
                    [ep["is_terminal"], np.zeros((pad,), dtype=np.float32)], axis=0
                )
                out["discount"][-1] = np.concatenate(
                    [ep["discount"], np.zeros((pad,), dtype=np.float32)], axis=0
                )

        # Stack to (B, L, ...)
        batch = {
            "image": np.stack(out["image"], axis=0),
            "action": np.stack(out["action"], axis=0),
            "reward": np.stack(out["reward"], axis=0),
            "discount": np.stack(out["discount"], axis=0),
            "is_first": np.stack(out["is_first"], axis=0),
            "is_terminal": np.stack(out["is_terminal"], axis=0),
        }

        # Hard checks: RSSM expects (B,T,A)
        assert batch["action"].ndim == 3, f"action must be (B,T,A), got {batch['action'].shape}"
        return batch


def main(
    batch_size: int = 8,
    seq_len: int = 16,
    train_steps: int = 20000,
    log_every: int = 200,
    max_success_eps: Optional[int] = None,
):
    # 1) Load success-only calibration episodes
    paths = sorted(glob.glob(CALIB_GLOB))
    print(f"Found calibration PKLs: {len(paths)}")

    episodes: List[Dict[str, np.ndarray]] = []
    action_dim = None
    image_shape = None  # (H,W,C)

    for p in paths:
        meta, steps = load_fiper_pkl(p)
        if not bool(meta.get("successful", False)):
            continue

        ep = stack_episode_rgb_only(steps)

        if action_dim is None:
            action_dim = int(ep["action"].shape[-1])
            image_shape = tuple(ep["image"].shape[1:])  # (H,W,C)

        episodes.append(ep)

        if max_success_eps is not None and len(episodes) >= max_success_eps:
            break

    if not episodes:
        raise RuntimeError("No SUCCESSFUL episodes found in calibration PKLs.")

    print(f"Success-only episodes: {len(episodes)}")
    print(f"Image shape: {image_shape}, action_dim (ignored but required): {action_dim}")

    # 2) Build WM + dummy obs_space/act_space objects
    class DummySpace:
        def __init__(self, shape): self.shape = shape

    class DummyDictSpace:
        def __init__(self, spaces): self.spaces = spaces

    obs_space = DummyDictSpace({"image": DummySpace(image_shape)})
    act_space = DummySpace((action_dim,))

    cfg = build_default_config(action_dim)
    step = torch.tensor(0, device=cfg.device)

    wm = WorldModel(obs_space, act_space, step, cfg).to(cfg.device).train()

    # 3) Offline sampler
    ds = OfflineEpisodeDataset(episodes)

    # 4) Train
    t0 = time.time()
    for it in trange(1, train_steps + 1):
        batch = ds.sample_batch(batch_size, seq_len)

        # additional sanity checks
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
                # numpy / torch arrays
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
                f"[{it:6d}] decoder={dec_loss:.4f} reward={r_loss:.4f} cont={c_loss:.4f} kl={kl:.4f}  ({dt:.1f}s)"
            )
            t0 = time.time()

    # 5) Save checkpoint
    ckpt = {
        "wm_state": wm.state_dict(),
        "config": asdict(cfg),
        "action_dim": action_dim,
        "image_shape": image_shape,
        "n_success_eps": len(episodes),
    }
    ckpt_path = os.path.join(OUT_DIR, "wm_success_only_rgb_only.pt")
    torch.save(ckpt, ckpt_path)
    print("Saved:", ckpt_path)

    with open(os.path.join(OUT_DIR, "wm_success_only_rgb_only_meta.json"), "w") as f:
        json.dump(
            {"n_success_eps": len(episodes), "action_dim": action_dim, "image_shape": image_shape},
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
