# extract_wm_feats_fiper_rgb_only.py
# - Loads your trained DreamerV3 WorldModel checkpoint
# - Extracts per-timestep RSSM features feat = dynamics.get_feat(post)
# - Uses RGB only + ZERO actions (same as training)
# - Saves:
#     calib_success_feats.npy  (object array of [Ti, D] arrays)
#     test_feats.npy           (object array of [Ti, D] arrays)
#     test_labels.npy          (1=failure, 0=success)
#     *_meta.json              (paths + episode ids + lengths)

import os, glob, pickle, json
import numpy as np
import torch
from tqdm import tqdm

from models import WorldModel


# =========================
# PATHS
# =========================
CALIB_GLOB = "/data/home/buddhig/projects/fiper/data/push_t/rollouts/calibration/*.pkl"
TEST_GLOB  = "/data/home/buddhig/projects/fiper/data/push_t/rollouts/test/*.pkl"
CKPT_PATH  = "/data/home/buddhig/projects/dreamer_fiper_offline/wm_success_only_rgb_only.pt"
OUT_DIR    = "/data/home/buddhig/projects/dreamer_fiper_offline"
os.makedirs(OUT_DIR, exist_ok=True)


# =========================
# FIPER loader (RGB only + zero actions)
# =========================
def load_fiper_rgb_only(path: str, action_dim: int):
    with open(path, "rb") as f:
        d = pickle.load(f)
    meta = d["metadata"]
    steps = d["rollout"]

    images = np.stack([s["rgb"] for s in steps], axis=0).astype(np.uint8)  # (T,H,W,C)
    T = images.shape[0]

    # IMPORTANT: match training (ignore dataset action; use zeros)
    actions = np.zeros((T, action_dim), dtype=np.float32)                  # (T,A)

    # reward must be (T,1) for reward head distribution shape
    reward = np.zeros((T, 1), dtype=np.float32)

    is_first = np.zeros((T,), dtype=np.float32); is_first[0] = 1.0
    is_terminal = np.zeros((T,), dtype=np.float32); is_terminal[-1] = 1.0

    discount = np.ones((T,), dtype=np.float32); discount[-1] = 0.0

    ep = {
        "image": images,
        "action": actions,
        "reward": reward,
        "discount": discount,
        "is_first": is_first,
        "is_terminal": is_terminal,
    }
    return meta, ep


# =========================
# Build WM from ckpt
# =========================
def build_wm_from_ckpt(ckpt: dict) -> WorldModel:
    cfgd = ckpt["config"]
    device = cfgd["device"]

    # same dummy spaces as training
    class DummySpace:
        def __init__(self, shape): self.shape = shape

    class DummyDictSpace:
        def __init__(self, spaces): self.spaces = spaces

    image_shape = tuple(ckpt["image_shape"])   # (H,W,C)
    action_dim  = int(ckpt["action_dim"])

    obs_space = DummyDictSpace({"image": DummySpace(image_shape)})
    act_space = DummySpace((action_dim,))
    step = torch.tensor(0, device=device)

    # Recreate cfg object with attribute access
    class Cfg: pass
    cfg = Cfg()
    for k, v in cfgd.items():
        setattr(cfg, k, v)

    wm = WorldModel(obs_space, act_space, step, cfg).to(device).eval()
    wm.load_state_dict(ckpt["wm_state"], strict=True)
    return wm


# =========================
# Feature extraction
# =========================
@torch.no_grad()
def embed_episode_feat(wm: WorldModel, ep: dict) -> np.ndarray:
    """
    Returns per-timestep RSSM feature: (T, D)
    Feature = dynamics.get_feat(post), typically concat(stoch, deter).
    """
    # preprocess expects dict of numpy arrays
    data = wm.preprocess(ep)  # tensors on device; image scaled; cont built; discount expanded to (T,1)

    # Add batch dim (WorldModel encoder/dynamics expect (B,T,...))
    for k in data:
        data[k] = data[k].unsqueeze(0)

    embed = wm.encoder(data)  # (1,T,E)
    post, prior = wm.dynamics.observe(embed, data["action"], data["is_first"])  # dicts (1,T,...)
    feat = wm.dynamics.get_feat(post)  # (1,T,D)

    return feat.squeeze(0).detach().cpu().numpy()  # (T,D)


def main():
    ckpt = torch.load(CKPT_PATH, map_location="cpu")

    action_dim = int(ckpt["action_dim"])
    device = ckpt["config"]["device"]
    print(f"Loaded ckpt. action_dim={action_dim}, device={device}")

    wm = build_wm_from_ckpt(ckpt)
    wm.eval()

    # -------------------------
    # CALIB: success-only feats
    # -------------------------
    calib_paths = sorted(glob.glob(CALIB_GLOB))
    calib_feats = []
    calib_info = []

    for p in tqdm(calib_paths, desc="CALIB (success only)"):
        meta, ep = load_fiper_rgb_only(p, action_dim=action_dim)
        if not bool(meta.get("successful", False)):
            continue

        feat = embed_episode_feat(wm, ep)
        calib_feats.append(feat)
        calib_info.append({
            "path": p,
            "episode": meta.get("episode", None),
            "T": int(feat.shape[0]),
            "successful": bool(meta.get("successful", False)),
        })

    calib_out = os.path.join(OUT_DIR, "calib_success_feats.npy")
    np.save(calib_out, np.array(calib_feats, dtype=object), allow_pickle=True)
    print(f"Saved {calib_out} (#eps={len(calib_feats)})")

    with open(os.path.join(OUT_DIR, "calib_success_feats_meta.json"), "w") as f:
        json.dump(calib_info, f, indent=2)

    # -------------------------
    # TEST feats + episode labels
    # -------------------------
    test_paths = sorted(glob.glob(TEST_GLOB))
    test_feats = []
    test_labels = []  # 1=failure, 0=success
    test_info = []

    for p in tqdm(test_paths, desc="TEST"):
        meta, ep = load_fiper_rgb_only(p, action_dim=action_dim)

        feat = embed_episode_feat(wm, ep)
        test_feats.append(feat)

        succ = bool(meta.get("successful", False))
        test_labels.append(0 if succ else 1)

        test_info.append({
            "path": p,
            "episode": meta.get("episode", None),
            "T": int(feat.shape[0]),
            "successful": succ,
        })

    test_feats_out = os.path.join(OUT_DIR, "test_feats.npy")
    test_labels_out = os.path.join(OUT_DIR, "test_labels.npy")
    np.save(test_feats_out, np.array(test_feats, dtype=object), allow_pickle=True)
    np.save(test_labels_out, np.array(test_labels, dtype=np.int64))
    print(f"Saved {test_feats_out} + {test_labels_out} (#eps={len(test_feats)})")

    with open(os.path.join(OUT_DIR, "test_feats_meta.json"), "w") as f:
        json.dump(test_info, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
