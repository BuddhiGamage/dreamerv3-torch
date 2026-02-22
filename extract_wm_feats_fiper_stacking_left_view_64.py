# extract_wm_feats_fiper_stacking_left_view_64.py
# - Loads trained DreamerV3 WorldModel checkpoint (stacking)
# - Extracts per-timestep RSSM features feat = dynamics.get_feat(post)
# - Uses RGB only (LEFT view) + ZERO actions (same as training)
# - Saves:
#     calib_success_feats.npy  (object array of [Ti, D] arrays)
#     test_feats.npy           (object array of [Ti, D] arrays)
#     test_labels.npy          (1=failure, 0=success)
#     *_meta.json              (paths + episode ids + lengths)

import os, glob, pickle, json
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

from models import WorldModel


# =========================
# PATHS (UPDATE THESE)
# =========================
CALIB_GLOB = "/data/home/buddhig/data_all/sorting/rollouts/calibration/*.pkl"
TEST_GLOB  = "/data/home/buddhig/data_all/sorting/rollouts/test/*.pkl"

# ✅ stacking checkpoint you trained with LEFT view resized to 64x64
CKPT_PATH  = "/data/home/buddhig/projects/dreamer_fiper_offline/sorting/wm_success_only_rgb_left_view_64.pt"
OUT_DIR    = "/data/home/buddhig/projects/dreamer_fiper_offline/sorting/feats_left64"
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_HW = (64, 64)  # (H,W) must match training


# =========================
# Preprocess: LEFT view + resize
# =========================
def _resize_uint8_hwc(img: np.ndarray, hw):
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    pil = Image.fromarray(img)
    pil = pil.resize((hw[1], hw[0]), resample=Image.BILINEAR)  # (W,H)
    out = np.asarray(pil, dtype=np.uint8)
    if out.ndim == 2:
        out = np.stack([out] * 3, axis=-1)
    return out

def left_view_resize(rgb: np.ndarray) -> np.ndarray:
    # rgb expected (96,192,3) = left+right concat
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB, got {rgb.shape}")
    H, W, _ = rgb.shape
    if W % 2 != 0:
        raise ValueError(f"Expected even width for 2-view concat, got W={W}")
    half = W // 2
    left = rgb[:, :half, :]       # (96,96,3)
    return _resize_uint8_hwc(left, TARGET_HW)  # (64,64,3)


# =========================
# FIPER loader (LEFT view + zero actions)
# =========================
def load_fiper_left_rgb_only(path: str, action_dim: int):
    with open(path, "rb") as f:
        d = pickle.load(f)
    meta = d["metadata"]
    steps = d["rollout"]

    # Build resized left-view frames
    images = np.stack([left_view_resize(np.asarray(s["rgb"], dtype=np.uint8)) for s in steps], axis=0).astype(np.uint8)
    T = images.shape[0]

    actions = np.zeros((T, action_dim), dtype=np.float32)  # action-free
    reward = np.zeros((T, 1), dtype=np.float32)

    is_first = np.zeros((T,), dtype=np.float32); is_first[0] = 1.0
    is_terminal = np.zeros((T,), dtype=np.float32); is_terminal[-1] = 1.0
    discount = np.ones((T,), dtype=np.float32); discount[-1] = 0.0

    ep = {
        "image": images,      # ✅ preprocess() expects "image"
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

    class DummySpace:
        def __init__(self, shape): self.shape = shape

    class DummyDictSpace:
        def __init__(self, spaces): self.spaces = spaces

    image_shape = tuple(ckpt["image_shape"])  # should be (64,64,3)
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
# Feature extraction
# =========================
@torch.no_grad()
def embed_episode_feat(wm: WorldModel, ep: dict) -> np.ndarray:
    data = wm.preprocess(ep)
    for k in data:
        data[k] = data[k].unsqueeze(0)  # add batch dim

    embed = wm.encoder(data)  # (1,T,E)
    post, prior = wm.dynamics.observe(embed, data["action"], data["is_first"])
    feat = wm.dynamics.get_feat(post)  # (1,T,D)
    return feat.squeeze(0).detach().cpu().numpy()  # (T,D)


def main():
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    action_dim = int(ckpt["action_dim"])
    device = ckpt["config"]["device"]
    print(f"Loaded ckpt. action_dim={action_dim}, device={device}, image_shape={ckpt['image_shape']}")

    wm = build_wm_from_ckpt(ckpt)
    wm.eval()

    # -------------------------
    # CALIB: success-only feats
    # -------------------------
    calib_paths = sorted(glob.glob(CALIB_GLOB))
    calib_feats = []
    calib_info = []

    for p in tqdm(calib_paths, desc="CALIB (success only)"):
        meta, ep = load_fiper_left_rgb_only(p, action_dim=action_dim)
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
    # TEST feats + labels
    # -------------------------
    test_paths = sorted(glob.glob(TEST_GLOB))
    test_feats = []
    test_labels = []
    test_info = []

    for p in tqdm(test_paths, desc="TEST"):
        meta, ep = load_fiper_left_rgb_only(p, action_dim=action_dim)

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