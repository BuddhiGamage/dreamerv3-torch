# extract_wm_feats_fiper_stacking_both_views_6ch_64.py
import os, glob, pickle, json
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

from models import WorldModel

CALIB_GLOB = "/data/home/buddhig/data_all/sorting/rollouts/calibration/*.pkl"
TEST_GLOB  = "/data/home/buddhig/data_all/sorting/rollouts/test/*.pkl"

CKPT_PATH  = "/data/home/buddhig/projects/dreamer_fiper_offline/sorting/wm_success_only_rgb_both_views_6ch_64.pt"
OUT_DIR    = "/data/home/buddhig/projects/dreamer_fiper_offline/sorting/feats_both6ch64"
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_HW = (64, 64)

def _resize_uint8_hwc(img: np.ndarray, hw):
    pil = Image.fromarray(img.astype(np.uint8))
    pil = pil.resize((hw[1], hw[0]), resample=Image.BILINEAR)
    out = np.asarray(pil, dtype=np.uint8)
    if out.ndim == 2:
        out = np.stack([out] * 3, axis=-1)
    return out

def both_views_to_6ch(rgb: np.ndarray) -> np.ndarray:
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
    return np.concatenate([left, right], axis=-1).astype(np.uint8)  # (64,64,6)

def load_fiper_both6(path: str, action_dim: int):
    with open(path, "rb") as f:
        d = pickle.load(f)
    meta = d["metadata"]
    steps = d["rollout"]

    images = np.stack([both_views_to_6ch(np.asarray(s["rgb"], np.uint8)) for s in steps], axis=0).astype(np.uint8)
    T = images.shape[0]

    actions = np.zeros((T, action_dim), dtype=np.float32)
    reward  = np.zeros((T, 1), dtype=np.float32)

    is_first = np.zeros((T,), dtype=np.float32); is_first[0] = 1.0
    is_terminal = np.zeros((T,), dtype=np.float32); is_terminal[-1] = 1.0
    discount = np.ones((T,), dtype=np.float32); discount[-1] = 0.0

    ep = dict(
        image=images,
        action=actions,
        reward=reward,
        discount=discount,
        is_first=is_first,
        is_terminal=is_terminal,
    )
    return meta, ep

def build_wm_from_ckpt(ckpt: dict) -> WorldModel:
    cfgd = ckpt["config"]
    device = cfgd["device"]

    class DummySpace:
        def __init__(self, shape): self.shape = shape
    class DummyDictSpace:
        def __init__(self, spaces): self.spaces = spaces

    image_shape = tuple(ckpt["image_shape"])  # (64,64,6)
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

@torch.no_grad()
def embed_episode_feat(wm: WorldModel, ep: dict) -> np.ndarray:
    data = wm.preprocess(ep)
    for k in data:
        data[k] = data[k].unsqueeze(0)
    embed = wm.encoder(data)
    post, prior = wm.dynamics.observe(embed, data["action"], data["is_first"])
    feat = wm.dynamics.get_feat(post)
    return feat.squeeze(0).detach().cpu().numpy()

def main():
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    action_dim = int(ckpt["action_dim"])
    print("Loaded ckpt:", "action_dim", action_dim, "image_shape", ckpt["image_shape"], "device", ckpt["config"]["device"])

    wm = build_wm_from_ckpt(ckpt)
    wm.eval()

    # calib success feats
    calib_paths = sorted(glob.glob(CALIB_GLOB))
    calib_feats, calib_info = [], []

    for p in tqdm(calib_paths, desc="CALIB (success only)"):
        meta, ep = load_fiper_both6(p, action_dim)
        if not bool(meta.get("successful", False)):
            continue
        feat = embed_episode_feat(wm, ep)
        calib_feats.append(feat)
        calib_info.append({"path": p, "episode": meta.get("episode", None), "T": int(feat.shape[0]), "successful": True})

    np.save(os.path.join(OUT_DIR, "calib_success_feats.npy"), np.array(calib_feats, dtype=object), allow_pickle=True)
    with open(os.path.join(OUT_DIR, "calib_success_feats_meta.json"), "w") as f:
        json.dump(calib_info, f, indent=2)
    print("Saved calib:", len(calib_feats))

    # test feats + labels
    test_paths = sorted(glob.glob(TEST_GLOB))
    test_feats, test_labels, test_info = [], [], []

    for p in tqdm(test_paths, desc="TEST"):
        meta, ep = load_fiper_both6(p, action_dim)
        feat = embed_episode_feat(wm, ep)
        test_feats.append(feat)

        succ = bool(meta.get("successful", False))
        test_labels.append(0 if succ else 1)
        test_info.append({"path": p, "episode": meta.get("episode", None), "T": int(feat.shape[0]), "successful": succ})

    np.save(os.path.join(OUT_DIR, "test_feats.npy"), np.array(test_feats, dtype=object), allow_pickle=True)
    np.save(os.path.join(OUT_DIR, "test_labels.npy"), np.array(test_labels, dtype=np.int64))
    with open(os.path.join(OUT_DIR, "test_feats_meta.json"), "w") as f:
        json.dump(test_info, f, indent=2)
    print("Saved test:", len(test_feats))

if __name__ == "__main__":
    main()