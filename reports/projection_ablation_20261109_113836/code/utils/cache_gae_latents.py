"""Pre-encode manifest scenes into GAE-64 latents (offline cache, resumable).

Cache layout:
  <cache-root>/<dataset>/<scene_id>.safetensor   keys "v<view_id>" -> fp16 [64,h,w]
  <cache-root>/<dataset>/<scene_id>.meta.json    resolution / view list / codec tag
  <cache-root>/<dataset>/latent_stats.pt         {"mean","std","count"} float32 [64]

Run (rgbd venv, from rgbd_video_gen root):
  python utils/cache_gae_latents.py --dataset-name bmvs \
      --manifest /mnt/sfs_turbo/t00845730/data/blendedmvs/manifest_all.jsonl \
      --data-root /mnt/sfs_turbo/t00845730/data/blendedmvs \
      --resolution 378x504 --max-views-per-scene 24

  python utils/cache_gae_latents.py --dataset-name re10k_test \
      --manifest /mnt/sfs_turbo/t00845730/re10k/manifest_test.jsonl \
      --data-root /mnt/sfs_turbo/t00845730/re10k \
      --resolution 378x672 --max-views-per-scene 16
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch_npu  # noqa: F401  (registers the npu device)
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.gae_codec import GAECodecWrapper  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-name", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--cache-root", default="/mnt/sfs_turbo/t00845730/gae_latent_cache")
    p.add_argument("--resolution", default="378x504", help="HxW, both multiples of 14")
    p.add_argument("--batch-views", type=int, default=8)
    p.add_argument("--max-views-per-scene", type=int, default=0, help="0 = all views")
    p.add_argument("--limit-scenes", type=int, default=0, help="0 = all scenes")
    p.add_argument("--device", default=None)
    p.add_argument("--stats-only", action="store_true",
                   help="rebuild latent_stats.pt from already-cached scenes, no encoding")
    return p.parse_args()


def select_views(views, max_views):
    if max_views <= 0 or len(views) <= max_views:
        return list(views)
    idx = np.round(np.linspace(0, len(views) - 1, max_views)).astype(int)
    return [views[i] for i in dict.fromkeys(idx.tolist())]


def load_rgb(path, hw):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    H, W = hw
    rgb = cv2.resize(bgr[:, :, ::-1], (W, H), interpolation=cv2.INTER_AREA)
    return (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)  # CHW


def encode_scene(wrapper, rgbs, batch):
    """rgbs [V,3,H,W] float32 -> latents [V,C,h,w] fp16 (cpu)."""
    outs = []
    for s in range(0, len(rgbs), batch):
        chunk = torch.from_numpy(rgbs[s : s + batch]).unsqueeze(0)  # [1,v,3,H,W]
        z = wrapper.encode(chunk)[0]  # [v,C,h,w]
        outs.append(z.cpu())
    return torch.cat(outs, 0).to(torch.float16)


def accumulate(stats, latents):
    f = latents.float()
    stats["sum"] += f.sum(dim=(0, 2, 3))
    stats["sumsq"] += (f ** 2).sum(dim=(0, 2, 3))
    stats["count"] += f.shape[0] * f.shape[2] * f.shape[3]


def save_stats(stats, out_path):
    mean = stats["sum"] / stats["count"]
    std = torch.sqrt((stats["sumsq"] / stats["count"] - mean ** 2).clamp_min(1e-12))
    torch.save({"mean": mean, "std": std, "count": int(stats["count"])}, out_path)
    return mean, std


def rebuild_stats_from_cache(cache_dir, hw_latent):
    """Scan cached safetensors and rebuild stats (used after resumed runs)."""
    import glob
    from safetensors.torch import load_file

    stats = {"sum": None, "sumsq": None, "count": 0}
    files = sorted(glob.glob(os.path.join(cache_dir, "*.safetensor")))
    for i, fp in enumerate(files):
        data = load_file(fp)
        z = torch.stack([v for k, v in data.items() if k.startswith("v")]).float()
        if stats["sum"] is None:
            C = z.shape[1]
            stats["sum"] = torch.zeros(C)
            stats["sumsq"] = torch.zeros(C)
        accumulate(stats, z)
        if (i + 1) % 200 == 0:
            print(f"[stats] {i+1}/{len(files)} scenes", flush=True)
    return stats


def main():
    args = parse_args()
    H, W = map(int, args.resolution.lower().split("x"))
    assert H % 14 == 0 and W % 14 == 0, "resolution must be multiples of 14"
    cache_dir = os.path.join(args.cache_root, args.dataset_name)
    os.makedirs(cache_dir, exist_ok=True)

    with open(args.manifest) as f:
        scenes = [json.loads(line) for line in f if line.strip()]
    if args.limit_scenes > 0:
        scenes = scenes[: args.limit_scenes]
    print(f"[plan] {len(scenes)} scenes, res {H}x{W}, cache -> {cache_dir}", flush=True)

    if args.stats_only:
        stats = rebuild_stats_from_cache(cache_dir, (H // 14, W // 14))
        if stats["count"] == 0:
            print("[stats] cache is empty, nothing to do")
            return
        mean, std = save_stats(stats, os.path.join(cache_dir, "latent_stats.pt"))
        print(f"[stats] saved from {stats['count']} latent positions, "
              f"mean[0,1,2]={mean[:3].tolist()}, std[0,1,2]={std[:3].tolist()}")
        return

    wrapper = GAECodecWrapper(device=torch.device(args.device) if args.device else None)
    print(f"[load] codec ready on {wrapper.device}", flush=True)

    stats = {"sum": None, "sumsq": None, "count": 0}
    t0, done, skipped = time.time(), 0, 0
    for si, scene in enumerate(scenes):
        sid = scene["scene_id"]
        views = select_views(scene["views"], args.max_views_per_scene)
        out_fp = os.path.join(cache_dir, f"{sid}.safetensor")
        meta_fp = out_fp.replace(".safetensor", ".meta.json")
        wanted = [v["view_id"] for v in views]
        if os.path.exists(meta_fp):
            try:
                with open(meta_fp) as f:
                    if json.load(f).get("views") == wanted:
                        continue
            except Exception:
                pass

        rgbs, kept = [], []
        for v in views:
            rgb = load_rgb(os.path.join(args.data_root, v["rgb_path"]), (H, W))
            if rgb is not None:
                rgbs.append(rgb)
                kept.append(v["view_id"])
        if len(rgbs) < 2:
            skipped += 1
            continue

        lat = encode_scene(wrapper, np.stack(rgbs), args.batch_views)
        save_file({f"v{vid}": lat[i].contiguous() for i, vid in enumerate(kept)}, out_fp)
        with open(meta_fp, "w") as f:
            json.dump({"scene_id": sid, "resolution": [H, W], "views": kept,
                       "codec": "gae_64", "n_views": len(kept)}, f)
        if stats["sum"] is None:
            stats["sum"] = torch.zeros(lat.shape[1])
            stats["sumsq"] = torch.zeros(lat.shape[1])
        accumulate(stats, lat.float())
        done += 1
        if done % 20 == 0:
            dt = time.time() - t0
            print(f"[{args.dataset_name}] {done} scenes encoded "
                  f"({dt:.0f}s, {dt/max(done,1):.2f}s/scene), skipped {skipped}", flush=True)

    mean, std = (None, None)
    if stats["count"] > 0:
        mean, std = save_stats(stats, os.path.join(cache_dir, "latent_stats.pt"))
    print(f"[done] {done} scenes cached, {skipped} skipped, {time.time()-t0:.0f}s total")
    if mean is not None:
        print(f"[stats] mean[:4]={mean[:4].tolist()} std[:4]={std[:4].tolist()}")


if __name__ == "__main__":
    main()
