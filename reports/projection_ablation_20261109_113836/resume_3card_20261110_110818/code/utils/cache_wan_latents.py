"""Pre-encode Wan-VAE latents for the geo/color-split experiment (offline cache).

Mirrors utils/cache_gae_latents.py but for the 16-channel Wan2.1 VAE, and — for
consistency with the geo stream — copies the per-scene view lists straight from
the GAE cache meta files, so both caches cover exactly the same (scene, view)
pairs and restrict_views_to_cache can guard both.

Cache layout:
  <cache-root>/<out-name>/<scene_id>.safetensor   keys "v<view_id>" -> fp16 [16,h,w]
  <cache-root>/<out-name>/<scene_id>.meta.json    copied from the GAE cache meta
  <cache-root>/<out-name>/latent_stats.pt         {"mean","std","count"} float32 [16]

Run (rgbd venv, 1 NPU, from rgbd_video_gen_gae root):
  ASCEND_RT_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/sfs_turbo/t00845730/zhangyiheng/Depth-Anything-3/src \
    /mnt/sfs_turbo/t00845730/rgbd_video_gen/.venv/bin/python utils/cache_wan_latents.py \
      --gae-cache-root /mnt/sfs_turbo/t00845730/gae_latent_cache \
      --gae-dataset-name re10k_train \
      --manifest /mnt/sfs_turbo/t00845730/re10k/manifest_train.jsonl \
      --data-root /mnt/sfs_turbo/t00845730/re10k/extracted \
      --limit-scenes 8 --samples-per-scene 8 \
      --out-name re10k_ovf8_wan
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch_npu  # noqa: F401  (registers the npu device)
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gae-cache-root", required=True)
    p.add_argument("--gae-dataset-name", required=True,
                   help="GAE cache dir whose meta.json view lists are reused")
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--cache-root", default="/mnt/sfs_turbo/t00845730/wan_latent_cache")
    p.add_argument("--out-name", required=True)
    p.add_argument("--limit-scenes", type=int, default=8)
    p.add_argument("--only-scene", default=None,
                   help="cache a single scene_id (overrides --limit-scenes)")
    p.add_argument("--min-views", type=int, default=9,
                   help="mirror restrict_views_to_cache: drop scenes with fewer")
    p.add_argument("--wan-model-root", default="/mnt/sfs_turbo/t00845730/wan21")
    p.add_argument("--resolution", default="378x672",
                   help="must match the training config; frames are loaded via the "
                        "dataset loader so the crop pipeline is pixel-identical")
    p.add_argument("--device", default="npu:0")
    return p.parse_args()


def main():
    args = parse_args()
    H, W = (int(x) for x in args.resolution.lower().split("x"))
    gae_dir = os.path.join(args.gae_cache_root, args.gae_dataset_name)
    out_dir = os.path.join(args.cache_root, args.out_name)
    os.makedirs(out_dir, exist_ok=True)

    # Same scene order + restriction as the trainer: manifest order, scenes kept
    # only if the GAE cache meta lists >= min_views views.
    with open(args.manifest) as f:
        manifest_scenes = [json.loads(l)["scene_id"] for l in f if l.strip()]
    scenes = []
    for sid in manifest_scenes:
        if args.only_scene and sid != args.only_scene:
            continue
        meta_fp = os.path.join(gae_dir, f"{sid}.meta.json")
        if not os.path.exists(meta_fp):
            continue
        with open(meta_fp) as f:
            meta = json.load(f)
        if len(meta["views"]) >= args.min_views:
            scenes.append((sid, meta))
        if len(scenes) >= args.limit_scenes:
            break
    print(f"[plan] {len(scenes)} scenes, res {H}x{W}, out -> {out_dir}", flush=True)
    for sid, _ in scenes:
        print(f"  scene {sid}")

    # Wan VAE through the project pipeline (same weights/dtype as training eval).
    from diffsynth.pipelines.wan_video_new_determine import (
        ModelConfig,
        WanVideoPipeline,
    )

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device="cpu",
        model_configs=[ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B",
                                   origin_file_pattern="Wan2.1_VAE.pth")],
        local_model_path=args.wan_model_root, skip_download=True,
    )
    pipe.vae = pipe.vae.to(args.device, torch.bfloat16)
    print("[load] wan vae ready", flush=True)

    # Frames must be pixel-identical to what training loads: reuse the dataset
    # loader (scale-resize + center crop + K adjust) instead of cv2.
    from examples.dataset import Kubric4DRGBDNVSDataset

    ds = Kubric4DRGBDNVSDataset(
        kubric4d_root=args.data_root,
        manifest_path=args.manifest,
        resolution=(H, W),
        num_source_views=6,
        num_target_views=3,
        random_permute_source_views=False,
        view_order="random",
        disparity_norm_mode="metric",
        samples_per_scene=1,
    )
    by_id = {r["scene_id"]: r for r in ds.records}

    stats = {"sum": None, "sumsq": None, "count": 0}
    t0, done, skipped = time.time(), 0, 0

    def accumulate(lat):
        f = lat.float()
        if f.ndim == 3:                      # single view [C,h,w]
            f = f.unsqueeze(0)
        if stats["sum"] is None:
            stats["sum"] = torch.zeros(f.shape[1])
            stats["sumsq"] = torch.zeros(f.shape[1])
        stats["sum"] += f.sum(dim=(0, 2, 3))
        stats["sumsq"] += (f ** 2).sum(dim=(0, 2, 3))
        stats["count"] += f.shape[0] * f.shape[2] * f.shape[3]

    for sid, meta in scenes:
        out_fp = os.path.join(out_dir, f"{sid}.safetensor")
        if os.path.exists(out_fp):
            skipped += 1
            continue
        record = by_id.get(sid)
        if record is None:
            print(f"[skip] scene {sid} not loadable by the dataset", flush=True)
            continue
        views = {str(v.view_id): v for v in record["views"]}
        latents, kept = [], []
        for vid in meta["views"]:
            view = views.get(str(vid))
            if view is None:
                continue
            img = ds._load_view(view)["image"]            # [3,H,W] in [0,1]
            vid_t = (img * 2 - 1).unsqueeze(0).unsqueeze(2).to(
                device=args.device, dtype=torch.bfloat16)  # [1,3,1,H,W]
            with torch.no_grad():
                lat = pipe.vae.encode(vid_t, device=args.device)[0, :, 0]  # [16,h,w]
            latents.append(lat.float().cpu())
            kept.append(vid)
            accumulate(lat.float().cpu())
        if len(latents) < 2:
            print(f"[skip] scene {sid}: only {len(latents)} views loadable", flush=True)
            continue
        save_file({f"v{vid}": lat.to(torch.float16).contiguous()
                   for vid, lat in zip(kept, latents)}, out_fp)
        meta_out = dict(meta)
        meta_out["views"] = kept
        with open(out_fp.replace(".safetensor", ".meta.json"), "w") as f:
            json.dump(meta_out, f)
        done += 1
        if done % 4 == 0 or done == len(scenes):
            dt = time.time() - t0
            print(f"[{args.out_name}] {done} scenes encoded ({dt:.0f}s, "
                  f"{dt/max(done,1):.2f}s/scene), skipped {skipped}", flush=True)

    if stats["count"] > 0:
        mean = stats["sum"] / stats["count"]
        std = torch.sqrt((stats["sumsq"] / stats["count"] - mean ** 2).clamp_min(1e-12))
        torch.save({"mean": mean, "std": std, "count": int(stats["count"])},
                   os.path.join(out_dir, "latent_stats.pt"))
        print(f"[stats] E[x^2]={float(mean.pow(2).mean() + std.pow(2).mean()):.4f} "
              f"mean[:4]={mean[:4].tolist()} std[:4]={std[:4].tolist()}")
    print(f"[done] {done} scenes cached, {skipped} skipped, {time.time()-t0:.0f}s total")


if __name__ == "__main__":
    main()
