"""Visualize the GT-depth RGB warp for ONE frozen-eval sample/target, to check
whether a weird-looking prediction in an eval_step_*/b{NN}_step*.jpg grid is
inherited from the warp (utils.warp.splat_sources_to_targets) rather than
produced by the model.

Reproduces the exact frozen-eval indexing from train_with_accelerate_rgbd_nvs.py
(eval_split="val", eval_indices = [i*samples_per_scene for i in
range(num_scenes)][:eval_max_samples], DataLoader batch_size=eval_batch_size,
shuffle=False) so that b{NN} in a saved filename maps back to the same
dataset item: overall eval index = chunk_idx*eval_batch_size + local_b, where
chunk_idx, local_b = divmod(NN, eval_batch_size).

Pure geometry -- no checkpoint, no DiT, no VAE. Same code path as
WanRGBDNVSTrainingModule._splat_sources_pixel.

Usage:
    python tools/vis_rgb_warp_one_sample.py \
        --run logs/bmvs_d_drocefp32_s42 --b 5 --target 2
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from examples.dataset.kubric4d_rgbd_nvs_dataset import Kubric4DRGBDNVSDataset
from utils.warp import splat_sources_to_targets


def build_eval_indices(cfg, eval_dataset):
    eval_max_samples = int(cfg.get("eval_max_samples", 2))
    samples_per_scene = getattr(eval_dataset, "samples_per_scene", 1)
    num_scenes = len(eval_dataset.records)
    eval_indices = [i * samples_per_scene for i in range(num_scenes)]
    return eval_indices[:eval_max_samples]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="e.g. logs/bmvs_d_drocefp32_s42")
    ap.add_argument("--b", type=int, required=True, help="b-index from the saved filename, e.g. 5 for b05")
    ap.add_argument("--target", type=int, required=True, help="0-indexed target view, e.g. 2 for tgt3")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = OmegaConf.load(os.path.join(args.run, "args.yaml"))
    OmegaConf.resolve(cfg)

    eval_batch_size = max(1, min(int(cfg.get("eval_max_samples", 2)), int(cfg.get("eval_batch_size", 4))))
    chunk_idx, local_b = divmod(args.b, eval_batch_size)
    eval_dataset = Kubric4DRGBDNVSDataset(
        kubric4d_root=cfg.kubric4d_root,
        manifest_path=cfg.manifest_path,
        resolution=tuple(cfg.resolution),
        num_source_views=cfg.num_source_views,
        num_target_views=cfg.num_target_views,
        random_permute_source_views=cfg.random_permute_source_views,
        view_order=cfg.get("view_order", "random"),
        disparity_norm_mode=cfg.get("disparity_norm_mode", "shared_scale"),
        disparity_norm_quantile=cfg.get("disparity_norm_quantile", 0.98),
        disparity_norm_headroom=cfg.get("disparity_norm_headroom", 1.5),
        min_view_valid_fraction=cfg.get(
            "eval_min_view_valid_fraction", cfg.get("min_view_valid_fraction", 0.0)),
        view_validity_path=cfg.get("view_validity_path", None),
        samples_per_scene=cfg.get("samples_per_scene", 1),
        split_path=cfg.get("split_path", None),
        split=cfg.get("eval_split", None),
    )
    eval_dataset.deterministic_view_seed = int(cfg.get("eval_view_seed", 1234))

    eval_indices = build_eval_indices(cfg, eval_dataset)
    overall_idx = chunk_idx * eval_batch_size + local_b
    ds_idx = eval_indices[overall_idx]
    item = eval_dataset[ds_idx]
    print(f"b{args.b:02d} -> chunk {chunk_idx}, local {local_b} -> eval_indices[{overall_idx}] "
          f"-> dataset idx {ds_idx} -> sample_id={item['sample_id']}")

    eps = 1e-3
    src_disp = item["source_disparity"][:, 0:1, :, :].squeeze(1).unsqueeze(0)  # [1,S,H,W]
    valid = src_disp > eps
    depth = torch.where(valid, 1.0 / src_disp.clamp_min(eps), torch.zeros_like(src_disp))

    warped, mask, tgt_disp = splat_sources_to_targets(
        src_rgb=item["source_images"].unsqueeze(0).float(),
        src_depth=depth.float(),
        src_valid=valid,
        src_K=item["source_K"].unsqueeze(0).float(),
        src_c2w=item["source_c2w"].unsqueeze(0).float(),
        tgt_K=item["target_K"].unsqueeze(0).float(),
        tgt_c2w=item["target_c2w"].unsqueeze(0).float(),
        return_target_disparity=True,
        geometry_fp32=bool(cfg.get("camera_geometry_fp32", False)),
    )
    warped, mask, tgt_disp = warped[0], mask[0], tgt_disp[0]  # -> [T,...]

    t = args.target
    rgb_gt = item["target_images"][t].numpy()          # [3,H,W]
    rgb_splat = warped[t].numpy()                        # [3,H,W]
    m = mask[t, 0].numpy()                                # [H,W]
    coverage = float(m.mean())
    disp_splat = tgt_disp[t, 0].numpy()
    print(f"target #{t}: coverage={coverage:.4f}")

    num_src = item["source_images"].shape[0]
    ncols = num_src + 3
    fig, axes = plt.subplots(1, ncols, figsize=(3.2 * ncols, 3.6))
    for s in range(num_src):
        axes[s].imshow(item["source_images"][s].permute(1, 2, 0).numpy().clip(0, 1))
        axes[s].set_title(f"src{s+1}", fontsize=10)
    axes[num_src].imshow(np.clip(rgb_gt.transpose(1, 2, 0), 0, 1))
    axes[num_src].set_title(f"tgt{t+1} GT", fontsize=10)
    axes[num_src + 1].imshow(np.clip(rgb_splat.transpose(1, 2, 0), 0, 1))
    axes[num_src + 1].set_title(f"tgt{t+1} RGB warp\ncov={coverage:.3f}", fontsize=10)
    axes[num_src + 2].imshow(m, cmap="gray", vmin=0, vmax=1)
    axes[num_src + 2].set_title(f"tgt{t+1} warp mask", fontsize=10)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(f"{args.run}  sample_id={item['sample_id']}  b{args.b:02d}  target#{t}", fontsize=11)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.02, wspace=0.05)

    out = args.out or f"tools/rgb_warp_{os.path.basename(args.run)}_b{args.b:02d}_t{t}.png"
    plt.savefig(out, dpi=140)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
