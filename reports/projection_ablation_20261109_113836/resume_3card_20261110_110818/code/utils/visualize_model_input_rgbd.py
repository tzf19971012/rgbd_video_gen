import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

# Ensure project-root absolute imports work regardless of current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.dataset import Kubric4DRGBDNVSDataset


def _disparity_to_color(disp_hw):
    disp_hw = np.clip(disp_hw, 0.0, 1.0)
    gray = (disp_hw * 255.0).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def _to_uint8_rgb(img_3hw):
    arr = img_3hw.detach().cpu().permute(1, 2, 0).numpy()
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def _make_pair(rgb_3hw, disp_3hw):
    rgb = _to_uint8_rgb(rgb_3hw)
    disp = disp_3hw[0].detach().cpu().numpy()
    disp_color = _disparity_to_color(disp)
    sep = np.full((rgb.shape[0], 6, 3), 200, dtype=np.uint8)
    return np.concatenate([rgb, sep, disp_color], axis=1)


def _save_sample_grid(sample, out_path):
    images = sample["images"]
    disparity = sample["disparity"]
    num_views = images.shape[0]
    rows = []
    for v in range(num_views):
        rows.append(_make_pair(images[v], disparity[v]))
    row_sep = np.full((6, rows[0].shape[1], 3), 220, dtype=np.uint8)
    canvas = rows[0]
    for row in rows[1:]:
        canvas = np.concatenate([canvas, row_sep, row], axis=0)
    Image.fromarray(canvas).save(out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize model-input RGB and disparity pairs after dataset preprocessing."
    )
    parser.add_argument("--config", type=str, required=True, help="Training config yaml path")
    parser.add_argument("--out_dir", type=str, default="logs/input_rgbd_vis", help="Output folder")
    parser.add_argument("--num_samples", type=int, default=6, help="Number of samples to export")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument(
        "--use_random_indices",
        action="store_true",
        help="Use random sample indices. Default uses first N indices.",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    dataset = Kubric4DRGBDNVSDataset(
        kubric4d_root=cfg.kubric4d_root,
        manifest_path=cfg.manifest_path,
        resolution=tuple(cfg.resolution),
        num_source_views=cfg.num_source_views,
        num_target_views=cfg.num_target_views,
        random_permute_source_views=cfg.random_permute_source_views,
    )

    os.makedirs(args.out_dir, exist_ok=True)

    n = min(args.num_samples, len(dataset))
    rng = random.Random(args.seed)
    if args.use_random_indices:
        indices = rng.sample(range(len(dataset)), n)
    else:
        indices = list(range(n))

    meta_lines = []
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        sample_id = sample.get("sample_id", f"idx_{idx}")
        out_name = f"sample_{i:02d}_idx_{idx:06d}.png"
        out_path = os.path.join(args.out_dir, out_name)
        _save_sample_grid(sample, out_path)

        roles = sample["view_roles"].tolist() if "view_roles" in sample else []
        meta_lines.append(f"{out_name}\tidx={idx}\tsample_id={sample_id}\tview_roles={roles}")
        print(f"saved: {out_path}")

    meta_path = os.path.join(args.out_dir, "index.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("\n".join(meta_lines) + "\n")
    print(f"wrote: {meta_path}")


if __name__ == "__main__":
    main()
