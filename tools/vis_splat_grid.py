"""Visualize GT-based forward splat vs. ground truth for the frozen BMVS eval set.

Each row is one (sample, target-view) pair: target RGB GT, target disparity GT,
RGB splat (source views forward-warped with GT source depth), disparity splat
(same warp's target-frame 1/z).  Rows are picked to span low/mid/high
`tgt_rgb_splat_coverage` so the coverage-vs-quality relationship discussed
earlier is visible directly in the images, not just in the correlation numbers.

Pure geometry -- no checkpoint, no DiT, no VAE.  Same code path as
`WanRGBDNVSTrainingModule._splat_sources_pixel` (utils.warp.splat_sources_to_targets),
called directly instead of through the training module to avoid loading the model.
"""
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

CFG_PATH = "train_config/normal_config/bmvs_c_base.yaml"
NUM_SCENES_TO_SCAN = 20   # eval scenes to pull candidate rows from
ROWS_PER_BUCKET = 2       # low / mid / high coverage
OUT_PATH = "tools/splat_grid_vis.png"


def main():
    cfg = OmegaConf.load(CFG_PATH)
    OmegaConf.resolve(cfg)

    dataset = Kubric4DRGBDNVSDataset(
        kubric4d_root=cfg.kubric4d_root,
        manifest_path=cfg.manifest_path,
        resolution=tuple(cfg.resolution),
        num_source_views=cfg.num_source_views,
        num_target_views=cfg.num_target_views,
        random_permute_source_views=cfg.random_permute_source_views,
        # Overridable so the D2 splat confound (plan §D2: the z-buffer tie-break
        # inside z_tolerance depends on source order) can be measured on fixed
        # data with no GPU training.
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
    dataset.deterministic_view_seed = int(cfg.get("eval_view_seed", 1234))
    samples_per_scene = dataset.samples_per_scene
    num_scenes = min(NUM_SCENES_TO_SCAN, len(dataset.records))
    eval_indices = [i * samples_per_scene for i in range(num_scenes)]
    print(f"scanning {num_scenes} eval scenes (of {len(dataset.records)} in split="
          f"{cfg.get('eval_split', None)!r}) for coverage spread")

    eps = 1e-3
    candidates = []  # (coverage, sample_id, target_idx, rgb_gt, disp_gt, rgb_splat, disp_splat)
    for raw_idx in eval_indices:
        item = dataset[raw_idx]
        src_disp = item["source_disparity"][:, 0:1, :, :].unsqueeze(0)   # [1,S,1,H,W]
        src_disp = src_disp.squeeze(2)                                   # [1,S,H,W]
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
        )
        warped, mask, tgt_disp = warped[0], mask[0], tgt_disp[0]  # drop batch dim -> [T,...]

        for t in range(item["target_images"].shape[0]):
            coverage = float(mask[t, 0].mean())
            candidates.append((
                coverage,
                item["sample_id"],
                t,
                item["target_images"][t].numpy(),                 # [3,H,W]
                item["target_disparity"][t, 0].numpy(),            # [H,W]
                warped[t].numpy(),                                 # [3,H,W]
                tgt_disp[t, 0].numpy(),                             # [H,W]
            ))

    candidates.sort(key=lambda c: c[0])
    n = len(candidates)
    low = candidates[:ROWS_PER_BUCKET]
    mid = candidates[n // 2 - ROWS_PER_BUCKET // 2: n // 2 - ROWS_PER_BUCKET // 2 + ROWS_PER_BUCKET]
    high = candidates[-ROWS_PER_BUCKET:]
    rows = low + mid + high
    print(f"{n} (sample, target-view) pairs scanned; coverage range "
          f"[{candidates[0][0]:.3f}, {candidates[-1][0]:.3f}]")
    for cov, sid, t, *_ in rows:
        print(f"  picked coverage={cov:.3f}  sample={sid}  target_idx={t}")

    height, width = rows[0][3].shape[1:]  # rgb_gt is [3,H,W]
    aspect = height / width
    col_w = 3.6
    fig_w = 4 * col_w + 2.2  # extra room for the left row-label column
    fig_h = len(rows) * col_w * aspect
    fig, axes = plt.subplots(
        len(rows), 4, figsize=(fig_w, fig_h),
        gridspec_kw={"wspace": 0.03, "hspace": 0.06},
    )
    col_titles = ["target RGB (GT)", "target disparity (GT)",
                  "target RGB (splat)", "target disparity (splat)"]
    for r, (cov, sid, t, rgb_gt, disp_gt, rgb_splat, disp_splat) in enumerate(rows):
        panels = [
            (np.clip(rgb_gt.transpose(1, 2, 0), 0, 1), None),
            (disp_gt, "plasma"),
            (np.clip(rgb_splat.transpose(1, 2, 0), 0, 1), None),
            (disp_splat, "plasma"),
        ]
        bucket = "low" if r < ROWS_PER_BUCKET else ("mid" if r < 2 * ROWS_PER_BUCKET else "high")
        for c, (img, cmap) in enumerate(panels):
            ax = axes[r, c]
            if cmap is None:
                ax.imshow(img)
            else:
                ax.imshow(img, cmap=cmap, vmin=0.0, vmax=1.0)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(col_titles[c], fontsize=12)
            if c == 0:
                ax.text(
                    -0.04, 0.5,
                    f"[{bucket}]\n{sid}\ntgt#{t}\ncov={cov:.2f}",
                    fontsize=9, ha="right", va="center",
                    transform=ax.transAxes,
                )

    fig.subplots_adjust(left=0.16, right=0.995, top=0.97, bottom=0.01)
    plt.savefig(OUT_PATH, dpi=130)
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
