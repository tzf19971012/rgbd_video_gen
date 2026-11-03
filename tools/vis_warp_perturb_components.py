"""Visualize the four components of FACTOR e-warpnoise's degradation operator
(`perturb_source_disparity` in WanRGBDNVSTrainingModule.py) on one real GT
source disparity map, plus what each does to the actual RGB warp.

Two figures:
  1. disparity space -- isolated ablations (each component alone) on top,
     the cumulative pipeline (fill -> blur -> affine -> lowfreq, the order the
     code applies them) on the bottom, for ONE source view.
  2. RGB space -- the target-view splat with GT disparity vs. with the fully
     degraded disparity, for all source views, so the disparity-space damage
     is tied to what the model actually has to denoise.

All amplitudes are read from `perturb_source_disparity`'s own defaults via
`inspect`, not re-typed here, so this script tracks the shipped calibration
(currently blur_sigma=2.0, lowfreq_std=0.05, ... -- see the module docstring
and tools/calibrate_warp_perturb.py) automatically.

Pure geometry -- no checkpoint, no DiT, no VAE.  Same dataset-loading and
splat code path as tools/vis_rgb_warp_one_sample.py.

Run as a MODULE from the project root:
  python -m tools.vis_warp_perturb_components --run logs/bmvs_d_droce_s42 --b 5
"""
import argparse
import inspect
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from examples.dataset import Kubric4DRGBDNVSDataset
from examples.wanvideo.model_training.WanRGBDNVSTrainingModule import (
    perturb_source_disparity,
)
from utils.warp import splat_sources_to_targets

DISPARITY_EPS = 1e-3


def build_eval_indices(cfg, eval_dataset):
    eval_max_samples = int(cfg.get("eval_max_samples", 2))
    samples_per_scene = getattr(eval_dataset, "samples_per_scene", 1)
    num_scenes = len(eval_dataset.records)
    eval_indices = [i * samples_per_scene for i in range(num_scenes)]
    return eval_indices[:eval_max_samples]


def default_kwargs():
    """The operator's own defaults -- the CALIBRATED values -- read off its
    signature so this script can't drift from what actually ships."""
    params = inspect.signature(perturb_source_disparity).parameters
    return {name: p.default for name, p in params.items()
            if name not in ("src_disparity", "generator")}


def apply(x, seed, **overrides):
    kwargs = default_kwargs()
    kwargs.update(overrides)
    torch.manual_seed(seed)
    return perturb_source_disparity(x, **kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="logs/bmvs_d_droce_s42",
                     help="only args.yaml is read; no checkpoint is loaded")
    ap.add_argument("--b", type=int, default=0, help="b-index into the frozen eval set")
    ap.add_argument("--src", type=int, default=0, help="source view to show in disparity space")
    ap.add_argument("--target", type=int, default=0, help="target view to show in RGB space")
    ap.add_argument("--seed", type=int, default=20260913)
    ap.add_argument("--out_prefix", default=None)
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
    print(f"b{args.b:02d} -> dataset idx {ds_idx} -> sample_id={item['sample_id']}")

    gt_disp = item["source_disparity"][:, 0:1, :, :].squeeze(1).unsqueeze(0).float()  # [1,S,H,W]
    defaults = default_kwargs()
    print("calibrated defaults:", defaults)

    # -------------------------------------------------------------------
    # Figure 1: disparity space
    # -------------------------------------------------------------------
    off = dict(defaults)
    off.update(fill_invalid=False, blur_sigma=0.0, gain_log_std=0.0,
               shift_std=0.0, lowfreq_std=0.0)

    isolated = {
        "GT": gt_disp,
        "+ fill_invalid\n(sky holes filled)": apply(gt_disp, args.seed, **dict(off, fill_invalid=True)),
        "+ blur_sigma\n(over-smoothing)": apply(gt_disp, args.seed, **dict(off, blur_sigma=defaults["blur_sigma"])),
        "+ gain/shift\n(per-view affine)": apply(gt_disp, args.seed, **dict(
            off, gain_log_std=defaults["gain_log_std"], shift_std=defaults["shift_std"])),
        "+ lowfreq field\n(spatial structure)": apply(gt_disp, args.seed, **dict(
            off, lowfreq_std=defaults["lowfreq_std"], lowfreq_cells=defaults["lowfreq_cells"])),
    }

    fill_cfg = dict(off, fill_invalid=True)
    fill_blur_cfg = dict(fill_cfg, blur_sigma=defaults["blur_sigma"])
    fill_blur_affine_cfg = dict(fill_blur_cfg, gain_log_std=defaults["gain_log_std"],
                                 shift_std=defaults["shift_std"])
    cumulative = {
        "GT": gt_disp,
        "1) fill": apply(gt_disp, args.seed, **fill_cfg),
        "2) fill+blur": apply(gt_disp, args.seed, **fill_blur_cfg),
        "3) +affine": apply(gt_disp, args.seed, **fill_blur_affine_cfg),
        "4) +lowfreq\n(= full pipeline)": apply(gt_disp, args.seed, **defaults),
    }

    s = args.src
    vmax = max(1e-6, float(gt_disp[0, s].max()) * 1.05)
    # A side-by-side absolute-value view hides blur/affine/fill on a sample whose
    # disparity range is dominated by one or two large structures (a building, the
    # ground plane) -- the component's effect is real but small relative to that
    # range.  A per-component (degraded - GT) row on a diverging, SYMMETRIC scale
    # makes every component visible and, unlike the absolute view, comparable
    # across components: it is the actual perturbation magnitude, not a guess.
    diff_scale = max(1e-6, max(
        float((x - gt_disp)[0, s].abs().max()) for x in list(isolated.values())[1:]))
    fig, axes = plt.subplots(3, 5, figsize=(16, 10))
    for c, (title, x) in enumerate(isolated.items()):
        axes[0, c].imshow(x[0, s].numpy(), cmap="plasma", vmin=0, vmax=vmax)
        axes[0, c].set_title(title, fontsize=9)
        diff = (x - gt_disp)[0, s].numpy()
        im = axes[1, c].imshow(diff, cmap="RdBu_r", vmin=-diff_scale, vmax=diff_scale)
        axes[1, c].set_title(f"mean|Δ|={np.abs(diff).mean():.4f}", fontsize=8)
    fig.colorbar(im, ax=list(axes[1, :]), fraction=0.02, pad=0.01, label="degraded - GT")
    for c, (title, x) in enumerate(cumulative.items()):
        axes[2, c].imshow(x[0, s].numpy(), cmap="plasma", vmin=0, vmax=vmax)
        axes[2, c].set_title(title, fontsize=9)
    axes[0, 0].set_ylabel("isolated\nabsolute value", fontsize=9)
    axes[1, 0].set_ylabel("isolated\n(degraded - GT)", fontsize=9)
    axes[2, 0].set_ylabel("cumulative\nfill→blur→affine→lowfreq", fontsize=9)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(
        f"{args.run}  sample_id={item['sample_id']}  src view #{s+1}  "
        f"(disparity, shared scale, plasma 0..{vmax:.2f})", fontsize=11)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.90, bottom=0.02, wspace=0.05, hspace=0.35)

    out1 = f"{args.out_prefix or ('tools/vis_warp_perturb_' + os.path.basename(args.run))}_disp_b{args.b:02d}_src{s}.png"
    plt.savefig(out1, dpi=140)
    print(f"saved {out1}")
    plt.close(fig)

    # -------------------------------------------------------------------
    # Figure 2: RGB space -- what the degraded disparity does to the splat
    # -------------------------------------------------------------------
    degraded_disp = apply(gt_disp, args.seed, **defaults)

    def splat(disp):
        valid = disp > DISPARITY_EPS
        depth = torch.where(valid, 1.0 / disp.clamp_min(DISPARITY_EPS), torch.zeros_like(disp))
        warped, mask, _ = splat_sources_to_targets(
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
        return warped[0], mask[0]

    warped_gt, mask_gt = splat(gt_disp)
    warped_deg, mask_deg = splat(degraded_disp)

    t = args.target
    rgb_target = item["target_images"][t].numpy()
    rgb_gt = warped_gt[t].numpy()
    rgb_deg = warped_deg[t].numpy()
    m_gt = mask_gt[t, 0].numpy()
    m_deg = mask_deg[t, 0].numpy()

    def psnr(pred, gt, mask):
        m = np.broadcast_to(mask, pred.shape)
        n = m.sum()
        if n < 1:
            return float("nan")
        mse = float((((pred - gt) ** 2) * m).sum() / n)
        return 100.0 if mse < 1e-10 else -10 * np.log10(mse)

    p_gt = psnr(rgb_gt, rgb_target, m_gt)
    p_deg = psnr(rgb_deg, rgb_target, m_deg)
    print(f"target #{t}: coverage gt={m_gt.mean():.4f} degraded={m_deg.mean():.4f}  "
          f"psnr_valid gt={p_gt:.2f} degraded={p_deg:.2f}")

    num_src = item["source_images"].shape[0]
    ncols = num_src + 3
    fig2, axes2 = plt.subplots(1, ncols, figsize=(3.0 * ncols, 3.4))
    for i in range(num_src):
        axes2[i].imshow(item["source_images"][i].permute(1, 2, 0).numpy().clip(0, 1))
        axes2[i].set_title(f"src{i+1}", fontsize=9)
    axes2[num_src].imshow(np.clip(rgb_target.transpose(1, 2, 0), 0, 1))
    axes2[num_src].set_title(f"tgt{t+1} GT image", fontsize=9)
    axes2[num_src + 1].imshow(np.clip(rgb_gt.transpose(1, 2, 0), 0, 1))
    axes2[num_src + 1].set_title(f"warp w/ GT disp\ncov={m_gt.mean():.3f} psnr={p_gt:.1f}", fontsize=9)
    axes2[num_src + 2].imshow(np.clip(rgb_deg.transpose(1, 2, 0), 0, 1))
    axes2[num_src + 2].set_title(f"warp w/ DEGRADED disp\ncov={m_deg.mean():.3f} psnr={p_deg:.1f}", fontsize=9)
    for ax in axes2:
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig2.suptitle(
        f"{args.run}  sample_id={item['sample_id']}  target#{t}  "
        f"(this is the regime ewarp/ewarp1 trains the model to see)", fontsize=11)
    fig2.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.02, wspace=0.06)

    out2 = f"{args.out_prefix or ('tools/vis_warp_perturb_' + os.path.basename(args.run))}_rgb_b{args.b:02d}_t{t}.png"
    plt.savefig(out2, dpi=140)
    print(f"saved {out2}")


if __name__ == "__main__":
    main()
