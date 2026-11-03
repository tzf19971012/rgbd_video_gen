"""Smoke-test the eval visualisation grid without a GPU.

`WanRGBDNVSTrainingModule.evaluate` cannot run here: it needs the DiT and the VAE
on a GPU.  But the part that changed -- the 3 x (S + 2T) layout, the placeholder
cells, the row/column labels, and now the metrics caption strip -- is pure image
assembly, so it lives in `_compose_eval_grid` and this script calls THAT
function, not a copy of it.  A test that reimplements the layout would pass
while the real one is broken.  Fake but plausible src/tgt caption numbers are
passed by default so the caption strip's alignment (one block per source
column, one block centered under each target's gt+pred pair) is visible without
a checkpoint -- pass --no_captions to see the grid without it.

Everything except the two prediction rows is real: RGB and GT disparity come from
Kubric4DRGBDNVSDataset at the training resolution, through the same
`disparity_norm_mode="shared_scale"` normalisation and the same plasma colormap the
eval uses.  The "pred" tiles are STAND-INS -- GT degraded by a blur, a global scale
error and a little noise -- because there is no checkpoint to run.  So this
validates the layout, not the model.

Usage:
  python -m tools.smoke_eval_grid --num_source_views 6 --num_target_views 3
"""
import argparse
import os

import numpy as np

from examples.dataset.kubric4d_rgbd_nvs_dataset import Kubric4DRGBDNVSDataset
from examples.wanvideo.model_training.WanRGBDNVSTrainingModule import _compose_eval_grid


def _plasma(gray):
    """[0,1] HxW float -> uint8 HxWx3, the same mapping the eval uses for depth."""
    import matplotlib.cm as cm
    return (cm.plasma(np.clip(gray, 0, 1))[:, :, :3] * 255).astype(np.uint8)


def _fake_pred(gray, rng, blur=9, scale=1.08, noise=0.02):
    """A plausible-looking wrong depth: blurred, mis-scaled, slightly noisy.

    Deliberately imperfect so the gt/pred columns are visibly different -- if they
    were identical the panel could not show whether the columns are ordered right.
    """
    pad = blur // 2
    padded = np.pad(gray, pad, mode="edge")
    kernel = np.ones(blur, np.float32) / blur
    smooth = np.apply_along_axis(lambda m: np.convolve(m, kernel, "valid"), 0, padded)
    smooth = np.apply_along_axis(lambda m: np.convolve(m, kernel, "valid"), 1, smooth)
    return np.clip(smooth * scale + rng.normal(0, noise, smooth.shape), 0, 1)


def main():
    parser = argparse.ArgumentParser()
    default_root = os.path.join(os.environ.get("DVD_DATA_ROOT", ""), "datasets", "BlendedMVS")
    parser.add_argument("--bmvs_root", default=default_root)
    parser.add_argument("--split_path", default="configs/bmvs_split_v1.json")
    parser.add_argument("--split", default="val")
    parser.add_argument("--num_source_views", type=int, default=6)
    parser.add_argument("--num_target_views", type=int, default=3)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--downscale", type=int, default=2,
                        help="Mirrors cfg eval_vis_downscale.")
    parser.add_argument("--out", default="logs/smoke_eval_grid.png")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_captions", action="store_true",
                        help="Render without the metrics caption strip.")
    args = parser.parse_args()

    dataset = Kubric4DRGBDNVSDataset(
        kubric4d_root=os.path.join(args.bmvs_root, "BlendedMVS"),
        manifest_path=os.path.join(args.bmvs_root, "manifest.jsonl"),
        num_source_views=args.num_source_views,
        num_target_views=args.num_target_views,
        samples_per_scene=1,
        min_view_valid_fraction=0.5,
        split_path=args.split_path,
        split=args.split,
        deterministic_view_seed=args.seed,
    )
    sample = dataset[args.index]
    rgb = sample["images"].float().numpy()        # [V, 3, H, W] in [0,1]
    # [V, 3, H, W]: disparity is replicated across channels, and the eval's
    # _gt_depth_to_np also takes channel 0, so index it the same way.
    disp = sample["disparity"].float().numpy()
    n_src, n_tgt = args.num_source_views, args.num_target_views
    print(f"scene {sample['sample_id']}  images {rgb.shape}  disparity {disp.shape} "
          f"[{disp.min():.3f}, {disp.max():.3f}]")

    rng = np.random.default_rng(args.seed)

    def rgb_tile(v):
        return (np.clip(rgb[v].transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)

    src_rgb = [rgb_tile(v) for v in range(n_src)]
    src_gt = [_plasma(disp[v, 0]) for v in range(n_src)]
    src_pred = [_plasma(_fake_pred(disp[v, 0], rng)) for v in range(n_src)]
    tgt_tiles = []
    for t in range(n_tgt):
        v = n_src + t
        # The pred RGB stand-in is the nearest source view's RGB: that is the
        # "copy" baseline the eval already scores, so it looks like a wrong-but-
        # plausible novel view rather than noise.
        tgt_tiles.append((
            rgb_tile(v),
            rgb_tile(min(t, n_src - 1)),
            _plasma(disp[v, 0]),
            _plasma(_fake_pred(disp[v, 0], rng)),
        ))

    src_captions = tgt_captions = None
    if not args.no_captions:
        src_captions = [[f"abs_rel {rng.uniform(0.02, 0.15):.3f}",
                         f"delta1 {rng.uniform(0.7, 1.0):.3f}"] for _ in range(n_src)]
        tgt_captions = [[f"cov {rng.uniform(0.0, 1.0):.3f}",
                         f"psnr {rng.uniform(10, 20):.2f} dB",
                         f"lpips {rng.uniform(0.3, 0.7):.3f}",
                         f"delta1 {rng.uniform(0.7, 1.0):.3f}",
                         f"abs_rel {rng.uniform(0.02, 0.2):.3f}"] for _ in range(n_tgt)]

    img, warning = _compose_eval_grid(src_rgb, src_gt, src_pred, tgt_tiles,
                                      downscale=args.downscale,
                                      src_captions=src_captions,
                                      tgt_captions=tgt_captions)
    if warning:
        print(f"[warn] {warning}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    img.save(args.out)
    print(f"{args.num_source_views} src + {args.num_target_views} tgt -> "
          f"3 x {n_src + 2 * n_tgt} grid, {img.size[0]}x{img.size[1]}  ->  {args.out}")


if __name__ == "__main__":
    main()
