"""Non-learned target-RGB baselines for the frozen NVS eval set.

Target-view DEPTH has had a trivial baseline since the beginning: `delta1_ramp`
scores a plain vertical ramp on the very same pixels, and `delta1_skill` reports
the fraction of the remaining headroom the model closed.  Target-view RGB has
had nothing of the sort, so `tgt_rgb_psnr = 13.1` is uninterpretable: it could
be excellent or it could be worse than pasting a source view.

Two references, both computable at inference time (no GT target information):

  copy   Paste the nearest source view, nearest = smallest camera-centre
         distance.  Zero geometry, zero learning.  This is the floor any NVS
         model must clear to justify existing.

  splat  Forward-splat the source images into the target view with the GT
         SOURCE disparity, z-buffered -- exactly the geometry the warp channel
         already carries -- and fill the holes with the mean splatted colour.
         The model has to beat this to justify its 24 s/step.

Neither depends on a checkpoint, so one run covers the whole A/B matrix.  The
eval set is reconstructed with the same frozen recipe the training script uses
(same scene stride, same deterministic_view_seed), so the numbers line up with
the `tgt_rgb_*` metrics in the run logs.

Scoring reference: the training-time metric compares the model against the VAE
ROUND-TRIP of the GT, not the raw pixels.  With --vae this script does the same,
so the comparison is apples-to-apples, and additionally reports the round-trip
PSNR itself as the ceiling of the whole pipeline.  Without --vae everything is
scored against the raw pixels instead (worth ~0.05 dB at these levels, but the
approximation is then stated rather than hidden).

Example:
    python tools/probe_rgb_baselines.py \
        --config logs/bmvs_ab_a8/args.yaml \
        --vae models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth
"""

import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf  # noqa: E402

from examples.dataset.kubric4d_rgbd_nvs_dataset import Kubric4DRGBDNVSDataset  # noqa: E402
from utils.warp import splat_sources_to_targets  # noqa: E402


def _psnr(pred, gt):
    mse = ((pred - gt) ** 2).mean().item()
    return 100.0 if mse < 1e-10 else -10 * math.log10(mse)


def _psnr_masked(pred, gt, mask):
    m = mask.expand_as(pred)
    n = m.sum()
    if n < 1:
        return float("nan")
    mse = ((((pred - gt) ** 2) * m).sum() / n).item()
    return 100.0 if mse < 1e-10 else -10 * math.log10(mse)


def build_eval_indices(dataset, eval_max_samples):
    """Mirror of the frozen eval-set selection in train_with_accelerate_rgbd_nvs.

    Strides over SCENES (not raw indices) and takes the first sample slot of
    each, so the selection is invariant to samples_per_scene.
    """
    samples_per_scene = getattr(dataset, "samples_per_scene", 1)
    num_scenes = len(dataset.records)
    scene_stride = max(1, num_scenes // eval_max_samples)
    return [
        scene_idx * samples_per_scene
        for scene_idx in range(0, num_scenes, scene_stride)
    ][:eval_max_samples]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="an arm's args.yaml")
    parser.add_argument("--vae", default=None, help="path to Wan2.1_VAE.pth")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lpips-net", default="alex")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="override eval_max_samples (smoke tests only -- changing it changes "
             "the scene stride and therefore WHICH scenes are scored)",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device(args.device)

    dataset = Kubric4DRGBDNVSDataset(
        kubric4d_root=cfg.kubric4d_root,
        manifest_path=cfg.manifest_path,
        resolution=tuple(cfg.resolution),
        num_source_views=cfg.num_source_views,
        num_target_views=cfg.num_target_views,
        random_permute_source_views=cfg.random_permute_source_views,
        disparity_norm_mode=cfg.get("disparity_norm_mode", "shared_scale"),
        disparity_norm_quantile=cfg.get("disparity_norm_quantile", 0.98),
        disparity_norm_headroom=cfg.get("disparity_norm_headroom", 1.5),
        min_view_valid_fraction=cfg.get("min_view_valid_fraction", 0.0),
        view_validity_path=cfg.get("view_validity_path", None),
        samples_per_scene=cfg.get("samples_per_scene", 1),
    )
    dataset.deterministic_view_seed = int(cfg.get("eval_view_seed", 1234))
    eval_max_samples = args.max_samples or int(cfg.get("eval_max_samples", 2))
    eval_indices = build_eval_indices(dataset, eval_max_samples)

    num_src = int(cfg.num_source_views)
    num_tgt = int(cfg.num_target_views)
    print(
        f"config={args.config}\n"
        f"scenes={len(dataset.records)}  eval samples={len(eval_indices)}  "
        f"s{num_src}t{num_tgt}  view_seed={dataset.deterministic_view_seed}\n"
        f"indices={eval_indices}"
    )

    vae = None
    if args.vae:
        from diffsynth.models.wan_video_vae import WanVideoVAE

        vae = WanVideoVAE()
        state = torch.load(args.vae, map_location="cpu", weights_only=True)
        missing, unexpected = vae.model.load_state_dict(state, strict=False)
        print(f"VAE loaded: missing={len(missing)} unexpected={len(unexpected)}")
        vae = vae.to(device).eval()

    try:
        import lpips as lpips_lib

        lpips_net = lpips_lib.LPIPS(net=args.lpips_net, verbose=False).to(device).eval()
    except ImportError:
        print("WARNING: `lpips` not installed, LPIPS columns will be NaN.")
        lpips_net = None

    def _lpips(pred, gt):
        if lpips_net is None:
            return float("nan")
        with torch.no_grad():
            return float(lpips_net(pred * 2 - 1, gt * 2 - 1).mean())

    @torch.no_grad()
    def _vae_roundtrip(img):
        """[1,3,H,W] in [0,1] -> VAE encode/decode round trip, same convention
        as _encode_views/_decode_view in the training module (independent mode:
        one view at a time, so the temporal convs never mix frames)."""
        lat = vae.encode(img[:, :, None], device=device)
        return vae.decode(lat, device=device)[:, :, 0].float().clamp(0, 1)

    keys = [
        "psnr_copy", "lpips_copy",
        "psnr_splat", "lpips_splat",
        "psnr_splat_valid", "splat_coverage",
        "psnr_vae",
    ]
    acc = {k: [] for k in keys}

    for n, idx in enumerate(eval_indices):
        item = dataset[idx]
        images = item["images"].to(device).float().clamp(0, 1)[None]       # [1,V,3,H,W]
        disparity = item["disparity"].to(device).float()[None]             # [1,V,3,H,W]
        K = item["K"].to(device).float()[None]                             # [1,V,3,3]
        c2w = item["c2w"].to(device).float()[None]                         # [1,V,4,4]

        # Same relative-pose convention as forward_preprocess.  Splatting is
        # invariant to a global rigid transform, so this only matters for
        # bit-level agreement with the training-time numbers.
        c2w = torch.linalg.inv(c2w[:, 0:1]) @ c2w

        warped, mask = splat_sources_to_targets(
            src_rgb=images[:, :num_src],
            src_depth=(1.0 / disparity[:, :num_src, 0].clamp_min(1e-3)),
            src_valid=disparity[:, :num_src, 0] > 1e-3,
            src_K=K[:, :num_src],
            src_c2w=c2w[:, :num_src],
            tgt_K=K[:, num_src:],
            tgt_c2w=c2w[:, num_src:],
        )

        centers = c2w[0, :, :3, 3]
        for tv in range(num_tgt):
            vi = num_src + tv
            raw_px = images[0, vi][None]                                   # [1,3,H,W]
            if vae is not None:
                gt_px = _vae_roundtrip(raw_px)
                acc["psnr_vae"].append(_psnr(gt_px, raw_px))
            else:
                gt_px = raw_px

            nearest = int(torch.argmin((centers[:num_src] - centers[vi]).norm(dim=-1)))
            copy_px = images[0, nearest][None]
            acc["psnr_copy"].append(_psnr(copy_px, gt_px))
            acc["lpips_copy"].append(_lpips(copy_px, gt_px))

            w, m = warped[0, tv], mask[0, tv]                              # [3,H,W], [1,H,W]
            fill = (w * m).sum(dim=(1, 2)) / m.sum().clamp_min(1.0)
            filled = (w * m + fill[:, None, None] * (1.0 - m))[None]
            acc["splat_coverage"].append(float(m.mean()))
            acc["psnr_splat"].append(_psnr(filled, gt_px))
            acc["lpips_splat"].append(_lpips(filled, gt_px))
            acc["psnr_splat_valid"].append(_psnr_masked(w[None], gt_px, m[None]))

        if (n + 1) % 8 == 0:
            print(f"  ...{n + 1}/{len(eval_indices)} scenes")

    print(f"\n=== target-RGB baselines  (n={len(acc['psnr_copy'])} target views) ===")
    ref = "VAE round-trip GT (same reference as tgt_rgb_psnr)" if vae else "RAW GT pixels"
    print(f"reference: {ref}")
    for k in keys:
        v = np.asarray(acc[k], dtype=np.float64)
        if v.size == 0 or np.all(np.isnan(v)):
            continue
        print(f"  {k:20s} mean={np.nanmean(v):8.4f}   std={np.nanstd(v):7.4f}")


if __name__ == "__main__":
    main()
