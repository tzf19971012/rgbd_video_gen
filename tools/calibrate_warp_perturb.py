"""Calibrate FACTOR e-warpnoise's degradation operator against the measured
statistics of the model's own pass-1 source disparity.

WHY THIS IS NOT OPTIONAL.  `perturb_source_disparity` manufactures the warp
regime that inference actually lands in, and its whole claim to being an
experiment rather than a guess is that its amplitudes reproduce a MEASURED
distribution.  Two of the four knobs have a closed form (see the docstring
there); the other two -- blur_sigma and lowfreq_std -- do not, and picking them
by eye would make a 3.8 node-day arm unreadable in the specific way this project
has already been burned by: it would look healthy either way.

The calibration target is the WARP, not the disparity L1.  Those two disagree,
and the disagreement is the whole finding of report_20260913_e_sweep_prep §3: L1
is dominated by the per-view affine error, while the DAMAGE (71.8% of the 4.84 dB
oracle gap) is dominated by residual structure.  Fit blur/lowfreq to L1 and you
get blur_sigma -> 0, i.e. an arm that trains on the one defect that does not
matter.  So the primary targets here are the splat's LPIPS/PSNR/coverage, with
the disparity statistics as guards.

Needs NO Wan weights and no checkpoint: the splat is pure geometry, so this
binds the shipped `_splat_sources_pixel` to a bare stub carrying only the two
attributes it reads.  ~2 min on one GPU for 16 batches.

Run as a MODULE from the project root:
  python -m tools.calibrate_warp_perturb --run logs/bmvs_d_droce_s42
  python -m tools.calibrate_warp_perturb --run logs/bmvs_d_droce_s42 \
      --blur_sigma 1:2:3:4 --lowfreq_std 0.0:0.02:0.04
"""
import argparse
import math
import os

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from examples.dataset import Kubric4DRGBDNVSDataset
from examples.wanvideo.model_training.train_with_accelerate_rgbd_nvs import custom_collate_fn
from examples.wanvideo.model_training.WanRGBDNVSTrainingModule import (
    DISPARITY_EPS,
    WanRGBDNVSTrainingModule,
    perturb_source_disparity,
    relative_c2w,
)

# Measured on 384 source / 192 target views of the model's OWN pass-1 estimate,
# bmvs_d_drocefp32_s42@3750 (tools/src_disp_scale_bmvs_d_drocefp32_s42_step3750.json,
# report_20260913_e_sweep_prep §3).  These are what the operator has to reproduce.
TARGETS = {
    "lpips_filled": 0.8947,   # primary: the structural damage
    "psnr_valid": 13.49,      # primary
    "coverage": 0.5371,       # primary: sky gets filled, so this EXCEEDS gt's 0.5266
    "affine_a_std": 0.3495,   # guard: closed form from gain_log_std
    "shift_b_std": 0.0590,    # guard: closed form from shift_std
    "raw_l1_median": 0.0437,  # guard ONLY -- see the module docstring
    "l1_over_spread_median": 0.1687,
}
# The oracle end of the same table, for reference when reading the output.
ORACLE = {"lpips_filled": 0.6344, "psnr_valid": 18.33, "coverage": 0.5266}


def num_list(text):
    """Colon-separated, because `sbatch --export` splits its own argument on
    commas and would silently turn a four-point sweep into a one-point one."""
    return [float(v) for v in str(text).split(":") if v != ""]


def fit_affine(p, g):
    """min_{a,b} ||a*p + b - g||^2, float64 closed form -- the same convention as
    evaluate()'s _fit_affine, so `a` is comparable to TARGETS."""
    p, g = p.double(), g.double()
    n = p.numel()
    sp, spp, sg, spg = p.sum(), (p * p).sum(), g.sum(), (p * g).sum()
    det = n * spp - sp * sp
    if torch.abs(det) < 1e-12:
        return 1.0, 0.0
    return float((n * spg - sp * sg) / det), float((spp * sg - sp * spg) / det)


def psnr_masked(pred, gt, mask):
    m = mask.expand_as(pred)
    n = m.sum()
    if n < 1:
        return float("nan")
    mse = ((((pred - gt) ** 2) * m).sum() / n).item()
    return 100.0 if mse < 1e-10 else -10 * math.log10(mse)


def build_dataset(cfg):
    dataset = Kubric4DRGBDNVSDataset(
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
    dataset.deterministic_view_seed = int(cfg.get("eval_view_seed", 1234))
    return dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="logs/bmvs_d_droce_s42",
                    help="only args.yaml is read; no checkpoint is loaded")
    ap.add_argument("--num_batches", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_lpips", action="store_true")
    # Sweep axes.  Defaults are single-valued, i.e. "score the shipped config".
    ap.add_argument("--blur_sigma", default=None, help="colon-separated sweep")
    ap.add_argument("--lowfreq_std", default=None, help="colon-separated sweep")
    ap.add_argument("--gain_log_std", default=None, help="colon-separated sweep")
    ap.add_argument("--shift_std", default=None, help="colon-separated sweep")
    ap.add_argument("--fill_invalid", default=None, help="colon-separated 0/1 sweep")
    ap.add_argument("--fill_scale", default=None, help="colon-separated sweep")
    args = ap.parse_args()

    cfg = OmegaConf.load(os.path.join(args.run, "args.yaml"))
    OmegaConf.resolve(cfg)
    device = torch.device(args.device)

    base = dict(
        gain_log_std=float(cfg.get("warp_perturb_gain_log_std", 0.34)),
        shift_std=float(cfg.get("warp_perturb_shift_std", 0.059)),
        blur_sigma=float(cfg.get("warp_perturb_blur_sigma", 2.0)),
        lowfreq_std=float(cfg.get("warp_perturb_lowfreq_std", 0.05)),
        lowfreq_cells=int(cfg.get("warp_perturb_lowfreq_cells", 6)),
        fill_invalid=bool(cfg.get("warp_perturb_fill_invalid", True)),
        fill_scale=float(cfg.get("warp_perturb_fill_scale", 1.0)),
    )
    grid_axes = {}
    for name, raw in (("blur_sigma", args.blur_sigma),
                      ("lowfreq_std", args.lowfreq_std),
                      ("gain_log_std", args.gain_log_std),
                      ("shift_std", args.shift_std),
                      ("fill_scale", args.fill_scale)):
        if raw is not None:
            grid_axes[name] = num_list(raw)
    if args.fill_invalid is not None:
        grid_axes["fill_invalid"] = [bool(int(v)) for v in args.fill_invalid.split(":")]

    configs = [dict(base)]
    for name, values in grid_axes.items():
        configs = [dict(c, **{name: v}) for c in configs for v in values]

    # Only geometry is needed, so bind the shipped method to a stub instead of
    # constructing the module: this keeps the code under test the SHIPPED code
    # (a reimplementation here could disagree with training and nobody would
    # notice) without paying for Wan2.1 or a checkpoint.
    class _SplatStub:
        num_source_views = int(cfg.num_source_views)
        num_target_views = int(cfg.num_target_views)
        camera_geometry_fp32 = bool(cfg.get("camera_geometry_fp32", False))
    stub = _SplatStub()
    splat = WanRGBDNVSTrainingModule._splat_sources_pixel.__get__(stub)
    num_src, num_tgt = stub.num_source_views, stub.num_target_views

    lpips_net = None
    if not args.no_lpips:
        try:
            import lpips as lpips_lib
            lpips_net = lpips_lib.LPIPS(
                net=cfg.get("eval_lpips_net", "alex"), verbose=False).to(device).eval()
        except ImportError:
            print("WARNING: lpips not installed -- the PRIMARY target is unavailable")

    dataset = build_dataset(cfg)
    eval_max_samples = int(cfg.get("eval_max_samples", 64))
    batch_size = max(1, min(eval_max_samples, int(cfg.get("eval_batch_size", 4))))
    samples_per_scene = cfg.get("samples_per_scene", 1)
    eval_indices = [i * samples_per_scene for i in range(len(dataset.records))][:eval_max_samples]
    loader = DataLoader(
        Subset(dataset, eval_indices), batch_size=batch_size, shuffle=False,
        num_workers=4, collate_fn=custom_collate_fn, drop_last=False,
    )

    # Materialise the batches ONCE so every grid cell sees byte-identical inputs;
    # a per-cell reload would re-roll the view selection and make the cells
    # incomparable at exactly the precision this calibration works at.
    batches = []
    for bi, batch in enumerate(loader):
        if bi >= args.num_batches:
            break
        batches.append(dict(
            rgb=batch["images"].to(device).float().clamp(0.0, 1.0),
            disp=batch["disparity"].to(device)[:, :num_src, 0].float(),
            K=batch["K"].to(device).float(),
            # relative_c2w, not the raw c2w: that is what forward_preprocess puts
            # in inputs["c2w"] and therefore what produced TARGETS.  The splat is
            # invariant to a global rigid transform in exact arithmetic, so this
            # matters only at the fp level -- which is exactly the level this
            # calibration works at, and `camera_geometry_fp32` is read off the run.
            c2w=relative_c2w(batch["c2w"].to(device).float(),
                             geometry_fp32=stub.camera_geometry_fp32),
        ))
    n_views = sum(b["disp"].shape[0] * num_src for b in batches)
    print(f"{len(batches)} batches, {n_views} source views, "
          f"{sum(b['disp'].shape[0] for b in batches) * num_tgt} target views, "
          f"{len(configs)} config(s)\n")

    rows = []
    for cfg_i, knobs in enumerate(configs):
        torch.manual_seed(args.seed)
        stats = {k: [] for k in ("affine_a", "shift_b", "raw_l1", "l1_over_spread",
                                 "coverage", "psnr_valid", "lpips_filled")}
        for b in batches:
            disp_gt, rgb = b["disp"], b["rgb"]
            with torch.no_grad():
                pert = perturb_source_disparity(disp_gt, **knobs)
                warped, mask = splat(rgb, pert, b["K"], b["c2w"])
            gt_valid = disp_gt > DISPARITY_EPS
            B = disp_gt.shape[0]
            for s in range(B):
                for v in range(num_src):
                    m = gt_valid[s, v]
                    if m.sum() < 64:
                        continue
                    p, g = pert[s, v][m], disp_gt[s, v][m]
                    a, _ = fit_affine(p, g)
                    spread = float(g.quantile(0.98) - g.quantile(0.02))
                    l1 = float((p - g).abs().mean())
                    stats["affine_a"].append(a)
                    stats["shift_b"].append(float((g.double() - p.double()).mean()))
                    stats["raw_l1"].append(l1)
                    stats["l1_over_spread"].append(l1 / max(spread, 1e-8))
            for s in range(B):
                for t in range(num_tgt):
                    gt_tile = rgb[s:s + 1, num_src + t]
                    wv, mv = warped[s:s + 1, t], mask[s:s + 1, t]
                    stats["coverage"].append(float(mv.mean()))
                    stats["psnr_valid"].append(psnr_masked(wv, gt_tile, mv))
                    if lpips_net is not None:
                        # Holes get the mean splatted colour, exactly as in
                        # diag_src_disp_scale.py: black would turn LPIPS into a
                        # coverage statistic instead of a geometry one, and the
                        # target number was produced that way.
                        fill = (wv * mv).sum(dim=(0, 2, 3)) / mv.sum().clamp_min(1.0)
                        filled = wv * mv + fill.reshape(1, 3, 1, 1) * (1.0 - mv)
                        with torch.no_grad():
                            stats["lpips_filled"].append(
                                float(lpips_net(filled * 2 - 1, gt_tile * 2 - 1).mean()))

        row = {"knobs": knobs}
        row["affine_a_std"] = float(np.std(stats["affine_a"]))
        row["affine_a_median"] = float(np.median(stats["affine_a"]))
        row["shift_b_std"] = float(np.std(stats["shift_b"]))
        row["raw_l1_median"] = float(np.median(stats["raw_l1"]))
        row["l1_over_spread_median"] = float(np.median(stats["l1_over_spread"]))
        row["coverage"] = float(np.mean(stats["coverage"]))
        row["psnr_valid"] = float(np.nanmean(stats["psnr_valid"]))
        row["lpips_filled"] = (float(np.mean(stats["lpips_filled"]))
                               if stats["lpips_filled"] else float("nan"))
        rows.append(row)
        print(f"[{cfg_i + 1}/{len(configs)}] "
              + " ".join(f"{k}={v}" for k, v in knobs.items()))
        print(f"      lpips={row['lpips_filled']:.4f} (target {TARGETS['lpips_filled']:.4f}, "
              f"oracle {ORACLE['lpips_filled']:.4f})  "
              f"psnr={row['psnr_valid']:.2f} (target {TARGETS['psnr_valid']:.2f}, "
              f"oracle {ORACLE['psnr_valid']:.2f})  "
              f"cov={row['coverage']:.4f} (target {TARGETS['coverage']:.4f})")
        print(f"      a_std={row['affine_a_std']:.4f} (target {TARGETS['affine_a_std']:.4f})  "
              f"a_med={row['affine_a_median']:.4f}  "
              f"shift_std={row['shift_b_std']:.4f} (target {TARGETS['shift_b_std']:.4f})  "
              f"l1_med={row['raw_l1_median']:.4f} (target {TARGETS['raw_l1_median']:.4f})  "
              f"l1/spread={row['l1_over_spread_median']:.4f} "
              f"(target {TARGETS['l1_over_spread_median']:.4f})")

    print("\n" + "=" * 78)
    print("ranked by the PRIMARY objective (relative error on LPIPS, then PSNR, then "
          "coverage);\nthe disparity statistics below the fold are guards, NOT targets "
          "to minimise")
    print("=" * 78)

    def cost(r):
        c = 0.0
        for key, weight in (("lpips_filled", 1.0), ("psnr_valid", 0.5), ("coverage", 0.5)):
            t = TARGETS[key]
            v = r[key]
            if v != v:                        # NaN, e.g. lpips unavailable
                continue
            c += weight * abs(v - t) / abs(t)
        return c

    for r in sorted(rows, key=cost):
        print(f"  cost={cost(r):.4f}  lpips={r['lpips_filled']:.4f}  "
              f"psnr={r['psnr_valid']:.2f}  cov={r['coverage']:.4f}  "
              f"a_std={r['affine_a_std']:.4f}  l1_med={r['raw_l1_median']:.4f}   "
              + " ".join(f"{k}={v}" for k, v in r["knobs"].items()))
    print("\nAdopt the winner by editing warp_perturb_* in "
          "train_config/normal_config/bmvs_c_base.yaml, then re-run "
          "`python -m tools.test_e_sweep_factors`.")


if __name__ == "__main__":
    main()
