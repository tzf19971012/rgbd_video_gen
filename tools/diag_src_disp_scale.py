"""Decompose the source-depth → RGB-warp failure into (a) global scale/shift,
(b) the missing sky/invalid mask, (c) residual structure error.

WHY.  At eval the model never gets a GT-depth warp: forward_preprocess zeroes
warp_block under `_eval_inputs` and sample_rgb rebuilds it from its own pass-1
source-depth estimate.  Over 192 val views that costs 18.22 dB (GT-depth splat)
→ 13.70 dB (predicted-depth splat).  The affine-aligned depth metrics say the
prediction is fine (src_depth_abs_rel≈0.06, delta1≈0.966) because _fit_affine
absorbs a global scale/shift that the warp -- which consumes RAW unaligned
disparity -- cannot tolerate.  On 57f8d9bbe73f6760f10e916a/tt0/s50 the raw
per-view disparity L1 is 44-152% of the view's own GT disparity spread
(p98-p2).

That is evidence the scale error is LARGE ENOUGH to wreck the warp.  It is NOT
evidence that fixing the scale would recover the warp: the same figure shows
two other defects an affine cannot repair -- the prediction is over-smooth
(layered archway structure gone) and it fills GT-invalid sky with plausible
disparity.  This script measures which of the three actually owns the 4.5 dB,
because that decides whether the E round should spend an arm on scale
supervision at all.

Section A -- VAE linearity.  A cheap latent-space "DC loss" on source depth is
only meaningful if a constant offset in pixel disparity moves the depth latent's
spatial mean monotonically.  The Wan VAE is a nonlinear conv encoder, so this is
an assumption, not a fact.  Measured first because a negative answer kills the
whole scale-loss idea before any training is spent on it.

Section B -- warp decomposition.  Splat with six source-disparity variants and
compare on the SAME samples/seed:
  raw            model's pass-1 prediction, i.e. what it actually receives
  shift          + per-view b            (1 DOF/view)
  affine_pv      a*p + b per view        (2 DOF/view)
  affine_shared  one (a,b) for all views (2 DOF total -- what a self-supervised
                 inference-time fix could realistically achieve)
  affine_gtmask  affine_pv AND GT validity imposed
  gt             oracle upper bound
Reading:  raw→affine_pv is the scale/shift share, affine_pv→affine_gtmask the
mask share, affine_gtmask→gt the residual structure share.

Uses PASS 1, not pass 2: pass 1's estimate is what builds the warp the model is
fed, and skipping pass 2 halves the DiT cost.

Run as a MODULE from the project root, one GPU:
  python -m tools.diag_src_disp_scale \
      --run logs/bmvs_d_drocefp32_s42 --step 3750 --num_batches 16
"""
import argparse
import inspect
import json
import math
import os

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Subset

from examples.dataset import Kubric4DRGBDNVSDataset
from examples.wanvideo.model_training.train_with_accelerate_rgbd_nvs import custom_collate_fn
from examples.wanvideo.model_training.WanRGBDNVSTrainingModule import WanRGBDNVSTrainingModule

EPS = 1e-3
VARIANTS = ("raw", "shift", "affine_pv", "affine_shared", "affine_gtmask", "gt")


def build_model(cfg, accelerator):
    """Construct the module with EVERY __init__ param the run's config sets.

    Introspection rather than a hand-written kwargs list: the shape-preserving
    flags (shared_rope_frame_ids, roce, camera_geometry_fp32, per_frame_timestep,
    ...) load cleanly when omitted and then silently describe a different model
    -- 1.45 dB of pure harness artefact in report_20260907_c_sweep §9-1a.
    """
    params = inspect.signature(WanRGBDNVSTrainingModule.__init__).parameters
    skip = {"self", "args", "accelerator", "model_paths"}
    kwargs = {n: cfg[n] for n in params if n not in skip and n in cfg}
    kwargs["use_gradient_checkpointing"] = False
    kwargs["use_gradient_checkpointing_offload"] = False
    kwargs["init_dvd_ckpt"] = None
    model = WanRGBDNVSTrainingModule(args=cfg, accelerator=accelerator, **kwargs)
    print(
        "model identity: "
        f"warp_mode={model.warp_mode} warp_two_pass={model.warp_two_pass} "
        f"shared_rope_frame_ids={model.shared_rope_frame_ids} "
        f"roce={model.roce_mode} camera_geometry_fp32={model.camera_geometry_fp32} "
        f"per_frame_timestep={model.per_frame_timestep} "
        f"clean_source_depth={model.clean_source_depth} "
        f"token_layout={model.token_layout} "
        f"eval_num_inference_steps={model.eval_num_inference_steps} "
        f"eval_flow_shift={model.eval_flow_shift} "
        f"eval_sample_seed={model.eval_sample_seed}"
    )
    return model


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


def nanmean(xs):
    vals = [float(x) for x in xs if x is not None and not math.isnan(float(x))]
    return float("nan") if not vals else sum(vals) / len(vals)


def psnr_masked(pred, gt, mask):
    m = mask.expand_as(pred)
    n = m.sum()
    if n < 1:
        return float("nan")
    mse = ((((pred - gt) ** 2) * m).sum() / n).item()
    return 100.0 if mse < 1e-10 else -10 * math.log10(mse)


def fit_affine(p, g):
    """Least squares min_{a,b} ||a*p + b - g||^2, float64 closed form.

    Same convention as evaluate()'s _fit_affine, so `a` is directly comparable
    to the alignment the reported src_depth_abs_rel/delta1 silently apply.
    """
    p, g = p.double(), g.double()
    n = p.numel()
    sp, spp, sg, spg = p.sum(), (p * p).sum(), g.sum(), (p * g).sum()
    det = n * spp - sp * sp
    if torch.abs(det) < 1e-12:
        return g.median() / p.median().clamp_min(1e-8), torch.zeros((), dtype=torch.float64)
    return (n * spg - sp * sg) / det, (spp * sg - sp * spg) / det


# --------------------------------------------------------------------------
# Section A: does a constant offset in pixel disparity move the latent DC?
# --------------------------------------------------------------------------
def vae_linearity(model, disp_gt, device):
    """disp_gt: [B,S,H,W] GT disparity in [0,1]. Prints DC-vs-offset slopes."""
    deltas = [-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10]
    scales = [0.80, 0.90, 1.00, 1.10, 1.25]

    def latent_dc(d):
        """d: [B,S,H,W] -> [16] mean latent DC over samples and views."""
        d3 = d.clamp(0.0, 1.0).unsqueeze(2).expand(-1, -1, 3, -1, -1)
        lat, _ = model._encode_views(d3.to(model.pipe.torch_dtype))   # [B,16,S,h,w]
        return lat.float().mean(dim=(0, 2, 3, 4))                     # [16]

    print("\n" + "=" * 78)
    print("SECTION A -- VAE linearity: pixel disparity offset/scale -> latent DC")
    print("=" * 78)
    for label, values, apply in (
        ("offset  disp + d", deltas, lambda d, v: disp_gt + v),
        ("scale   disp * a", scales, lambda d, v: disp_gt * v),
    ):
        dcs = torch.stack([latent_dc(apply(disp_gt, v)) for v in values]).cpu()  # [K,16]
        x = torch.tensor(values, dtype=torch.float64)
        xc = x - x.mean()
        print(f"\n{label}   values={values}")
        rows = []
        for c in range(dcs.shape[1]):
            y = dcs[:, c].double()
            yc = y - y.mean()
            denom = (xc.pow(2).sum().sqrt() * yc.pow(2).sum().sqrt()).clamp_min(1e-12)
            r = float((xc * yc).sum() / denom)
            slope = float((xc * yc).sum() / xc.pow(2).sum().clamp_min(1e-12))
            dy = y.diff()
            mono = bool((dy >= -1e-6).all() or (dy <= 1e-6).all())
            rows.append((abs(r), c, r, slope, mono, y))
        rows.sort(reverse=True)
        print(f"  {'ch':>3} {'pearson_r':>10} {'slope':>10} {'monotonic':>10}   DC values")
        for _, c, r, slope, mono, y in rows[:6]:
            vals = " ".join(f"{float(v):+.3f}" for v in y)
            print(f"  {c:>3} {r:>10.4f} {slope:>10.3f} {str(mono):>10}   {vals}")
        strong = sum(1 for a, *_ in rows if a > 0.99)
        print(f"  channels with |r|>0.99: {strong}/16   "
              f"(a latent-DC loss can constrain this DOF only if this is >0)")


# --------------------------------------------------------------------------
# Section B: warp decomposition
# --------------------------------------------------------------------------
def make_variants(pred_disp, disp_gt):
    """Build the six source-disparity variants. All [B,S,H,W].

    The GT validity mask is imposed by zeroing, because _splat_sources_pixel
    derives validity from `src_disparity > eps` internally -- so a zero is the
    only way to tell it a pixel is invalid without changing its signature.
    """
    B, S = pred_disp.shape[:2]
    gt_valid = disp_gt > EPS
    out = {"raw": pred_disp.clone(), "gt": disp_gt.clone()}
    shift = pred_disp.clone()
    aff_pv = pred_disp.clone()
    fits = []                                  # (b_only, a, b) per (sample, view)
    for b in range(B):
        for v in range(S):
            m = gt_valid[b, v]
            if m.sum() < 64:
                fits.append((0.0, 1.0, 0.0))
                continue
            p, g = pred_disp[b, v][m], disp_gt[b, v][m]
            b_only = float((g.double() - p.double()).mean())
            a, bb = fit_affine(p, g)
            shift[b, v] = pred_disp[b, v] + b_only
            aff_pv[b, v] = float(a) * pred_disp[b, v] + float(bb)
            fits.append((b_only, float(a), float(bb)))
    out["shift"] = shift
    out["affine_pv"] = aff_pv

    # One (a, b) pooled over all source views of a sample: the 2-DOF fix an
    # inference-time self-supervised search could plausibly find, since it has
    # no per-view GT to fit against.
    shared = pred_disp.clone()
    for b in range(B):
        m = gt_valid[b]
        if m.sum() >= 64:
            a, bb = fit_affine(pred_disp[b][m], disp_gt[b][m])
            shared[b] = float(a) * pred_disp[b] + float(bb)
    out["affine_shared"] = shared

    # A negative aligned disparity would silently become an invalid pixel and
    # smuggle mask information into the scale-only variants; floor them instead.
    for k in ("raw", "shift", "affine_pv", "affine_shared"):
        out[k] = out[k].clamp_min(EPS * 2)

    # affine_pv with GT validity imposed: isolates the "sky filled with
    # plausible disparity" defect from the scale defect.  Built from the FLOORED
    # affine_pv so the only difference between the two variants is the mask.
    out["affine_gtmask"] = torch.where(
        gt_valid, out["affine_pv"], torch.zeros_like(aff_pv))
    return out, fits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--num_batches", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = OmegaConf.load(os.path.join(args.run, "args.yaml"))
    OmegaConf.resolve(cfg)
    set_seed(0)
    accelerator = Accelerator(mixed_precision=cfg.mixed_precision)

    model = build_model(cfg, accelerator)
    ckpt = os.path.join(args.run, f"checkpoint-step-{args.step}")
    nfiles = len(os.listdir(ckpt))
    if nfiles != 11:
        raise SystemExit(f"{ckpt} has {nfiles} files, expected 11 -- write was interrupted")
    state = load_file(os.path.join(ckpt, "model.safetensors"), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"loaded {ckpt}: missing={len(missing)} unexpected={len(unexpected)}")
    model.to(accelerator.device)
    model.eval()
    device = model.pipe.device
    num_src = model.num_source_views

    dataset = build_dataset(cfg)
    eval_max_samples = int(cfg.get("eval_max_samples", 2))
    eval_batch_size = max(1, min(eval_max_samples, int(cfg.get("eval_batch_size", 4))))
    samples_per_scene = cfg.get("samples_per_scene", 1)
    eval_indices = [i * samples_per_scene for i in range(len(dataset.records))][:eval_max_samples]
    loader = DataLoader(
        Subset(dataset, eval_indices), batch_size=eval_batch_size, shuffle=False,
        num_workers=0, collate_fn=custom_collate_fn, drop_last=False,
    )
    print(f"frozen eval: {len(eval_indices)} samples, batch {eval_batch_size}, "
          f"scoring {args.num_batches} batches")

    lpips_net = model._get_lpips(device)
    rows, view_rows = [], []
    did_section_a = False

    autocast_ctx = torch.autocast(device_type="cuda", dtype=model.pipe.torch_dtype, enabled=True)
    for bi, batch in enumerate(loader):
        if bi >= args.num_batches:
            break
        with torch.no_grad(), autocast_ctx:
            model._eval_inputs = True
            try:
                inputs = model.forward_preprocess(batch)
            finally:
                model._eval_inputs = False

            rgb_px = batch["images"].to(device).float().clamp(0.0, 1.0)      # [B,V,3,H,W]
            disp_gt = batch["disparity"].to(device)[:, :num_src, 0].float()  # [B,S,H,W]
            K = batch["K"].to(device).float()

            if not did_section_a:
                vae_linearity(model, disp_gt, device)
                did_section_a = True
                print("\n" + "=" * 78)
                print("SECTION B -- warp decomposition (pass-1 predicted source disparity)")
                print("=" * 78)

            # -- pass 1: blind (zeroed warp), exactly what sample_rgb does ----
            gen = torch.Generator(device=device)
            gen.manual_seed(model.eval_sample_seed)
            blind = dict(inputs)
            blind["warp_block"] = torch.zeros_like(inputs["warp_block"])
            if inputs.get("depth_warp_block") is not None:
                blind["depth_warp_block"] = torch.zeros_like(inputs["depth_warp_block"])
            pred_depth_p1, _ = model._sample_rgb_once(
                blind, model.eval_num_inference_steps, gen)

            lat = pred_depth_p1[:, :, :num_src] / model.depth_latent_scale
            pred_disp = torch.stack([
                model.pipe.vae.decode(
                    lat[:, :, v:v + 1].to(model.pipe.torch_dtype), device=device
                )[:, :, 0].float().mean(dim=1)
                for v in range(num_src)
            ], dim=1).clamp(0.0, 1.0)                                       # [B,S,H,W]

            variants, fits = make_variants(pred_disp, disp_gt)
            splats = {k: model._splat_sources_pixel(rgb_px, v, K, inputs["c2w"])
                      for k, v in variants.items()}

        B = pred_disp.shape[0]
        # --- per source view: how big is the raw error relative to the scene? --
        gt_valid = disp_gt > EPS
        for b in range(B):
            for v in range(num_src):
                m = gt_valid[b, v]
                if m.sum() < 64:
                    continue
                g = disp_gt[b, v][m]
                spread = float(g.quantile(0.98) - g.quantile(0.02))
                l1 = float((pred_disp[b, v][m] - g).abs().mean())
                b_only, a, bb = fits[b * num_src + v]
                view_rows.append(dict(
                    sample=batch["sample_id"][b], view=v, raw_l1=l1, spread=spread,
                    l1_over_spread=l1 / max(spread, 1e-8),
                    shift_b=b_only, affine_a=a, affine_b=bb,
                    valid_frac=float(m.float().mean()),
                ))

        # --- per target view: the six warps --------------------------------
        for b in range(B):
            for t in range(model.num_target_views):
                gt_tile = rgb_px[b:b + 1, num_src + t]
                row = dict(sample=batch["sample_id"][b], target=t)
                for k in VARIANTS:
                    w, msk = splats[k]
                    wv, mv = w[b:b + 1, t], msk[b:b + 1, t]
                    cov = float(mv.mean())
                    row[f"cov_{k}"] = cov
                    row[f"psnr_{k}"] = psnr_masked(wv, gt_tile, mv)
                    # Holes get the mean splatted colour, matching evaluate()'s
                    # tgt_rgb_lpips_splat: black would make LPIPS a coverage
                    # statistic instead of a geometry one.
                    denom = mv.sum().clamp_min(1.0)
                    fill = (wv * mv).sum(dim=(0, 2, 3)) / denom
                    filled = wv * mv + fill.reshape(1, 3, 1, 1) * (1.0 - mv)
                    if lpips_net is not None:
                        with torch.no_grad():
                            row[f"lpips_{k}"] = float(
                                lpips_net(filled * 2 - 1, gt_tile * 2 - 1).mean())
                    else:
                        row[f"lpips_{k}"] = float("nan")
                rows.append(row)
        print(f"[batch {bi}] {len(rows)} target views  "
              + "  ".join(f"{k}={nanmean([r[f'psnr_{k}'] for r in rows]):.2f}" for k in VARIANTS))

    # ------------------------------- report --------------------------------
    print("\n" + "=" * 78)
    print(f"WARP DECOMPOSITION -- {len(rows)} target views, {len(view_rows)} source views")
    print("=" * 78)
    print(f"{'variant':<16}{'coverage':>10}{'psnr_valid':>12}{'lpips_filled':>14}{'d_psnr':>10}")
    base = nanmean([r["psnr_raw"] for r in rows])
    for k in VARIANTS:
        p = nanmean([r[f"psnr_{k}"] for r in rows])
        print(f"{k:<16}{nanmean([r[f'cov_{k}'] for r in rows]):>10.4f}{p:>12.2f}"
              f"{nanmean([r[f'lpips_{k}'] for r in rows]):>14.4f}{p - base:>+10.2f}")

    p = {k: nanmean([r[f"psnr_{k}"] for r in rows]) for k in VARIANTS}
    total = p["gt"] - p["raw"]
    print(f"\nshare of the {total:.2f} dB oracle gap:")
    for label, lo, hi in (("global scale/shift (raw->affine_pv)", "raw", "affine_pv"),
                          ("  of which 2 DOF shared (raw->affine_shared)", "raw", "affine_shared"),
                          ("missing GT mask (affine_pv->affine_gtmask)", "affine_pv", "affine_gtmask"),
                          ("residual structure (affine_gtmask->gt)", "affine_gtmask", "gt")):
        d = p[hi] - p[lo]
        pct = 100.0 * d / total if abs(total) > 1e-6 else float("nan")
        print(f"  {label:<48}{d:>+8.2f} dB  ({pct:>5.1f}%)")

    print("\nfitted alignment over source views "
          "(a=1, b=0 would mean the raw prediction needs no alignment):")
    for key in ("affine_a", "affine_b", "shift_b", "raw_l1", "spread", "l1_over_spread"):
        vals = np.array([r[key] for r in view_rows], dtype=float)
        print(f"  {key:<16} mean={vals.mean():+.4f}  median={np.median(vals):+.4f}  "
              f"std={vals.std():.4f}  p10={np.percentile(vals,10):+.4f}  "
              f"p90={np.percentile(vals,90):+.4f}")
    frac = float((np.array([r["l1_over_spread"] for r in view_rows]) > 0.5).mean())
    print(f"  source views whose raw L1 exceeds HALF their own GT disparity "
          f"spread: {frac*100:.1f}%")

    out = args.out or (f"tools/src_disp_scale_{os.path.basename(args.run)}"
                       f"_step{args.step}.json")
    with open(out, "w") as f:
        json.dump({"targets": rows, "views": view_rows}, f)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
