"""Can the SOURCE disparity's scale be fixed at INFERENCE time, GT-free?

report_20260916_pass1_prior_diag.md §15 split the 0.1402 LPIPS that oracle source
depth is worth (§14) into its parts, and the answer moved the target:

    warp_oscale  -0.0690  49.2%   per-view scale fixed to GT, model's shape
    warp_orel    -0.0101   7.2%   only the BETWEEN-view part fixed
    warp_oshape  -0.0093   6.6%   GT shape, model's scale error kept

So 42% of the gap is the GLOBAL scale of the source disparity -- a single scalar
per sample, measured at |geomean(a) - 1| = 0.1193 -- and the remaining per-view
spread is worth only 7.2%.  That reframes the problem: `fcons` (§12/§13) compared
the model's depth in view i against the model's OWN depth in view j, so it asked
for agreement between two things that are both wrong, and agreement does not
imply correctness.  All six views can be consistently wrong.

This script tests the other anchor.  A PHOTOMETRIC objective compares view i's
reprojection against view j's RGB IMAGE, which is a conditioning input and is
ground truth.  Because depth and the pose translations share one unit
(`disp = (1/d)/s` with `c2w[:3,3] *= s`) and the poses are correct, scaling every
disparity by k does not commute with the camera translations: the reprojection
lands somewhere else and the photometric error rises.  That is ordinary
triangulation against a known baseline, and it is exactly the absolute anchor
`fcons` lacks.  utils/depth_consistency.py:63-77 already measured the same
sensitivity for the depth-vs-depth form and found it only WEAK (|a-1| 0.0076 /
0.0206 / 0.0385 at baselines 0.15 / 0.4 / 0.8 against 0.20 for a one-view error)
-- but those baselines are far below BMVS's actual 3-4 source spacing.

TWO BLOCKS, because block 1 is the gate on block 2 and both need the same pass-1
readout:

  1. The pairwise COVERAGE matrix, GT depth beside pass-1 predicted depth.  §10c
     left this open and called it a one-minute job.  It decides two things:
       * the overlap/baseline tension.  Global-scale sensitivity grows with
         baseline, but §10c measured that with GT depth only 6/12/12 of 30
         ordered pairs clear a 2%-of-frame support floor, because the cameras sit
         3-4 apart.  If support and baseline are strongly ANTI-correlated there
         may be no usable middle band and block 2 cannot work.
       * §10c's unresolved confound.  Predicted depth gave n_pairs 21.0 against
         GT's 6-12, and the "GT holes are masked out" explanation was falsified
         (validity 0.99-1.00, filling holes changed nothing).  The surviving
         explanation is that wrong depth puts points inside a neighbour's frustum
         where the real points are not.  For a diagnostic feature that is a
         confound; for an OPTIMISED objective it is an exploit, so block 2 has to
         report whether its minimum sits where the support collapses.

  2. The GT-free photometric scale fit itself, as a 1-DOF global scan and then a
     6-DOF per-view refinement.  The decisive question is not "what does the
     optimiser return" but "IS THE OBJECTIVE INFORMATIVE" -- does it prefer the
     GT-optimal scale over k=1, and does a 12% error (the measured level) produce
     a rise that clears the objective's own noise?  Both are reported per sample,
     not just in aggregate, because a mean can hide a coin flip.

No LPIPS here and no sampling beyond pass 1: this is the cheap gate that runs
BEFORE the ~2h `warp_mvs` arm (§16d), and it can kill the idea on its own.

    sbatch --export=ALL,RUN=logs/bmvs_d_drocefp32_s42,STEP=3750 \
           sbatch/diag_mvs_scale_fit.sbatch

The decoded source disparity is cached, so block 2 can be re-run without a model:

    python -m tools.diag_mvs_scale_fit --from-cache tools/mvs_scale_fit_.../cache
"""
import argparse
import json
import math
import os
import statistics
import time

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from utils.warp import _pixel_rays_to_world

EPS = 1e-3
# A pair is dropped below this fraction of the frame.  Not a knob: it is
# utils/depth_consistency.py's `min_support`, and §10c learned the expensive way
# that a scale fitted on 0.1% of the pixels in one corner is noise.
MIN_SUPPORT = 0.02


# --------------------------------------------------------------------------
# geometry.  Deliberately the same readout as utils/depth_consistency.py's
# `_project_into` + `_pixel_rays_to_world`, imported rather than copied where
# possible, so this measures the geometry the loss and the warp actually use.
# --------------------------------------------------------------------------
def project_into(world, K, c2w, z_near=1e-3):
    """world [N, HW, 3] -> pixel uv [N, HW, 2] and camera-frame depth [N, HW]."""
    w2c = torch.linalg.inv(c2w.float())
    cam = world @ w2c[:, :3, :3].transpose(1, 2) + w2c[:, :3, 3].unsqueeze(1)
    z = cam[..., 2]
    uvw = cam @ K.float().transpose(1, 2)
    uv = uvw[..., :2] / z.clamp_min(z_near).unsqueeze(-1)
    return uv, z


def uv_to_grid(uv, height, width):
    """Pixel uv (already +0.5 centred) -> grid_sample coords for align_corners=False."""
    gx = 2.0 * uv[..., 0] / width - 1.0
    gy = 2.0 * uv[..., 1] / height - 1.0
    return gx, gy


def reproject(disp_i, valid_i, K_i, c2w_i, K_j, c2w_j, k_i=1.0):
    """View i's pixels, scaled by k_i, projected into view j.

    Returns (gx, gy, z, inside) each [HW].  `k_i` scales DISPARITY, so depth
    scales by 1/k_i -- the same parameterisation the warp arms use
    (`warp_oscale` feeds `disp * a`).
    """
    height, width = disp_i.shape
    depth = 1.0 / (disp_i * k_i).clamp_min(EPS)
    world = _pixel_rays_to_world(
        depth[None], valid_i[None], K_i[None], c2w_i[None]
    )
    uv, z = project_into(world, K_j[None], c2w_j[None])
    gx, gy = uv_to_grid(uv[0], height, width)
    inside = (gx.abs() < 1.0) & (gy.abs() < 1.0) & (z[0] > 1e-3)
    return gx, gy, z[0], inside


def sample_at(img, gx, gy, height, width):
    """img [C, H, W] sampled at flat grid coords [N, HW] -> [N, C, HW]."""
    n = gx.shape[0]
    grid = torch.stack([gx, gy], dim=-1).reshape(n, height, width, 2)
    out = F.grid_sample(
        img[None].expand(n, -1, -1, -1), grid,
        mode="bilinear", padding_mode="zeros", align_corners=False,
    )
    return out.reshape(n, img.shape[0], -1)


# --------------------------------------------------------------------------
# block 1: the coverage matrix
# --------------------------------------------------------------------------
@torch.no_grad()
def coverage(disp, valid, K, c2w):
    """Support fraction of every ordered pair, plus the pair baselines.

    support[i, j] = fraction of the FRAME that view i's valid pixels occupy
    inside view j, counting only where view j is itself valid.  This is the
    quantity utils/depth_consistency.py's `min_support` gates on, so the numbers
    here are directly the `n_pairs` that loss would see.
    """
    num, height, width = disp.shape
    sup = torch.zeros((num, num))
    base = torch.zeros((num, num))
    for i in range(num):
        for j in range(num):
            if i == j:
                continue
            base[i, j] = float((c2w[i, :3, 3] - c2w[j, :3, 3]).norm())
            gx, gy, _, inside = reproject(
                disp[i], valid[i], K[i], c2w[i], K[j], c2w[j])
            vj = sample_at(valid[j][None].float(), gx[None], gy[None],
                           height, width)[0, 0]
            keep = valid[i].reshape(-1) & inside & (vj > 0.999)
            sup[i, j] = float(keep.sum()) / (height * width)
    return sup, base


def wide_pairs(base, sup, pairs, floor=MIN_SUPPORT):
    """Pairs that clear the support floor AND have above-median baseline.

    report §16e's "usable middle band": global-scale sensitivity grows with
    baseline, but this dataset's cameras sit close enough together that most
    high-overlap pairs are narrow-baseline and barely see scale at all.  Both
    `base` (poses) and `sup` (support computed from PREDICTED disparity, as in
    block 1) are GT-free, so this filter is itself deployable, not a diagnostic
    luxury.  Falls back to the un-filtered list if fewer than 3 pairs survive
    (an empty fit is worse than a slightly-narrow one).
    """
    vals = [base[i, j] for i, j in pairs]
    if not vals:
        return list(pairs)
    med = statistics.median(vals)
    wide = [(i, j) for i, j in pairs
            if sup[i, j] >= floor and base[i, j] >= med]
    return wide if len(wide) >= 3 else list(pairs)


# --------------------------------------------------------------------------
# block 2: the photometric scale fit
# --------------------------------------------------------------------------
def keep_intersection(disp, valid, K, c2w, pairs, grid):
    """Pixels that stay usable across the WHOLE scan, per pair.

    The control for the exploit named in the module docstring.  The free keep
    mask shrinks monotonically with k -- the smoke run measured n_keep 2.06M at
    k=0.5 against 0.38M at k=2.0 -- so a curve computed on the free mask compares
    a different pixel set at every grid point, and the minimum could in principle
    be bought by changing WHICH pixels are scored rather than how well they
    align.  Intersecting over the grid removes that degree of freedom entirely,
    at the cost of a smaller and more central pixel set.
    """
    out = {}
    for idx, (i, j) in enumerate(pairs):
        acc = None
        for k in grid:
            gx, gy, _, inside = reproject(
                disp[i], valid[i], K[i], c2w[i], K[j], c2w[j], k_i=k)
            vj = sample_at(valid[j][None].float(), gx[None], gy[None],
                           *disp.shape[-2:])[0, 0]
            m = valid[i].reshape(-1) & inside & (vj > 0.999)
            acc = m if acc is None else (acc & m)
        out[idx] = acc
    return out


def photo_error(rgb, disp, valid, K, c2w, pairs, log_k, trim=0.9, max_px=20000,
                fixed_keep=None):
    """Trimmed pairwise photometric error under per-view disparity scales exp(log_k).

    Differentiable in `log_k`.  Returns (error, n_keep_total, n_pairs_used).

    The reference is view j's RGB IMAGE -- a conditioning input, i.e. ground
    truth -- which is the whole point: unlike utils/depth_consistency.py this
    does not ask two wrong predictions to agree with each other.
    """
    num, height, width = disp.shape
    min_px = max(256, int(MIN_SUPPORT * height * width))
    errs, weights, keeps = [], [], []
    for pi, (i, j) in enumerate(pairs):
        gx, gy, _, inside = reproject(
            disp[i], valid[i], K[i], c2w[i], K[j], c2w[j],
            k_i=log_k[i].exp())
        samp = sample_at(torch.cat([rgb[j], valid[j][None].float()], dim=0),
                         gx[None], gy[None], height, width)[0]
        vj = samp[3]
        if fixed_keep is not None:
            # The k-INDEPENDENT pixel set.  `inside`/`vj` are deliberately not
            # re-applied: the point of the control is that the scored set does
            # not move with k.
            keep = fixed_keep[pi]
        else:
            keep = valid[i].reshape(-1) & inside & (vj > 0.999)
        n_keep = int(keep.sum())
        if n_keep < min_px:
            continue
        idx = keep.nonzero(as_tuple=True)[0]
        if idx.numel() > max_px:
            # Strided, not random.  The kept set already shifts with `log_k`
            # (that is intrinsic), so the subsample must not add noise of its
            # own on top -- a re-drawn random subset would make the scan curve
            # jitter between grid points and hide exactly the shallow minimum
            # this script exists to detect.  A stride over the raster-ordered
            # kept indices is uniform over that set and deterministic.
            idx = idx[:: max(1, idx.numel() // max_px)][:max_px]
        tgt = samp[:3, idx]
        own = rgb[i].reshape(3, -1)[:, idx]
        resid = (tgt - own).abs().mean(dim=0)
        with torch.no_grad():
            thr = torch.quantile(resid.float(), trim)
        # Occlusion survives any mask in small numbers; trimming the worst decile
        # is utils/depth_consistency.py's own treatment of the same problem.
        errs.append(resid[resid <= thr].mean())
        weights.append(float(n_keep))
        keeps.append(n_keep)
    if not errs:
        return None, 0, 0
    total = sum(weights)
    err = sum(e * (w / total) for e, w in zip(errs, weights))
    return err, sum(keeps), len(errs)


def _wbox_mean(img, weight, win, ones_c=None, ones_1=None):
    """Weighted local mean via a box filter.  `img` [C,H,W], `weight` [H,W].

    Plain (unweighted) box-filtering the warped image is wrong here: outside
    each pair's overlap, `grid_sample(..., padding_mode="zeros")` returns
    exact zeros, and an unweighted box mean blends those zeros into every
    patch that touches the boundary -- which is worst exactly on the
    high-baseline pairs `wide_pairs` selects, since low overlap means more of
    the frame IS boundary.  Weighting by validity and dividing by the summed
    weight (not win*win) removes that contamination.
    """
    c = img.shape[0]
    if ones_c is None:
        ones_c = img.new_ones(c, 1, win, win)
    if ones_1 is None:
        ones_1 = img.new_ones(1, 1, win, win)
    wsum = F.conv2d(weight[None, None], ones_1, padding=win // 2)[0, 0]
    num = F.conv2d((img * weight[None])[None], ones_c, padding=win // 2,
                   groups=c)[0]
    return num / wsum.clamp_min(1e-6), wsum


def ncc_error(rgb, disp, valid, K, c2w, pairs, log_k, trim=0.9, max_px=20000,
             fixed_keep=None, win=5, eps=1e-3, min_win_frac=0.5):
    """Trimmed pairwise 1-minus-ZNCC error over `win`x`win` patches.

    §16g flagged the per-pixel abs-diff objective as "the weakest kind of
    photometric term -- no patch, no NCC/census, zero robustness to lighting".
    This is the patch-NCC alternative: local zero-mean normalised cross
    correlation between own RGB and the reprojected RGB, weighted-box-filtered
    per channel (see `_wbox_mean`) then averaged.  Same calling convention as
    `photo_error` (and the same trimmed, support-weighted aggregation) so it
    drops into `scan_global` / `fit_per_view` via `error_fn=`.  Differentiable
    in `log_k`.  `min_win_frac` additionally drops pixels whose window has
    less than that fraction of `win*win` actually supported -- a low-support
    NCC is not just noisy, it is the same boundary-contamination risk this
    function exists to remove, just partial instead of total.
    """
    num, height, width = disp.shape
    min_px = max(256, int(MIN_SUPPORT * height * width))
    min_wsum = min_win_frac * win * win
    errs, weights, keeps = [], [], []
    for pi, (i, j) in enumerate(pairs):
        gx, gy, _, inside = reproject(
            disp[i], valid[i], K[i], c2w[i], K[j], c2w[j],
            k_i=log_k[i].exp())
        grid = torch.stack([gx, gy], dim=-1).reshape(1, height, width, 2)
        warped = F.grid_sample(rgb[j][None], grid, mode="bilinear",
                               padding_mode="zeros", align_corners=False)[0]
        vj_map = F.grid_sample(valid[j][None, None].float(), grid,
                               mode="bilinear", padding_mode="zeros",
                               align_corners=False)[0, 0]
        own = rgb[i]
        w = valid[i].float() * (vj_map > 0.999).float()
        om, wsum = _wbox_mean(own, w, win)
        wm, _ = _wbox_mean(warped, w, win)
        oc, wc = own - om, warped - wm
        cov, _ = _wbox_mean(oc * wc, w, win)
        vo, _ = _wbox_mean(oc * oc, w, win)
        vw, _ = _wbox_mean(wc * wc, w, win)
        ncc = cov / (vo * vw).clamp_min(eps).sqrt()
        resid_map = (1.0 - ncc.mean(dim=0)) * 0.5
        if fixed_keep is not None:
            keep = fixed_keep[pi]
        else:
            keep = (valid[i] & inside.reshape(height, width)
                    & (vj_map > 0.999) & (wsum >= min_wsum)).reshape(-1)
        n_keep = int(keep.sum())
        if n_keep < min_px:
            continue
        idx = keep.nonzero(as_tuple=True)[0]
        if idx.numel() > max_px:
            idx = idx[:: max(1, idx.numel() // max_px)][:max_px]
        resid = resid_map.reshape(-1)[idx]
        with torch.no_grad():
            thr = torch.quantile(resid.float(), trim)
        errs.append(resid[resid <= thr].mean())
        weights.append(float(n_keep))
        keeps.append(n_keep)
    if not errs:
        return None, 0, 0
    total = sum(weights)
    err = sum(e * (w / total) for e, w in zip(errs, weights))
    return err, sum(keeps), len(errs)


@torch.no_grad()
def scan_global(rgb, disp, valid, K, c2w, pairs, grid, fixed_keep=None,
                error_fn=photo_error):
    """1-DOF scan: one shared scale for all views.  Returns (errs, n_keeps)."""
    num = disp.shape[0]
    errs, nks = [], []
    for k in grid:
        log_k = torch.full((num,), math.log(k), device=disp.device)
        e, nk, _ = error_fn(rgb, disp, valid, K, c2w, pairs, log_k,
                            fixed_keep=fixed_keep)
        errs.append(float("nan") if e is None else float(e))
        nks.append(nk)
    return errs, nks


def fit_per_view(rgb, disp, valid, K, c2w, pairs, log_k0, iters=120, lr=0.02,
                 error_fn=photo_error):
    """6-DOF refinement from `log_k0` by Adam on the same objective."""
    log_k = log_k0.clone().requires_grad_(True)
    opt = torch.optim.Adam([log_k], lr=lr)
    last = float("nan")
    for _ in range(iters):
        opt.zero_grad()
        e, _, npair = error_fn(rgb, disp, valid, K, c2w, pairs, log_k)
        if e is None or not e.requires_grad:
            break
        e.backward()
        opt.step()
        last = float(e.detach())
    return log_k.detach(), last


# --------------------------------------------------------------------------
def ordered_pairs(n):
    return [(i, j) for i in range(n) for j in range(n) if i != j]


def per_view_scale(pred, gt, valid):
    """a_v = median(gt_v)/median(pred_v), per view.  §15's definition exactly."""
    num = pred.shape[0]
    a = pred.new_ones((num,))
    for v in range(num):
        m = valid[v]
        if not bool(m.any()):
            continue
        p = pred[v][m].median()
        if float(p) > 1e-8:
            a[v] = gt[v][m].median() / p
    return a


def geomean(x):
    return float(x.clamp_min(1e-8).log().mean().exp())


def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x == x and y == y]
    if len(pairs) < 3:
        return float("nan")
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def summarise(vals, label, indent="   "):
    vals = sorted(v for v in vals if v == v)
    if not vals:
        print(f"{indent}{label:28s} (no data)")
        return
    n = len(vals)
    extra = f"  std {statistics.stdev(vals):.4f}" if n > 1 else ""
    print(f"{indent}{label:28s} mean {statistics.fmean(vals):.4f}  "
          f"median {vals[n // 2]:.4f}  p10 {vals[int(.1 * (n - 1))]:.4f}  "
          f"p90 {vals[int(.9 * (n - 1))]:.4f}  min {vals[0]:.4f} "
          f"max {vals[-1]:.4f}{extra}")


# --------------------------------------------------------------------------
def build_cache(args):
    """Run pass 1 once and cache everything block 1 and 2 need."""
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from safetensors.torch import load_file
    from torch.utils.data import DataLoader, Subset

    from examples.wanvideo.model_training.train_with_accelerate_rgbd_nvs import (
        custom_collate_fn)
    from tools.diag_cross_stream_rails import (
        decode_src_disparity, sample_rails)
    from tools.vis_rgb_warp_pred_depth import build_dataset, build_model

    cfg = OmegaConf.load(os.path.join(args.run, "args.yaml"))
    OmegaConf.resolve(cfg)
    set_seed(0)
    accelerator = Accelerator(mixed_precision=cfg.mixed_precision)
    model = build_model(cfg, accelerator)
    ckpt = os.path.join(args.run, f"checkpoint-step-{args.step}")
    nfiles = len(os.listdir(ckpt))
    if nfiles != 11:
        raise SystemExit(f"{ckpt} has {nfiles} files, expected 11 -- "
                         "write was interrupted")
    missing, unexpected = model.load_state_dict(
        load_file(os.path.join(ckpt, "model.safetensors"), device="cpu"),
        strict=False)
    print(f"loaded {ckpt}: missing={len(missing)} unexpected={len(unexpected)}",
          flush=True)
    model.to(accelerator.device)
    model.eval()
    device = model.pipe.device

    dataset = build_dataset(cfg)
    eval_max_samples = int(cfg.get("eval_max_samples", 2))
    eval_batch_size = max(1, min(eval_max_samples,
                                 int(cfg.get("eval_batch_size", 4))))
    samples_per_scene = cfg.get("samples_per_scene", 1)
    eval_indices = [i * samples_per_scene
                    for i in range(len(dataset.records))][:eval_max_samples]
    loader = DataLoader(Subset(dataset, eval_indices), batch_size=eval_batch_size,
                        shuffle=False, num_workers=0,
                        collate_fn=custom_collate_fn, drop_last=False)

    num_src = model.num_source_views
    num_steps = model.eval_num_inference_steps
    seed = model.eval_sample_seed
    autocast_ctx = torch.autocast(device_type="cuda",
                                  dtype=model.pipe.torch_dtype, enabled=True)
    os.makedirs(args.cache, exist_ok=True)
    t0 = time.time()

    for chunk_idx, batch in enumerate(loader):
        if args.max_chunks is not None and chunk_idx >= args.max_chunks:
            break
        path = os.path.join(args.cache, f"chunk{chunk_idx:03d}.pt")
        if os.path.exists(path) and not args.overwrite:
            print(f"[chunk {chunk_idx}] cached, skip", flush=True)
            continue
        with torch.no_grad(), autocast_ctx:
            model._eval_inputs = True
            try:
                inputs = model.forward_preprocess(batch)
            finally:
                model._eval_inputs = False
            rgb_gt_lat, dep_gt_lat = inputs["rgb_gt"], inputs["depth_gt"]
            # Same single noise draw, same order, as diag_cross_stream_rails so
            # `dep1` here is the SAME pass-1 readout §15's arms were built on.
            g = torch.Generator(device=device)
            g.manual_seed(seed)
            eps_rgb = torch.randn(rgb_gt_lat.shape, device=device,
                                  dtype=rgb_gt_lat.dtype, generator=g)
            eps_dep = torch.randn(dep_gt_lat.shape, device=device,
                                  dtype=dep_gt_lat.dtype, generator=g)
            blind = dict(inputs)
            blind["warp_block"] = torch.zeros_like(inputs["warp_block"])
            if inputs.get("depth_warp_block") is not None:
                blind["depth_warp_block"] = torch.zeros_like(
                    inputs["depth_warp_block"])
            dep1, _ = sample_rails(model, blind, num_steps, eps_rgb, eps_dep)

            sdisp_p1 = decode_src_disparity(model, dep1)
            sdisp_gt = decode_src_disparity(model, dep_gt_lat)
            # The dataset's own disparity, NOT through the VAE: block 1's "GT
            # geometry" column has to be the real geometry, while `sdisp_gt`
            # exists only so `a_v` matches §15's definition bit for bit.
            gt_px = batch["disparity"][:, :num_src, 0].to(device).float()
            blob = {
                "sample_id": batch["sample_id"],
                "sdisp_p1": sdisp_p1.half().cpu(),
                "sdisp_gt": sdisp_gt.half().cpu(),
                "gt_px": gt_px.half().cpu(),
                "valid": (inputs["loss_masks"][:, 0, :num_src, 0] > 0.5).cpu(),
                "rgb": inputs["source_images"].float().half().cpu(),
                "K": inputs["pixel_K"][:, :num_src].float().cpu(),
                "c2w": inputs["c2w"][:, :num_src].float().cpu(),
            }
        torch.save(blob, path)
        print(f"[chunk {chunk_idx}] {time.time() - t0:.0f}s -> {path}  "
              f"disp{tuple(sdisp_p1.shape)} rgb{tuple(blob['rgb'].shape)}",
              flush=True)
    print(f"cache written to {args.cache}", flush=True)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--step", type=int)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--from-cache", action="store_true",
                    help="skip pass 1, analyse an existing cache")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-chunks", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-fit", action="store_true", help="block 1 only")
    args = ap.parse_args()

    if not args.from_cache:
        if not (args.run and args.step):
            raise SystemExit("--run and --step are required unless --from-cache")
        build_cache(args)

    out_dir = args.out_dir or os.path.dirname(args.cache.rstrip("/")) or "."
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(args.device)
    chunks = sorted(f for f in os.listdir(args.cache) if f.endswith(".pt"))
    if not chunks:
        raise SystemExit(f"no cache in {args.cache}")

    # A log grid: a scale error is multiplicative, so the scan has to be too, or
    # it spends most of its points on one side of 1.
    grid = [math.exp(x) for x in
            torch.linspace(math.log(1 / 1.6), math.log(1.6), 33).tolist()]
    rows = []
    t0 = time.time()

    for cf in chunks:
        blob = torch.load(os.path.join(args.cache, cf))
        bsz = blob["sdisp_p1"].shape[0]
        for b in range(bsz):
            p1 = blob["sdisp_p1"][b].to(device).float().clamp(0, 1)
            gtv = blob["sdisp_gt"][b].to(device).float().clamp(0, 1)
            gtp = blob["gt_px"][b].to(device).float().clamp(0, 1)
            val = blob["valid"][b].to(device)
            rgb = blob["rgb"][b].to(device).float()
            K = blob["K"][b].to(device)
            c2w = blob["c2w"][b].to(device)
            num = p1.shape[0]
            pairs = ordered_pairs(num)
            row = {"sample_id": blob["sample_id"][b]}

            # ---- block 1 ----
            sup_gt, base = coverage(gtp, val, K, c2w)
            sup_p1, _ = coverage(p1, val, K, c2w)
            off = ~torch.eye(num, dtype=torch.bool)
            row["base"] = base[off].tolist()
            row["sup_gt"] = sup_gt[off].tolist()
            row["sup_p1"] = sup_p1[off].tolist()
            row["npair_gt"] = int((sup_gt[off] >= MIN_SUPPORT).sum())
            row["npair_p1"] = int((sup_p1[off] >= MIN_SUPPORT).sum())

            # ---- the GT reference for block 2 ----
            a = per_view_scale(p1, gtv, val)
            row["a_view"] = a.tolist()
            row["a_geo"] = geomean(a)
            row["a_cv"] = float(a.std() / a.mean())

            if not args.no_fit:
                # ---- block 2 ----
                errs, nks = scan_global(rgb, p1, val, K, c2w, pairs, grid)
                row["scan"] = errs
                row["scan_nkeep"] = nks
                # THE CONTROL for the exploit.  Same curve, but every grid point
                # scored on the SAME pixels, so a minimum here cannot have been
                # bought by moving the scored set.  If the free curve's argmin
                # survives on the fixed mask, the anchor is real geometry.
                fk = keep_intersection(p1, val, K, c2w, pairs, grid)
                errs_f, nks_f = scan_global(rgb, p1, val, K, c2w, pairs, grid,
                                            fixed_keep=fk)
                row["scan_fixed"] = errs_f
                row["scan_fixed_nkeep"] = nks_f
                fin_f = [(e, k) for e, k in zip(errs_f, grid) if e == e]
                if fin_f:
                    row["k_global_fixed"] = min(fin_f, key=lambda t: t[0])[1]
                fin = [(e, k, n) for e, k, n in zip(errs, grid, nks) if e == e]
                if fin:
                    e_min, k_hat, nk_at = min(fin, key=lambda t: t[0])
                    row["k_global"] = k_hat
                    row["e_min"] = e_min
                    row["nkeep_at_min"] = nk_at
                    # The three numbers that decide whether the anchor exists.
                    # `e_at_1` is no correction; `e_at_gt` is the GT-optimal
                    # global scale.  If e_at_gt > e_at_1 the objective points
                    # the WRONG WAY on this sample and no optimiser can help.
                    for tag, kk in (("e_at_1", 1.0), ("e_at_gt", row["a_geo"])):
                        log_k = torch.full((num,), math.log(max(kk, 1e-6)),
                                           device=device)
                        with torch.no_grad():
                            e, nk, _ = photo_error(rgb, p1, val, K, c2w,
                                                   pairs, log_k)
                            ef, _, _ = photo_error(rgb, p1, val, K, c2w,
                                                   pairs, log_k, fixed_keep=fk)
                        row[tag] = float("nan") if e is None else float(e)
                        row[tag + "_nkeep"] = nk
                        row[tag + "_fixed"] = (
                            float("nan") if ef is None else float(ef))
                    log_k, e_fit = fit_per_view(
                        rgb, p1, val, K, c2w, pairs,
                        torch.full((num,), math.log(k_hat), device=device))
                    kv = log_k.exp()
                    row["k_view"] = kv.tolist()
                    row["e_fit"] = e_fit
                    # Residual error AFTER applying the fit: a/k is what is left.
                    row["resid_geo_global"] = geomean(a / k_hat)
                    row["resid_geo_view"] = geomean(a / kv)
                    row["resid_cv_view"] = float((a / kv).std() / (a / kv).mean())

                # ---- block 2c: patch-NCC on the high-baseline/high-support
                # subset only.  §16g's open question -- is 24-33% the ceiling
                # of the anchor, or the floor of the weakest photometric term?
                # Both `wide_pairs`' inputs (base from poses, sup_p1 from
                # predicted disparity) are GT-free, so this stays deployable.
                pairs_wide = wide_pairs(base, sup_p1, pairs)
                row["npair_wide"] = len(pairs_wide)
                errs_n, nks_n = scan_global(rgb, p1, val, K, c2w, pairs_wide,
                                            grid, error_fn=ncc_error)
                fin_n = [(e, k) for e, k in zip(errs_n, grid) if e == e]
                if fin_n:
                    e_min_n, k_hat_n = min(fin_n, key=lambda t: t[0])
                    row["k_global_ncc"] = k_hat_n
                    row["resid_geo_global_ncc"] = geomean(a / k_hat_n)
                    for tag, kk in (("e_at_1_ncc", 1.0),
                                    ("e_at_gt_ncc", row["a_geo"])):
                        log_k = torch.full((num,), math.log(max(kk, 1e-6)),
                                           device=device)
                        with torch.no_grad():
                            e, _, _ = ncc_error(rgb, p1, val, K, c2w,
                                                pairs_wide, log_k)
                        row[tag] = float("nan") if e is None else float(e)
            rows.append(row)
            print(f"  [{len(rows):3d}] {row['sample_id']}  "
                  f"npair gt={row['npair_gt']} p1={row['npair_p1']}  "
                  f"a_geo={row['a_geo']:.4f}"
                  + (f"  k_hat={row.get('k_global', float('nan')):.4f}"
                     if not args.no_fit else "")
                  + f"  {time.time() - t0:.0f}s", flush=True)

    path = os.path.join(out_dir, "rows.jsonl")
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    report(rows, grid, args.no_fit)
    print(f"\nper-sample rows -> {path}")


def report(rows, grid, no_fit):
    n = len(rows)
    print(f"\n=== {n} samples ===")

    print("\n--- BLOCK 1: pairwise coverage (§10c's open question) ---")
    print("support = fraction of the FRAME view i occupies inside view j, "
          f"counting only where j is valid.  floor = {MIN_SUPPORT:.0%}")
    summarise([r["npair_gt"] for r in rows], "n_pairs >= floor, GT depth")
    summarise([r["npair_p1"] for r in rows], "n_pairs >= floor, p1 depth")
    summarise([statistics.fmean(r["sup_gt"]) for r in rows], "mean support, GT")
    summarise([statistics.fmean(r["sup_p1"]) for r in rows], "mean support, p1")
    summarise([r["npair_p1"] - r["npair_gt"] for r in rows], "n_pairs p1 - GT")
    print("   §10c measured n_pairs 6/12/12 on GT (3 samples, CPU) and 21.0 mean")
    print("   on p1.  If p1 - GT is reliably POSITIVE, wrong depth is inventing")
    print("   overlap -- a confound for a feature, an EXPLOIT for an objective.")

    # RISK 1: does support fight baseline?  Pooled over every ordered pair.
    allb = [b for r in rows for b in r["base"]]
    allg = [s for r in rows for s in r["sup_gt"]]
    allp = [s for r in rows for s in r["sup_p1"]]
    print(f"\n   pooled over {len(allb)} ordered pairs:")
    print(f"     pearson(baseline, support_GT) = {pearson(allb, allg):+.3f}")
    print(f"     pearson(baseline, support_p1) = {pearson(allb, allp):+.3f}")
    summarise(allb, "baseline |t_i - t_j|", indent="     ")
    # The middle band: pairs that BOTH clear the floor and have a real baseline.
    med_b = statistics.median(allb)
    both = [(b, s) for b, s in zip(allb, allg) if s >= MIN_SUPPORT]
    wide = [(b, s) for b, s in both if b >= med_b]
    print(f"     GT pairs clearing the floor: {len(both)}/{len(allb)} "
          f"({100 * len(both) / max(len(allb), 1):.1f}%)")
    print(f"     ... of those, with above-median baseline: {len(wide)} "
          f"({100 * len(wide) / max(len(both), 1):.1f}%)")
    print("   RISK 1 reads here: global-scale sensitivity grows with baseline,")
    print("   so the usable band is pairs that clear the floor AND are wide.")

    if no_fit:
        return

    print("\n--- BLOCK 2: is the photometric objective INFORMATIVE about scale? ---")
    # The single most decisive read: does the objective prefer the GT-optimal
    # global scale over no correction at all?
    better = [r for r in rows if r.get("e_at_gt", float("nan")) == r.get("e_at_gt")
              and r["e_at_gt"] < r["e_at_1"]]
    have = [r for r in rows
            if r.get("e_at_gt") == r.get("e_at_gt")
            and r.get("e_at_1") == r.get("e_at_1")]
    print(f"   objective prefers the GT scale over k=1: "
          f"**{len(better)}/{len(have)}** samples "
          f"({100 * len(better) / max(len(have), 1):.1f}%)")
    print("   (50% = a coin flip = the anchor is not there; this is the gate)")
    summarise([100 * (r["e_at_1"] - r["e_at_gt"]) / max(r["e_at_1"], 1e-9)
               for r in have], "% error drop, k=1 -> GT")

    print("\n   the fit itself:")
    summarise([r["k_global"] for r in rows if "k_global" in r], "k_hat (1-DOF scan)")
    summarise([r["a_geo"] for r in rows], "GT geomean(a) to recover")
    print(f"   pearson(log k_hat, log geomean(a)) = "
          f"{pearson([math.log(max(r['k_global'], 1e-6)) for r in rows if 'k_global' in r], [math.log(max(r['a_geo'], 1e-6)) for r in rows if 'k_global' in r]):+.3f}")

    print("\n   DOES IT REDUCE THE ERROR?  §15e measured |geomean(a)-1| = 0.1193.")
    summarise([abs(r["a_geo"] - 1) for r in rows], "before: |geomean(a)-1|")
    summarise([abs(r["resid_geo_global"] - 1) for r in rows
               if "resid_geo_global" in r], "after 1-DOF: |geomean(a/k)-1|")
    summarise([abs(r["resid_geo_view"] - 1) for r in rows
               if "resid_geo_view" in r], "after 6-DOF: |geomean(a/k_v)-1|")
    print("   §15e measured the cross-view spread cv(a) = 0.0971 as well:")
    summarise([r["a_cv"] for r in rows], "before: cv(a)")
    summarise([r["resid_cv_view"] for r in rows if "resid_cv_view" in r],
              "after 6-DOF: cv(a/k_v)")

    print("\n   RISK 2, the exploit: does the minimum sit where support collapses?")
    summarise([r["nkeep_at_min"] / max(r["e_at_1_nkeep"], 1) for r in rows
               if "nkeep_at_min" in r and r.get("e_at_1_nkeep")],
              "n_keep(k_hat) / n_keep(1)")
    print("   Below ~1 means the optimiser bought its improvement by pushing")
    print("   pixels out of the neighbour's frame rather than by aligning them.")
    print("   But n_keep falls monotonically in k regardless of the fit, so the")
    print("   free curve compares a DIFFERENT pixel set at every grid point.")
    print("   The control below removes that degree of freedom: the keep mask is")
    print("   intersected over the whole grid, so every k is scored on the same")
    print("   pixels.  These are the numbers to believe.")
    fx = [r for r in rows if "k_global_fixed" in r]
    if fx:
        summarise([r["k_global_fixed"] for r in fx], "k_hat, FIXED mask")
        print(f"     pearson(log k_hat_fixed, log geomean(a)) = "
              f"{pearson([math.log(max(r['k_global_fixed'], 1e-6)) for r in fx], [math.log(max(r['a_geo'], 1e-6)) for r in fx]):+.3f}")
        # The control is NOT free: intersecting over the grid can starve a pair
        # below min_support and drop the sample entirely, and the samples it
        # drops are the low-coverage ones -- which are also the ones with the
        # largest scale error, i.e. where the fit matters most.  So comparing
        # the fixed residual against the free residual over ALL samples is a
        # selection effect, not an improvement.  Print the same-subset numbers.
        drop = [r for r in rows if "k_global" in r and "k_global_fixed" not in r]
        print(f"     samples surviving the intersection: {len(fx)}/"
              f"{len([r for r in rows if 'k_global' in r])}"
              f"  (dropped: too little overlap left)")
        summarise([abs(r["a_geo"] / r["k_global_fixed"] - 1) for r in fx],
                  "after 1-DOF FIXED: |geomean(a/k)-1|")
        print("     same-subset comparison, the only fair one:")
        print(f"       before                 "
              f"{statistics.fmean([abs(r['a_geo'] - 1) for r in fx]):.4f}")
        print(f"       free  fit, these {len(fx):3d}   "
              f"{statistics.fmean([abs(r['a_geo'] / r['k_global'] - 1) for r in fx]):.4f}")
        print(f"       fixed fit, these {len(fx):3d}   "
              f"{statistics.fmean([abs(r['a_geo'] / r['k_global_fixed'] - 1) for r in fx]):.4f}")
        if drop:
            print(f"     the {len(drop)} DROPPED sample(s), on the free fit -- check "
                  "whether the")
            print("     control is excluding exactly where the fit does its work:")
            summarise([r["npair_gt"] for r in drop], "npair_gt, dropped",
                      indent="       ")
            summarise([abs(r["a_geo"] - 1) for r in drop],
                      "|geomean(a)-1| before, dropped", indent="       ")
            summarise([abs(r["a_geo"] / r["k_global"] - 1) for r in drop],
                      "|geomean(a/k)-1| after, dropped", indent="       ")
        gf = [r for r in fx
              if r.get("e_at_gt_fixed") == r.get("e_at_gt_fixed")
              and r.get("e_at_1_fixed") == r.get("e_at_1_fixed")]
        nb = [r for r in gf if r["e_at_gt_fixed"] < r["e_at_1_fixed"]]
        print(f"     prefers GT scale over k=1, FIXED mask: "
              f"**{len(nb)}/{len(gf)}** ({100 * len(nb) / max(len(gf), 1):.1f}%)")
        summarise([100 * (r["e_at_1_fixed"] - r["e_at_gt_fixed"])
                   / max(r["e_at_1_fixed"], 1e-9) for r in gf],
                  "% error drop, FIXED mask")
        # How much of the frame is left after intersecting: if this collapses the
        # control is measuring a sliver and proves nothing either way.
        mid = len(grid) // 2
        summarise([r["scan_fixed_nkeep"][mid] / max(r["scan_nkeep"][mid], 1)
                   for r in fx if r.get("scan_fixed_nkeep")
                   and r["scan_nkeep"][mid]],
                  "n_keep fixed / free, at k=1")

    print("\n   the scan curve, meaned over samples (relative to each sample's min):")
    print("     k       free: err/min  n_keep   |  fixed-mask: err/min  n_keep")
    for idx in range(0, len(grid), 4):
        vals = [r["scan"][idx] / max(r["e_min"], 1e-9)
                for r in rows if "scan" in r and r["scan"][idx] == r["scan"][idx]]
        nk = [r["scan_nkeep"][idx] for r in rows if "scan_nkeep" in r]
        if not vals:
            continue
        fv = [r["scan_fixed"][idx] for r in fx
              if r["scan_fixed"][idx] == r["scan_fixed"][idx]
              and min(e for e in r["scan_fixed"] if e == e) > 0]
        fmin = [min(e for e in r["scan_fixed"] if e == e) for r in fx
                if r["scan_fixed"][idx] == r["scan_fixed"][idx]
                and min(e for e in r["scan_fixed"] if e == e) > 0]
        fnk = [r["scan_fixed_nkeep"][idx] for r in fx if r.get("scan_fixed_nkeep")]
        ftxt = "      --"
        if fv:
            ftxt = (f"  {statistics.fmean([a / b for a, b in zip(fv, fmin)]):.4f}"
                    f"  {statistics.fmean(fnk):.0f}")
        print(f"     k={grid[idx]:.3f}    {statistics.fmean(vals):.4f}  "
              f"{statistics.fmean(nk):.0f}   |{ftxt}")
    print("   A curve that is FLAT near k=1 means a 12% scale error is not")
    print("   photometrically visible and the idea is dead regardless of the fit.")
    print("   The free and fixed columns disagreeing in ARGMIN means the free")
    print("   curve was reading support, not geometry.")

    print("\n--- BLOCK 2c: patch-NCC on the high-baseline/high-support pairs only ---")
    print("   §16g's open question: is 24-33% the ceiling of the photometric")
    print("   anchor, or the floor of the weakest possible photometric term")
    print("   (per-pixel abs-diff, every pair)?  Both changes are GT-free.")
    nc = [r for r in rows if "k_global_ncc" in r]
    print(f"   samples with >=3 wide pairs: {len(nc)}/{len([r for r in rows if 'npair_wide' in r])}")
    if nc:
        summarise([r["npair_wide"] for r in rows if "npair_wide" in r],
                  "n_pairs used (wide subset)")
        better_n = [r for r in nc
                    if r.get("e_at_gt_ncc") == r.get("e_at_gt_ncc")
                    and r.get("e_at_1_ncc") == r.get("e_at_1_ncc")
                    and r["e_at_gt_ncc"] < r["e_at_1_ncc"]]
        have_n = [r for r in nc if r.get("e_at_gt_ncc") == r.get("e_at_gt_ncc")
                  and r.get("e_at_1_ncc") == r.get("e_at_1_ncc")]
        print(f"   objective prefers the GT scale over k=1 (NCC, wide pairs): "
              f"**{len(better_n)}/{len(have_n)}** ({100 * len(better_n) / max(len(have_n), 1):.1f}%)")
        summarise([r["k_global_ncc"] for r in nc], "k_hat (NCC, wide pairs)")
        print(f"   pearson(log k_hat_ncc, log geomean(a)) = "
              f"{pearson([math.log(max(r['k_global_ncc'], 1e-6)) for r in nc], [math.log(max(r['a_geo'], 1e-6)) for r in nc]):+.3f}")
        summarise([abs(r["a_geo"] - 1) for r in nc], "before, same subset: |geomean(a)-1|")
        summarise([abs(r["a_geo"] / r["k_global_ncc"] - 1) for r in nc],
                  "after NCC/wide: |geomean(a/k)-1|")
        summarise([abs(r["a_geo"] / r["k_global"] - 1) for r in nc if "k_global" in r],
                  "after L1/all-pairs, same subset (block 2's number)")
        print("   If NCC/wide beats block 2's L1/all-pairs number on this same")
        print("   subset, the weak-photometric-term hypothesis is confirmed and")
        print("   patch NCC is worth carrying into a `warp_mvs` arm.  If it does")
        print("   not, 24-33% was the anchor's ceiling, not the term's floor.")


if __name__ == "__main__":
    main()
