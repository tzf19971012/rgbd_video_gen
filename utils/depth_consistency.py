"""Differentiable cross-view consistency of a predicted disparity field.

FACTOR f-consist.  report_20260916_pass1_prior_diag.md §10 measured that pass-1's
predicted SOURCE disparity is multi-view scale-INCONSISTENT -- the six source
views each want their own scale, spread 8.7% (544x the within-view fit noise),
worst/best ratio up to 1.87 -- and §10b showed that this inconsistency, and not
the error magnitude, is what predicts pass-2 degradation: controlling for
`scale_spread` collapses `src_disp_l1_gt`'s +0.274 correlation to -0.036, while
the reverse survives (+0.420 -> +0.333).  §11c(3) then showed the architecture
CAN propagate depth across views (railing only the target depth improved source
`src_disp_l1` 0.0588 -> 0.0530), so the pathway a consistency loss would use
exists.  Nothing in the objective asks for it: the depth loss is an independent
per-frame MSE (train_with_accelerate_rgbd_nvs.py:665).

This module is the training-time counterpart of block 1b in
tools/diag_src_depth_scale_consistency.py, and the numbers it reports are
deliberately the SAME statistics that diagnostic reports, so the loss can be read
against measured levels rather than against nothing:

    scale_dev  <->  pair_scale_dev        0.1309   mean |a_ij - 1|
    repro      <->  pair_res_after_scale  0.1213   L1 after scale, / own level
    support    <->  support_mean          0.2049   fraction of the frame per pair
    n_pairs    <->  n_pairs               21.0     of 30 ordered pairs

TWO DIFFERENCES FROM THE DIAGNOSTIC, both forced:

  1. The diagnostic FORWARD-splats (utils/warp.py), which is not differentiable
     with respect to depth in the way that matters: the landing pixel goes
     through `col.floor().long()` and an index_put, so the position -- the main
     way a depth error shows itself -- gets no gradient at all.  So this uses the
     BACKWARD direction instead: unproject view i's pixels with view i's own
     predicted depth, project into view j, and `grid_sample` view j's own
     predicted disparity there.  Fully differentiable, same convention (pixel
     centres at +0.5, `_pixel_rays_to_world` is imported rather than copied).

  2. There is no ground truth anywhere.  The reference in view j is view j's OWN
     prediction.  The only external inputs are the camera intrinsics and poses,
     which are already conditioning inputs, plus the dataset's depth validity
     mask -- the same mask the existing depth MSE is computed under.

WHAT THE TWO TERMS SEPARATE.  Fitting one scalar `a_ij` per ordered pair and then
looking at what is left splits "view i's disparity is uniformly mis-scaled
relative to view j's" from "view i's disparity is wrong in a way no single number
can express".  The split is sharper than it may look: a PURE scale error is
absorbed by `a_ij` alone, exactly and even across a real baseline, because scaling
i's depths is a uniform scaling of the scene about i's centre and the induced
sample shift is itself proportional to disparity (verified on a plane in
tools/test_f_sweep_factors.py, repro = 0.0000 at k = 1.5 and baseline 0.4).  What
reaches `repro` is therefore the SPATIALLY VARYING part -- a smooth +-10%
multiplicative field leaves |a-1| at 0.0015 and drives repro to 0.026 -- which is
§10c's scale-independent component and the defect shape a depth network actually
produces.

  * `scale` = (a_ij - 1)^2 -- the dominant defect.  §10a showed the misfit is
    scale-TYPE and not shift-type (allowing a per-view shift barely shrinks the
    gap, 0.0147 -> 0.0137), so this single scalar per pair is where the mass is.
    Squared and not |a-1|: the penalty has to vanish smoothly at a_ij = 1, or it
    keeps pushing on pairs that already agree.
  * `repro` = the residual after removing `a_ij`, at `a_ij.detach()` so the two
    terms cannot trade against each other -- the reprojection term is not allowed
    to buy anything by moving the global scale.

WHAT THIS DOES NOT FIX, stated because the pairwise form invites the opposite
assumption.  §10a's `res_gap` is defined relative to a SHARED scale and is
therefore structurally blind to an error all six views make together.  The
pairwise form is not blind to it -- scaling every disparity by k does not commute
with the camera translations -- but it is only WEAKLY sensitive: the response is
first order in (disparity gradient) x (focal) x (baseline), so on gentle geometry
it nearly cancels.  Measured on a synthetic plane at k = 1.4
(tools/test_f_sweep_factors.py::a_shared_scale_error_registers_only_WEAKLY):

    baseline 0.15 -> |a-1| 0.0076    0.4 -> 0.0206    0.8 -> 0.0385

against 0.20 for the same error applied to ONE view.  So this is a detector of
DISAGREEMENT between views -- which is exactly what §10 measured, spread 8.7% and
worst/best 1.87 -- and the supervised depth MSE remains the only thing holding the
common scale.

THE DEGENERATE SOLUTION, which is why the caller must keep the weight small and
watch delta1/abs_rel: a constant disparity field satisfies both terms exactly
(a_ij = 1, residual 0).  This is a regulariser on top of a supervised MSE, never
a standalone objective.  `flatness()` is provided so the caller can log the thing
that would move first if it started happening.
"""
import torch
import torch.nn.functional as F

from diffsynth.models.wan_video_camera_controller import fp32_geometry
# Imported, not reimplemented: the +0.5 pixel-centre convention and the fp32
# `.float()` placement have to match the splat exactly, or the loss optimises a
# geometry that differs from the one warp_block is built with.
from utils.warp import _pixel_rays_to_world

EPS = 1e-3


def _project_into(world, K, c2w, z_near=1e-3):
    """world [N, HW, 3] -> pixel coords [N, HW, 2] and camera-frame depth [N, HW].

    Mirrors the projection half of `_splat_sources_to_targets`, minus the
    quantisation: no floor(), no z-buffer, because the consumer is `grid_sample`.
    """
    w2c = torch.linalg.inv(c2w.float())
    cam = world @ w2c[:, :3, :3].transpose(1, 2) + w2c[:, :3, 3].unsqueeze(1)
    z = cam[..., 2]
    uvw = cam @ K.float().transpose(1, 2)
    uv = uvw[..., :2] / z.clamp_min(z_near).unsqueeze(-1)
    return uv, z


def flatness(disp, valid):
    """Spatial std of the disparity over valid pixels, per view, meaned.

    The monitor for the degenerate solution named in the module docstring: a
    constant field scores zero on both consistency terms, so this is the quantity
    that collapses first if the weight is too high.  No gradient.
    """
    with torch.no_grad():
        out = []
        for v in range(disp.shape[0]):
            sel = valid[v]
            if int(sel.sum()) < 2:
                continue
            out.append(disp[v][sel].float().std())
        if not out:
            return float("nan")
        return float(torch.stack(out).mean())


def pairwise_consistency(
    disp,
    valid,
    K,
    c2w,
    pairs,
    min_support=0.02,
    ratio_clip=4.0,
    trim_quantile=0.9,
    charbonnier_eps=0.01,
    z_near=1e-3,
):
    """Cross-view scale + reprojection consistency over a list of ordered pairs.

    Args:
        disp:  [S, H, W] float, the predicted disparity under test.  Carries
            gradient; everything else here does not.  Shared-scale convention, so
            depth is exactly 1/disp (see kubric4d_rgbd_nvs_dataset.py).
        valid: [S, H, W] bool, where the depth loss is defined.
        K:     [S, 3, 3] PIXEL intrinsics (not the latent ones).
        c2w:   [S, 4, 4] camera-to-world, translations already in the same
            normalised units as `disp` implies.
        pairs: list of ordered (i, j).  Both directions of a pair are different
            measurements and both are worth having.
        min_support: a pair is DROPPED below this fraction of the frame.  Not a
            tuning knob but scene geometry: BMVS source views are spread widely
            enough that a given ordered pair often barely overlaps (measured on
            GT depth, 6/10/12 of 30 ordered pairs cleared even 256 px), and a
            scale fitted on 0.1% of the pixels in one corner is noise.  The
            diagnostic learned this the expensive way -- its first version, at a
            0.1% floor and unweighted, reported a max |a-1| of 32.8.
        ratio_clip: pixels whose implied/own disparity ratio is outside
            [1/clip, clip] are dropped before the fit.  This is the OCCLUSION
            filter: a point of i's that lands behind j's surface reads as a large
            ratio and would dominate a least-squares fit.  4.0 and not something
            tight like 1.5 on purpose -- the worst per-view scale error measured
            is 1.87x, so a tight clip would throw away exactly the signal.
        trim_quantile: after the scale is removed, the largest (1-q) of the
            residuals are dropped from the reprojection term only.  Applied AFTER
            the fit so a global scale error, which moves every residual together,
            is not trimmed away.
        charbonnier_eps: the residual is `sqrt(r^2+e^2)-e`, i.e. L1-like far from
            zero and smooth at it.  Occlusion boundaries survive the trim in
            small numbers and a squared residual would let them set the gradient.

    Returns:
        None when no pair clears `min_support` (a legitimate outcome -- see the
        note about overlap above), else a dict with differentiable `scale` and
        `repro` and detached monitors `n_pairs`, `support`, `scale_dev`.
    """
    if disp.dim() != 3:
        raise ValueError(f"disp must be [S, H, W], got {tuple(disp.shape)}")
    num_views, height, width = disp.shape
    min_px = max(256, int(min_support * height * width))

    # Every matmul below is autocast-eligible and would silently run in bf16;
    # `.float()` on the operands does not prevent that, which is the defect
    # fp32_geometry exists for.  This is a loss and not a model input, so unlike
    # the `camera_geometry_fp32` sweep factor there is no reason to leave it off.
    with fp32_geometry(disp.device, True):
        depth = 1.0 / disp.float().clamp_min(EPS)
        scales, repros, weights, devs = [], [], [], []
        for i, j in pairs:
            world = _pixel_rays_to_world(
                depth[i:i + 1], valid[i:i + 1], K[i:i + 1], c2w[i:i + 1]
            )                                                    # [1, HW, 3]
            uv, z = _project_into(world, K[j:j + 1], c2w[j:j + 1], z_near=z_near)

            # `uv` already carries the +0.5 pixel-centre offset, which is exactly
            # what align_corners=False expects, so the mapping is this and not the
            # (W-1) variant.
            gx = 2.0 * uv[0, :, 0] / width - 1.0
            gy = 2.0 * uv[0, :, 1] / height - 1.0
            grid = torch.stack([gx, gy], dim=-1).reshape(1, height, width, 2)
            # Sampled together so the mask is interpolated by the same kernel as
            # the values: a bilinear mask below 1 means one of the four taps was
            # invalid, and the tap already contaminated the sampled disparity.
            src = torch.stack(
                [disp[j].float(), valid[j].float()], dim=0
            ).unsqueeze(0)                                       # [1, 2, H, W]
            samp = F.grid_sample(
                src, grid, mode="bilinear", padding_mode="zeros", align_corners=False
            )
            own = samp[0, 0].reshape(-1)          # view j's own prediction
            own_valid = samp[0, 1].reshape(-1)
            implied = 1.0 / z[0].clamp_min(z_near)  # what view i says j should see

            keep = (
                valid[i].reshape(-1)
                & (z[0] > z_near)
                & (gx.abs() < 1.0) & (gy.abs() < 1.0)
                & (own_valid > 0.999)
                & (own > EPS)
            )
            with torch.no_grad():
                ratio = implied / own.clamp_min(EPS)
                keep = keep & (ratio > 1.0 / ratio_clip) & (ratio < ratio_clip)
            n_keep = int(keep.sum())
            if n_keep < min_px:
                continue

            p, g = own[keep], implied[keep]
            # Least squares in the same form as the diagnostic's fit_scale:
            # a = <p,g>/<p,p>, scale-only because the GT convention is scale-only.
            # Closed form, so the gradient of `a` reaches every kept pixel -- a
            # median would send it to one.
            a = (p * g).sum() / (p * p).sum().clamp_min(1e-12)
            scales.append((a - 1.0) ** 2)

            # a.detach(): the reprojection term measures what a scale CANNOT
            # explain, so it must not be reducible by moving the scale.
            resid = (a.detach() * p - g).abs()
            with torch.no_grad():
                thr = torch.quantile(resid, trim_quantile)
            trimmed = resid[resid <= thr]
            charb = (
                torch.sqrt(trimmed * trimmed + charbonnier_eps ** 2) - charbonnier_eps
            ).mean()
            # Normalised by the reference's own level, exactly as the diagnostic's
            # `pair_res_after_scale` is, so samples with different disparity
            # magnitudes contribute comparably and the logged number can be read
            # against the measured 0.1213.
            repros.append(charb / g.detach().mean().clamp_min(1e-6))

            weights.append(float(n_keep))
            devs.append(float((a - 1.0).abs().detach()))

        if not scales:
            return None

        total = sum(weights)
        w = [x / total for x in weights]
        return {
            "scale": sum(s * wi for s, wi in zip(scales, w)),
            "repro": sum(r * wi for r, wi in zip(repros, w)),
            "n_pairs": float(len(scales)),
            "support": total / len(scales) / (height * width),
            "scale_dev": sum(d * wi for d, wi in zip(devs, w)),
        }


def ordered_pairs(views):
    """Every ordered pair of distinct entries, i.e. both directions of each pair."""
    return [(i, j) for i in views for j in views if i != j]
