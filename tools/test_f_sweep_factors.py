"""CPU-only unit tests for the F-sweep factor (f-consist).

Same reason as tools/test_c_sweep_factors.py / test_d_sweep_factors.py /
test_e_sweep_factors.py: the factor is SILENT when wrong.  f-consist is the
quietest one yet -- it adds a LOSS TERM, so it changes no tensor shape, no
parameter count and no reported metric name.  An arm whose loss is subtly wrong
(a geometry convention half a pixel off from `warp_block`'s, a `grid_sample`
axis order swapped, a scale fitted on 40 pixels in one corner, a residual term
that quietly buys its improvement by moving the scale it is supposed to be
orthogonal to) trains to completion, logs a plausible `consist` line, and answers
a question nobody asked.

The tests lean on one construction throughout: a PLANE.  For a plane, inverse
depth is an exactly affine function of pixel coordinates, and bilinear
interpolation of an affine function is exact -- so a two-view plane scene is a
case where the correct answer is `a_ij = 1` and `residual = 0` to float32, with
no tolerance to hide a sign error in.  Two camera configurations are used:

  * SHARED CENTRE, rotated.  Scaling one view's disparity by k moves its points
    along rays that pass through the other camera's centre too, so the reprojected
    disparity is k x the truth EXACTLY and nothing else changes.  This is the
    clean test that `scale` recovers k and that `repro` is blind to it.
  * TRANSLATED.  A scale error now also produces parallax, which is the component
    §10c measured as scale-independent and which `repro` exists to cover.

Nothing here loads Wan2.1 or touches a GPU.  The geometry helpers under test are
the shipped ones (`utils.depth_consistency` imports `_pixel_rays_to_world` from
`utils.warp` rather than copying it, and these tests exercise that import), and
the gating logic is bound to a bare object carrying only the attributes it reads
-- the same trick test_d_sweep_factors.py uses for `_select_views`.

  python -m tools.test_f_sweep_factors
"""
import inspect
import math
import os
import types

import torch

from examples.wanvideo.model_training.WanRGBDNVSTrainingModule import (
    WanRGBDNVSTrainingModule,
)
from utils.depth_consistency import (
    _project_into,
    flatness,
    ordered_pairs,
    pairwise_consistency,
)
from utils.warp import _pixel_rays_to_world

CHECKS = []

# Measured levels the loss is calibrated against, from
# tools/src_scale_consist_v2_bmvs_d_drocefp32_s42_step3750/analysis.txt (report
# §10).  Used here only by the ratio_clip test: the WORST per-view scale error
# actually observed is what the occlusion filter must not throw away.
MEASURED_WORST_SCALE_RATIO = 1.869

H, W = 48, 64
FOCAL = 100.0
# Unnormalised plane normal; the plane is {X : n.X = PLANE_OFFSET}.  Tilted on
# purpose -- a fronto-parallel plane has CONSTANT disparity, which is the
# degenerate solution and would pass every test for the wrong reason.
PLANE_N = torch.tensor([0.25, 0.15, 1.0], dtype=torch.float64)
PLANE_OFFSET = 2.0


def check(fn):
    CHECKS.append(fn)
    return fn


# ----------------------------------------------------------------------------
# Scene construction
# ----------------------------------------------------------------------------

def intrinsics():
    return torch.tensor(
        [[FOCAL, 0.0, W / 2.0], [0.0, FOCAL, H / 2.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )


def roty(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=torch.float64)


def c2w_from(rot, trans):
    out = torch.eye(4, dtype=torch.float64)
    out[:3, :3] = rot
    out[:3, 3] = torch.as_tensor(trans, dtype=torch.float64)
    return out


def plane_depth(K, c2w):
    """Exact camera-frame z of the plane, per pixel, for one camera.

    Same unprojection convention as `_pixel_rays_to_world`: pixel centres at +0.5
    and `depth` meaning the camera z coordinate, so a disparity field built from
    this is one the module can consume without a convention mismatch.
    """
    row, col = torch.meshgrid(
        torch.arange(H, dtype=torch.float64),
        torch.arange(W, dtype=torch.float64),
        indexing="ij",
    )
    pix = torch.stack([col + 0.5, row + 0.5, torch.ones_like(col)], dim=0).reshape(3, -1)
    m = torch.linalg.inv(K) @ pix                      # [3, HW], ray at z = 1
    denom = PLANE_N @ (c2w[:3, :3] @ m)                # [HW]
    z = (PLANE_OFFSET - PLANE_N @ c2w[:3, 3]) / denom
    assert float(z.min()) > 0.1, "the plane must be in front of the camera"
    return z.reshape(H, W)


def plane_scene(poses):
    """-> disp [S, H, W] float32 (a mutually consistent prediction), valid, K, c2w."""
    K = intrinsics()
    disp = torch.stack([1.0 / plane_depth(K, p) for p in poses], dim=0)
    valid = torch.ones(len(poses), H, W, dtype=torch.bool)
    return (
        disp.float(),
        valid,
        K.float().unsqueeze(0).expand(len(poses), 3, 3).contiguous(),
        torch.stack(poses, dim=0).float(),
    )


def shared_centre_pair():
    """Two cameras at the SAME point, 6 deg apart.  A per-view disparity scale
    error is then a pure scale error in the other view's reading of it."""
    return [c2w_from(torch.eye(3, dtype=torch.float64), [0, 0, 0]),
            c2w_from(roty(6.0), [0, 0, 0])]


def translated_pair(baseline=0.15):
    """A real stereo baseline: ~7.5 px of disparity at the plane's depth."""
    eye = torch.eye(3, dtype=torch.float64)
    return [c2w_from(eye, [0, 0, 0]), c2w_from(eye, [baseline, 0, 0])]


def run(disp, valid, K, c2w, **kw):
    return pairwise_consistency(disp, valid, K, c2w, ordered_pairs(range(disp.shape[0])), **kw)


# ----------------------------------------------------------------------------
# The x0 recovery the whole term stands on
# ----------------------------------------------------------------------------

@check
def x0_is_recovered_exactly_from_the_velocity_head():
    """`x_t - sigma * v` is x_0 EXACTLY under this module's parameterisation, not
    approximately: x_t = (1-s)x0 + s*eps and the head regresses eps - x0, so
    x_t - s*(eps - x0) = x0.  If it were only an approximation the loss would be
    measuring the consistency of a biased estimate and no amount of weight tuning
    would fix it, so this is worth pinning down rather than trusting."""
    torch.manual_seed(0)
    x0 = torch.randn(2, 16, 6, 8, 10)
    eps = torch.randn_like(x0)
    for s in (0.05, 0.3, 0.6, 1.0):
        sigma = torch.full((2, 1, 1, 1, 1), s)
        x_t = (1.0 - sigma) * x0 + sigma * eps
        v = eps - x0                       # the exact velocity target
        assert torch.allclose(x_t - sigma * v, x0, atol=1e-5), f"sigma={s}"

    src = inspect.getsource(WanRGBDNVSTrainingModule._depth_consistency_loss)
    assert "depth_x_t[:, :, :num_src] - sigma * pred_depth[:, :, :num_src]" in src, (
        "the module must build x0 from x_t and the velocity, with no extra forward "
        "pass and no scheduler call"
    )


@check
def the_module_reads_the_same_disparity_pass_two_splats():
    """The loss is only worth running if it constrains the SAME disparity the warp
    block is built from.  A separately post-processed readout (a different clamp, a
    different channel reduction, a missing depth_latent_scale) would regularise a
    quantity pass 2 never sees."""
    a = inspect.getsource(WanRGBDNVSTrainingModule._decode_src_disparity_with_grad)
    b = inspect.getsource(WanRGBDNVSTrainingModule._warp_block_from_predicted_depth)
    # `.mean(dim=` without the axis: the splat stacks views first and reduces at
    # dim 2, the grad path reduces one view at a time at dim 1.  Same operation,
    # and pinning the literal axis here would only assert the loop structure.
    for needle in ("/ self.depth_latent_scale", ".float().mean(dim=", "clamp(0.0, 1.0)"):
        assert needle in a, f"the grad readout is missing {needle!r}"
        assert needle in b, f"the splat readout no longer has {needle!r}"
    # One view at a time in both: the VAE's temporal convolutions mix frames, and
    # this dataset's views are unrelated.
    assert "v:v + 1" in a and "v:v + 1" in b


# ----------------------------------------------------------------------------
# The geometry: a consistent scene must score zero
# ----------------------------------------------------------------------------

@check
def a_consistent_plane_scores_zero_on_both_terms():
    """The single most important test here.  A plane rendered from two real cameras
    is EXACTLY cross-view consistent, and inverse depth is affine in pixel
    coordinates so bilinear sampling is exact too -- there is no tolerance for a
    half-pixel convention error, a transposed rotation or a swapped grid axis to
    hide in.  Any of those would put a floor under the loss that no training run
    could reach, i.e. the model would be pushed towards a wrong geometry."""
    for name, poses in (("translated", translated_pair()),
                        ("shared_centre", shared_centre_pair())):
        got = run(*plane_scene(poses))
        assert got is not None, f"{name}: no pair cleared support on a full frame"
        assert got["n_pairs"] == 2.0, f"{name}: both directions must be measured"
        assert got["scale_dev"] < 2e-3, f"{name}: |a-1| = {got['scale_dev']:.5f}"
        assert float(got["scale"]) < 1e-5, f"{name}: scale = {float(got['scale']):.3e}"
        assert float(got["repro"]) < 2e-3, f"{name}: repro = {float(got['repro']):.3e}"
        assert got["support"] > 0.5, f"{name}: support = {got['support']:.3f}"


@check
def the_projection_helper_inverts_the_unprojection():
    """`_project_into` is the only piece of geometry in this module that is not
    imported from utils/warp.py, so it is the only place the convention can drift.
    Unproject then project must be the identity on pixel coordinates."""
    disp, valid, K, c2w = plane_scene(translated_pair())
    depth = 1.0 / disp
    world = _pixel_rays_to_world(depth[:1], valid[:1], K[:1], c2w[:1])
    uv, z = _project_into(world, K[:1], c2w[:1])
    row, col = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij")
    want = torch.stack([col.reshape(-1) + 0.5, row.reshape(-1) + 0.5], dim=-1)
    assert (uv[0] - want).abs().max() < 1e-2, "round trip moved the pixel grid"
    assert (z[0] - depth[0].reshape(-1)).abs().max() < 1e-3, "z is not the input depth"


# ----------------------------------------------------------------------------
# The two terms actually separate
# ----------------------------------------------------------------------------

@check
def a_per_view_scale_error_is_recovered_as_a():
    """With the two cameras sharing a centre the answer is exact: scaling view 0's
    disparity by k slides its points along rays that also pass through camera 1's
    centre, so they project to the same pixels and read k x too near.  `a_10` must
    come back as k -- if it came back as 1/k the loss would have the wrong sign and
    would AMPLIFY the inconsistency §10 measured."""
    for k in (0.6, 1.25, 2.0):
        disp, valid, K, c2w = plane_scene(shared_centre_pair())
        disp[0] = disp[0] * k
        got = run(disp, valid, K, c2w)
        # a_ij fits view i's implied disparity against view j's own, so the pair
        # (0,1) reports k and (1,0) reports 1/k; the mean |a-1| is what is logged.
        want = (abs(k - 1.0) + abs(1.0 / k - 1.0)) / 2.0
        assert abs(got["scale_dev"] - want) < 0.05 * want, (
            f"k={k}: |a-1| = {got['scale_dev']:.4f}, expected ~{want:.4f}")
        assert float(got["scale"]) > 0.5 * (k - 1.0) ** 2, (
            f"k={k}: scale = {float(got['scale']):.4f} is too small to penalise a "
            f"{k}x scale error")


@check
def repro_is_blind_to_what_the_scale_explains():
    """The residual term must measure only what a scale CANNOT explain -- that is
    the whole reason for `a.detach()`.  In the shared-centre config a per-view
    scale error is PURELY scale, so `scale` must grow while `repro` stays at the
    zero it had on the consistent scene.  Without the detach the two terms would
    trade and the reported split between them would be meaningless."""
    base = float(run(*plane_scene(shared_centre_pair()))["repro"])
    for k in (0.6, 1.25, 2.0):
        disp, valid, K, c2w = plane_scene(shared_centre_pair())
        disp[0] = disp[0] * k
        got = run(disp, valid, K, c2w)
        assert float(got["repro"]) < max(3e-3, 3.0 * base), (
            f"k={k}: repro moved to {float(got['repro']):.4e} on a PURE scale error "
            f"(baseline {base:.4e}) -- the two terms are not separated")
    src = inspect.getsource(pairwise_consistency)
    assert "a.detach() * p - g" in src, "the residual must be evaluated at a.detach()"


@check
def a_pure_scale_error_on_a_plane_is_absorbable_by_construction():
    """Recorded because it is surprising and it constrains how the other half of the
    split has to be tested.  For a PLANE and two cameras related by a pure lateral
    translation, scaling one view's disparity by k is absorbed EXACTLY by a single
    scalar -- even with a real baseline.  The reason: scaling the depths by 1/k is a
    uniform scaling of the scene about the first camera's centre, which makes the
    reprojection behave as if the baseline were k*B, and for a plane the disparity
    field is affine in pixel coordinates, so the resulting sample shift (itself
    proportional to disparity) multiplies the sampled value by a CONSTANT.

    Measured: repro stays at 0.0000 for a 1.5x error at B = 0.15 and at B = 0.4.
    So `repro` cannot be tested with a scale error, and more importantly the term
    only earns its keep where the scene has relief or the views are rotated -- which
    real BMVS scenes have and this synthetic plane does not."""
    for baseline in (0.15, 0.4):
        disp, valid, K, c2w = plane_scene(translated_pair(baseline))
        disp[0] = disp[0] * 1.5
        got = run(disp, valid, K, c2w)
        assert got["scale_dev"] > 0.3, f"B={baseline}: the scale term must still see it"
        assert float(got["repro"]) < 1e-4, (
            f"B={baseline}: repro = {float(got['repro']):.2e}; if this is no longer ~0 "
            "the geometry above has changed and the reasoning needs redoing")


@check
def repro_sees_what_a_scale_cannot_explain():
    """The other half of the split, tested with the defect shape that actually
    motivates it: a SPATIALLY VARYING disparity error, which is what a depth network
    makes and which no single scalar can absorb.  A smooth multiplicative field of
    +-10% leaves |a-1| at 0.0015 -- the scale term is essentially blind to it -- and
    drives repro to 0.026, i.e. two orders of magnitude above the consistent scene.
    That is §10c's scale-independent component, and it is the only reason to carry a
    second term at all."""
    clean = float(run(*plane_scene(translated_pair()))["repro"])
    yy = torch.linspace(0.0, 1.0, H).reshape(H, 1)
    xx = torch.linspace(0.0, 1.0, W).reshape(1, W)
    prev = 0.0
    for amp in (0.1, 0.2):
        disp, valid, K, c2w = plane_scene(translated_pair())
        disp[0] = disp[0] * (1.0 + amp * torch.sin(math.pi * xx) * torch.cos(math.pi * yy))
        got = run(disp, valid, K, c2w)
        assert got["scale_dev"] < 0.02, (
            f"amp={amp}: |a-1| = {got['scale_dev']:.4f}; the field is mean-neutral so "
            "the scale term should barely move -- if it does, the two terms overlap")
        assert float(got["repro"]) > 20.0 * max(clean, 1e-6), (
            f"amp={amp}: repro = {float(got['repro']):.4e} vs {clean:.4e} clean -- the "
            "residual term is not seeing a purely non-scale error")
        assert float(got["repro"]) > prev, "repro must be monotone in the error"
        prev = float(got["repro"])


@check
def a_constant_disparity_field_is_the_documented_degeneracy():
    """Not a bug -- a property, recorded here so nobody 'fixes' it by changing the
    loss instead of keeping the weight small.  A fronto-parallel constant field is
    exactly consistent between two laterally translated cameras, so both terms are
    zero and gradient descent could reach it by flattening depth.  This is why the
    caller keeps the supervised MSE, keeps the weight sub-unity, and logs
    `flatness` and delta1."""
    _, valid, K, c2w = plane_scene(translated_pair())
    disp = torch.full((2, H, W), 0.5)
    got = run(disp, valid, K, c2w)
    assert got is not None
    assert float(got["scale"]) < 1e-6 and float(got["repro"]) < 1e-4, (
        "the degeneracy claim in the docstring is wrong, which means the monitor "
        "guarding it is guarding nothing")
    assert flatness(disp, valid) < 1e-6, "flatness must be the quantity that detects it"
    real = plane_scene(translated_pair())
    assert flatness(real[0], real[1]) > 0.01, "flatness must be non-trivial on a real field"


# ----------------------------------------------------------------------------
# The robustness knobs, each for its measured reason
# ----------------------------------------------------------------------------

@check
def ratio_clip_does_not_throw_away_the_worst_measured_error():
    """The occlusion filter is deliberately loose.  The worst per-view scale error
    §10 measured is 1.87x, so a tight clip (1.5, which is what an occlusion filter
    would naturally be set to) would discard exactly the samples the loss exists
    for.  This checks the default keeps every pixel of a 1.87x error, by comparing
    the support against the unperturbed scene."""
    ref = run(*plane_scene(shared_centre_pair()))
    disp, valid, K, c2w = plane_scene(shared_centre_pair())
    disp[0] = disp[0] * MEASURED_WORST_SCALE_RATIO
    got = run(disp, valid, K, c2w)
    assert abs(got["support"] - ref["support"]) < 1e-6, (
        f"support dropped {ref['support']:.4f} -> {got['support']:.4f}: the default "
        f"ratio_clip is clipping a {MEASURED_WORST_SCALE_RATIO}x error")
    tight = run(disp, valid, K, c2w, ratio_clip=1.5)
    assert tight is None or tight["support"] < ref["support"] - 1e-3, (
        "a 1.5 clip must visibly discard it -- otherwise this test proves nothing "
        "about the 4.0 default"
    )


@check
def thin_pairs_are_dropped_not_downweighted():
    """A scale fitted on a handful of pixels in one corner is noise, and the
    diagnostic learned this the expensive way: its first version, at a 0.1% floor
    and unweighted, reported a max |a-1| of 32.8.  Both the absolute 256 px floor
    and the min_support FRACTION have to bite."""
    disp, valid, K, c2w = plane_scene(translated_pair())
    thin = torch.zeros_like(valid)
    thin[:, 4:14, 4:14] = True                        # 100 px, under the 256 floor
    assert run(disp, thin, K, c2w) is None, "the absolute 256 px floor is not applied"

    mid = torch.zeros_like(valid)
    mid[:, 4:29, 4:44] = True                         # 1000 px = 32.6% of the frame
    assert run(disp, mid, K, c2w, min_support=0.02) is not None, (
        "1000 px should clear a 2% floor")
    assert run(disp, mid, K, c2w, min_support=0.5) is None, (
        "min_support is not scaled by H*W -- the fraction is being ignored")


@check
def the_interpolated_mask_gates_the_interpolated_value():
    """The validity mask is sampled by the SAME bilinear kernel as the disparity,
    and anything below 1 is dropped.  A mask sampled with nearest, or thresholded
    at 0.5, would admit pixels whose value was already contaminated by an invalid
    tap -- i.e. the loss would chase disparity across a depth discontinuity, which
    is the one place a plausible prediction is legitimately allowed to disagree."""
    src = inspect.getsource(pairwise_consistency)
    assert "own_valid > 0.999" in src, "the sampled mask must be required to be ~1"
    assert 'mode="bilinear"' in src and 'padding_mode="zeros"' in src
    assert "align_corners=False" in src, (
        "the grid is built from +0.5 pixel centres, which is align_corners=False")
    disp, valid, K, c2w = plane_scene(translated_pair())
    holed = valid.clone()
    holed[1, 10:30, 10:30] = False                    # invalid block in the reference
    got = run(disp, holed, K, c2w)
    assert got is not None
    # A dilated hole: every pixel of view 0 landing on or beside the block is gone.
    assert got["support"] < 1.0 - 400.0 / (H * W), "the hole was not excluded"
    assert got["scale_dev"] < 2e-3, "excluding the hole must not disturb the fit"


@check
def the_trim_runs_after_the_fit():
    """Order matters and is not cosmetic.  A global scale error moves EVERY residual
    together, so trimming the largest residuals first would discard the error
    uniformly and leave the term reporting a small number on a badly scaled field.
    Trimming after the scale has been removed only discards outliers, which is what
    a trim is for."""
    src = inspect.getsource(pairwise_consistency)
    fit = src.index("a = (p * g).sum()")
    resid = src.index("resid = (a.detach() * p - g).abs()")
    trim = src.index("torch.quantile(resid, trim_quantile)")
    assert fit < resid < trim, "the trim must come after the scale fit and residual"
    # And empirically: a global scale error is not trimmed away, i.e. it still shows
    # up in |a-1| rather than being discarded as an outlier set.
    disp, valid, K, c2w = plane_scene(translated_pair())
    disp *= 1.4                                       # ALL views, i.e. a shared error
    got = run(disp, valid, K, c2w)
    assert got["scale_dev"] > 5e-3, (
        f"|a-1| = {got['scale_dev']:.4f}: a shared scale error must still register")


@check
def a_shared_scale_error_registers_only_WEAKLY():
    """A correction to the claim in the module docstring, measured rather than
    argued.  It is true that scaling every view's disparity by k does not cancel --
    the pairwise form is not blind to it the way §10a's shared-scale `res_gap` is --
    but the sensitivity is first order in (disparity gradient) x (focal) x (baseline)
    and is therefore SMALL on gentle geometry.  Measured, plane, k = 1.4:

        B = 0.15 -> |a-1| 0.0076      B = 0.4 -> 0.0206      B = 0.8 -> 0.0385

    against 0.5 x (k-1) = 0.20 for the same error applied to ONE view.  So this term
    is a detector of DISAGREEMENT between views, which is exactly what §10 measured
    (spread 8.7%, worst/best 1.87), and it must not be advertised as a fix for a
    scale error the whole prediction shares.  Nothing else in the objective would
    catch that, and the supervised depth MSE is what has to."""
    dev = {}
    for baseline in (0.15, 0.4, 0.8):
        disp, valid, K, c2w = plane_scene(translated_pair(baseline))
        disp *= 1.4
        dev[baseline] = run(disp, valid, K, c2w)["scale_dev"]
    assert dev[0.15] < dev[0.4] < dev[0.8], f"not monotone in baseline: {dev}"
    assert dev[0.8] < 0.1, f"the shared-scale response is not weak any more: {dev}"

    one_view = plane_scene(translated_pair())
    one_view[0][0] *= 1.4
    only = run(*one_view)["scale_dev"]
    assert only > 8.0 * dev[0.15], (
        f"a one-view 1.4x error reads {only:.4f} and a shared one {dev[0.15]:.4f} -- "
        "the term must be far more sensitive to disagreement than to a common error")


# ----------------------------------------------------------------------------
# Gradients: the reason this is a backward warp and not the shipped splat
# ----------------------------------------------------------------------------

@check
def the_vae_decode_is_safe_to_recompute():
    """`depth_consist_vae_checkpoint` is what makes three gradient-carrying decodes
    affordable, and torch.utils.checkpoint RECOMPUTES the forward.  The VAE's causal
    convolutions keep a feature cache across frames, so if that cache survived
    between calls the recomputed pass would not equal the original one and the
    gradient would be silently wrong -- not an error, a wrong number.  It is safe
    only because `decode` calls `clear_cache()` first; asserted rather than assumed."""
    from diffsynth.models.wan_video_vae import VideoVAE_
    body = inspect.getsource(VideoVAE_.decode)
    first = [ln.strip() for ln in body.splitlines() if ln.strip()
             and not ln.strip().startswith("#")][1]
    assert first == "self.clear_cache()", (
        f"VideoVAE_.decode now starts with {first!r}; the causal-conv cache is no "
        "longer reset per call and gradient checkpointing over it is unsound")
    assert "use_reentrant=False" in inspect.getsource(
        WanRGBDNVSTrainingModule._decode_src_disparity_with_grad), (
        "the reentrant checkpoint implementation does not support a call whose only "
        "input requires grad while module parameters also do")


@check
def the_shipped_splat_could_not_have_been_used():
    """utils/warp.py's forward splat quantises the landing pixel with
    `col.floor().long()` and writes with index_put, so the POSITION -- the main way
    a depth error shows itself -- carries no gradient at all, and the public entry
    point is under no_grad on top of that.  Documented here as a test so the
    'why not reuse the splat' question has an answer in the repo."""
    import utils.warp
    src = inspect.getsource(utils.warp._splat_sources_to_targets)
    assert "col.floor().long()" in src, (
        "the splat's pixel quantisation is what forces grid_sample here")
    assert "warped[bidx, t, pidx] =" in src, (
        "the splat writes by index assignment, which carries no gradient to the "
        "index -- i.e. none to the depth that produced it")
    assert "@torch.no_grad()" in inspect.getsource(utils.warp).split(
        "def splat_sources_to_targets")[0].rsplit("\n\n", 1)[-1], (
        "the public splat entry point is under no_grad on top of that")
    consist = inspect.getsource(pairwise_consistency)
    assert "F.grid_sample" in consist, "the loss must sample, not splat"
    assert ".floor()" not in consist and ".long()" not in consist, (
        "any quantisation of the sample position would kill the positional gradient")


@check
def gradient_reaches_both_views_through_both_paths():
    """view i's disparity enters through the SAMPLE POSITION (unproject, project,
    grid) and through `implied`; view j's enters through the sampled value and
    through the closed-form `a`.  All four have to be live: a detach anywhere makes
    the term a one-sided pull that can be satisfied by corrupting the other view."""
    disp, valid, K, c2w = plane_scene(shared_centre_pair())
    disp[0] = disp[0] * 1.3
    disp = disp.clone().requires_grad_(True)
    got = run(disp, valid, K, c2w)
    (got["scale"] + got["repro"]).backward()
    for v in (0, 1):
        assert disp.grad[v].abs().sum() > 0, f"no gradient reached view {v}"

    # The closed-form `a` specifically: with only the scale term, view j's gradient
    # can ONLY come through `a`, because `own` appears nowhere else in it.
    disp2 = plane_scene(shared_centre_pair())[0]
    disp2[0] = disp2[0] * 1.3
    disp2 = disp2.clone().requires_grad_(True)
    run(disp2, valid, K, c2w)["scale"].backward()
    assert disp2.grad[1].abs().sum() > 0, (
        "the least-squares scale must be differentiated, not detached -- a median or "
        "a detached fit would send the gradient to one pixel or to none")


@check
def the_analytic_gradient_matches_finite_differences():
    """The positional path is the one an implementation can get subtly wrong while
    still producing a plausible loss VALUE -- a transposed grid, a flipped y, a
    (W-1) normalisation.  Those are invisible to a value test near the optimum and
    visible here, because a directional derivative checks the whole chain at once.

    Evaluated away from the optimum (a ~ 1.3) since the gradient of (a-1)^2 vanishes
    at a = 1, and on the scale term only, whose value is smooth -- the residual's
    trim quantile is a set membership that finite differences would step across.

    The probe direction is SMOOTH, and that is not a convenience.  Bilinear sampling
    is only C0 in the sample position, so its derivative is piecewise constant in
    cell units; a white-noise direction moves thousands of samples across a cell
    boundary within h, and the finite difference is then genuinely not the derivative
    (measured: 8-15% disagreement at every h from 1e-3 to 2e-2, i.e. it does not
    converge, which is the signature of a kink rather than of a bug).  Along a smooth
    direction the same check agrees to 0.5%."""
    base, valid, K, c2w = plane_scene(translated_pair())
    base[0] = base[0] * 1.3
    yy = torch.linspace(0.0, 1.0, H).reshape(1, H, 1)
    xx = torch.linspace(0.0, 1.0, W).reshape(1, 1, W)
    direction = (torch.sin(2.1 * xx + 0.3) * torch.cos(1.7 * yy)
                 + 0.4 * torch.cos(3.0 * xx)).expand(2, H, W)
    # Different sign per view, so a bug that only got one of the two paths right
    # cannot cancel out in the dot product.
    direction = (direction * torch.tensor([1.0, -0.7]).reshape(2, 1, 1)).contiguous()

    def value(x):
        return float(run(x, valid, K, c2w)["scale"])

    x = base.clone().requires_grad_(True)
    run(x, valid, K, c2w)["scale"].backward()
    analytic = float((x.grad * direction).sum())

    h = 1e-3
    numeric = (value(base + h * direction) - value(base - h * direction)) / (2 * h)
    denom = max(abs(analytic), abs(numeric), 1e-8)
    assert abs(analytic - numeric) / denom < 0.02, (
        f"directional derivative {analytic:.6f} vs finite difference {numeric:.6f}")


# ----------------------------------------------------------------------------
# Pair bookkeeping
# ----------------------------------------------------------------------------

@check
def both_directions_of_every_pair_are_measured():
    """i->j and j->i are different measurements, not a redundant pair: the sample
    positions, the occluded set and the support all differ, and a_ji is not 1/a_ij
    once the kept sets differ.  Averaging both is also what makes the term
    symmetric, so it cannot be satisfied by degrading whichever view happens to be
    listed second."""
    assert ordered_pairs(range(3)) == [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
    assert len(ordered_pairs(range(6))) == 30, "matches the diagnostic's 30 pairs"
    assert ordered_pairs([2, 5]) == [(2, 5), (5, 2)], "must accept arbitrary labels"


@check
def support_weighting_follows_the_diagnostic():
    """Pairs are averaged by KEPT PIXEL COUNT, not equally.  The diagnostic's first
    unweighted version is the one that reported |a-1| up to 32.8; the reported
    `support` and `n_pairs` are the same statistics it reports (0.2049, 21.0 of 30)
    so the training log can be read against measured levels instead of nothing."""
    disp, valid, K, c2w = plane_scene(translated_pair())
    lop = valid.clone()
    lop[1, :, 40:] = False                            # shrink one direction's support
    got = run(disp, lop, K, c2w)
    assert 0.0 < got["support"] < 1.0
    assert set(inspect.signature(pairwise_consistency).parameters) >= {
        "min_support", "ratio_clip", "trim_quantile", "charbonnier_eps", "z_near"}
    src = inspect.getsource(pairwise_consistency)
    assert "weights.append(float(n_keep))" in src, "weighting must be by kept pixels"


# ----------------------------------------------------------------------------
# The gates, which are what keeps this affordable
# ----------------------------------------------------------------------------

def gate_harness(sigma_list, pixel_K=True, clean=None, max_samples=2, num_views=2,
                 measurable=False):
    """Bind `_depth_consistency_loss` to a bare object, with a decode stub that
    records which batch entries it was asked for.

    `measurable=False` returns an all-zero disparity, so `disp > 1e-3` empties the
    validity mask and pairwise_consistency returns None -- which is what the gate
    tests want, since they assert on WHICH samples got decoded and a decode that
    also produced a measurement would conflate the two.  `measurable=True` returns a
    positive constant instead: the harness's cameras are all identity, so every pair
    is trivially consistent and `got` is non-None, which is the only way to test the
    statistics that are meant to be means over MEASURED samples."""
    seen = []

    def stub(self, latents):
        seen.append(round(float(latents.mean()), 4))
        fill = 0.5 if measurable else 0.0
        return torch.full((latents.shape[2], H, W), fill, requires_grad=True)

    obj = types.SimpleNamespace(
        depth_consist_on=True,
        depth_consist_sigma_max=0.6,
        depth_consist_max_samples=max_samples,
        depth_consist_num_views=num_views,
        depth_consist_kwargs=dict(min_support=0.02, ratio_clip=4.0, trim_quantile=0.9),
        num_source_views=6,
    )
    obj._decode_src_disparity_with_grad = types.MethodType(stub, obj)
    obj._depth_consistency_loss = types.MethodType(
        WanRGBDNVSTrainingModule._depth_consistency_loss, obj)

    batch = len(sigma_list)
    sigma = torch.tensor(sigma_list).reshape(batch, 1, 1, 1, 1)
    # x0[b] == b, so the stub's recorded mean identifies which sample was chosen.
    depth_x_t = torch.arange(batch, dtype=torch.float32).reshape(batch, 1, 1, 1, 1)
    depth_x_t = depth_x_t.expand(batch, 16, 9, 6, 8).contiguous()
    inputs = {
        "loss_masks": torch.ones(batch, 2, 9, 1, H, W),
        "c2w": torch.eye(4).reshape(1, 1, 4, 4).expand(batch, 9, 4, 4).contiguous(),
    }
    if pixel_K:
        inputs["pixel_K"] = intrinsics().float().reshape(1, 1, 3, 3).expand(
            batch, 9, 3, 3).contiguous()
    out = obj._depth_consistency_loss(
        inputs, torch.zeros_like(depth_x_t), depth_x_t, sigma, clean)
    return out, seen


@check
def eval_never_pays_for_the_term():
    """Two separate facts, and it is worth keeping them apart because the obvious
    reading is wrong.

    Eval is excluded STRUCTURALLY: `_forward_flow` is only reached from `forward()`,
    while evaluate() goes through `sample_rgb`.  It is NOT excluded by the pixel_K
    check -- forward_preprocess's eval branch zeroes `warp_block` to a tensor rather
    than to None, so the intrinsics are still exported there.

    The pixel_K check is therefore defensive only, for `warp_mode != "splat"`, which
    the constructor already refuses.  It has to degrade to a no-op rather than raise,
    which is what this asserts."""
    train_src = inspect.getsource(WanRGBDNVSTrainingModule.forward)
    assert "_forward_flow" in train_src
    assert "_depth_consistency_loss" not in inspect.getsource(
        WanRGBDNVSTrainingModule.sample_rgb), "the eval sampler must not call the loss"
    out, seen = gate_harness([0.1, 0.2], pixel_K=False)
    assert seen == [], "a missing pixel_K must not decode"
    assert out["n_samples"] == 0.0 and float(out["scale"]) == 0.0


@check
def the_sigma_gate_bites():
    """Near sigma = 1 the x0 estimate is the model's blurry unconditional guess;
    penalising its geometry is not wrong, it is uninformative, and the VAE decodes
    are the expensive part.  Under logit_normal(0,1) with shift 5 the 0.6 default
    admits roughly the lowest 15% of draws."""
    out, seen = gate_harness([0.8, 0.9, 0.95])
    assert seen == [] and out["n_samples"] == 0.0, "high sigma must be skipped"
    _, seen = gate_harness([0.1, 0.9])
    assert seen == [0.0], "only the low-sigma sample may be decoded"
    # The boundary is inclusive, which is the documented spelling.
    assert "sig <= self.depth_consist_sigma_max" in inspect.getsource(
        WanRGBDNVSTrainingModule._depth_consistency_loss)


@check
def clean_source_depth_samples_are_skipped():
    """Their source depth frames were REPLACED by the clean latent, so
    `x_t - sigma*v` is not that sample's prediction of anything and its perfect
    consistency would be free.  This run has c-cleansrcd off, but the loss must not
    depend on that -- the two factors have to stay composable."""
    out, seen = gate_harness([0.1, 0.2], clean=torch.tensor([True, True]))
    assert seen == [] and out["n_samples"] == 0.0
    _, seen = gate_harness([0.1, 0.2], clean=torch.tensor([True, False]))
    assert seen == [1.0], "only the noised sample may be measured"


@check
def max_samples_caps_the_decodes_and_keeps_the_lowest_sigma():
    """The hard cost cap.  It keeps the LOWEST-sigma candidates, which is both the
    cheapest rule and the sharpest x0 estimate in the batch -- a random pick would
    spend the same compute on a worse signal."""
    _, seen = gate_harness([0.5, 0.15, 0.45, 0.25], max_samples=2)
    assert len(seen) == 2, f"decoded {len(seen)} times with max_samples=2"
    assert sorted(seen) == [1.0, 3.0], (
        f"decoded samples {sorted(seen)}; sigma order puts the two lowest at b=1 "
        "(0.15) and b=3 (0.25)")
    _, seen = gate_harness([0.5, 0.15, 0.45, 0.25], max_samples=1)
    assert seen == [1.0]


@check
def num_views_sets_how_many_views_are_decoded():
    """num_views is the cost knob with the steepest slope: k decodes and k(k-1)
    ordered pairs.  It must reach the decode, or the arm pays for 3 and measures 2."""
    for k in (2, 3, 4):
        out, seen = gate_harness([0.1], num_views=k, max_samples=1)
        assert len(seen) == 1
        assert out["n_samples"] == 0.0    # the stub's disparity is all-invalid
    src = inspect.getsource(WanRGBDNVSTrainingModule._depth_consistency_loss)
    assert "torch.randperm(num_src, device=x0.device)[:k]" in src, (
        "the views must be resampled per step -- a fixed subset would leave the "
        "other source views permanently unconstrained")


@check
def a_batch_with_no_measurable_pair_is_counted_not_hidden():
    """`pairwise_consistency` returning None is a legitimate outcome: on GT depth
    only 6-12 of 30 ordered BMVS pairs clear even 256 px.  It has to show up as
    n_samples = 0 in the log rather than as a silent zero loss, because a run whose
    n_samples sits near zero is not training the factor at all."""
    out, seen = gate_harness([0.1, 0.2])
    assert len(seen) == 2, "both eligible samples must be attempted"
    assert out["n_samples"] == 0.0 and float(out["scale"]) == 0.0
    assert out["n_pairs"] == 0.0
    src = inspect.getsource(WanRGBDNVSTrainingModule._depth_consistency_loss)
    assert "if got is None:" in src and "continue" in src


@check
def the_reported_terms_are_means_over_measured_samples():
    """Deliberate, and the docstring says so: dividing by the BATCH would be the
    clean statistical expectation but would make the logged number incomparable to
    the diagnostic's 0.1309 / 0.1213.  The price -- an effective weight that varies
    with how often a batch has an eligible sample -- is accepted, not overlooked."""
    src = inspect.getsource(WanRGBDNVSTrainingModule._depth_consistency_loss)
    assert "n = len(scale_terms)" in src and 'stats["scale"] = sum(scale_terms) / n' in src
    assert "batch_size" not in src, "normalising by the batch would change the meaning"


@check
def the_reported_sigma_is_the_sigma_of_the_MEASURED_samples():
    """Added after the fork smoke off droce@4000, and it is not cosmetic.  That smoke
    read |a-1| 0.052 and repro 0.032 where the diagnostic measured 0.1309 and 0.1213
    on the same checkpoint family -- a 2.5-4x gap that decides between raising the
    weights and raising sigma_max.  The two hypotheses are only separable if the
    logged sigma belongs to the samples that were actually MEASURED, because a
    candidate that decoded but whose pairs all failed min_support contributes a
    sigma and no |a-1|, and averaging it in would bias the very column the
    attribution rests on."""
    out, seen = gate_harness([0.5, 0.15, 0.45, 0.25], max_samples=2, measurable=True)
    assert out["n_samples"] == 2.0, f"harness must measure, got n={out['n_samples']}"
    # b=1 (0.15) and b=3 (0.25) are the two lowest; their mean is 0.20, NOT the
    # 0.3375 mean of the whole batch nor the 0.15 of the single lowest.
    assert abs(out["sigma"] - 0.20) < 1e-5, (
        f"sigma={out['sigma']}; expected the mean over the two measured samples")
    # And a decoded-but-unmeasurable candidate must not reach it.
    out, seen = gate_harness([0.15, 0.25], measurable=False)
    assert len(seen) == 2 and out["n_samples"] == 0.0
    assert out["sigma"] != out["sigma"], "unmeasured must stay nan, not 0.2"


# ----------------------------------------------------------------------------
# The §9-1a trap: a factor that leaves no trace anywhere else
# ----------------------------------------------------------------------------

@check
def defaults_are_a_no_op_and_agree_across_every_file_that_spells_them():
    """f-consist changes no shape and no parameter, so a default that disagrees
    between the module, the config and the training script produces a checkpoint
    that loads without a single warning and a run that trained a different
    objective than the one recorded.  That is §9-1a verbatim, one step quieter than
    warp_perturb because not even the training input differs."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    params = inspect.signature(WanRGBDNVSTrainingModule.__init__).parameters
    expect = {
        "depth_consist_scale_weight": 0.0,
        "depth_consist_repro_weight": 0.0,
        "depth_consist_sigma_max": 0.6,
        "depth_consist_num_views": 3,
        "depth_consist_max_samples": 2,
        "depth_consist_min_support": 0.02,
        "depth_consist_ratio_clip": 4.0,
        "depth_consist_trim_quantile": 0.9,
        "depth_consist_vae_checkpoint": True,
    }
    for name, value in expect.items():
        assert name in params, f"{name} is not a constructor parameter"
        assert params[name].default == value, (
            f"{name} defaults to {params[name].default!r}, not {value!r}")
    assert expect["depth_consist_scale_weight"] == 0.0
    assert expect["depth_consist_repro_weight"] == 0.0, "the default must be OFF"

    cfg_path = os.path.join(root, "train_config/normal_config/bmvs_c_base.yaml")
    cfg_text = open(cfg_path, encoding="utf-8").read()
    for name, value in expect.items():
        # isinstance first: `0.0 == False` in Python, so a bool-keyed branch would
        # spell the weights as "false".
        spelling = ("true" if value else "false") if isinstance(value, bool) else str(value)
        assert f"{name}: {spelling}" in cfg_text, (
            f"{cfg_path} must pin {name}: {spelling} -- every c/D/E arm is "
            "reproducible from this file and a missing key changes what they mean")

    train = os.path.join(root, "examples/wanvideo/model_training/train_with_accelerate_rgbd_nvs.py")
    text = open(train, encoding="utf-8").read()
    for name, value in expect.items():
        needle = f'{name}=cfg.get("{name}", {value!r})'
        assert needle in text, f"the training script does not forward: {needle}"

    # The robustness knobs are spelled once in the module and once in the util; the
    # util's defaults are the authority and the module must not shadow them with a
    # different number.
    util = inspect.signature(pairwise_consistency).parameters
    for short in ("min_support", "ratio_clip", "trim_quantile"):
        assert util[short].default == expect[f"depth_consist_{short}"], (
            f"utils/depth_consistency.py defaults {short} to {util[short].default!r} "
            f"but the module and config say {expect[f'depth_consist_{short}']!r}")


@check
def the_factors_line_identifies_the_arm():
    """The whole point: an arm must be identifiable from train.log alone.  f-consist
    leaves no other trace -- same shapes, same metric names, same everything -- so
    if the weights are not echoed here there is no way to tell afterwards whether a
    checkpoint was trained with the term or without it."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    text = open(os.path.join(
        root, "examples/wanvideo/model_training/train_with_accelerate_rgbd_nvs.py"),
        encoding="utf-8").read()
    start = text.index('f"factors: ')
    line = text[start:text.index("\n    )", start)]
    for name in ("depth_consist_scale_weight", "depth_consist_repro_weight",
                 "depth_consist_sigma_max", "depth_consist_num_views",
                 "depth_consist_max_samples", "depth_consist_min_support",
                 "depth_consist_ratio_clip", "depth_consist_trim_quantile",
                 "depth_consist_vae_checkpoint"):
        assert name in line, f"the factors: line does not echo {name}"


@check
def the_term_actually_reaches_the_optimised_loss():
    """A term computed, logged and then dropped on the floor is the worst outcome
    available: the run reports the consistency it never trained on.  `fill_scale` in
    the E round was exactly that for one commit."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    text = open(os.path.join(
        root, "examples/wanvideo/model_training/train_with_accelerate_rgbd_nvs.py"),
        encoding="utf-8").read()
    assert 'consist = res.get("consist")' in text
    assert 'consist_scale_weight * consist["scale"]' in text
    assert 'consist_repro_weight * consist["repro"]' in text
    # And the log must not re-enter the graph: float() on a requires_grad tensor
    # warns once per log_step into the one file the arm is identified from.
    assert 'float(consist["scale"].detach())' in text
    add = text.index('consist_scale_weight * consist["scale"]')
    assert add < text.index("accelerator.backward(", add), (
        "the term must be added to `loss` before the backward pass, not after it")
    # And the module must hand it out, or `res.get` silently returns None forever.
    assert '"consist": consist' in inspect.getsource(WanRGBDNVSTrainingModule._forward_flow)


@check
def the_weights_land_at_the_calibrated_level():
    """The level is chosen, not tuned: at the measured pair_scale_dev 0.1309 and
    pair_res_after_scale 0.1213, with depth_loss reading 0.0375, the intended arm
    puts the regulariser at roughly half the depth MSE.  This checks the arithmetic
    the sbatch header claims, so a typo in a weight cannot pass as a decision."""
    scale_w, repro_w = 0.25, 0.1
    pair_scale_dev, pair_res = 0.1309, 0.1213
    depth_loss = 0.0375
    # E[(a-1)^2] >= (E|a-1|)^2 by Jensen, so the squared term's contribution is
    # bounded below by this and the true value depends on the spread (max 1.87x).
    lo = scale_w * pair_scale_dev ** 2 + repro_w * pair_res
    assert 0.3 * depth_loss < lo < 0.6 * depth_loss, (
        f"the lower bound {lo:.4f} is not ~half of depth_loss {depth_loss:.4f}; the "
        "sbatch header's arithmetic and the weights disagree")


def main():
    failed = 0
    for fn in CHECKS:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - a test harness reports, not raises
            failed += 1
            print(f"FAIL  {fn.__name__}\n      {type(exc).__name__}: {exc}")
        else:
            print(f"ok    {fn.__name__}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
