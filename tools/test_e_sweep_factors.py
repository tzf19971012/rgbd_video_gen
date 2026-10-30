"""CPU-only unit tests for the E-sweep factors.

Same reason as tools/test_c_sweep_factors.py and tools/test_d_sweep_factors.py:
the factor is SILENT when wrong.  E-warpnoise is worse than most, because its
whole job is to make the training input slightly wrong on purpose -- so a bug
that makes it wrong in the WRONG WAY (blurring across the sky boundary, gain and
shift drawn per-batch instead of per-view, a perturbation that zeroes pixels and
thereby leaks a validity mask, an amplitude that drifted off the measured
distribution) produces a run that converges, reports a full set of metrics, and
answers a question nobody asked.

Nothing here loads Wan2.1 or touches a GPU.  The perturbation operator is a
module-level function on purpose, so these tests and
tools/calibrate_warp_perturb.py exercise the SHIPPED code rather than a copy of
it; the mixing wrapper is bound to a bare object carrying only the attributes it
reads, the same trick test_d_sweep_factors.py uses for `_select_views`.

  python -m tools.test_e_sweep_factors
"""
import inspect
import math
import re

import torch

from examples.wanvideo.model_training.WanRGBDNVSTrainingModule import (
    DISPARITY_EPS,
    WanRGBDNVSTrainingModule,
    normalized_gaussian_blur,
    perturb_source_disparity,
)

CHECKS = []

# The measured distribution the operator has to reproduce -- 384 source views of
# bmvs_d_drocefp32_s42@3750's own pass-1 estimate (report_20260913_e_sweep_prep §3).
MEASURED_AFFINE_A_STD = 0.3495
MEASURED_SHIFT_B_STD = 0.0590


def check(fn):
    CHECKS.append(fn)
    return fn


def synthetic_disparity(B=3, S=4, H=48, W=64, sky_rows=10, seed=0):
    """A disparity map with the two features that matter: layered structure the
    blur can destroy, and a block of GT-invalid `sky` at the top."""
    g = torch.Generator().manual_seed(seed)
    y = torch.linspace(0.2, 0.9, H).reshape(1, 1, H, 1).expand(B, S, H, W).clone()
    # Three depth layers with hard edges -- the thing over-smoothing eats.
    y[..., W // 3:2 * W // 3] += 0.15
    y[..., 2 * W // 3:] -= 0.10
    y = y + 0.01 * torch.randn(B, S, H, W, generator=g)
    y = y.clamp(0.05, 0.95)
    y[:, :, :sky_rows, :] = 0.0          # GT-invalid
    return y


def fit_a(p, g):
    p, g = p.double().flatten(), g.double().flatten()
    n = p.numel()
    sp, spp, sg, spg = p.sum(), (p * p).sum(), g.sum(), (p * g).sum()
    return float((n * spg - sp * sg) / (n * spp - sp * sp))


# ----------------------------------------------------------------------------
# The blur
# ----------------------------------------------------------------------------

@check
def blur_ignores_invalid_pixels():
    """A plain blur would drag disparity towards the zeros that mark sky, i.e.
    manufacture a depth cliff at every sky boundary.  The real defect is the
    opposite -- the prediction fills sky with plausible values -- so a normalized
    convolution here is load bearing, not a nicety."""
    # A flat valid region isolates the effect: any deviation from 0.6 next to the
    # sky boundary can only have come from averaging in the invalid zeros.
    x = torch.full((1, 1, 48, 64), 0.6)
    x[:, :, :10, :] = 0.0
    mask = (x > DISPARITY_EPS).float()
    out = normalized_gaussian_blur(x, mask, sigma=3.0)
    row = 11                                    # first valid row, next to the sky
    assert abs(float(out[0, 0, row].mean()) - 0.6) < 1e-4, (
        f"row {row} came back at {float(out[0, 0, row].mean()):.4f} instead of 0.6 -- "
        "the blur is averaging in the zeros"
    )
    plain = torch.nn.functional.avg_pool2d(x, 7, 1, 3)
    assert float(plain[0, 0, row].mean()) < 0.6 - 0.05, (
        "the control is not controlling: a plain blur should visibly darken this row"
    )


@check
def blur_preserves_a_constant():
    """Normalized convolution must be exact on constants, including at the image
    border -- otherwise every source view acquires a frame-shaped disparity
    artefact and the splat gets a fake wall around the picture."""
    x = torch.full((1, 1, 32, 40), 0.6)
    mask = torch.ones_like(x)
    out = normalized_gaussian_blur(x, mask, sigma=4.0)
    assert torch.allclose(out, x, atol=1e-5), (
        f"max deviation {float((out - x).abs().max()):.2e}; replicate padding is "
        "not being used, or the kernel is not normalised"
    )


@check
def blur_sigma_zero_is_the_identity_on_valid_pixels():
    x = synthetic_disparity(B=1, S=2)
    mask = (x > DISPARITY_EPS).float()
    assert torch.equal(normalized_gaussian_blur(x, mask, 0.0), x * mask)


# ----------------------------------------------------------------------------
# The four measured defects
# ----------------------------------------------------------------------------

@check
def gain_matches_the_measured_affine_spread():
    """The closed form the default is derived from: the operator applies
    p = mu + g*(d - mu) with g = exp(N(0, s)), so the diagnostic's fit
    d ~ a*p + b recovers a = 1/g, lognormal with median 1 and
    std = sqrt(exp(s^2) - 1).  If this drifts, the arm trains on a scale error
    the model will never actually see, and nothing in the run says so."""
    s = 0.34
    predicted = math.sqrt(math.exp(s * s) - 1.0)
    assert abs(predicted - MEASURED_AFFINE_A_STD) < 0.01, (
        f"the closed form gives std(a)={predicted:.4f} for gain_log_std={s}, which no "
        f"longer matches the measured {MEASURED_AFFINE_A_STD}: recalibrate the default"
    )

    torch.manual_seed(0)
    x = synthetic_disparity(B=24, S=6, seed=1)
    out = perturb_source_disparity(
        x, gain_log_std=s, shift_std=0.0, blur_sigma=0.0, lowfreq_std=0.0,
        fill_invalid=False)
    valid = x > DISPARITY_EPS
    a = [fit_a(out[b, v][valid[b, v]], x[b, v][valid[b, v]])
         for b in range(x.shape[0]) for v in range(x.shape[1])]
    a = torch.tensor(a)
    # n=144 views, so allow a generous band; the point is order of magnitude and
    # centring, not a precise moment match.
    assert 0.5 * predicted < float(a.std()) < 1.8 * predicted, (
        f"realised std(a)={float(a.std()):.4f}, expected around {predicted:.4f}"
    )
    assert 0.85 < float(a.median()) < 1.15, (
        f"median(a)={float(a.median()):.4f}: the gain is not centred, so this factor "
        "also applies a systematic bias the measurement does not support"
    )


@check
def shift_matches_the_measured_offset_spread():
    torch.manual_seed(0)
    x = synthetic_disparity(B=32, S=6, seed=2)
    out = perturb_source_disparity(
        x, gain_log_std=0.0, shift_std=MEASURED_SHIFT_B_STD, blur_sigma=0.0,
        lowfreq_std=0.0, fill_invalid=False)
    valid = x > DISPARITY_EPS
    b_only = torch.tensor([
        float((x[b, v][valid[b, v]] - out[b, v][valid[b, v]]).mean())
        for b in range(x.shape[0]) for v in range(x.shape[1])])
    assert 0.5 * MEASURED_SHIFT_B_STD < float(b_only.std()) < 1.8 * MEASURED_SHIFT_B_STD, (
        f"realised shift spread {float(b_only.std()):.4f}, expected around "
        f"{MEASURED_SHIFT_B_STD}"
    )


@check
def gain_and_shift_are_drawn_per_view_not_per_batch_or_per_sample():
    """The measurement is per SOURCE VIEW (std(a)=0.35 across 384 views, and
    `affine_shared` proves the six sources of one sample disagree with each other).
    A per-batch draw would make the whole batch's warp wrong in the same direction
    -- a much easier problem than the real one, and one that trains a model to
    expect a coherent global scale error."""
    torch.manual_seed(0)
    x = synthetic_disparity(B=2, S=6, seed=3)
    out = perturb_source_disparity(
        x, gain_log_std=0.34, shift_std=0.059, blur_sigma=0.0, lowfreq_std=0.0,
        fill_invalid=False)
    valid = x > DISPARITY_EPS
    a = [[fit_a(out[b, v][valid[b, v]], x[b, v][valid[b, v]]) for v in range(6)]
         for b in range(2)]
    for b in range(2):
        assert len(set(round(v, 4) for v in a[b])) == 6, (
            f"sample {b}: the six source views share a gain {a[b]} -- the draw is not "
            "per view"
        )
    assert round(a[0][0], 4) != round(a[1][0], 4), "the draw is shared across samples"


@check
def invalid_pixels_are_filled_with_something_plausible():
    """The prediction never emits disparity <= eps: it fills GT sky, and its splat
    mask therefore comes out LARGER than the oracle's (0.5371 vs 0.5266).  If this
    stops happening the arm loses the one defect that changes the mask, which is
    also the only one visible in `tgt_rgb_splat_pred_coverage`."""
    torch.manual_seed(0)
    x = synthetic_disparity(B=2, S=3, sky_rows=12, seed=4)
    invalid = x <= DISPARITY_EPS
    assert invalid.any()

    out = perturb_source_disparity(
        x, blur_sigma=2.0, lowfreq_std=0.0, gain_log_std=0.0, shift_std=0.0,
        fill_invalid=True)
    assert bool((out[invalid] > DISPARITY_EPS).all()), "fill_invalid=True left holes"
    # Plausible = continuous with the surrounding geometry, not an arbitrary
    # constant and not out of range.
    near = out[0, 0, 12:16].mean()
    filled = out[0, 0, 8:12].mean()
    assert abs(float(filled - near)) < 0.25, (
        f"filled {float(filled):.3f} vs neighbouring {float(near):.3f}: the fill is "
        "not a continuation of the surrounding disparity"
    )
    assert float(out.min()) >= 2.0 * DISPARITY_EPS and float(out.max()) <= 1.0

    off = perturb_source_disparity(
        x, blur_sigma=2.0, lowfreq_std=0.0, gain_log_std=0.0, shift_std=0.0,
        fill_invalid=False)
    assert torch.equal(off[invalid], torch.zeros_like(off[invalid])), (
        "fill_invalid=False must leave invalid pixels at exactly 0: "
        "_splat_sources_pixel reads validity off `disparity > eps`, so any positive "
        "leakage silently changes the warp MASK too"
    )


@check
def output_never_lands_on_an_accidentally_invalid_value():
    """A perturbed pixel that happens to land at or below eps would silently become
    an invalid pixel, i.e. the perturbation would leak a *mask* change into what is
    supposed to be a value change.  With a large shift pushing everything negative
    this is the failure that must not happen."""
    x = synthetic_disparity(B=4, S=4, seed=5)
    valid = x > DISPARITY_EPS
    torch.manual_seed(0)
    out = perturb_source_disparity(
        x, gain_log_std=0.5, shift_std=0.5, blur_sigma=1.0, lowfreq_std=0.2,
        fill_invalid=True)
    assert bool((out > DISPARITY_EPS).all()), "a pixel fell to or below eps"
    assert torch.isfinite(out).all()
    torch.manual_seed(0)
    out = perturb_source_disparity(
        x, gain_log_std=0.5, shift_std=0.5, blur_sigma=1.0, lowfreq_std=0.2,
        fill_invalid=False)
    assert bool((out[valid] > DISPARITY_EPS).all()), (
        "a GT-valid pixel became invalid: the perturbation is changing the mask"
    )


@check
def blur_actually_removes_structure():
    """The residual-structure share is 71.8% of the oracle gap, so this is the
    component that carries the factor.  A blur that does not measurably flatten
    the layer edges would make the arm a scale-error arm by accident -- and §3
    says scale is worth only a quarter of the gap."""
    x = synthetic_disparity(B=1, S=1, seed=6)
    out = perturb_source_disparity(
        x, gain_log_std=0.0, shift_std=0.0, blur_sigma=3.0, lowfreq_std=0.0,
        fill_invalid=True)
    grad = lambda t: (t[..., 1:] - t[..., :-1]).abs()[..., 14:, :].mean()
    assert float(grad(out)) < 0.6 * float(grad(x)), (
        f"horizontal gradient energy only went {float(grad(out)):.4f} vs "
        f"{float(grad(x)):.4f}; blur_sigma is not reaching the image"
    )


@check
def lowfreq_field_is_low_frequency_and_zero_mean():
    """Spatially correlated error, not per-pixel noise: white noise on a disparity
    map is removed by the splat's own z-buffer and by the VAE encode, so it would
    train the model against a defect it never receives."""
    torch.manual_seed(0)
    x = torch.full((8, 6, 48, 64), 0.5)
    out = perturb_source_disparity(
        x, gain_log_std=0.0, shift_std=0.0, blur_sigma=0.0, lowfreq_std=0.05,
        lowfreq_cells=6, fill_invalid=True)
    field = out - 0.5
    assert abs(float(field.mean())) < 0.01, f"field mean {float(field.mean()):.4f}"
    fine = (field[..., 1:] - field[..., :-1]).abs().mean()
    coarse = (field[..., 8:] - field[..., :-8]).abs().mean()
    assert float(fine) < 0.35 * float(coarse), (
        f"neighbouring-pixel variation {float(fine):.5f} is not small against the "
        f"8-pixel one {float(coarse):.5f}: the field is not low frequency"
    )


# ----------------------------------------------------------------------------
# The mixing wrapper and the config surface
# ----------------------------------------------------------------------------

class _Stub:
    """Only what `_maybe_perturb_src_disparity` reads."""

    def __init__(self, prob, **knobs):
        self.warp_perturb_prob = float(prob)
        self.warp_perturb_kwargs = dict(
            gain_log_std=0.34, shift_std=0.059, blur_sigma=2.0,
            lowfreq_std=0.05, lowfreq_cells=6, fill_invalid=True, fill_scale=1.0)
        self.warp_perturb_kwargs.update(knobs)


def bind(stub):
    return WanRGBDNVSTrainingModule._maybe_perturb_src_disparity.__get__(stub)


@check
def prob_zero_returns_the_input_object_untouched():
    """The no-op path the entire baseline reuse rests on: droce's two seeds are
    already on disk at step 4000, and every E arm is read against them.  If the
    factor's default did anything at all -- even numerically identical work
    through a different code path -- those checkpoints would stop being the
    anchor."""
    x = synthetic_disparity()
    out = bind(_Stub(0.0))(x)
    assert out is x, "prob=0 must short-circuit, not recompute"


@check
def mixing_is_per_sample_and_leaves_the_rest_bit_identical():
    """Per-sample, not per-batch, for the same reason warp_dropout_prob is: a mixed
    batch keeps both regimes inside every gradient step.  And the unperturbed
    samples must be BIT-identical, or the factor quietly perturbs 100% of the batch
    by a small amount instead of 50% by the measured amount."""
    x = synthetic_disparity(B=16, S=4, seed=7)
    torch.manual_seed(0)
    out = bind(_Stub(0.5))(x)
    same = torch.tensor([bool(torch.equal(out[b], x[b])) for b in range(16)])
    assert 0 < int(same.sum()) < 16, (
        f"{int(same.sum())}/16 samples untouched -- the mixture is degenerate"
    )
    for b in range(16):
        if same[b]:
            continue
        assert not torch.equal(out[b], x[b])


@check
def all_views_of_a_perturbed_sample_are_perturbed_together():
    """The warp block for one target view is built from all six sources at once, so
    degrading a subset would produce a splat that is internally inconsistent in a
    way inference never is."""
    x = synthetic_disparity(B=8, S=6, seed=8)
    torch.manual_seed(1)
    out = bind(_Stub(0.5))(x)
    for b in range(8):
        per_view = [bool(torch.equal(out[b, v], x[b, v])) for v in range(6)]
        assert len(set(per_view)) == 1, f"sample {b} is half perturbed: {per_view}"


@check
def prob_one_perturbs_everything():
    x = synthetic_disparity(B=6, S=3, seed=9)
    torch.manual_seed(2)
    out = bind(_Stub(1.0))(x)
    for b in range(6):
        assert not torch.equal(out[b], x[b])


@check
def shape_and_dtype_survive():
    """The block downstream is cast to the latents' dtype, so a silent float64 or a
    transposed view here would either cost memory or splat garbage."""
    for dtype in (torch.float32, torch.bfloat16):
        x = synthetic_disparity(B=2, S=6, seed=10).to(dtype)
        out = bind(_Stub(1.0))(x)
        assert out.shape == x.shape and out.dtype == dtype, (out.shape, out.dtype)
        assert torch.isfinite(out.float()).all()


@check
def the_factor_refuses_the_configurations_that_would_measure_nothing():
    """Every one of these is a 46-hour run that converges, reports a full metric
    set, and duplicates the arm it is supposed to be read against."""
    src = inspect.getsource(WanRGBDNVSTrainingModule.__init__)
    guards = (
        ("warp_mode != \"splat\"", "no warp block to degrade"),
        ("not 0.0 < self.warp_perturb_prob <= 1.0", "out-of-range probability"),
        ("self.warp_dropout_prob >= 1.0", "every degraded block then zeroed"),
        ("is the identity map", "all amplitudes zero"),
    )
    for needle, why in guards:
        assert needle in src, f"__init__ no longer refuses: {why} ({needle!r})"


@check
def defaults_are_a_no_op_and_agree_across_every_file_that_spells_them():
    """A default that disagrees between the module, the config and an eval harness
    is the §9-1a failure verbatim: the checkpoint loads without a warning and the
    numbers describe a different model.  Note warp_perturb_* affects TRAINING only,
    which makes a mismatch here even quieter than usual -- no eval metric moves."""
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    params = inspect.signature(WanRGBDNVSTrainingModule.__init__).parameters
    expect = {
        "warp_perturb_prob": 0.0,
        "warp_perturb_gain_log_std": 0.34,
        "warp_perturb_shift_std": 0.059,
        "warp_perturb_blur_sigma": 2.0,
        "warp_perturb_lowfreq_std": 0.05,
        "warp_perturb_lowfreq_cells": 6,
        "warp_perturb_fill_invalid": True,
        "warp_perturb_fill_scale": 1.0,
    }
    for name, value in expect.items():
        assert name in params, f"{name} is not a constructor parameter"
        assert params[name].default == value, (
            f"{name} defaults to {params[name].default!r}, not {value!r}")
    assert expect["warp_perturb_prob"] == 0.0, "the default must be OFF"

    cfg_path = os.path.join(root, "train_config/normal_config/bmvs_c_base.yaml")
    cfg_text = open(cfg_path, encoding="utf-8").read()
    for name, value in expect.items():
        # isinstance before the lookup: `0.0 == False` in Python, so a dict keyed on
        # booleans would spell warp_perturb_prob's default as "false".
        spelling = ("true" if value else "false") if isinstance(value, bool) else str(value)
        assert f"{name}: {spelling}" in cfg_text, (
            f"{cfg_path} must pin {name}: {spelling} -- the c/d-round arms are "
            "reproducible from this file and a missing key changes what they mean"
        )

    for path in ("examples/wanvideo/model_training/train_with_accelerate_rgbd_nvs.py",
                 "examples/wanvideo/model_training/eval_rgbd_nvs.py",
                 "tools/diag_posterior_mean.py",
                 "tools/diag_warp_sensitivity.py",
                 "tools/diag_pose_sensitivity.py"):
        text = open(os.path.join(root, path), encoding="utf-8").read()
        for name, value in expect.items():
            needle = f'{name}=cfg.get("{name}", {value!r})'
            assert needle in text, f"{path} does not forward: {needle}"


@check
def every_amplitude_knob_actually_reaches_the_operator():
    """A ctor parameter that is accepted, echoed into the `factors:` line and then
    dropped on the floor is the worst kind of knob: the sweep cell reports the value
    it asked for and trains on the default.  `fill_scale` was exactly that for one
    edit.  So: the operator's keyword names, the dict the ctor builds, and the
    warp_perturb_* parameters must be the same set, with the same defaults."""
    operator = set(inspect.signature(perturb_source_disparity).parameters)
    operator -= {"src_disparity", "generator"}

    src = inspect.getsource(WanRGBDNVSTrainingModule.__init__)
    body = src[src.index("self.warp_perturb_kwargs = dict("):]
    body = body[: body.index("\n        )")]
    forwarded = set(re.findall(r"^\s+(\w+)=", body, flags=re.M))
    assert forwarded == operator, (
        f"warp_perturb_kwargs forwards {sorted(forwarded)} but the operator takes "
        f"{sorted(operator)}; missing: {sorted(operator - forwarded)}"
    )

    params = inspect.signature(WanRGBDNVSTrainingModule.__init__).parameters
    op_defaults = inspect.signature(perturb_source_disparity).parameters
    for name in sorted(operator):
        ctor_name = f"warp_perturb_{name}"
        assert ctor_name in params, f"{ctor_name} is not a constructor parameter"
        assert params[ctor_name].default == op_defaults[name].default, (
            f"{ctor_name} defaults to {params[ctor_name].default!r} but the operator "
            f"defaults {name} to {op_defaults[name].default!r} -- two spellings of "
            "the same calibrated amplitude"
        )


@check
def the_oracle_variant_forces_the_factor_off():
    """`evaluate_warp_sensitivity`'s oracle rebuilds the warp from GT depth by
    calling forward_preprocess in TRAINING mode.  For an e-warpnoise checkpoint
    that path would hand back a DEGRADED oracle -- an upper bound that is quietly
    too low, on exactly the axis the arm is testing, in a diagnostic that is read
    as a delta so nothing looks out of range."""
    src = inspect.getsource(WanRGBDNVSTrainingModule.evaluate_warp_sensitivity)
    assert "self.warp_perturb_prob = 0.0" in src
    assert "self.warp_perturb_prob = saved_perturb" in src, (
        "the factor is forced off but never restored -- validate_step would then "
        "silently disable the factor for the REST OF TRAINING"
    )


@check
def the_perturbation_is_upstream_of_the_splat():
    """Degrading the finished latent block instead of the disparity would produce
    values no splat can emit (that is what the diagnostic-only
    `_corrupt_warp_block` is for).  The regime inference lands in is a
    geometrically CONSISTENT wrong warp: wrong occlusions, wrong parallax, a mask
    that covers the sky."""
    src = inspect.getsource(WanRGBDNVSTrainingModule.forward_preprocess)
    assert "_maybe_perturb_src_disparity(" in src
    # The CALL form, not the bare name: the comment at the call site names
    # _corrupt_warp_block to explain why it is the wrong tool here.
    assert "self._corrupt_warp_block(" not in src, (
        "forward_preprocess must not use the diagnostic corruption path"
    )
    build = src.index("_build_warp_block")
    perturb = src.index("_maybe_perturb_src_disparity(")
    assert build < perturb, (
        "the perturbation must be an ARGUMENT to _build_warp_block, i.e. applied to "
        "the disparity, not to the block it returns"
    )
    dropout = src.index("self.warp_dropout_prob > 0.0")
    assert perturb < dropout, (
        "the perturbation Bernoulli must run BEFORE the dropout one, so that "
        "warp_dropout_prob keeps its tuned meaning and the two factors stay "
        "orthogonal"
    )


@check
def eval_never_perturbs():
    """At eval the warp block is zeroed and rebuilt by sample_rgb from the model's
    own estimate; the degradation must not also apply there or the arm would be
    scored on a different input than every checkpoint it is compared with."""
    src = inspect.getsource(WanRGBDNVSTrainingModule.forward_preprocess)
    head = src.index("if self.warp_mode == \"splat\":")
    branch = src.index("if self._eval_inputs:", head)
    else_at = src.index("            else:", branch)
    assert "_maybe_perturb_src_disparity" not in src[branch:else_at], (
        "the eval branch of the warp path must stay free of the perturbation"
    )


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
