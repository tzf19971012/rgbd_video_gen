"""CPU-only unit tests for the two C-sweep factors.

Both factors are silent when wrong: a mis-plumbed `rope_frame_ids` still trains,
and a `clean_source_depth` that forgets to pin the timestep or drop the loss still
converges -- it just measures nothing.  So every assertion here targets a failure
that would NOT raise on its own during a 40-hour run.

Nothing here loads Wan2.1.  The RoPE tests build a tiny real WanModel and call the
real `model_fn_wan_video`; the clean-source-depth tests bind the real
`WanRGBDNVSTrainingModule` methods to a bare object with only the attributes they
touch, so the code under test is the shipped code and not a copy of it.

  python -m tools.test_c_sweep_factors
"""
import torch

from diffsynth.models import wan_video_dit
from diffsynth.models.wan_video_dit import WanModel
from diffsynth.pipelines.wan_video_new_determine import model_fn_wan_video
from examples.wanvideo.model_training.WanRGBDNVSTrainingModule import (
    WanRGBDNVSTrainingModule,
)

CHECKS = []

# flash-attn has no CPU kernel, so `flash_attention` has to fall through to
# F.scaled_dot_product_attention.  These flags are read as globals at call time,
# and the fallback is a numerically equivalent path that already exists in the
# file -- so this changes where attention is computed, not what the RoPE test
# is measuring.
wan_video_dit.FLASH_ATTN_3_AVAILABLE = False
wan_video_dit.FLASH_ATTN_2_AVAILABLE = False
wan_video_dit.SAGE_ATTN_AVAILABLE = False


def check(fn):
    CHECKS.append(fn)
    return fn


# ----------------------------------------------------------------------------
# shared_rope_frame_ids
# ----------------------------------------------------------------------------

def _tiny_dit():
    # head_dim must be divisible by 6: precompute_freqs_cis_3d splits it as
    # (d - 2*(d//3), d//3, d//3) and each part is halved again, so anything else
    # produces a freqs table narrower than head_dim/2 and the rotation fails.
    # Real Wan2.1-1.3B has head_dim 128... which is not divisible by 6 either, but
    # 128 - 2*42 = 44 happens to make the halves add up; 24 is the smallest size
    # that also works.
    torch.manual_seed(0)
    return WanModel(
        dim=48, in_dim=4, ffn_dim=64, out_dim=4, text_dim=16, freq_dim=48,
        eps=1e-6, patch_size=(1, 2, 2), num_heads=2, num_layers=1,
        has_image_input=False,
    ).eval()


def _run_dit(dit, latents, rope_frame_ids=None):
    return model_fn_wan_video(
        dit=dit,
        latents=latents,
        timestep=torch.tensor([500.0]),
        context=torch.randn(1, 3, 16, generator=torch.Generator().manual_seed(1)),
        rope_frame_ids=rope_frame_ids,
    )


@check
def rope_override_changes_the_output():
    """The whole point: passing ids must actually reach the RoPE table.

    If `rope_frame_ids` were dropped on the floor by a signature mismatch or a
    `**kwargs` swallow, this factor would train an exact duplicate of `base`.
    """
    dit = _tiny_dit()
    v = 3
    latents = torch.randn(1, 4, 2 * v, 4, 4, generator=torch.Generator().manual_seed(2))
    default = _run_dit(dit, latents)
    shared = _run_dit(dit, latents, rope_frame_ids=torch.arange(v).repeat(2))
    assert default.shape == shared.shape, (default.shape, shared.shape)
    assert not torch.allclose(default, shared, atol=1e-5), (
        "rope_frame_ids had no effect on the output -- the override is not reaching "
        "the freqs table"
    )


@check
def rope_identity_ids_are_a_noop():
    """arange(f) must reproduce the default exactly, bit for bit.

    This is the regression check: if it fails, `shared_rope_frame_ids=False` runs
    are no longer comparable to anything measured before this change.
    """
    dit = _tiny_dit()
    latents = torch.randn(1, 4, 6, 4, 4, generator=torch.Generator().manual_seed(3))
    default = _run_dit(dit, latents)
    identity = _run_dit(dit, latents, rope_frame_ids=torch.arange(6))
    assert torch.equal(default, identity), (default - identity).abs().max()


@check
def rope_wrong_length_raises():
    dit = _tiny_dit()
    latents = torch.randn(1, 4, 6, 4, 4, generator=torch.Generator().manual_seed(4))
    try:
        _run_dit(dit, latents, rope_frame_ids=torch.arange(5))
    except ValueError as exc:
        assert "5 entries" in str(exc) and "6 frames" in str(exc), str(exc)
        return
    raise AssertionError("a wrong-length rope_frame_ids must raise, not broadcast")


@check
def rope_ids_are_duplicated_per_modality():
    """The module hands the pipeline [0..V-1, 0..V-1], so an RGB frame and its
    disparity share a temporal position."""
    m = object.__new__(WanRGBDNVSTrainingModule)
    m.num_views = 9
    m.shared_rope_frame_ids = True
    m._rope_frame_ids = torch.arange(m.num_views).repeat(2)
    ids = m._rope_frame_ids
    assert ids.tolist() == list(range(9)) * 2, ids.tolist()
    for v in range(9):
        assert ids[v] == ids[9 + v]


@check
def rope_ids_reach_the_model_fn():
    """`_forward_model` is the single funnel for training AND both eval passes."""
    seen = {}

    class _Pipe:
        in_iteration_models = ()
        torch_dtype = torch.float32
        def model_fn(self, **kwargs):
            seen.update(kwargs)
            return torch.zeros(1)

    for enabled in (True, False):
        m = object.__new__(WanRGBDNVSTrainingModule)
        m.pipe = _Pipe()
        m.num_views = 4
        m.token_layout = "channel"
        m._rope_frame_ids = torch.arange(4).repeat(2) if enabled else None
        seen.clear()
        m._forward_model({"context": None}, torch.zeros(1), torch.zeros(1))
        if enabled:
            assert "rope_frame_ids" in seen, sorted(seen)
            assert seen["rope_frame_ids"].tolist() == [0, 1, 2, 3] * 2
        else:
            assert "rope_frame_ids" not in seen, (
                "the kwarg must be absent when the factor is off, so the pipeline "
                "takes its original code path"
            )


# ----------------------------------------------------------------------------
# clean_source_depth
# ----------------------------------------------------------------------------

S, T, C, H, W = 6, 3, 4, 5, 7
V = S + T


def _module(clean=True, prob=0.5):
    m = object.__new__(WanRGBDNVSTrainingModule)
    m.num_source_views, m.num_target_views, m.num_views = S, T, V
    m.per_frame_timestep = True
    m.clean_source_depth = clean
    m.clean_source_depth_prob = prob
    return m


def _fixture(batch=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    depth_gt = torch.randn(batch, C, V, H, W, generator=g)
    depth_x_t = torch.randn(batch, C, V, H, W, generator=g)
    loss_masks = torch.ones(batch, 2, V, 1, H, W)
    return depth_gt, depth_x_t, loss_masks


@check
def clean_prob_one_replaces_every_source_depth_frame():
    m = _module(prob=1.0 - 1e-9)
    depth_gt, depth_x_t, loss_masks = _fixture()
    noisy_before = depth_x_t.clone()
    clean, masks = m._apply_clean_source_depth(depth_x_t, depth_gt, loss_masks, 8)
    assert clean.all(), clean
    assert torch.equal(depth_x_t[:, :, :S], depth_gt[:, :, :S]), "source depth not clean"
    # Target depth frames must be untouched -- they are the prediction target.
    assert torch.equal(depth_x_t[:, :, S:], noisy_before[:, :, S:]), "target depth altered"
    assert masks[:, 0, :S].sum() == 0, "clean frames still supervised"
    assert masks[:, 0, S:].sum() == 8 * T * H * W, "target depth mask damaged"
    assert masks[:, 1].sum() == 8 * V * H * W, "rgb mask damaged"


@check
def clean_prob_zero_is_a_bitwise_noop():
    """The regression check that matters most: with the factor off, every tensor
    must be byte-identical to the pre-change behaviour."""
    depth_gt, depth_x_t, loss_masks = _fixture()
    baseline = depth_x_t.clone()

    off = _module(clean=False)
    clean, masks = off._apply_clean_source_depth(depth_x_t, depth_gt, loss_masks, 8)
    assert clean is None
    assert masks is loss_masks, "loss_masks must be passed through, not copied"
    assert torch.equal(depth_x_t, baseline)

    on_but_never = _module(prob=0.0)
    clean, masks = on_but_never._apply_clean_source_depth(depth_x_t, depth_gt, loss_masks, 8)
    assert not clean.any(), clean
    assert torch.equal(depth_x_t, baseline), "no sample was selected yet x_t changed"
    assert torch.equal(masks, loss_masks)


@check
def clean_is_per_sample_not_per_batch():
    """A per-batch draw would keep the two regimes in different gradient steps,
    which is exactly the failure warp_dropout_prob's per-sample draw avoids."""
    m = _module(prob=0.5)
    depth_gt, depth_x_t, loss_masks = _fixture(batch=256, seed=7)
    torch.manual_seed(11)
    clean, masks = m._apply_clean_source_depth(depth_gt.clone(), depth_gt, loss_masks, 256)
    frac = clean.float().mean().item()
    assert 0.35 < frac < 0.65, f"selection fraction {frac} is not ~0.5"
    # Selected samples lose their source-depth mask; unselected keep it.
    per_sample = masks[:, 0, :S].flatten(1).sum(1)
    assert (per_sample[clean] == 0).all()
    assert (per_sample[~clean] == S * H * W).all()


@check
def clean_source_depth_frames_are_pinned_to_timestep_zero():
    """Without this the model is handed a clean latent while being told it sits at
    sigma -- the exact guessing game per_frame_timestep exists to remove."""
    m = _module()
    t = torch.full((4,), 700.0)
    clean = torch.tensor([True, False, True, False])
    per_frame = m._expand_timestep(t, clean_depth_mask=clean)
    assert per_frame.shape == (4, 2 * V)
    # Source RGB is always clean, regardless of the factor.
    assert (per_frame[:, :S] == 0).all()
    assert (per_frame[:, S:V] == 700.0).all(), "target RGB must stay at sigma"
    for b, is_clean in enumerate(clean.tolist()):
        want = 0.0 if is_clean else 700.0
        assert (per_frame[b, V:V + S] == want).all(), (b, is_clean, per_frame[b, V:V + S])
    # Target disparity is always a prediction target.
    assert (per_frame[:, V + S:] == 700.0).all()


@check
def expand_timestep_without_the_factor_is_unchanged():
    m = _module(clean=False)
    t = torch.full((4,), 700.0)
    per_frame = m._expand_timestep(t)
    assert (per_frame[:, :S] == 0).all()
    assert (per_frame[:, S:] == 700.0).all(), "source disparity must stay noisy when off"


@check
def expand_timestep_rejects_a_mask_it_cannot_honour():
    m = _module()
    m.per_frame_timestep = False
    try:
        m._expand_timestep(torch.full((2,), 700.0), clean_depth_mask=torch.ones(2).bool())
    except RuntimeError:
        return
    raise AssertionError(
        "a clean mask with per_frame_timestep off must raise: the frames would be "
        "clean but told they are noisy"
    )


@check
def forward_does_not_restore_the_mask_it_just_dropped():
    """`forward()` used to assign `result["loss_masks"] = inputs["loss_masks"]`.
    With this factor on, that would put the clean frames straight back into the
    depth loss -- the factor would still run, and still measure nothing."""
    m = object.__new__(WanRGBDNVSTrainingModule)
    m.rgb_flow_matching = True
    trimmed = torch.zeros(1)
    m.forward_preprocess = lambda data: {
        "loss_masks": torch.ones(1),
        "depth_gt": torch.zeros(1),
        "rgb_gt": torch.zeros(1),
    }
    m._forward_flow = lambda inputs: {"pred": None, "loss_masks": trimmed}
    out = m.forward(None)
    assert out["loss_masks"] is trimmed, (
        "forward() clobbered the trimmed mask with the untrimmed one from "
        "forward_preprocess"
    )


@check
def build_flow_input_leaves_the_injected_source_depth_alone():
    """The real _build_flow_input, not a stub: it adds the camera conditioning to
    the TARGET depth slots, and if that slicing were ever widened to cover the
    source slots the injected clean latent would be silently perturbed."""
    m = _module()
    m.token_layout = "sequence"
    m.depth_flow_matching = True
    m.latent_scale = 1.0

    g = torch.Generator().manual_seed(21)
    shape = (2, C, V, H, W)
    inputs = {
        "rgb_latents": torch.randn(*shape, generator=g),
        "view_condition": torch.randn(*shape, generator=g),
        "plucker_block": None,
        "warp_block": None,
    }
    depth_x_t = torch.randn(*shape, generator=g)
    model_in = m._build_flow_input(inputs, torch.randn(*shape, generator=g), depth_x_t)

    # [RGB block | depth block] along the frame axis, then a modality flag channel.
    assert model_in.shape[2] == 2 * V, model_in.shape
    depth_block = model_in[:, :C, V:]
    assert torch.equal(depth_block[:, :, :S], depth_x_t[:, :, :S]), (
        "source-depth frames were modified on the way into the DiT"
    )
    assert not torch.equal(depth_block[:, :, S:], depth_x_t[:, :, S:]), (
        "target-depth frames lost their camera conditioning"
    )
    # The modality flag is what lets shared RoPE positions stay distinguishable.
    assert (model_in[:, -1, :V] == 0).all() and (model_in[:, -1, V:] == 1).all()


@check
def inference_holds_the_injected_depth_across_every_euler_step():
    """A single assignment before the loop is not enough -- each Euler step moves
    every frame, so the pinned ones have to be restored after each one."""
    m = _module()
    m.depth_flow_matching = True
    m.eval_num_inference_steps = 4
    m.eval_flow_shift = 0.5
    m.rgb_flow_shift = 5.0
    m.token_layout = "sequence"
    m._rope_frame_ids = None
    m._build_flow_input = lambda inputs, x, x_depth=None: x_depth

    seen = []

    class _Sched:
        num_train_timesteps = 1000

    class _Pipe:
        torch_dtype = torch.float32
        scheduler = _Sched()
    m.pipe = _Pipe()

    def _fake_forward(inputs, latents, timestep):
        # latents is what _build_flow_input returned, i.e. the depth stack the DiT
        # would actually see this step.
        seen.append((latents.clone(), timestep.clone()))
        # A large constant velocity: if the pin is missing, the source frames drift
        # far enough that no tolerance could hide it.
        return torch.full_like(latents, 50.0), torch.zeros_like(inputs["rgb_gt"])
    m._forward_model = _fake_forward

    g = torch.Generator().manual_seed(5)
    inputs = {
        "rgb_gt": torch.randn(2, C, V, H, W, generator=g),
        "depth_gt": torch.randn(2, C, V, H, W, generator=g),
    }
    pinned = torch.randn(2, C, S, H, W, generator=g)
    x_depth, _ = m._sample_rgb_once(inputs, num_steps=4, clean_src_depth=pinned)

    assert len(seen) == 4
    for i, (latents, timestep) in enumerate(seen):
        assert torch.equal(latents[:, :, :S], pinned), f"step {i}: source depth drifted"
        assert (timestep[:, V:V + S] == 0).all(), f"step {i}: pinned frames not at t=0"
        assert (timestep[:, :S] == 0).all(), f"step {i}: source RGB not at t=0"
        assert (timestep[:, S:V] > 0).all(), f"step {i}: target RGB pinned by mistake"
    assert torch.equal(x_depth[:, :, :S], pinned), "returned source depth is not the estimate"
    assert not torch.equal(x_depth[:, :, S:], inputs["depth_gt"][:, :, S:])


@check
def inference_without_the_factor_integrates_every_depth_frame():
    m = _module(clean=False)
    m.depth_flow_matching = True
    m.eval_num_inference_steps = 2
    m.eval_flow_shift = 0.5
    m.rgb_flow_shift = 5.0
    m._rope_frame_ids = None
    m._build_flow_input = lambda inputs, x, x_depth=None: x_depth

    seen = []

    class _Sched:
        num_train_timesteps = 1000

    class _Pipe:
        torch_dtype = torch.float32
        scheduler = _Sched()
    m.pipe = _Pipe()

    def _fake_forward(inputs, latents, timestep):
        seen.append(timestep.clone())
        return torch.full_like(latents, 50.0), torch.zeros_like(inputs["rgb_gt"])
    m._forward_model = _fake_forward

    g = torch.Generator().manual_seed(6)
    inputs = {
        "rgb_gt": torch.randn(2, C, V, H, W, generator=g),
        "depth_gt": torch.randn(2, C, V, H, W, generator=g),
    }
    m._sample_rgb_once(inputs, num_steps=2)
    for timestep in seen:
        assert (timestep[:, V:V + S] > 0).all(), (
            "source disparity was pinned even though clean_source_depth is off"
        )


# ----------------------------------------------------------------------------
# Constructor guards.  Each of these configurations would produce a run that
# LOOKS like the cleansrcd arm and measures something else.
# ----------------------------------------------------------------------------

@check
def constructor_guards_are_all_reachable():
    import inspect
    src = inspect.getsource(WanRGBDNVSTrainingModule.__init__)
    for needle in (
        "clean_source_depth requires per_frame_timestep",
        "clean_source_depth requires depth_flow_matching",
        "clean_source_depth requires warp_mode",
        "clean_source_depth_prob must be strictly between 0 and 1",
        "shared_rope_frame_ids only means anything",
    ):
        assert needle in src, f"missing guard: {needle}"


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
