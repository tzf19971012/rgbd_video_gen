"""CPU-only unit tests for the three D-sweep factors (rope-geometry plan §D1-§D3).

Same reason the c-sweep needed tools/test_c_sweep_factors.py: every factor is
SILENT when wrong.  A `rope_frame_axis` that never reaches the freqs table trains
an exact duplicate of the arm it is supposed to be read against, a `view_order`
that also disturbs the view SELECTION quietly unfreezes the eval set and breaks
pairing with every number measured before it, and a CaPE that is fed c2w instead
of w2c encodes "which view happens to be view 0" instead of the relative pose.
None of them raise, all of them converge, all of them produce a full report.  So
every assertion here targets a failure that would not show up during a 40-hour
run.

Nothing here loads Wan2.1.  The RoPE tests build a tiny real WanModel and call the
real `model_fn_wan_video`; the ordering tests call the real
`_order_source_views_by_pose` and bind the real `_select_views` to a bare object
carrying only the attributes it touches, so the code under test is the shipped
code and not a copy of it.

  python -m tools.test_d_sweep_factors
"""
import contextlib
import inspect
import itertools

import torch

from diffsynth.models import wan_video_dit
from diffsynth.models.wan_video_dit import (
    ROCE_MODES,
    WanModel,
    cape_apply,
    num_frame_rope_dims,
    num_hw_rope_phase_dims,
    precompute_freqs_cis_3d,
    rope_apply,
)
from diffsynth.pipelines.wan_video_new_determine import model_fn_wan_video
from examples.dataset.kubric4d_rgbd_nvs_dataset import (
    Kubric4DRGBDNVSDataset,
    _ViewRecord,
    _order_source_views_by_pose,
)
from examples.wanvideo.model_training.WanRGBDNVSTrainingModule import (
    ROPE_FRAME_AXES,
    WanRGBDNVSTrainingModule,
    cape_pose_matrices,
    relative_c2w,
    resolve_roce_mode,
    resolve_rope_frame_axis,
    resolve_rope_pose_translation_scale,
    roce_camera_latents,
    rope_frame_ids_for_axis,
)
from utils.warp import splat_sources_to_targets

CHECKS = []

# flash-attn has no CPU kernel, so `flash_attention` has to fall through to
# F.scaled_dot_product_attention.  Same reasoning as in test_c_sweep_factors: this
# changes where attention is computed, not what the RoPE test measures.
wan_video_dit.FLASH_ATTN_3_AVAILABLE = False
wan_video_dit.FLASH_ATTN_2_AVAILABLE = False
wan_video_dit.SAGE_ATTN_AVAILABLE = False


def check(fn):
    CHECKS.append(fn)
    return fn


# ----------------------------------------------------------------------------
# D1 -- rope_frame_axis
# ----------------------------------------------------------------------------

@check
def axis_ids_are_what_the_documentation_says():
    V = 9
    assert rope_frame_ids_for_axis("view_modality", V) is None, (
        "the default must return None, not arange(2V): passing the override at all "
        "takes the pipeline down a different code path and stops being bit-identical "
        "to every run measured before the override existed"
    )
    assert rope_frame_ids_for_axis("view", V).tolist() == list(range(V)) * 2
    assert rope_frame_ids_for_axis("modality", V).tolist() == [0] * V + [1] * V
    assert rope_frame_ids_for_axis("constant", V).tolist() == [0] * (2 * V)
    # `pose` must be all-ZERO, not merely constant: CaPE composes with whatever
    # the frame RoPE block already applied, and only freqs[0][0] is 1+0j.  A
    # constant-but-nonzero id would leave a real rotation underneath the SE(3)
    # matrix and silently break the exact relative-pose identity.
    assert rope_frame_ids_for_axis("pose", V).tolist() == [0] * (2 * V)
    for axis in ROPE_FRAME_AXES:
        ids = rope_frame_ids_for_axis(axis, V)
        if ids is not None:
            assert ids.numel() == 2 * V, (axis, ids.numel())


@check
def view_axis_is_bit_identical_to_the_sharedrope_arm():
    """`view` must reproduce the c-sweep factor exactly.

    The whole D round is read against those checkpoints; if this drifts, `d2`
    differs from `sharedrope` in two ways and measures neither.
    """
    for V in (3, 4, 9):
        assert torch.equal(
            rope_frame_ids_for_axis("view", V), torch.arange(V).repeat(2)
        )


@check
def d1_axes_erase_view_identity():
    """The point of D1: no two views may be positionally distinguishable."""
    V = 9
    for axis in ("modality", "constant"):
        ids = rope_frame_ids_for_axis(axis, V).tolist()
        assert len(set(ids[:V])) == 1, (axis, ids[:V])
        assert len(set(ids[V:])) == 1, (axis, ids[V:])
    # ...and the one thing that separates them: `modality` keeps an offset
    # between the two halves, `constant` does not.  This is why D1 uses
    # `constant` against the sharedrope anchor, whose modality offset is zero.
    mod = rope_frame_ids_for_axis("modality", V)
    const = rope_frame_ids_for_axis("constant", V)
    assert mod[0] != mod[V] and const[0] == const[V]


@check
def unknown_axis_raises_in_both_entry_points():
    for fn, args in (
        (rope_frame_ids_for_axis, ("veiw", 9)),
        (resolve_rope_frame_axis, ("veiw", False)),
    ):
        try:
            fn(*args)
        except ValueError as exc:
            assert "rope_frame_axis" in str(exc), str(exc)
        else:
            raise AssertionError(f"{fn.__name__} accepted a typo'd axis name")


@check
def shared_rope_frame_ids_stays_an_exact_alias():
    """Every c-sweep checkpoint's config spells the factor as the bool.

    eval_rgbd_nvs / diag_posterior_mean rebuild the model from that config, and a
    RoPE mismatch there is silent -- it already cost one posterior run 1.45 dB of
    harness artefact.  So the bool must keep resolving to `view`, and a
    contradiction must raise rather than silently pick one.
    """
    assert resolve_rope_frame_axis(None, True) == "view"
    assert resolve_rope_frame_axis(None, False) == "view_modality"
    assert resolve_rope_frame_axis("view", True) == "view"
    assert resolve_rope_frame_axis("constant", False) == "constant"
    try:
        resolve_rope_frame_axis("constant", True)
    except ValueError as exc:
        assert "exactly one" in str(exc), str(exc)
        return
    raise AssertionError(
        "shared_rope_frame_ids=True together with a different rope_frame_axis must "
        "raise: silently honouring one of them is how an arm trains something other "
        "than its name"
    )


def _tiny_dit():
    # head_dim must be divisible by 6 -- see test_c_sweep_factors._tiny_dit for
    # why 24 is the smallest workable size.
    torch.manual_seed(0)
    return WanModel(
        dim=48, in_dim=4, ffn_dim=64, out_dim=4, text_dim=16, freq_dim=48,
        eps=1e-6, patch_size=(1, 2, 2), num_heads=2, num_layers=1,
        has_image_input=False,
    ).eval()


def _run_dit(dit, latents, rope_frame_ids=None, rope_pose_matrices=None,
             roce_camera_latents=None):
    return model_fn_wan_video(
        dit=dit,
        latents=latents,
        timestep=torch.tensor([500.0]),
        context=torch.randn(1, 3, 16, generator=torch.Generator().manual_seed(1)),
        rope_frame_ids=rope_frame_ids,
        rope_pose_matrices=rope_pose_matrices,
        roce_camera_latents=roce_camera_latents,
    )


def _random_se3(n, seed=0, translation=1.0):
    """n random SE(3) matrices, as c2w."""
    g = torch.Generator().manual_seed(seed)
    out = torch.eye(4).repeat(n, 1, 1)
    for i in range(n):
        q, r = torch.linalg.qr(torch.randn(3, 3, generator=g))
        q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
        if torch.det(q) < 0:
            q[:, 0] = -q[:, 0]
        out[i, :3, :3] = q
        out[i, :3, 3] = torch.randn(3, generator=g) * translation
    return out


def _lookat_c2w(n, seed=0, distance=4.0, baseline=0.15):
    """n c2w matrices on a small patch of a sphere, all aimed at the world origin.

    `_random_se3` is useless for anything that has to actually *project*: with an
    independent random rotation per camera, a target view almost never contains the
    source content, and a splat test built on it silently compares two all-zero
    tensors.  This instead reproduces the regime plan §10's cancellation lives in --
    cameras a long way from the origin (`distance`) and close to each other
    (`baseline`) -- while guaranteeing a shared field of view.

    OpenCV convention, matching `_pixel_rays_to_world`: camera x right, y down,
    z forward, so a point at the origin projects to the principal point.
    """
    g = torch.Generator().manual_seed(seed)
    out = torch.eye(4).repeat(n, 1, 1)
    base = torch.tensor([0.0, 0.0, -1.0]) * distance
    for i in range(n):
        eye = base + torch.randn(3, generator=g) * baseline
        f = -eye / eye.norm()                                # look at the origin
        up = torch.tensor([0.0, -1.0, 0.0])
        r = torch.linalg.cross(up, f)
        r = r / r.norm()
        d = torch.linalg.cross(f, r)
        out[i, :3, :3] = torch.stack([r, d, f], dim=1)        # columns
        out[i, :3, 3] = eye
    return out


def _pose_for_axis(axis, V, translation_scale=0.2, seed=11, translation=1.0):
    """The `cape_pose` a training step would build, or None for the other axes."""
    if axis != "pose":
        return None
    c2w = _random_se3(V, seed=seed, translation=translation).unsqueeze(0)
    c2w_rel = torch.linalg.inv(c2w[:, 0:1]) @ c2w
    return cape_pose_matrices(c2w_rel, translation_scale)


@check
def every_axis_is_a_distinct_attention_geometry():
    """All five settings must produce five different outputs on a real DiT.

    Two failures hide here.  One: the ids never reach the freqs table, and D1 is
    a duplicate of its anchor.  Two: `constant` accidentally coincides with the
    default (it does not -- position 0 for all frames is not position 0..2V-1),
    which would make a null result unreadable.  `pose` shares its ids with
    `constant`, so the only thing that can separate those two is the CaPE matrices
    actually arriving -- which is exactly the D3 plumbing under test.
    """
    dit = _tiny_dit()
    V = 3
    latents = torch.randn(1, 4, 2 * V, 4, 4, generator=torch.Generator().manual_seed(2))
    outs = {}
    for axis in ROPE_FRAME_AXES:
        outs[axis] = _run_dit(dit, latents,
                              rope_frame_ids=rope_frame_ids_for_axis(axis, V),
                              rope_pose_matrices=_pose_for_axis(axis, V))
    for a, b in itertools.combinations(ROPE_FRAME_AXES, 2):
        assert outs[a].shape == outs[b].shape
        assert not torch.allclose(outs[a], outs[b], atol=1e-5), (
            f"rope_frame_axis={a!r} and {b!r} give the same output -- one of them is "
            f"not reaching the RoPE table"
        )


@check
def axis_ids_reach_the_model_fn():
    """`_forward_model` is the single funnel for training AND both eval passes."""
    seen = {}

    class _Pipe:
        in_iteration_models = ()
        torch_dtype = torch.float32

        def model_fn(self, **kwargs):
            seen.update(kwargs)
            return torch.zeros(1)

    for axis in ROPE_FRAME_AXES:
        m = object.__new__(WanRGBDNVSTrainingModule)
        m.pipe = _Pipe()
        m.num_views = 4
        m.token_layout = "channel"
        m._rope_frame_ids = rope_frame_ids_for_axis(axis, m.num_views)
        seen.clear()
        m._forward_model({"context": None}, torch.zeros(1), torch.zeros(1))
        if axis == "view_modality":
            assert "rope_frame_ids" not in seen, (
                "the kwarg must be absent for the default axis, so the pipeline takes "
                "its original code path"
            )
        else:
            assert "rope_frame_ids" in seen, (axis, sorted(seen))
            assert seen["rope_frame_ids"].tolist() == \
                rope_frame_ids_for_axis(axis, m.num_views).tolist()


@check
def rope_constructor_guards_are_all_reachable():
    src = inspect.getsource(WanRGBDNVSTrainingModule.__init__)
    for needle in (
        "shared_rope_frame_ids only means anything",
        "only means anything with",
        "resolve_rope_frame_axis(rope_frame_axis, shared_rope_frame_ids)",
        "resolve_rope_pose_translation_scale(",
    ):
        assert needle in src, f"missing guard: {needle}"


# ----------------------------------------------------------------------------
# D3 -- rope_frame_axis="pose": EscherNet 6-DoF CaPE on the frame block
# ----------------------------------------------------------------------------

def _cape_logits(q, k, pose):
    """The V x V attention-logit matrix CaPE produces, via the shipped cape_apply.

    q, k are [V, D] -- one token per frame, one head -- so cape_apply sees
    [1, V, D] with frames=V and tokens_per_frame=1.
    """
    frame_dims = num_frame_rope_dims(q.shape[-1])
    pose_q = torch.linalg.inv(pose).transpose(-1, -2)
    qq = cape_apply(q.unsqueeze(0), pose_q.unsqueeze(0), 1, q.shape[0], frame_dims)[0]
    kk = cape_apply(k.unsqueeze(0), pose.unsqueeze(0), 1, k.shape[0], frame_dims)[0]
    return qq @ kk.T


@check
def frame_rope_dims_match_the_freqs_table():
    """num_frame_rope_dims must mirror precompute_freqs_cis_3d, in REAL dims.

    Off by a factor of two (complex channels vs real dims) and CaPE would write
    over half the height block while leaving half the frame block rotary -- which
    is a perfectly trainable, completely wrong encoding.
    """
    for head_dim in (24, 48, 128):
        n = num_frame_rope_dims(head_dim)
        f, h, w = precompute_freqs_cis_3d(head_dim)
        assert f.shape[-1] * 2 == n, (head_dim, f.shape[-1], n)
        assert 2 * (f.shape[-1] + h.shape[-1] + w.shape[-1]) == head_dim, head_dim
        assert n % 4 == 0, f"head_dim {head_dim} gives {n} frame dims, not a multiple of 4"


@check
def cape_logit_is_exactly_the_relative_pose():
    """q'.k' == sum_groups g_q (P_k P_q^-1)^T g_k, for real SE(3) matrices.

    This is the whole reason CaPE was chosen over a phase encoding of the camera
    centre, and it is what fails silently if the query/key roles are swapped (P on
    the query instead of P^-T): the logit stays smooth, stays a function of the
    poses, and stops being a function of the relative pose alone.
    """
    D = 24
    V = 4
    g = torch.Generator().manual_seed(5)
    q = torch.randn(V, D, generator=g, dtype=torch.float32)
    k = torch.randn(V, D, generator=g, dtype=torch.float32)
    pose = torch.linalg.inv(_random_se3(V, seed=6))          # w2c
    got = _cape_logits(q, k, pose)

    frame_dims = num_frame_rope_dims(D)
    want = torch.zeros(V, V)
    for i in range(V):
        for j in range(V):
            rel = pose[j] @ torch.linalg.inv(pose[i])
            gq = q[i, :frame_dims].reshape(-1, 4)
            gk = k[j, :frame_dims].reshape(-1, 4)
            want[i, j] = (gq @ rel.T * gk).sum() + (q[i, frame_dims:] * k[j, frame_dims:]).sum()
    assert torch.allclose(got, want, atol=1e-4), (got - want).abs().max()


@check
def cape_on_w2c_is_blind_to_the_world_frame():
    """x_world -> G x_world sends w2c -> w2c G^-1 and must leave every logit alone.

    Measured on real BlendedMVS poses: 0.0000 relative error for w2c, 1.77 for c2w
    (tools/probe_pose_rope_scale.py §3).  Feeding c2w would make the encoding
    depend on which view happens to be view 0 -- which D2 permutes.
    """
    D = 24
    V = 4
    g = torch.Generator().manual_seed(7)
    q = torch.randn(V, D, generator=g)
    k = torch.randn(V, D, generator=g)
    c2w = _random_se3(V, seed=8)
    G = _random_se3(1, seed=9, translation=3.0)[0]

    w2c = torch.linalg.inv(c2w)
    ref = _cape_logits(q, k, w2c)
    moved = _cape_logits(q, k, w2c @ torch.linalg.inv(G))
    assert torch.allclose(ref, moved, atol=1e-3), (ref - moved).abs().max()

    # ...and the c2w form must NOT be invariant, otherwise this test would pass
    # for a broken implementation that ignores the pose entirely.
    ref_c = _cape_logits(q, k, c2w)
    moved_c = _cape_logits(q, k, G @ c2w)
    assert not torch.allclose(ref_c, moved_c, atol=1e-3)


@check
def rotation_only_cape_is_orthogonal():
    """lambda=0 must preserve ||q||, ||k|| exactly: it is R (+) 1 on 4-vectors.

    That is the d3rot arm's entire safety argument -- Wan's pretrained blocks have
    only ever seen unit-modulus factors on these dims.
    """
    D = 24
    V = 5
    g = torch.Generator().manual_seed(10)
    q = torch.randn(V, D, generator=g)
    c2w_rel = _random_se3(V, seed=12, translation=5.0).unsqueeze(0)
    pose0 = cape_pose_matrices(c2w_rel, 0.0)[0, :V]
    out = cape_apply(q.unsqueeze(0), pose0.unsqueeze(0), 1, V, num_frame_rope_dims(D))[0]
    assert torch.allclose(out.norm(dim=-1), q.norm(dim=-1), atol=1e-5)

    # ...and lambda=1 must NOT, so the inflation this project caps with lambda=0.2
    # is a real effect and not an artefact of the probe.
    pose1 = cape_pose_matrices(c2w_rel, 1.0)[0, :V]
    out1 = cape_apply(q.unsqueeze(0), pose1.unsqueeze(0), 1, V, num_frame_rope_dims(D))[0]
    assert out1.norm(dim=-1).max() > q.norm(dim=-1).max() * 1.2


@check
def cape_survives_bf16_at_the_real_head_width():
    """Training runs in bf16, and plan §10 already found one bf16 pose defect.

    The risk here is different from §10's: CaPE's own matmul is done in fp32
    inside cape_apply, so what is at stake is only the cast back to bf16 and the
    attention matmul.  Measured at head_dim 128 with the baselines the probe found
    on real data, the logit error is ~0.3% of full scale at every lambda -- the
    same order as a plain bf16 dot product, i.e. CaPE adds no bf16 hazard.  A
    regression that dropped the internal fp32 cast would show up here.
    """
    D, V = 128, 9
    frame_dims = num_frame_rope_dims(D)
    g = torch.Generator().manual_seed(19)
    q = torch.randn(V, D, generator=g)
    k = torch.randn(V, D, generator=g)
    c2w = _random_se3(V, seed=20, translation=2.0).unsqueeze(0)
    c2w = torch.linalg.inv(c2w[:, 0:1]) @ c2w
    for lam in (0.0, 0.2, 1.0):
        P = cape_pose_matrices(c2w, lam)[:, :V]
        Pq = torch.linalg.inv(P).transpose(-1, -2)

        def logits(dtype):
            qq = cape_apply(q.to(dtype).unsqueeze(0), Pq, 1, V, frame_dims)[0]
            kk = cape_apply(k.to(dtype).unsqueeze(0), P, 1, V, frame_dims)[0]
            return qq.float() @ kk.float().T

        exact = logits(torch.float32)
        rel = (exact - logits(torch.bfloat16)).abs().max() / exact.abs().max()
        assert rel < 0.01, (lam, float(rel))


@check
def cape_applies_each_pose_to_its_own_frame():
    """The frame-major token view is an ASSUMPTION about patchify, not a checked fact.

    If it were wrong -- or if the reshape were (tokens, frames) instead of
    (frames, tokens) -- every token would get some other view's pose and the model
    would still train.  P_v = c_v * I isolates that: it must scale frame v's frame
    dims by c_v and touch nothing else.
    """
    b, V, tokens, heads, D = 1, 3, 5, 2, 24
    frame_dims = num_frame_rope_dims(D)
    x = torch.randn(b, V * tokens, heads * D, generator=torch.Generator().manual_seed(13))
    scales = torch.tensor([2.0, 3.0, 5.0])
    pose = torch.eye(4).repeat(b, V, 1, 1) * scales.view(1, V, 1, 1)
    out = cape_apply(x, pose, heads, V, frame_dims)

    xv = x.reshape(b, V, tokens, heads, D)
    ov = out.reshape(b, V, tokens, heads, D)
    for v in range(V):
        assert torch.allclose(ov[:, v, ..., :frame_dims],
                              xv[:, v, ..., :frame_dims] * scales[v], atol=1e-5), v
        assert torch.allclose(ov[:, v, ..., frame_dims:],
                              xv[:, v, ..., frame_dims:], atol=1e-6), (
            f"frame {v}: CaPE wrote outside the frame block, i.e. into the "
            f"height/width RoPE dims"
        )


@check
def pose_axis_degenerates_to_the_constant_arm():
    """lambda=0 with identity rotations must reproduce axis='constant' exactly.

    This is what makes d1/`constant` the right anchor for reading d3: the two arms
    differ by a continuous knob that starts at the anchor.  If they differed by
    anything else, a d3 win would be unattributable.
    """
    dit = _tiny_dit()
    V = 3
    latents = torch.randn(1, 4, 2 * V, 4, 4, generator=torch.Generator().manual_seed(14))
    ids = rope_frame_ids_for_axis("pose", V)
    const = _run_dit(dit, latents, rope_frame_ids=ids)
    identity = _run_dit(dit, latents, rope_frame_ids=ids,
                        rope_pose_matrices=torch.eye(4).repeat(1, 2 * V, 1, 1))
    assert torch.allclose(const, identity, atol=1e-6), (const - identity).abs().max()

    # ...and lambda must be a live knob: rotation-only, 0.2 and 1.0 are three
    # different geometries, none of them equal to the anchor.
    outs = [const]
    for lam in (0.0, 0.2, 1.0):
        outs.append(_run_dit(dit, latents, rope_frame_ids=ids,
                             rope_pose_matrices=_pose_for_axis("pose", V, lam)))
    for a in range(len(outs)):
        for b in range(a + 1, len(outs)):
            assert not torch.allclose(outs[a], outs[b], atol=1e-5), (a, b)


@check
def model_fn_refuses_to_stack_cape_on_a_live_frame_rope():
    """CaPE and the rotary frame block fight over real dims 0..43.

    A 2x2 rotation and a 4x4 SE(3) matrix do not commute, so stacking them
    destroys the exactness identity -- and nothing downstream can tell.  The
    pipeline has to refuse, because the training module could be wired to `view`
    while passing poses and the run would look fine.
    """
    dit = _tiny_dit()
    V = 3
    latents = torch.randn(1, 4, 2 * V, 4, 4, generator=torch.Generator().manual_seed(15))
    pose = _pose_for_axis("pose", V)
    cases = [
        (None, "rope_frame_ids"),                             # default 0..2V-1 ladder
        (torch.arange(2 * V), "min="),                        # a real ladder
        (torch.ones(2 * V, dtype=torch.long), "== 0"),        # constant but not 1+0j
    ]
    for ids, needle in cases:
        try:
            _run_dit(dit, latents, rope_frame_ids=ids, rope_pose_matrices=pose)
        except ValueError as exc:
            assert needle in str(exc), (ids, str(exc))
        else:
            raise AssertionError(
                f"rope_frame_ids={ids} was accepted together with CaPE"
            )
    # Frame-count and shape mismatches must raise too: with **kwargs in the
    # signature there is no other line of defence.
    ids = rope_frame_ids_for_axis("pose", V)
    for bad in (pose[:, : 2 * V - 1], pose[..., :3]):
        try:
            _run_dit(dit, latents, rope_frame_ids=ids, rope_pose_matrices=bad)
        except (ValueError, RuntimeError):
            pass
        else:
            raise AssertionError(f"a {tuple(bad.shape)} pose tensor was accepted")


@check
def cape_pose_matrices_are_world_to_camera_and_shared_across_modalities():
    V = 4
    c2w_rel = _random_se3(V, seed=16, translation=2.0).unsqueeze(0)
    c2w_rel = torch.linalg.inv(c2w_rel[:, 0:1]) @ c2w_rel
    P = cape_pose_matrices(c2w_rel, 1.0)
    assert P.shape == (1, 2 * V, 4, 4), P.shape
    eye = torch.eye(4).expand(1, V, 4, 4)
    assert torch.allclose(P[:, :V] @ c2w_rel, eye, atol=1e-4), "not world-to-camera"
    assert torch.equal(P[:, V:], P[:, :V]), (
        "view v's disparity frame must carry view v's pose; the sequence layout is "
        "[RGB x V, disparity x V], so a wrong tile would pair each disparity with "
        "another view's camera"
    )
    # lambda touches the translation column and nothing else.
    P2 = cape_pose_matrices(c2w_rel, 0.2)
    assert torch.allclose(P2[..., :3, :3], P[..., :3, :3], atol=1e-6)
    assert torch.allclose(P2[..., :3, 3], 0.2 * P[..., :3, 3], atol=1e-6)
    assert torch.allclose(P2[..., 3, :], P[..., 3, :], atol=1e-6)
    # lambda=0 is exactly orthogonal.
    P0 = cape_pose_matrices(c2w_rel, 0.0)
    assert torch.allclose(P0 @ P0.transpose(-1, -2),
                          torch.eye(4).expand(1, 2 * V, 4, 4), atol=1e-5)
    # Referencing to view 0 makes the matrices independent of the world frame, so
    # the arm cannot be sensitive to a global rescale/rotation of the scene.
    G = _random_se3(1, seed=17, translation=9.0)
    c2w = _random_se3(V, seed=18, translation=2.0).unsqueeze(0)
    rel_a = torch.linalg.inv(c2w[:, 0:1]) @ c2w
    moved = G.unsqueeze(0) @ c2w
    rel_b = torch.linalg.inv(moved[:, 0:1]) @ moved
    assert torch.allclose(cape_pose_matrices(rel_a, 0.2),
                          cape_pose_matrices(rel_b, 0.2), atol=1e-4)


@check
def lambda_must_be_spelled_out_by_the_pose_arm_and_by_nobody_else():
    assert resolve_rope_pose_translation_scale("pose", 0.0) == 0.0
    assert resolve_rope_pose_translation_scale("pose", 0.2) == 0.2
    assert resolve_rope_pose_translation_scale("view", None) is None
    for axis, scale, needle in (
        ("pose", None, "requires an explicit"),
        ("view", 0.2, "only used by"),
        ("constant", 0.0, "only used by"),   # 0.0 is not None -- the axis is still off
        ("pose", -0.1, ">= 0"),
    ):
        try:
            resolve_rope_pose_translation_scale(axis, scale)
        except ValueError as exc:
            assert needle in str(exc), (axis, scale, str(exc))
        else:
            raise AssertionError(f"resolve accepted axis={axis!r} scale={scale!r}")


@check
def pose_matrices_reach_the_model_fn_through_the_single_funnel():
    """`cape_pose` is built in forward_preprocess and renamed in _forward_model.

    model_fn's signature ends in **kwargs, so if the rename were missing the poses
    would be silently swallowed and the arm would train as plain `constant`.
    """
    seen = {}

    class _Pipe:
        in_iteration_models = ()
        torch_dtype = torch.float32

        def model_fn(self, **kwargs):
            seen.update(kwargs)
            return torch.zeros(1)

    V = 4
    m = object.__new__(WanRGBDNVSTrainingModule)
    m.pipe = _Pipe()
    m.num_views = V
    m.token_layout = "channel"
    m.rope_frame_axis = "pose"
    m._rope_frame_ids = rope_frame_ids_for_axis("pose", V)
    pose = _pose_for_axis("pose", V)
    m._forward_model({"context": None, "cape_pose": pose}, torch.zeros(1), torch.zeros(1))
    assert "cape_pose" not in seen, (
        "the internal key must be renamed, not passed through: model_fn would "
        "swallow it into **kwargs"
    )
    assert torch.equal(seen["rope_pose_matrices"], pose)

    # The other axes must not send it at all.
    seen.clear()
    m.rope_frame_axis = "constant"
    m._rope_frame_ids = rope_frame_ids_for_axis("constant", V)
    m._forward_model({"context": None, "cape_pose": None}, torch.zeros(1), torch.zeros(1))
    assert "rope_pose_matrices" not in seen, sorted(seen)


# ----------------------------------------------------------------------------
# D-extra -- roce: ReDirector's camera-conditioned RoPE phase on the h/w axes
# ----------------------------------------------------------------------------

ROCE_CAM_CHANNELS = 6


def _roce_dit(mode="qk_vo", phase_std=0.0, seed=0):
    """A tiny DiT with RoCE enabled; phase_std>0 un-zeroes the phase MLPs.

    At the shipped zero init the arm is bit-identical to its anchor, which is the
    point -- but it also makes every behavioural assertion vacuous, so the tests
    that are about what RoCE DOES have to break the zero init on purpose.
    """
    dit = _tiny_dit()
    dit.enable_roce(ROCE_CAM_CHANNELS, mode)
    if phase_std:
        g = torch.Generator().manual_seed(seed)
        for block in dit.blocks:
            for mlp in (block.self_attn.rope_phase_qk, block.self_attn.rope_phase_vo):
                if mlp is not None:
                    mlp[-1].weight.data.normal_(0.0, phase_std, generator=g)
    return dit


def _cam_field(frames, h=4, w=4, seed=5):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, ROCE_CAM_CHANNELS, frames, h, w, generator=g)


@check
def hw_phase_width_is_exactly_the_h_and_w_blocks():
    """The whole safety of RoCE-next-to-CaPE rests on this one number.

    rope_apply LEFT-pads the phase, so a width that is too large slides the learned
    field onto the frame block -- where CaPE lives.  Nothing about that raises
    inside attention, so the arithmetic is asserted against the real freqs tables
    rather than against ReDirector's `2 * head_dim // 3 // 2`, which is right only
    when head_dim//3 happens to be even.
    """
    for head_dim in (24, 48, 128):
        f, h, w = precompute_freqs_cis_3d(head_dim)
        assert num_hw_rope_phase_dims(head_dim) == h.shape[-1] + w.shape[-1], head_dim
        assert (
            num_hw_rope_phase_dims(head_dim) + num_frame_rope_dims(head_dim) // 2
            == head_dim // 2
        ), head_dim
    # 128 is the only width the 1.3B model runs, and it is the one where upstream's
    # formula agrees; keep the literal so a refactor cannot quietly drift off it.
    assert num_hw_rope_phase_dims(128) == 42
    assert 2 * 128 // 3 // 2 == 42
    # And the width upstream would get wrong, to prove the two formulas differ.
    assert num_hw_rope_phase_dims(64) == 20 and 2 * 64 // 3 // 2 == 21


@check
def roce_leaves_the_frame_block_bit_exact():
    """The answer to "does RoCE touch the frame axis": no, and bit-exactly so.

    If it did, it would stop being orthogonal to D3 and the two could never be
    read apart -- and a phase on the frame block also destroys CaPE's exact
    relative-pose identity without raising.
    """
    head_dim, num_heads = 24, 2
    dim = head_dim * num_heads
    S = 6
    g = torch.Generator().manual_seed(7)
    x = torch.randn(1, S, dim, generator=g)
    f, h, w = precompute_freqs_cis_3d(head_dim)
    freqs = torch.cat([f[:S], h[:S], w[:S]], dim=-1).reshape(S, 1, -1)
    phase = torch.randn(
        1, S, num_heads * num_hw_rope_phase_dims(head_dim), generator=g
    )
    plain = rope_apply(x, freqs, num_heads).view(1, S, num_heads, head_dim)
    with_phase = rope_apply(x, freqs, num_heads, phase).view(1, S, num_heads, head_dim)
    n_frame = num_frame_rope_dims(head_dim)
    assert torch.equal(plain[..., :n_frame], with_phase[..., :n_frame]), (
        "RoCE changed the frame block: the left zero-pad is the wrong width"
    )
    assert not torch.allclose(plain[..., n_frame:], with_phase[..., n_frame:]), (
        "RoCE changed nothing on h/w -- the phase is not reaching the freqs"
    )


@check
def roce_phase_is_norm_preserving():
    """A phase shift is a rotation, so |q| and |k| are untouched.

    This is the structural difference from CaPE, which is an SE(3) matrix and
    inflates the attention logit (x1.37 worst-case even at lambda=0.2).  RoCE can
    therefore be handed to Wan's pretrained blocks without changing the logit
    scale they were trained under, and this test is what pins that claim down.
    """
    head_dim, num_heads = 24, 2
    S = 5
    g = torch.Generator().manual_seed(8)
    x = torch.randn(1, S, head_dim * num_heads, generator=g)
    phase = torch.randn(
        1, S, num_heads * num_hw_rope_phase_dims(head_dim), generator=g
    ) * 3.0
    out = rope_apply(x, None, num_heads, phase)
    def _mag(t):
        return torch.view_as_complex(
            t.view(1, S, num_heads, head_dim // 2, 2).contiguous()
        ).abs()
    assert torch.allclose(_mag(x), _mag(out), atol=1e-6), "phase changed a magnitude"


@check
def a_too_wide_phase_is_refused():
    """The one silent failure rope_apply can still see, so it raises on it."""
    head_dim, num_heads = 24, 2
    x = torch.randn(1, 3, head_dim * num_heads)
    for extra in (1, -1):
        phase = torch.zeros(
            1, 3, num_heads * (num_hw_rope_phase_dims(head_dim) + extra)
        )
        try:
            rope_apply(x, None, num_heads, phase)
        except ValueError as exc:
            assert "frame block" in str(exc), str(exc)
        else:
            raise AssertionError(f"a phase {extra} channels off was accepted")


@check
def roce_is_bit_identical_to_its_anchor_at_init():
    """Zero-init phase MLPs => exp(1j*0) == 1 => the same numbers, exactly.

    Not merely close: this is what makes droce readable against the arm it is a
    superset of.  Both modes, because "qk_vo" also rotates v and o and those two
    rotations must cancel at init rather than merely be small.
    """
    latents = torch.randn(1, 4, 6, 4, 4, generator=torch.Generator().manual_seed(2))
    cam = _cam_field(6)
    base = _run_dit(_tiny_dit(), latents)
    for mode in ("qk", "qk_vo"):
        dit = _tiny_dit()
        dit.enable_roce(ROCE_CAM_CHANNELS, mode)
        out = _run_dit(dit, latents, roce_camera_latents=cam)
        assert torch.equal(base, out), (
            f"roce={mode} is not bit-identical at zero init; max |diff| "
            f"{(base - out).abs().max().item():.3e}"
        )
    # ...and once the MLPs are non-zero it must actually do something, or the test
    # above would be passing for the wrong reason.
    out = _run_dit(_roce_dit("qk_vo", phase_std=0.5), latents, roce_camera_latents=cam)
    assert not torch.allclose(base, out, atol=1e-5), "a non-zero phase changed nothing"


@check
def the_value_rotation_exists_only_in_qk_vo():
    """ReDirector's ablation says RoCE without it LOWERS geometric consistency, so
    the two modes have to be distinguishable, and "qk" must not silently get it."""
    for mode, has_vo in (("qk", False), ("qk_vo", True)):
        dit = _tiny_dit()
        dit.enable_roce(ROCE_CAM_CHANNELS, mode)
        for block in dit.blocks:
            assert block.self_attn.rope_phase_qk is not None
            assert (block.self_attn.rope_phase_vo is not None) is has_vo, mode
    latents = torch.randn(1, 4, 6, 4, 4, generator=torch.Generator().manual_seed(2))
    cam = _cam_field(6)
    outs = {
        mode: _run_dit(_roce_dit(mode, phase_std=0.5), latents, roce_camera_latents=cam)
        for mode in ("qk", "qk_vo")
    }
    assert not torch.allclose(outs["qk"], outs["qk_vo"], atol=1e-5), (
        "the two modes agree -- the value rotation is not being applied"
    )


@check
def roce_modules_are_absent_until_enabled():
    """A plain Wan2.1 checkpoint must keep its exact key set, so the off arm loads
    the pretrained weights with no missing/unexpected keys at all."""
    dit = _tiny_dit()
    assert dit.roce_mode == "off" and dit.roce_cam_embedding is None
    assert not [k for k in dit.state_dict() if "rope_phase" in k or "roce" in k]
    dit.enable_roce(ROCE_CAM_CHANNELS, "qk_vo")
    keys = [k for k in dit.state_dict() if "rope_phase" in k or "roce" in k]
    # 2 phase MLPs per block (one Linear each, bias=False) + the camera conv's
    # weight and bias.
    assert len(keys) == 2 * len(dit.blocks) + 2, keys
    for block in dit.blocks:
        for mlp in (block.self_attn.rope_phase_qk, block.self_attn.rope_phase_vo):
            assert torch.count_nonzero(mlp[-1].weight) == 0, "phase MLP is not zero-init"


@check
def camera_tokens_are_aligned_with_patchify():
    """Token i of the camera field must describe the patch token i of x came from.

    A transposed or frame-minor camera field trains perfectly happily on shuffled
    geometry.  Feeding a field that is CONSTANT within each frame and different
    across frames makes the alignment observable: the token sequence then has to be
    piecewise constant in blocks of h'*w'.
    """
    dit = _roce_dit("qk_vo")
    frames, h, w = 6, 4, 4
    per_frame = torch.arange(1.0, frames + 1).view(1, 1, frames, 1, 1)
    cam = per_frame.expand(1, ROCE_CAM_CHANNELS, frames, h, w).contiguous()
    tokens = dit.roce_camera_tokens(cam)
    tokens_per_frame = (h // 2) * (w // 2)
    assert tokens.shape[1] == frames * tokens_per_frame, tokens.shape
    blocks = tokens.view(1, frames, tokens_per_frame, -1)
    for i in range(frames):
        assert torch.allclose(blocks[0, i], blocks[0, i, :1].expand_as(blocks[0, i])), (
            f"frame {i} is not constant across its tokens -- the rearrange is not "
            f"patchify's"
        )
        for j in range(i + 1, frames):
            assert not torch.allclose(blocks[0, i, 0], blocks[0, j, 0]), (i, j)


@check
def roce_composes_with_cape_and_neither_shadows_the_other():
    """D3 and D-extra are different arms, so the code must not force a choice.

    They act on disjoint channels of every head (CaPE right-multiplies the frame
    4-groups, RoCE rotates h/w), so combining them is legal -- and if the
    combination ever silently equalled one of the two, the plan's claim that they
    are independent axes would be wrong.
    """
    V = 3
    latents = torch.randn(1, 4, 2 * V, 4, 4, generator=torch.Generator().manual_seed(2))
    cam = _cam_field(2 * V)
    pose = _pose_for_axis("pose", V)
    ids = rope_frame_ids_for_axis("pose", V)
    dit = _roce_dit("qk_vo", phase_std=0.5)
    cape_only = _run_dit(dit, latents, rope_frame_ids=ids, rope_pose_matrices=pose)
    roce_only = _run_dit(dit, latents, roce_camera_latents=cam)
    both = _run_dit(dit, latents, rope_frame_ids=ids, rope_pose_matrices=pose,
                    roce_camera_latents=cam)
    assert not torch.allclose(both, cape_only, atol=1e-5)
    assert not torch.allclose(both, roce_only, atol=1e-5)


@check
def model_fn_refuses_a_camera_field_it_cannot_align():
    """Three refusals, all of which would otherwise be silent misalignment."""
    latents = torch.randn(1, 4, 6, 4, 4, generator=torch.Generator().manual_seed(2))
    # 1. no RoCE modules on the model at all.
    try:
        _run_dit(_tiny_dit(), latents, roce_camera_latents=_cam_field(6))
    except ValueError as exc:
        assert "enable_roce" in str(exc), str(exc)
    else:
        raise AssertionError("a camera field was accepted by a DiT without RoCE")
    # 2. the wrong frame count, i.e. the sequence-layout tiling was forgotten.
    dit = _roce_dit("qk_vo")
    try:
        _run_dit(dit, latents, roce_camera_latents=_cam_field(3))
    except ValueError as exc:
        assert "tokens" in str(exc), str(exc)
    else:
        raise AssertionError("a camera field with 3 frames was paired with 6 latent frames")
    # 3. the wrong spatial grid.
    try:
        _run_dit(dit, latents, roce_camera_latents=_cam_field(6, h=8, w=8))
    except ValueError as exc:
        assert "tokens" in str(exc), str(exc)
    else:
        raise AssertionError("an 8x8 camera field was paired with a 4x4 latent")


@check
def roce_camera_latents_tiles_over_the_sequence_layout():
    """[B, V, C, h, w] -> [B, C, 2V, h, w], with view v's camera on BOTH of its
    frames.  Get the permute wrong and channels become views; get the tiling wrong
    and every disparity frame is told it was rendered from another view's camera."""
    B, V, C, h, w = 2, 3, 6, 4, 5
    plucker = torch.arange(B * V * C * h * w, dtype=torch.float32).reshape(B, V, C, h, w)
    cam = roce_camera_latents(plucker, 2)
    assert cam.shape == (B, C, 2 * V, h, w), cam.shape
    for v in range(V):
        assert torch.equal(cam[:, :, v], plucker[:, v])
        assert torch.equal(cam[:, :, V + v], plucker[:, v]), (
            "the disparity half is not view-aligned with the RGB half"
        )
    # The channel layout has one frame per view and must not tile.
    assert roce_camera_latents(plucker, 1).shape == (B, C, V, h, w)


@check
def roce_mode_is_a_mode_and_not_a_bool():
    """`roce: true` in a YAML is a plausible typo whose meaning is ambiguous
    (with or without the value rotation?), so it raises instead of guessing."""
    assert resolve_roce_mode(None) == "off"
    assert resolve_roce_mode(False) == "off"
    assert resolve_roce_mode("off") == "off"
    for mode in ("qk", "qk_vo"):
        assert resolve_roce_mode(mode) == mode
    for bad in (True, "yes", "on", "vo", "qkvo"):
        try:
            resolve_roce_mode(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"resolve_roce_mode accepted {bad!r}")
    assert ROCE_MODES == ("off", "qk", "qk_vo")


@check
def camera_field_reaches_the_model_fn_and_the_optimizer():
    """Two ways this arm can train as an exact duplicate of its anchor: the field
    never arrives, or it arrives at frozen zero-init MLPs.  Both are silent."""
    seen = {}

    class _Pipe:
        in_iteration_models = ()
        torch_dtype = torch.float32

        def model_fn(self, **kwargs):
            seen.update(kwargs)
            return torch.zeros(1)

    m = object.__new__(WanRGBDNVSTrainingModule)
    m.pipe = _Pipe()
    m.num_views = 4
    m.token_layout = "channel"
    m._rope_frame_ids = None
    cam = _cam_field(4)
    m._forward_model(
        {"context": None, "roce_camera_latents": cam}, torch.zeros(1), torch.zeros(1)
    )
    assert "roce_camera_latents" in seen, sorted(seen)
    assert torch.equal(seen["roce_camera_latents"], cam)

    src = inspect.getsource(WanRGBDNVSTrainingModule.set_training_param)
    assert '"rope_phase" in name or "roce_cam_embedding" in name' in src, (
        "set_training_param does not unfreeze the RoCE parameters; LoRA mode "
        "freezes everything it is not told about and the arm would stay at its "
        "zero init for the whole run"
    )
    ctor = inspect.getsource(WanRGBDNVSTrainingModule.__init__)
    assert "resolve_roce_mode(roce)" in ctor, "the roce kwarg is not validated"


@check
def the_plucker_field_is_shared_and_carries_no_gradient():
    """RoCE consumes the encoder's RAW field so the arm changes exactly one thing.

    Routing it through the encoder's learned parts instead would give RoCE a second
    gradient path into `channel_scale`, i.e. it would also perturb the OTHER
    channel block -- two interventions in one arm.  And passing the field in must
    be bit-identical to letting the encoder build it, or every c-sweep number
    measured before this refactor stops reproducing.
    """
    from diffsynth.models.view_condition import ViewConditionEncoder

    enc = ViewConditionEncoder(
        plucker_resolution="latent", mode="concat", use_role=False,
        normalize_moment=True,
    )
    B, V, h, w = 1, 3, 4, 6
    K = torch.tensor([[2.0, 0.0, 2.0], [0.0, 2.0, 3.0], [0.0, 0.0, 1.0]])
    K = K.view(1, 1, 3, 3).expand(B, V, 3, 3).contiguous()
    c2w = _random_se3(V, seed=3).unsqueeze(0)
    field = enc.plucker_field(K, c2w, latent_height=h, latent_width=w)
    assert field.shape == (B, V, enc.plucker_channels, h, w), field.shape
    assert not field.requires_grad, "the raw Plucker field carries an autograd graph"
    with_field = enc(K, c2w, None, latent_height=h, latent_width=w, plucker=field)
    without = enc(K, c2w, None, latent_height=h, latent_width=w)
    assert torch.equal(with_field, without), (
        "passing the field in is not bit-identical to building it inside forward"
    )


# ----------------------------------------------------------------------------
# D4 -- camera_geometry_fp32: take the camera math out of autocast
# ----------------------------------------------------------------------------

def _bf16_autocast():
    """CPU bf16 autocast.  The dispatch LISTS are what these tests are about, and
    those are shared between the CPU and CUDA backends, so a CPU context exercises
    the same `matmul -> low precision, linalg.inv -> fp32` split the training run
    hits.  Nothing here needs a GPU."""
    return torch.autocast(device_type="cpu", dtype=torch.bfloat16)


@check
def autocast_downcasts_a_matmul_whose_operands_are_both_fp32():
    """The premise of this whole factor, asserted rather than assumed.

    If a torch upgrade ever changed the dispatch lists, every test below would
    still pass while measuring nothing, and the arm would be a duplicate of its
    anchor.  So pin the two halves of the split that makes the defect invisible:
    the inverse stays fp32, the matmul next to it does not.
    """
    c2w = torch.eye(4).repeat(1, 4, 1, 1).float()
    with _bf16_autocast():
        assert torch.linalg.inv(c2w[:, 0:1]).dtype == torch.float32, (
            "linalg.inv left autocast's fp32 list"
        )
        assert (c2w @ c2w).dtype == torch.bfloat16, (
            "matmul is no longer downcast by autocast -- this factor now fixes "
            "nothing and the arm would duplicate its anchor"
        )


@check
def the_default_path_is_bit_identical_to_every_earlier_arm():
    """The load-bearing property: false must reproduce the shipped two lines.

    Every c-sweep and D-sweep checkpoint is read on covariates computed the old
    way.  If the default drifted, `camera_geometry_fp32=false` would stop meaning
    "the anchor" and the D4 contrast would measure the drift instead.
    """
    c2w = _random_se3(9, seed=21, translation=4.0).unsqueeze(0)
    for ctx in (contextlib.nullcontext(), _bf16_autocast()):
        with ctx:
            legacy = torch.linalg.inv(c2w[:, 0:1]) @ c2w
            got = relative_c2w(c2w, geometry_fp32=False)
            assert got.dtype == legacy.dtype, (got.dtype, legacy.dtype)
            assert torch.equal(got, legacy), (got - legacy).abs().max()


@check
def the_fp32_path_is_invariant_to_autocast():
    """Inside or outside autocast must give the SAME numbers, bit for bit.

    This is the actual definition of the fix.  A `.float()` on the result would
    pass a closeness test and fail this one, which is precisely the trap the
    original code fell into.
    """
    c2w = _random_se3(9, seed=22, translation=4.0).unsqueeze(0)
    outside = relative_c2w(c2w, geometry_fp32=True)
    with _bf16_autocast():
        inside = relative_c2w(c2w, geometry_fp32=True)
    assert inside.dtype == torch.float32, inside.dtype
    assert torch.equal(outside, inside), (outside - inside).abs().max()


@check
def the_fp32_path_fixes_the_cancellation_it_exists_for():
    """Error vs an exact fp64 reference, on poses shaped like the real data.

    Cameras ~4 from the world origin with baselines 0.05-2.0: plan §10's regime.
    The bf16 path must be visibly bad and the fp32 path must be ~1e5 times
    better, measured RELATIVE TO THE BASELINE because that is the scale the
    downstream reprojection actually cares about.
    """
    g = torch.Generator().manual_seed(23)
    bad, good, worst_bad = [], [], 0.0
    for _ in range(300):
        c2w = _random_se3(2, seed=int(torch.randint(0, 10**6, (1,), generator=g)))
        centre = torch.randn(3, generator=g)
        centre = centre / centre.norm() * 4.0
        baseline = float(10 ** torch.empty(1).uniform_(-1.3, 0.3, generator=g))
        offset = torch.randn(3, generator=g)
        c2w[0, :3, 3] = centre
        c2w[1, :3, 3] = centre + offset / offset.norm() * baseline
        c2w = c2w.unsqueeze(0)

        exact = (torch.linalg.inv(c2w[:, 0:1].double()) @ c2w.double())[0, 1, :3, 3]
        with _bf16_autocast():
            legacy = relative_c2w(c2w, geometry_fp32=False)[0, 1, :3, 3].double()
            fixed = relative_c2w(c2w, geometry_fp32=True)[0, 1, :3, 3].double()
        bad.append(float((exact - legacy).norm()) / baseline)
        good.append(float((exact - fixed).norm()) / baseline)
        worst_bad = max(worst_bad, bad[-1])

    bad_med = sorted(bad)[len(bad) // 2]
    good_max = max(good)
    assert bad_med > 0.005, (
        f"the legacy path's median error is only {bad_med:.2%} of the baseline -- "
        f"either the poses here stopped resembling the real data or autocast "
        f"stopped downcasting, and this factor is measuring nothing"
    )
    assert worst_bad > 0.05, f"legacy worst case only {worst_bad:.2%}"
    assert good_max < 1e-5, f"the fp32 path is off by {good_max:.2e} of the baseline"


@check
def the_fp32_path_survives_bf16_even_without_the_autocast_guard():
    """Belt and braces: the REORDERING alone must help, not just the dtype.

    Subtracting the translations before rotating is what makes the expression
    well conditioned.  If someone later moves this back inside an autocast region
    -- or drops the guard while keeping the arithmetic -- the result should degrade
    to "merely imprecise" rather than back to "catastrophic".
    """
    g = torch.Generator().manual_seed(24)
    legacy_err, reordered_err = [], []
    for _ in range(300):
        c2w = _random_se3(2, seed=int(torch.randint(0, 10**6, (1,), generator=g)))
        centre = torch.randn(3, generator=g)
        centre = centre / centre.norm() * 4.0
        baseline = float(10 ** torch.empty(1).uniform_(-1.3, 0.3, generator=g))
        offset = torch.randn(3, generator=g)
        c2w[0, :3, 3] = centre
        c2w[1, :3, 3] = centre + offset / offset.norm() * baseline

        exact = (torch.linalg.inv(c2w[0:1].double()) @ c2w.double())[1, :3, 3]
        r_ref_t = c2w[0, :3, :3].T
        legacy = (torch.linalg.inv(c2w[0:1]).bfloat16() @ c2w.bfloat16()).float()[1, :3, 3]
        d = c2w[1, :3, 3] - c2w[0, :3, 3]          # fp32 subtraction, O(baseline)
        reordered = (r_ref_t.bfloat16() @ d.bfloat16()).float()
        legacy_err.append(float((exact - legacy.double()).norm()) / baseline)
        reordered_err.append(float((exact - reordered.double()).norm()) / baseline)

    assert max(reordered_err) < max(legacy_err) / 5, (
        f"reordering bought only {max(legacy_err) / max(reordered_err):.1f}x on the "
        f"worst case; the subtract-then-rotate form is not doing its job"
    )


@check
def the_plucker_pixel_grid_is_quantised_in_bf16_and_not_in_fp32():
    """Defect 2, at the resolution this project actually runs.

    `ray_condition` builds its grid with `dtype=c2w.dtype`, so a bf16 c2w -- which
    is what the legacy relative-pose matmul returns -- silently collapses the
    384x672 grid.  This asserts the magnitude directly, because it is the part of
    the defect that is easiest to dismiss as rounding noise and is in fact a
    multi-pixel, spatially non-uniform warp.
    """
    for size, want_max_err in ((672, 3.0), (384, 2.0)):
        exact = torch.linspace(0, size - 1, size, dtype=torch.float32)
        got = torch.linspace(0, size - 1, size, dtype=torch.bfloat16).float()
        assert (got - exact).abs().max() == want_max_err, (
            size, float((got - exact).abs().max())
        )
        assert len(torch.unique(got)) < size, size
    # 672 is the load-bearing number: 247 columns lose their own coordinate.
    got = torch.linspace(0, 671, 672, dtype=torch.bfloat16).float()
    assert len(torch.unique(got)) == 425, len(torch.unique(got))


@check
def the_encoder_rasterises_the_field_in_fp32_when_asked():
    """The flag has to reach `_plucker`, and it has to change the field.

    Two failure modes, both silent: the constructor arg is stored and never read
    (the arm duplicates its anchor), or the field is cast AFTER rasterisation (the
    dtype looks right and the pixel grid is still quantised).  The second is
    caught by feeding a bf16 c2w -- exactly what the legacy path hands over -- and
    requiring the fp32 encoder to ignore its dtype.
    """
    from diffsynth.models.view_condition import ViewConditionEncoder

    B, V, h, w = 1, 3, 4, 6
    K = torch.tensor([[600.0, 0.0, 336.0], [0.0, 600.0, 192.0], [0.0, 0.0, 1.0]])
    K = K.view(1, 1, 3, 3).expand(B, V, 3, 3).contiguous()
    c2w = _random_se3(V, seed=25, translation=4.0).unsqueeze(0)

    def field(geometry_fp32, c2w_in, ctx):
        enc = ViewConditionEncoder(
            plucker_resolution="latent", mode="concat", use_role=False,
            normalize_moment=True, geometry_fp32=geometry_fp32,
        )
        with ctx:
            return enc.plucker_field(K, c2w_in, latent_height=h, latent_width=w)

    assert not ViewConditionEncoder(geometry_fp32=False).geometry_fp32
    assert ViewConditionEncoder(geometry_fp32=True).geometry_fp32

    legacy = field(False, c2w.bfloat16(), _bf16_autocast())
    fixed = field(True, c2w.bfloat16(), _bf16_autocast())
    assert fixed.dtype == torch.float32, fixed.dtype
    assert not torch.allclose(legacy.float(), fixed, atol=1e-3), (
        "the fp32 encoder produced the bf16 field -- geometry_fp32 is inert"
    )
    # A bf16 c2w must not survive as a bf16 pixel grid: the fp32 field has to be
    # the same whether it is handed fp32 or bf16 poses that round-trip to it.
    assert torch.equal(fixed, field(True, c2w.bfloat16().float(), _bf16_autocast()))
    # ...and outside autocast the fp32 path must agree with itself.
    assert torch.equal(
        field(True, c2w, contextlib.nullcontext()),
        field(True, c2w, _bf16_autocast()),
    )


@check
def the_splat_is_taken_out_of_autocast_too():
    """`warp_mode: splat` is on in the base config, so this is a model input.

    utils/warp.py is written entirely in `.float()` and is downcast anyway, for
    the same dispatch reason.  The default must stay bit-identical and the flag
    must change the result.
    """
    B, S, T, H, W = 1, 2, 1, 24, 32
    g = torch.Generator().manual_seed(26)
    rgb = torch.rand(B, S, 3, H, W, generator=g)
    valid = torch.ones(B, S, H, W, dtype=torch.bool)
    K = torch.tensor([[30.0, 0.0, 16.0], [0.0, 30.0, 12.0], [0.0, 0.0, 1.0]])
    K = K.view(1, 1, 3, 3).expand(B, S + T, 3, 3).contiguous()
    # Cameras 4 units from the origin, 0.15 apart -- plan §10's cancellation regime
    # -- and all aimed at the origin, so the target view actually CONTAINS the source
    # content.  Depth is centred on the same 4 units so the points land near it.
    c2w = _lookat_c2w(S + T, seed=27, distance=4.0, baseline=0.15).unsqueeze(0)
    depth = 4.0 + torch.rand(B, S, H, W, generator=g) * 0.5

    def run(geometry_fp32, ctx):
        with ctx:
            return splat_sources_to_targets(
                src_rgb=rgb, src_depth=depth, src_valid=valid,
                src_K=K[:, :S], src_c2w=c2w[:, :S],
                tgt_K=K[:, S:], tgt_c2w=c2w[:, S:],
                return_target_disparity=True,
                geometry_fp32=geometry_fp32,
            )

    # `tgt_disp` is the sensitive channel: the RGB output only moves when a splat
    # lands in a DIFFERENT pixel, and floor() hides sub-pixel error, whereas 1/z is
    # continuous in the reprojection.  Asserting on RGB alone would pass on a grid
    # too coarse to resolve the defect.
    out_fixed = run(True, contextlib.nullcontext())
    # Guard the guard: an all-zero splat would satisfy every assertion below.
    assert out_fixed[1].sum() > 0.5 * B * T * H * W, (
        "the fixture's target view sees almost none of the source content, so this "
        "test would compare two near-empty tensors"
    )
    in_fixed = run(True, _bf16_autocast())
    for a, b in zip(out_fixed, in_fixed):
        assert torch.equal(a, b), "geometry_fp32 splat is not autocast-invariant"

    in_legacy = run(False, _bf16_autocast())
    assert not torch.equal(in_legacy[2], in_fixed[2]), (
        "the splat is unchanged by geometry_fp32 -- either the kwarg is dropped or "
        "autocast is not reaching it, and the arm's warp_block duplicates the anchor"
    )
    # Outside autocast the default path must equal the fixed path: there is no
    # reordering here, only the dtype, so any difference would mean the wrapper
    # changed the arithmetic rather than just the precision it runs at.
    for a, b in zip(run(False, contextlib.nullcontext()), out_fixed):
        assert torch.equal(a, b), (
            "the geometry_fp32 wrapper altered the splat outside autocast, so the "
            "default path is no longer the shipped computation"
        )


@check
def camera_geometry_fp32_reaches_the_three_consumers():
    """The flag is only useful if all four sites see it, and three of them are
    reached through objects the constructor builds or calls it makes."""
    src = inspect.getsource(WanRGBDNVSTrainingModule.__init__)
    assert "self.camera_geometry_fp32 = bool(camera_geometry_fp32)" in src
    assert "geometry_fp32=self.camera_geometry_fp32" in src, (
        "ViewConditionEncoder is built without the flag, so the Plucker field -- the "
        "biggest of the three defects in pixels -- stays quantised"
    )
    pre = inspect.getsource(WanRGBDNVSTrainingModule.forward_preprocess)
    assert "relative_c2w(c2w, self.camera_geometry_fp32)" in pre
    splat = inspect.getsource(WanRGBDNVSTrainingModule._splat_sources_pixel)
    assert "geometry_fp32=self.camera_geometry_fp32" in splat


# ----------------------------------------------------------------------------
# D2 -- view_order
# ----------------------------------------------------------------------------

def _view(view_id, center):
    c2w = torch.eye(4)
    c2w[:3, 3] = torch.tensor(center, dtype=torch.float32)
    return _ViewRecord(
        scene_id="scene", time_id="t0", view_id=view_id,
        rgb_path=f"{view_id}.png", depth_path=f"{view_id}.pfm",
        K=torch.eye(3), c2w=c2w,
    )


def _ring(angles_deg, radius=1.0, prefix="s"):
    out = []
    for a in angles_deg:
        t = torch.deg2rad(torch.tensor(float(a)))
        out.append(_view(f"{prefix}{int(a):03d}",
                         (radius * float(torch.cos(t)), radius * float(torch.sin(t)), 0.0)))
    return out


def _path_length(views):
    return sum(
        float((views[i + 1].c2w[:3, 3] - views[i].c2w[:3, 3]).norm())
        for i in range(len(views) - 1)
    )


# Deliberately UNEVEN spacing: on a uniform ring every greedy step is a tie and
# the expected answer would be an artefact of the tie-break rather than of the
# geometry.  Angles ascend while the expected output descends, so an
# implementation that merely sorts by view_id (the tie-break, not the ordering)
# fails this test.
_RING_ANGLES = (0, 40, 90, 150, 220, 300)
_TARGETS_NEAR_ZERO = _ring((-10, 0, 10), radius=0.9, prefix="t")


@check
def pose_order_walks_the_ring():
    sources = _ring(_RING_ANGLES)
    ordered = _order_source_views_by_pose(sources, _TARGETS_NEAR_ZERO)
    assert [v.view_id for v in ordered] == ["s300", "s220", "s150", "s090", "s040", "s000"], \
        [v.view_id for v in ordered]


@check
def pose_order_ends_next_to_the_target_block():
    """Frame distance from a target at position S+j to a source at i is S+j-i, so
    WHICH SOURCE SITS LAST is what decides whether a target's nearest camera is
    also its nearest frame.  Any ordering that gets this backwards makes the frame
    axis anti-correlated with geometry, which is worse than random."""
    for target_angles in ((-10, 0, 10), (140, 150, 160), (290, 300, 310)):
        targets = _ring(target_angles, radius=0.9, prefix="t")
        ordered = _order_source_views_by_pose(_ring(_RING_ANGLES), targets)
        centroid = torch.stack([v.c2w[:3, 3] for v in targets]).mean(dim=0)
        nearest = min(ordered, key=lambda v: float((v.c2w[:3, 3] - centroid).norm()))
        assert ordered[-1] is nearest, (
            target_angles, [v.view_id for v in ordered], nearest.view_id
        )


@check
def pose_order_is_a_pure_function_of_the_set():
    """The incoming list comes out of rng.sample in a RANDOM order.

    If the result depended on that order -- through a nearest-neighbour tie, or
    through two near-coincident cameras, both of which BlendedMVS contains -- the
    frame axis would still carry a random label and the factor would be measuring
    nothing.
    """
    sources = _ring(_RING_ANGLES)
    reference = _order_source_views_by_pose(sources, _TARGETS_NEAR_ZERO)
    for permutation in itertools.permutations(range(len(sources))):
        shuffled = [sources[i] for i in permutation]
        got = _order_source_views_by_pose(shuffled, _TARGETS_NEAR_ZERO)
        assert [v.view_id for v in got] == [v.view_id for v in reference], permutation

    # The same, with an exact tie present: two cameras at the same centre.  Ties
    # are where an order-dependent implementation actually breaks.
    twins = _ring(_RING_ANGLES) + [_view("s000b", (1.0, 0.0, 0.0))]
    ref_twins = _order_source_views_by_pose(twins, _TARGETS_NEAR_ZERO)
    for permutation in ((6, 0, 1, 2, 3, 4, 5), (3, 6, 1, 0, 5, 2, 4)):
        got = _order_source_views_by_pose([twins[i] for i in permutation], _TARGETS_NEAR_ZERO)
        assert [v.view_id for v in got] == [v.view_id for v in ref_twins], permutation


@check
def pose_order_shortens_the_path_on_a_scattered_cloud():
    """BlendedMVS is not a clean ring, so check the property that actually has to
    hold in general: consecutive frames are closer than they would be by chance."""
    g = torch.Generator().manual_seed(7)
    for trial in range(5):
        sources = [
            _view(f"v{i}", tuple(torch.randn(3, generator=g).tolist()))
            for i in range(6)
        ]
        targets = [
            _view(f"t{i}", tuple(torch.randn(3, generator=g).tolist()))
            for i in range(3)
        ]
        ordered = _order_source_views_by_pose(sources, targets)
        chance = sum(_path_length([sources[i] for i in p])
                     for p in itertools.permutations(range(6))) / 720.0
        assert _path_length(ordered) < chance, (trial, _path_length(ordered), chance)


def _bare_dataset(view_order, random_permute=True):
    ds = object.__new__(Kubric4DRGBDNVSDataset)
    ds.num_source_views = 6
    ds.num_target_views = 3
    ds.num_views = 9
    ds.deterministic_view_seed = 1234
    ds.random_permute_source_views = random_permute
    ds.view_order = view_order
    return ds


# 12 views, so the selection has something to leave out; distinct view_ids because
# `remaining = [v for v in views if v not in target_views]` relies on the dataclass
# comparing an unequal field before it reaches a tensor field.
_RECORD = {"views": _ring(tuple(range(0, 360, 30)))}


@check
def pose_order_leaves_the_frozen_eval_set_alone():
    """The load-bearing property of D2.

    The eval set is a pure function of (eval_view_seed, idx), and every paired
    comparison against the c-sweep history -- per scene AND per target_index --
    assumes the same views come out in the same roles.  `view_order` is allowed to
    permute the source block and nothing else.
    """
    for idx in range(20):
        src_r, tgt_r = _bare_dataset("random")._select_views(_RECORD, idx=idx)
        src_p, tgt_p = _bare_dataset("pose")._select_views(_RECORD, idx=idx)
        assert [v.view_id for v in tgt_p] == [v.view_id for v in tgt_r], (
            f"idx={idx}: the target block moved. target_index is positional and is "
            f"the key of every paired per-target analysis, and per_view_metrics.jsonl "
            f"carries no view id to re-pair on."
        )
        assert {v.view_id for v in src_p} == {v.view_id for v in src_r}, (
            f"idx={idx}: the source SET changed, so this arm is not measuring order"
        )
    # ...and it does have to change something.
    changed = 0
    for idx in range(20):
        src_r, _ = _bare_dataset("random")._select_views(_RECORD, idx=idx)
        src_p, _ = _bare_dataset("pose")._select_views(_RECORD, idx=idx)
        changed += [v.view_id for v in src_p] != [v.view_id for v in src_r]
    assert changed >= 15, f"only {changed}/20 draws reordered -- the factor is inert"


@check
def pose_order_supersedes_random_permute_source_views():
    """The D2 arm keeps `random_permute_source_views: true` in its config, because
    it inherits the shared base.  If that flag still had a say, the arm would be
    half-random and unreadable."""
    for idx in range(10):
        with_permute, _ = _bare_dataset("pose", random_permute=True)._select_views(
            _RECORD, idx=idx)
        without, _ = _bare_dataset("pose", random_permute=False)._select_views(
            _RECORD, idx=idx)
        assert [v.view_id for v in with_permute] == [v.view_id for v in without], idx


@check
def view_order_guard_is_reachable():
    src = inspect.getsource(Kubric4DRGBDNVSDataset.__init__)
    assert "view_order must be one of" in src
    select = inspect.getsource(Kubric4DRGBDNVSDataset._select_views)
    assert "_order_source_views_by_pose" in select
    # The elif, not a second if: leaving the shuffle reachable under
    # view_order="pose" would put the random label straight back.
    assert "elif self.random_permute_source_views:" in select


@check
def factors_are_plumbed_into_every_harness_that_rebuilds_a_model():
    """The failure this repeats: a shape-preserving flag that the config sets and
    an eval harness forgets, so the checkpoint loads without a warning and reports
    numbers for a different model (report_20260907_c_sweep §9-1a)."""
    needles = (
        "rope_frame_axis=cfg.get",
        "rope_pose_translation_scale=cfg.get",
        'view_order=cfg.get("view_order", "random")',
        # roce is the one factor that changes parameter SHAPES, so an eval harness
        # that forgets it fails loudly on the off/on boundary -- but "qk" vs
        # "qk_vo" is shape-compatible one way (the vo phase MLP simply stays at its
        # zero init), so half of this factor is silent too.
        'roce=cfg.get("roce", "off")',
        # D4 is shape-preserving AND changes model inputs, so an eval harness that
        # forgets it scores a D4 checkpoint on the anchor's geometry with no
        # warning -- the exact failure mode of §9-1a, which cost 1.45 dB.
        'camera_geometry_fp32=cfg.get("camera_geometry_fp32", False)',
    )
    wants = {
        "examples/wanvideo/model_training/train_with_accelerate_rgbd_nvs.py": needles,
        "examples/wanvideo/model_training/eval_rgbd_nvs.py": needles,
        "tools/diag_posterior_mean.py": needles,
        # Both sensitivity diagnostics rebuild the model from args.yaml the same
        # way, and both are read as dB DELTAS between variants -- which is
        # exactly the shape of error a silent flag mismatch hides, because every
        # variant is wrong by the same amount and the deltas still look sane.
        "tools/diag_warp_sensitivity.py": needles,
        "tools/diag_pose_sensitivity.py": needles,
    }
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for path, needles in wants.items():
        text = open(os.path.join(root, path), encoding="utf-8").read()
        for needle in needles:
            assert needle in text, f"{path} does not forward: {needle}"
    # The training script builds TWO datasets -- train and the held-out eval -- and
    # the factor has to be on for both.
    text = open(os.path.join(
        root, "examples/wanvideo/model_training/train_with_accelerate_rgbd_nvs.py"),
        encoding="utf-8").read()
    assert text.count('view_order=cfg.get("view_order", "random")') == 2, (
        "the train script must pass view_order to the train dataset AND the "
        "held-out eval dataset"
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
