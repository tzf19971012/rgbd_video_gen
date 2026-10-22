import copy
import math
import os

import torch
import torch.nn.functional as F
from einops import rearrange

from diffsynth.models.view_condition import ViewConditionEncoder
from diffsynth.models.wan_video_camera_controller import fp32_geometry
from diffsynth.models.wan_video_dit import ROCE_MODES
from diffsynth.pipelines.wan_video_new_determine import WanVideoPipeline
from utils.depth_consistency import flatness, ordered_pairs, pairwise_consistency
from utils.projection import intrins_resize
from utils.warp import splat_sources_to_targets

from .WanTrainingModule import WanTrainingModule


_FONT_CACHE = {}


def _label_font(size):
    """A real TrueType face at `size` px, falling back to PIL's bitmap default.

    PIL's built-in font is 11 px, which is illegible beside a 384 px tile, so
    prefer the DejaVuSans that ships with matplotlib -- already a dependency here
    for the depth colormaps, and available offline.
    """
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    from PIL import ImageFont

    def _dejavu():
        from matplotlib import font_manager
        return ImageFont.truetype(
            font_manager.findfont("DejaVu Sans", fallback_to_default=True), size)

    font = None
    for attempt in (_dejavu,
                    lambda: ImageFont.load_default(size=size),  # Pillow >= 10.1
                    ImageFont.load_default):                    # Pillow < 10.1: bitmap
        try:
            font = attempt()
            break
        except Exception:
            continue
    if font is None:
        raise RuntimeError("no usable font for grid labels")
    _FONT_CACHE[size] = font
    return font


def _annotate_grid(img, row_labels, col_labels, tile_hw, sep=4.0, captions=None):
    """Add row labels down the left margin, column labels across the top, and
    an optional per-span caption strip across the bottom.

    Positions come from the tile geometry (`tile_hw`, `sep`) rather than from
    anything measured in the image, so a column whose tile is a grey placeholder
    still gets labelled -- which is the whole point, since the reader needs to
    know that the blank under "tgt1 pred" in the "depth gt" row means "not
    applicable", not "the model output nothing".

    tile_hw and sep may be fractional: after `eval_vis_downscale` the tile pitch
    is not an integer, and rounding it per column would let labels drift away from
    their tiles across a wide grid.

    `captions`, if given, is a list of (start_col, span, lines) -- `lines` a
    list of short strings drawn centered under columns [start_col,
    start_col + span), stacked top to bottom, in a smaller font than the column
    labels so a 5-line target caption does not dominate the panel.  A caption
    spans multiple columns (e.g. a target's gt+pred pair) so one set of numbers
    for that view, not two duplicated halves, sits under it.
    """
    from PIL import Image as PILImage, ImageDraw
    tile_h, tile_w = float(tile_hw[0]), float(tile_hw[1])
    size = max(9, int(round(tile_h * 0.11)))
    font = _label_font(size)
    probe = ImageDraw.Draw(PILImage.new("RGB", (1, 1)))

    def _box(text):
        return probe.textbbox((0, 0), text, font=font)

    gap = max(3, size // 3)
    left = max((_box(t)[2] - _box(t)[0] for t in row_labels), default=0) + 2 * gap
    top = size + 2 * gap

    cap_font = cap_gap = None
    bottom = 0
    if captions:
        cap_size = max(8, int(round(tile_h * 0.09)))
        cap_font = _label_font(cap_size)
        cap_gap = max(2, cap_size // 4)
        n_lines = max((len(lines) for _, _, lines in captions), default=0)
        bottom = n_lines * (cap_size + cap_gap) + 2 * cap_gap

    out = PILImage.new("RGB", (left + img.width, top + img.height + bottom),
                       (255, 255, 255))
    out.paste(img, (left, top))
    draw = ImageDraw.Draw(out)

    # Placed by hand from textbbox instead of via draw.text(anchor=...): anchors
    # raise on bitmap fonts, which is exactly the fallback path above.
    for c, text in enumerate(col_labels):
        x0, y0, x1, y1 = _box(text)
        cx = left + c * (tile_w + sep) + tile_w / 2.0
        draw.text((round(cx - (x1 - x0) / 2.0 - x0), round(top - gap - (y1 - y0) - y0)),
                  text, fill=(0, 0, 0), font=font)
    for r, text in enumerate(row_labels):
        x0, y0, x1, y1 = _box(text)
        cy = top + r * (tile_h + sep) + tile_h / 2.0
        draw.text((round(left - gap - (x1 - x0) - x0), round(cy - (y1 - y0) / 2.0 - y0)),
                  text, fill=(0, 0, 0), font=font)

    if captions:
        sep_y = top + img.height
        draw.line([(left, sep_y), (left + img.width, sep_y)], fill=(160, 160, 160))
        for start_col, span, lines in captions:
            x_left = left + start_col * (tile_w + sep)
            x_right = x_left + span * tile_w + (span - 1) * sep
            cx = (x_left + x_right) / 2.0
            y = sep_y + cap_gap
            for line in lines:
                x0, y0, x1, y1 = probe.textbbox((0, 0), line, font=cap_font)
                draw.text((round(cx - (x1 - x0) / 2.0 - x0), round(y - y0)),
                          line, fill=(30, 30, 30), font=cap_font)
                y += (y1 - y0) + cap_gap
    return out


def _compose_eval_grid(src_rgb, src_gt_depth, src_pred_depth, tgt_tiles, downscale=1,
                       src_captions=None, tgt_captions=None):
    """Assemble one labelled eval image: 3 rows x (S + 2T) columns.

    Rows are rgb / depth gt / depth pred.  Columns are the S source views followed
    by a (gt, pred) pair per target view.  Source depth is predicted too -- that is
    the spatial-understanding half of the task -- so source columns fill all three
    rows; a target's GT depth exists only in its gt column and its predicted depth
    only in its pred column, so in the depth rows each target repeats its own
    content across both columns (gt row shows gt_depth, pred row shows pred_depth)
    instead of a grey placeholder.

    `tgt_tiles` is a list of (gt_rgb, pred_rgb, gt_depth, pred_depth) uint8 HxWx3
    arrays, one per target.  Returns (PIL image, warning or None); a labelling
    failure yields the unlabelled image plus a message rather than raising, since
    the tiles are the data and their VAE decodes are already paid for.

    `src_captions` / `tgt_captions`, if given, are lists of per-view caption
    lines (one entry per source / target view, each entry itself a list of
    short strings) drawn in a strip below the grid -- a source's under its
    single column, a target's centered under its gt+pred pair.  These are the
    caller's OWN per-view metrics for THIS sample (not the eval-set mean), so
    the numbers under a tile are what produced it; this function only knows
    how to position text, not what a metric means.

    Kept module-level and free of `self` so the layout can be exercised without a
    GPU -- see tools/smoke_eval_grid.py.
    """
    import numpy as np
    from PIL import Image as PILImage

    tiles = list(src_rgb) or [t[0] for t in tgt_tiles]
    H, W = tiles[0].shape[:2]
    n_src, n_tgt = len(src_rgb), len(tgt_tiles)
    n_cols = n_src + 2 * n_tgt
    empty = np.ones((H, W, 3), dtype=np.uint8) * 240      # light grey placeholder
    col_sep = np.ones((H, 4, 3), dtype=np.uint8) * 200

    def _hstack(imgs):
        result = imgs[0]
        for img in imgs[1:]:
            result = np.concatenate([result, col_sep, img], axis=1)
        return result

    row_rgb, row_gt, row_pred = list(src_rgb), list(src_gt_depth), list(src_pred_depth)
    for gt_rgb, pred_rgb, gt_depth, pred_depth in tgt_tiles:
        row_rgb += [gt_rgb, pred_rgb]
        row_gt += [gt_depth, gt_depth]
        row_pred += [pred_depth, pred_depth]

    row_sep = np.ones((4, W * n_cols + 4 * (n_cols - 1), 3), np.uint8) * 200
    grid = np.concatenate(
        [_hstack(row_rgb), row_sep, _hstack(row_gt), row_sep, _hstack(row_pred)],
        axis=0,
    )

    img = PILImage.fromarray(grid)
    tile_h, tile_w, sep = float(H), float(W), 4.0
    if downscale > 1:
        new_w = max(1, grid.shape[1] // downscale)
        new_h = max(1, grid.shape[0] // downscale)
        # Label AFTER downscaling, at a size set by the final tile, so the text is
        # drawn crisp instead of being resampled along with the image.  The pitch
        # is fractional once scaled; _annotate_grid takes floats for that reason.
        tile_h *= new_h / grid.shape[0]
        tile_w *= new_w / grid.shape[1]
        sep *= new_w / grid.shape[1]
        img = img.resize((new_w, new_h), PILImage.LANCZOS)

    col_labels = [f"src{i + 1}" for i in range(n_src)]
    for t in range(n_tgt):
        col_labels += [f"tgt{t + 1} gt", f"tgt{t + 1} pred"]

    captions = []
    for i, lines in enumerate(src_captions or []):
        if lines:
            captions.append((i, 1, lines))
    for t, lines in enumerate(tgt_captions or []):
        if lines:
            captions.append((n_src + 2 * t, 2, lines))

    try:
        return _annotate_grid(img, ["rgb", "depth gt", "depth pred"], col_labels,
                              (tile_h, tile_w), sep, captions=captions or None), None
    except Exception as exc:
        return img, f"grid labels unavailable ({exc}); saving unlabelled images"


# ---------------------------------------------------------------------------
# What the RoPE frame axis encodes
# ---------------------------------------------------------------------------
# The frame axis owns 44 of every head's 128 dims (precompute_freqs_cis_3d splits
# head_dim as dim-2*(dim//3), dim//3, dim//3), and in the sequence layout it is
# handed ids 0..2V-1 -- which is a RANDOM LABEL, because
# random_permute_source_views reshuffles which view lands on which index on every
# sample.  So the video prior "adjacent frames are more correlated" is not merely
# uninformative here, it is systematically wrong
# (reports/plan_20260906_rope_geometry.md §0, §1.1).  These are the five things
# that label can be:
#
#   view_modality  0..2V-1            default.  View v's disparity sits V
#                                     positions from its own RGB -- further than
#                                     any two RGB views are from each other.
#   view           [0..V-1]*2         == shared_rope_frame_ids, the c-sweep
#                                     factor: an RGB frame and its disparity are
#                                     positionally IDENTICAL, and the modality
#                                     flag channel is what still tells them apart.
#   modality       [0]*V + [1]*V      view identity gone, modality kept.
#   constant       [0]*2V             the frame axis carries nothing at all.
#   pose           [0]*2V + CaPE      the discrete label is gone AND the same 44
#                                     dims instead carry the CONTINUOUS 6-DoF
#                                     camera pose (plan §12, arm d3).
#
# `modality`/`constant` are D1 of the rope-geometry plan: they test whether the
# axis is worth ENRICHING (D2/D3) or worth REMOVING.  View identity does not
# disappear from the model when they are on -- it still arrives through the 384
# per-latent-pixel Plucker channels, the modality flag channel and the warp block.
# Judged against `view` rather than against `view_modality`, because `view` is the
# only arm of the c-sweep whose four metrics all moved the right way
# (report_20260907_c_sweep §8-6) and is therefore the D round's anchor.
#
# `pose` is D3.  It pins the ids to a constant on purpose: EscherNet's CaPE and
# the rotary frame block act on the SAME real dims 0..43, and a 2x2 rotation does
# not commute with a 4x4 SE(3) matrix, so stacking them would destroy the exact
# relative-pose identity.  With ids == 0 the frame block is exactly 1+0j, so
# `pose` with translation scale 0 and no rotation degenerates to EXACTLY the
# `constant` arm -- which is what makes d1 the right anchor to read d3 against.
# model_fn_wan_video re-checks all of this and raises rather than trusting us.
#
# Spelled "constant" and not "none" on purpose: `rope_frame_axis=none` in an
# OmegaConf dotlist reads as "no override" to a human even though PyYAML parses
# it as the string.
ROPE_FRAME_AXES = ("view_modality", "view", "modality", "constant", "pose")


def resolve_rope_frame_axis(rope_frame_axis, shared_rope_frame_ids):
    """Fold the older `shared_rope_frame_ids` bool into `rope_frame_axis`.

    Every c-sweep checkpoint's config spells the factor as the bool, and
    eval_rgbd_nvs / diag_posterior_mean must keep reproducing those runs' RoPE
    geometry exactly -- a mismatch there is silent and already cost one
    posterior-mean run 1.45 dB of pure harness artefact (report_20260907_c_sweep
    §9-1a).  So the bool stays supported and means exactly axis="view"; setting
    both to different things is a contradiction and raises rather than picking a
    winner.
    """
    if rope_frame_axis is None:
        return "view" if bool(shared_rope_frame_ids) else "view_modality"
    if rope_frame_axis not in ROPE_FRAME_AXES:
        raise ValueError(
            f"rope_frame_axis must be one of {ROPE_FRAME_AXES}, got {rope_frame_axis!r}"
        )
    if bool(shared_rope_frame_ids) and rope_frame_axis != "view":
        raise ValueError(
            f"shared_rope_frame_ids=True is the same factor as "
            f"rope_frame_axis='view', but rope_frame_axis={rope_frame_axis!r} was "
            f"also given. Set exactly one of them."
        )
    return rope_frame_axis


def rope_frame_ids_for_axis(axis, num_views):
    """Temporal RoPE ids for the 2*V frames of the sequence layout, or None.

    None means "do not pass the override at all", so the pipeline keeps its
    original code path and the default arange is bit-identical to every run
    measured before the override existed.
    """
    if axis == "view_modality":
        return None
    if axis == "view":
        return torch.arange(num_views).repeat(2)
    if axis == "modality":
        return torch.arange(2).repeat_interleave(num_views)
    if axis in ("constant", "pose"):
        # "pose" MUST be all-zero, not merely constant: cape_apply composes with
        # whatever the frame RoPE block already put on dims 0..43, and only
        # dit.freqs[0][0] is exactly 1+0j.
        return torch.zeros(2 * num_views, dtype=torch.long)
    raise ValueError(f"rope_frame_axis must be one of {ROPE_FRAME_AXES}, got {axis!r}")


def resolve_rope_pose_translation_scale(axis, translation_scale):
    """Validate lambda against the axis.  Both directions are load-bearing.

    Requiring it when axis='pose' stops a default from silently picking an arm;
    refusing it otherwise stops `rope_pose_translation_scale: 0.2` from sitting in
    a config whose axis was never switched -- which would train, converge and
    report numbers for the wrong arm without a single warning (plan §6-2).
    """
    if axis != "pose":
        if translation_scale is not None:
            raise ValueError(
                f"rope_pose_translation_scale is only used by "
                f"rope_frame_axis='pose', but the axis is {axis!r}. Leave it unset."
            )
        return None
    if translation_scale is None:
        raise ValueError(
            "rope_frame_axis='pose' requires an explicit "
            "rope_pose_translation_scale (lambda in (R,t)->(R,lambda t)): 0.0 is "
            "the rotation-only, exactly orthogonal arm, 0.2 is the measured knee "
            "of the logit-inflation curve, 1.0 is EscherNet as published (4.2x "
            "worst-case inflation)."
        )
    translation_scale = float(translation_scale)
    if translation_scale < 0.0:
        raise ValueError(
            f"rope_pose_translation_scale must be >= 0, got {translation_scale}"
        )
    return translation_scale


def resolve_roce_mode(roce):
    """Validate the D-extra factor `roce` (ReDirector's camera-conditioned RoPE).

    None and "off" both mean off, so a YAML may spell it either way; anything else
    must be one of the two live modes.  Unlike the frame-axis factors this one
    changes PARAMETER SHAPES -- it adds two zero-init phase MLPs per block and one
    conv -- so a checkpoint cannot silently cross the boundary the way
    rope_frame_axis can: accelerator.load_state raises on the missing keys.  That
    is the one place in the D round where the usual silent-flag hazard does not
    apply, and it is why "off" is still forwarded by every harness anyway (the
    mode also decides whether the value rotation exists, and qk vs qk_vo IS
    shape-compatible in the wrong direction: a qk checkpoint loads into a qk_vo
    model with rope_phase_vo left at its zero init, i.e. silently degraded).
    """
    if roce is None:
        return "off"
    if roce is False:
        return "off"
    if roce is True:
        raise ValueError(
            "roce is a mode, not a bool: use 'qk_vo' (ReDirector as published, "
            "with the GTA-style value rotation) or 'qk' (phase on q/k only)."
        )
    roce = str(roce)
    if roce not in ROCE_MODES:
        raise ValueError(f"roce must be one of {ROCE_MODES}, got {roce!r}")
    return roce


def roce_camera_latents(plucker, num_repeats):
    """Lay the per-view Plucker field out over the DiT's FRAME axis.

    `plucker` is [B, V, C, h, w] as `ViewConditionEncoder.plucker_field` returns
    it; the DiT wants [B, C, frames, h, w] because its RoCE conv is a Conv3d of
    the same geometry as patch_embedding.  `num_repeats` is 2 for the sequence
    layout, whose frames are [RGB x V, disparity x V] -- the same tiling
    cape_pose_matrices does, and for the same reason: view v's disparity frame is
    rendered from view v's camera.
    """
    cam = plucker.permute(0, 2, 1, 3, 4).contiguous()
    if num_repeats == 1:
        return cam
    return cam.repeat(1, 1, num_repeats, 1, 1)


def relative_c2w(c2w, geometry_fp32=False):
    """Express every camera-to-world matrix relative to view 0.  [B, V, 4, 4].

    The obvious two lines -- invert view 0, left-multiply -- are what every arm up
    to and including the D round ran, and under bf16 autocast they are wrong in a
    way that has nothing to do with the input dtype.  The translation column
    evaluates R_ref^T t - R_ref^T t_ref: two quantities on the order of the
    distance to the world origin (~4 on BlendedMVS) subtracted to produce one on
    the order of the camera baseline (~0.1).  bf16 carries 8 significant bits, so
    each term arrives with absolute error ~0.016, which lands undiminished on a
    difference of ~0.1.  Measured over 64 val scenes x 8 non-reference views:
    camera centre off by a median 0.37% of the sample baseline, p90 2.91%, max
    75.3%, with 5.1% of relative poses past 5%.  The closer two cameras are, the
    worse it gets -- i.e. it is worst on exactly the pairs whose correspondence
    the model most needs.

    `geometry_fp32=True` fixes it twice over, because either fix alone is fragile:
      1. the arithmetic is reordered to subtract the translations *before*
         rotating, so the cancellation never happens at all and the result would
         survive even in bf16 (measured: max error 3.80% of baseline instead of
         35.4% on the same synthetic sweep);
      2. it runs outside autocast, so the product is fp32 and -- just as
         importantly -- the *returned dtype* is fp32.  Downstream, `ray_condition`
         sizes its pixel grid with `dtype=c2w.dtype`; a bf16 dtype silently
         quantises a 384x672 grid to 425 distinct columns.
    Doing only (1) would leave that dtype leak in place; doing only (2) would
    leave a correct-but-needlessly-ill-conditioned expression for the next person
    who moves it back inside an autocast region.

    Off by default: this changes model inputs *and* the `tgt_view_min_src_dist` /
    `_angle_deg` covariates, so no run using it is comparable to one that does
    not.  It is plan §10's D4 candidate, wired as a sweep factor.
    """
    if not geometry_fp32:
        c2w_ref_inv = torch.linalg.inv(c2w[:, 0:1])
        return c2w_ref_inv @ c2w
    with fp32_geometry(c2w.device, True):
        c2w = c2w.float()
        r_ref_t = c2w[:, 0:1, :3, :3].transpose(-1, -2)      # [B, 1, 3, 3]
        # Subtract first, rotate second: the difference is O(baseline) before it
        # ever reaches a matmul, so there is nothing left to cancel.
        d = c2w[..., :3, 3] - c2w[:, 0:1, :3, 3]             # [B, V, 3]
        rel = torch.zeros_like(c2w)
        rel[..., :3, :3] = r_ref_t @ c2w[..., :3, :3]
        rel[..., :3, 3] = (r_ref_t @ d.unsqueeze(-1)).squeeze(-1)
        rel[..., 3, 3] = 1.0
        return rel


def cape_pose_matrices(c2w_rel, translation_scale, num_repeats=2):
    """World-to-camera matrices for CaPE, one per latent FRAME.

    `c2w_rel` is [B, V, 4, 4] already expressed relative to view 0.  CaPE's logit
    is q^T (P_k P_q^-1)^T k, so it depends only on the RELATIVE pose and the
    choice of reference view provably cancels (probe_pose_rope_scale.py §3 reads
    0.0000 for w2c, 1.77 for c2w -- the c2w form conjugates the relative pose by
    the world frame and is NOT invariant, so this must stay world-to-camera).
    Referencing to view 0 anyway keeps ||t|| on the order of the baseline instead
    of the distance to an arbitrary world origin, which is what bf16 attention
    needs to not lose the relative part to cancellation.

    `translation_scale` is lambda in (R, t) -> (R, lambda t).  That map is a group
    AUTOMORPHISM of SE(3), so exactness and frame-invariance survive verbatim for
    any lambda while the logit inflation shrinks; lambda=0 is rotation-only and
    exactly orthogonal.  Measured inflation: lambda=1 -> 2.26x mean / 4.20x worst
    on the 44-dim block, lambda=0.2 -> 1.10x / 1.37x, lambda=0 -> 1.000x.

    `num_repeats` tiles the V poses over the frame layout.  The sequence layout is
    [RGB x V, disparity x V], so 2 gives each disparity frame its own view's pose.
    """
    w2c = torch.linalg.inv(c2w_rel.float())
    w2c[..., :3, 3] = w2c[..., :3, 3] * float(translation_scale)
    return w2c.repeat(1, num_repeats, 1, 1)


DISPARITY_EPS = 1e-3


def normalized_gaussian_blur(values, mask, sigma):
    """Separable Gaussian blur that IGNORES masked-out pixels.

    values/mask: [B, S, H, W], mask in {0, 1}.  Returns blur(values*mask)/blur(mask),
    i.e. normalized convolution.  A plain blur would drag disparity towards the
    zeros that mark GT-invalid pixels, which reads as a fake depth cliff around
    every sky boundary -- the opposite of the defect this is meant to imitate,
    since the model's own prediction fills those regions with plausible values.

    Pixels whose entire neighbourhood is masked out come back as 0 with a 0
    denominator; the caller decides what to put there.
    """
    sigma = float(sigma)
    if sigma <= 0.0:
        return values * mask
    radius = max(1, int(math.ceil(3.0 * sigma)))
    t = torch.arange(-radius, radius + 1, device=values.device, dtype=torch.float32)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    B, S, H, W = values.shape
    num, den = (values * mask).reshape(B * S, 1, H, W), mask.reshape(B * S, 1, H, W)
    for kernel, pad in ((k.reshape(1, 1, 1, -1), (radius, radius, 0, 0)),
                        (k.reshape(1, 1, -1, 1), (0, 0, radius, radius))):
        # replicate, not zero: a zero pad is a claim that the pixels outside the
        # image are invalid, which would darken the disparity along all four
        # borders of every source view.
        num = F.conv2d(F.pad(num, pad, mode="replicate"), kernel)
        den = F.conv2d(F.pad(den, pad, mode="replicate"), kernel)
    out = num / den.clamp_min(1e-6)
    return (out * (den > 1e-6)).reshape(B, S, H, W)


def perturb_source_disparity(
    src_disparity,
    gain_log_std=0.34,
    shift_std=0.059,
    blur_sigma=2.0,
    lowfreq_std=0.05,
    lowfreq_cells=6,
    fill_invalid=True,
    fill_scale=1.0,
    generator=None,
):
    """Degrade GT source disparity into something shaped like the model's own
    pass-1 estimate.  [B, S, H, W] in the shared-scale convention, in [0, 1].

    WHY THIS EXISTS.  Training shows the warp block in exactly two regimes --
    the GT-depth splat and all zeros (`warp_dropout_prob`).  Inference gets a
    third one 100% of the time: the splat rebuilt from the model's own pass-1
    disparity, which is structurally wrong in ways no zeroed block prepares it
    for.  That is textbook exposure bias, and this function manufactures the
    missing regime at the cost of one extra blur (the splat and VAE encode
    downstream are paid either way).

    THE FOUR COMPONENTS ARE THE FOUR MEASURED DEFECTS, not a guess
    (report_20260913_e_sweep_prep §3, 384 source views of
    bmvs_d_drocefp32_s42@3750):

      fill_invalid  the prediction never emits disparity <= eps, so it fills GT
                    sky with plausible values and the splat mask comes out
                    LARGER than the oracle's (cov 0.5371 vs 0.5266).  Filled
                    from a wide normalized blur, i.e. a smooth continuation of
                    the surrounding geometry, which is what the model does.
      fill_scale    how FAR the filled sky goes, as a fraction of that
                    continuation.  This is the knob the first calibration pass
                    said was missing: filling at the local background value
                    overshot coverage badly (0.585-0.60 against the measured
                    0.5371), because a near-disparity sky gets large parallax
                    and scatters over the target view, whereas the real
                    prediction puts sky far away, where the splatted pixels
                    barely move and pile up on each other.  <1 also happens to
                    be the physically right guess for sky.
      blur_sigma    over-smoothing: the residual after per-view affine
                    alignment owns 71.8% of the 4.84 dB oracle gap, and it
                    looks like lost layered/thin structure.  MEASURED TO BE
                    NEARLY IRRELEVANT to the warp: 1.0 vs 5.0 moves the splat's
                    LPIPS by 0.003.  What actually carries the structural
                    damage is the low-frequency field below -- geometric
                    displacement, not local smoothness.  Kept because it is the
                    cheapest way to also lose thin structure, not because the
                    calibration demanded it.
      gain_log_std  per-view scale error.  The diagnostic fits GT ~ a*pred + b
                    and measures std(a)=0.3495 with median 0.928, so this is
                    NOT a constant bias that could be calibrated away.  Applied
                    to the MEAN-CENTRED disparity with g=exp(N(0,s)), so that
                    a=1/g is lognormal with median 1 and
                    std = sqrt(exp(s^2)-1); s=0.34 gives std(a)=0.350.
      shift_std     per-view offset; the b-only fit measures std=0.0590.

    Centring the gain is what makes these two knobs independent: the raw (a, b)
    least-squares pair has corr=-0.87, which is an artefact of fitting an
    uncentred x, and drawing them independently would produce combinations that
    never occur.

    lowfreq adds a smooth random field on top, because the real error is
    spatially correlated rather than the deterministic result of a blur.

    Every default here is the CALIBRATED value; re-derive with
    tools/calibrate_warp_perturb.py after touching any of them.
    """
    x = src_disparity.float()
    valid = x > DISPARITY_EPS
    mask = valid.to(x.dtype)
    B, S, H, W = x.shape

    if fill_invalid:
        # 4x the structural blur: wide enough to reach across a sky region, and
        # it only ever supplies the pixels the GT has nothing for.
        background = normalized_gaussian_blur(x, mask, 4.0 * max(blur_sigma, 1.0))
        per_view_mean = ((x * mask).sum(dim=(2, 3)) / mask.sum(dim=(2, 3)).clamp_min(1.0))
        background = torch.where(
            background > 0, background, per_view_mean.reshape(B, S, 1, 1).expand_as(x))
        x = torch.where(valid, x, background * float(fill_scale))
        mask = torch.ones_like(x)

    x = normalized_gaussian_blur(x, mask, blur_sigma)

    mu = ((x * mask).sum(dim=(2, 3)) / mask.sum(dim=(2, 3)).clamp_min(1.0)).reshape(B, S, 1, 1)
    noise = torch.randn(B, S, 2, device=x.device, dtype=x.dtype, generator=generator)
    gain = torch.exp(noise[..., 0] * float(gain_log_std)).reshape(B, S, 1, 1)
    shift = (noise[..., 1] * float(shift_std)).reshape(B, S, 1, 1)
    x = mu + gain * (x - mu) + shift

    if lowfreq_std > 0.0 and lowfreq_cells >= 1:
        cells_h = int(lowfreq_cells)
        cells_w = max(1, int(round(cells_h * W / max(H, 1))))
        field = torch.randn(
            B * S, 1, cells_h, cells_w, device=x.device, dtype=x.dtype, generator=generator)
        field = F.interpolate(field, size=(H, W), mode="bilinear", align_corners=False)
        x = x + field.reshape(B, S, H, W) * float(lowfreq_std)

    # Floor rather than clamp to 0: `_splat_sources_pixel` reads validity off
    # `disparity > eps`, so a value that lands at 0 would silently become an
    # invalid pixel and smuggle mask information back in.
    x = x.clamp(2.0 * DISPARITY_EPS, 1.0)
    if not fill_invalid:
        x = torch.where(valid, x, torch.zeros_like(x))
    return x.to(src_disparity.dtype)


class WanRGBDNVSTrainingModule(WanTrainingModule):
    def __init__(
        self,
        args,
        accelerator,
        model_paths=None,
        model_id_with_origin_paths=None,
        trainable_models=None,
        lora_rank=32,
        lora_base_model="dit",
        lora_target_modules="q,k,v,o,ffn.0,ffn.2",
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        local_model_path="./models",
        skip_download=False,
        strip_i2v_image_input=False,
        num_source_views=2,
        num_target_views=1,
        vae_encoding_mode="temporal_padding",
        init_dvd_ckpt=None,
        view_condition_out_channels=16,
        view_condition_role_dim=8,
        view_condition_hidden_channels=64,
        view_condition_plucker_resolution="latent",
        view_condition_pixel_scale=8,
        view_condition_mode="add",
        view_condition_use_role=True,
        view_condition_normalize_moment=False,
        latent_scale=1.0,
        depth_latent_scale=None,
        token_layout="channel",
        warp_mode="none",
        warp_splat_footprint="nearest",
        warp_dropout_prob=0.5,
        warp_two_pass=True,
        warp_perturb_prob=0.0,
        warp_perturb_gain_log_std=0.34,
        warp_perturb_shift_std=0.059,
        warp_perturb_blur_sigma=2.0,
        warp_perturb_lowfreq_std=0.05,
        warp_perturb_lowfreq_cells=6,
        warp_perturb_fill_invalid=True,
        warp_perturb_fill_scale=1.0,
        depth_warp_block=False,
        depth_consist_scale_weight=0.0,
        depth_consist_repro_weight=0.0,
        depth_consist_sigma_max=0.6,
        depth_consist_num_views=3,
        depth_consist_max_samples=2,
        depth_consist_min_support=0.02,
        depth_consist_ratio_clip=4.0,
        depth_consist_trim_quantile=0.9,
        depth_consist_vae_checkpoint=True,
        per_frame_timestep=False,
        clean_source_depth=False,
        clean_source_depth_prob=0.5,
        shared_rope_frame_ids=False,
        rope_frame_axis=None,
        rope_pose_translation_scale=None,
        roce="off",
        camera_geometry_fp32=False,
        rgb_loss_weight=1.0,
        depth_loss_weight=1.0,
        rgb_flow_matching=True,
        rgb_flow_shift=5.0,
        rgb_flow_timestep_sampling="logit_normal",
        rgb_flow_logit_mean=0.0,
        rgb_flow_logit_std=1.0,
        depth_flow_matching=False,
        depth_head_init="pretrained",
        eval_num_inference_steps=4,
        eval_flow_shift=None,
        eval_sample_seed=1234,
        eval_lpips_net="alex",
        eval_depth_vae_ceiling=False,
        eval_vis_downscale=2,
        eval_vis_format="jpg",
        eval_vis_depth_norm="shared",
    ):
        super().__init__(
            args=args,
            accelerator=accelerator,
            model_paths=model_paths,
            model_id_with_origin_paths=model_id_with_origin_paths,
            trainable_models=trainable_models,
            lora_rank=lora_rank,
            lora_base_model=lora_base_model,
            lora_target_modules=lora_target_modules,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            local_model_path=local_model_path,
            skip_download=skip_download,
        )
        # Optional: convert an I2V-structured base (e.g. Wan2.1-I2V-14B-720P)
        # into the T2V structure IN PLACE.  Must run before anything sizes
        # parameters off the DiT (rgb_head copy, patch-embedding widening),
        # i.e. before enable_rgb_head / _expand_patch_embedding further below.
        if strip_i2v_image_input:
            self._strip_i2v_image_input(accelerator)
        self.num_source_views = int(num_source_views)
        self.num_target_views = int(num_target_views)
        self.num_views = self.num_source_views + self.num_target_views
        self.vae_encoding_mode = vae_encoding_mode
        self.rgb_loss_weight = float(rgb_loss_weight)
        self.depth_loss_weight = float(depth_loss_weight)

        # --- Target-view RGB branch: flow matching (rectified flow) ---------
        # The deterministic single-timestep x-prediction inherited from DVD is a
        # good fit for disparity (a genuinely deterministic quantity) but it
        # collapses target-view RGB to the conditional mean.  Flow matching is
        # both the objective Wan2.1 was pretrained with and the standard choice
        # for NVS (CAT3D / ReconX / ViewCrafter).
        self.rgb_flow_matching = bool(rgb_flow_matching)
        self.rgb_flow_shift = float(rgb_flow_shift)
        if rgb_flow_timestep_sampling not in ("logit_normal", "uniform"):
            raise ValueError(
                "rgb_flow_timestep_sampling must be 'logit_normal' or 'uniform', "
                f"got {rgb_flow_timestep_sampling!r}"
            )
        self.rgb_flow_timestep_sampling = rgb_flow_timestep_sampling
        self.rgb_flow_logit_mean = float(rgb_flow_logit_mean)
        self.rgb_flow_logit_std = float(rgb_flow_logit_std)

        # --- Joint RGB-D flow matching ("option A") -------------------------
        # Mixing an x-prediction disparity head with a v-prediction RGB head on
        # a shared trunk has three structural defects: the timestep is a pure
        # nuisance variable for the depth head, target-view depth is regressed
        # from an appearance that is ~79% noise during training, and the two
        # losses live on different scales.  Putting disparity on the same
        # rectified-flow objective removes all three: both heads regress a
        # velocity, share one sigma, and both modalities are noised into the
        # input so depth and appearance can co-denoise.
        self.depth_flow_matching = bool(depth_flow_matching)
        if self.depth_flow_matching and not self.rgb_flow_matching:
            raise ValueError(
                "depth_flow_matching=True requires rgb_flow_matching=True "
                "(the two modalities share a single sigma and timestep)."
            )
        if depth_head_init not in ("pretrained", "dvd"):
            raise ValueError(
                f"depth_head_init must be 'pretrained' or 'dvd', got {depth_head_init!r}"
            )
        self.depth_head_init = depth_head_init
        self.eval_num_inference_steps = max(1, int(eval_num_inference_steps))
        # The shift map serves two unrelated purposes: in _sample_sigma it sets
        # WHERE the training noise levels concentrate, in inference_sigmas it
        # sets WHERE the ODE puts its steps.  One number cannot be right for
        # both -- shift=5 with few steps leaves a single 0.55-wide sigma jump at
        # the end of the schedule.  Defaults to rgb_flow_shift so nothing
        # changes unless a run asks for it.
        self.eval_flow_shift = (
            self.rgb_flow_shift if eval_flow_shift is None else float(eval_flow_shift)
        )
        self.eval_sample_seed = int(eval_sample_seed)
        # PSNR systematically rewards blur: a model that collapses to the
        # conditional mean beats a sharp model whose sample is slightly
        # misaligned.  For a generative target-view branch that is exactly the
        # wrong incentive, so LPIPS is reported alongside it.  Built lazily on
        # first use so the AlexNet weights are not loaded on ranks that never
        # evaluate.
        self.eval_lpips_net = eval_lpips_net
        # Measures the VAE round-trip floor for the depth metrics -- see the block
        # in evaluate().  Checkpoint-independent, so this stays off during
        # training and is switched on for one diagnostic pass.
        self.eval_depth_vae_ceiling = bool(eval_depth_vae_ceiling)
        # Held inside a plain dict on purpose.  `nn.Module.__setattr__`
        # intercepts any value that is itself an nn.Module and registers it in
        # `self._modules`, which would put the AlexNet weights into every
        # checkpoint (~6 MB of dead weight plus tied-tensor warnings on save)
        # and into `.train()`/DDP traversal.  A dict is not an nn.Module, so
        # the network inside it stays invisible to all of that.
        self._lpips_holder = {}
        # The raw eval grid is (num_source+2) x 3 full-resolution tiles, i.e.
        # 4052x1160 for s4t3 -> several MB per PNG, times num_target_views per
        # sample, times every validate_step.  Downscale and use JPEG instead.
        self.eval_vis_downscale = max(1, int(eval_vis_downscale))
        if eval_vis_format not in ("jpg", "png"):
            raise ValueError(f"eval_vis_format must be 'jpg' or 'png', got {eval_vis_format!r}")
        self.eval_vis_format = eval_vis_format
        # "shared": render predicted disparity with the SAME fixed [0,1] -> plasma
        # mapping as the GT row.  Under disparity_norm_mode="shared_scale" the GT
        # row is deliberately low-contrast (headroom 1.5 puts most disparity in
        # ~[0.17, 0.56]), so the legacy "per_view" 2%/98% stretch made every
        # prediction span the full colour ramp and looked far more structured --
        # and correctly scaled -- than it was.
        if eval_vis_depth_norm not in ("shared", "per_view"):
            raise ValueError(
                f"eval_vis_depth_norm must be 'shared' or 'per_view', got {eval_vis_depth_norm!r}"
            )
        self.eval_vis_depth_norm = eval_vis_depth_norm

        # --- Latent scaling ------------------------------------------------
        # Rectified flow assumes Var(x0) ~ 1: sigma is then a meaningful SNR and
        # the velocity target noise - x0 has both terms on the same scale.  The
        # Wan VAE emits latents with std ~0.47-0.52 for RGB and E[x^2] ~0.355
        # for shared-scale disparity, so the signal sits ~4x below where the
        # objective expects it and every sigma is effectively noisier than it
        # claims.  Multiplying the latents restores the assumption; the inverse
        # is applied before every VAE decode so pixel space is untouched.
        self.latent_scale = float(latent_scale)
        self.depth_latent_scale = (
            self.latent_scale if depth_latent_scale is None else float(depth_latent_scale)
        )

        # --- Token layout ---------------------------------------------------
        # "channel":  RGB and disparity of one view share a token (depth is a
        #             channel block).  Cheap -- token count unchanged -- but one
        #             token, one AdaLN modulation and one 1536-d residual stream
        #             have to carry both modalities.
        # "sequence": disparity becomes its own set of frames, doubling the
        #             sequence to 2*V*h*w tokens.  Attention is quadratic, so
        #             this costs ~4x in the attention and ~2x elsewhere.
        if token_layout not in ("channel", "sequence"):
            raise ValueError(f"token_layout must be 'channel' or 'sequence', got {token_layout!r}")
        self.token_layout = token_layout

        # --- Warped-source conditioning -------------------------------------
        if warp_mode not in ("none", "splat"):
            raise ValueError(f"warp_mode must be 'none' or 'splat', got {warp_mode!r}")
        self.warp_mode = warp_mode
        # How a z-buffer-kept splat deposits its colour: "nearest" is the
        # historical single-pixel last-write-wins path (bit-identical, all
        # pre-existing arms reproduce exactly); "bilinear" splits each kept
        # splat over a 2x2 footprint with a continuous coverage mask.  This
        # changes warp_block, i.e. a model input, so arms must not flip it
        # mid-comparison -- same caution class as warp_mode itself.
        if warp_splat_footprint not in ("nearest", "bilinear"):
            raise ValueError(
                f"warp_splat_footprint must be 'nearest' or 'bilinear', "
                f"got {warp_splat_footprint!r}"
            )
        self.warp_splat_footprint = warp_splat_footprint
        # The warp block is dropped this often during training.  Without it the
        # model would never have seen a zeroed block, and the first pass of
        # two-pass inference -- which has no depth estimate to warp with yet --
        # would be out of distribution.
        self.warp_dropout_prob = float(warp_dropout_prob)
        self.warp_two_pass = bool(warp_two_pass)
        # --- FACTOR e-warpnoise: the third warp regime ------------------------
        # Orthogonal to warp_dropout_prob BY CONSTRUCTION: the Bernoulli below
        # runs BEFORE the dropout one and only decides what the non-zero block
        # contains, so `warp_dropout_prob` keeps its tuned value and the training
        # mixture becomes 50% zeroed / 25% GT splat / 25% degraded splat at
        # prob=0.5.  Note the GT-splat regime is the one that NEVER occurs at
        # inference; prob=1.0 drops it entirely and is a legitimate (more
        # aggressive) level of this factor.
        self.warp_perturb_prob = float(warp_perturb_prob)
        self.warp_perturb_kwargs = dict(
            gain_log_std=float(warp_perturb_gain_log_std),
            shift_std=float(warp_perturb_shift_std),
            blur_sigma=float(warp_perturb_blur_sigma),
            lowfreq_std=float(warp_perturb_lowfreq_std),
            lowfreq_cells=int(warp_perturb_lowfreq_cells),
            fill_invalid=bool(warp_perturb_fill_invalid),
            fill_scale=float(warp_perturb_fill_scale),
        )
        if self.warp_perturb_prob > 0.0:
            # Each of these would train a silent duplicate of the arm this one is
            # read against, and the run would look perfectly healthy for 46 hours.
            if self.warp_mode != "splat":
                raise ValueError(
                    "warp_perturb_prob>0 requires warp_mode='splat': with no warp "
                    "block there is nothing to degrade, so the factor would be a "
                    "silent no-op."
                )
            if not 0.0 < self.warp_perturb_prob <= 1.0:
                raise ValueError(
                    f"warp_perturb_prob must be in (0, 1], got {self.warp_perturb_prob}"
                )
            k = self.warp_perturb_kwargs
            if not (k["gain_log_std"] > 0.0 or k["shift_std"] > 0.0
                    or k["blur_sigma"] > 0.0 or k["lowfreq_std"] > 0.0
                    or k["fill_invalid"]):
                raise ValueError(
                    "warp_perturb_prob>0 with every degradation amplitude at zero "
                    "is the identity map -- this arm would be a bit-identical "
                    "duplicate of its own baseline."
                )
            if self.warp_dropout_prob >= 1.0:
                raise ValueError(
                    "warp_perturb_prob>0 with warp_dropout_prob>=1 degrades a block "
                    "that is then zeroed for every sample; the factor would measure "
                    "nothing."
                )
        # In the sequence layout the depth frames get the RGB warp block's 17
        # channels filled with literal zeros, purely so both halves of the
        # sequence share one patch embedding.  Filling them with the splatted
        # target-frame disparity turns that dead weight into the same geometric
        # prior the RGB frames already get, at no extra input channels -- so a
        # checkpoint trained without it stays shape-compatible.
        self.depth_warp_block = bool(depth_warp_block) and self.warp_mode == "splat"

        # --- FACTOR f-consist: cross-view consistency of the SOURCE disparity --
        # report_20260916 §10: the six predicted source disparities each want
        # their own scale (spread 8.7%, 544x the within-view fit noise, worst/best
        # up to 1.87), and §10b: it is that DISAGREEMENT and not the error
        # magnitude that predicts pass-2 degradation.  Nothing in the objective
        # asks for consistency -- the depth loss is an independent per-frame MSE.
        # §11 is the other half of the decision: the RGB<->depth cross-modal
        # channel was measured at a total capacity of 0.0090 LPIPS, identical to
        # what a third pass already buys for free, so this factor is deliberately
        # cross-VIEW only and adds nothing cross-modal.  See
        # utils/depth_consistency.py for the two terms and their calibration.
        self.depth_consist_scale_weight = float(depth_consist_scale_weight)
        self.depth_consist_repro_weight = float(depth_consist_repro_weight)
        self.depth_consist_on = (
            self.depth_consist_scale_weight > 0.0 or self.depth_consist_repro_weight > 0.0
        )
        self.depth_consist_sigma_max = float(depth_consist_sigma_max)
        self.depth_consist_num_views = int(depth_consist_num_views)
        self.depth_consist_max_samples = int(depth_consist_max_samples)
        self.depth_consist_kwargs = dict(
            min_support=float(depth_consist_min_support),
            ratio_clip=float(depth_consist_ratio_clip),
            trim_quantile=float(depth_consist_trim_quantile),
        )
        self.depth_consist_vae_checkpoint = bool(depth_consist_vae_checkpoint)
        if self.depth_consist_on:
            # Every one of these would train a silent duplicate of the arm this
            # one is read against, which is the most expensive failure mode a
            # sweep has: the run converges, reports a full metric set, and answers
            # a question nobody asked.
            if not self.depth_flow_matching:
                raise ValueError(
                    "depth_consist_* requires depth_flow_matching=true: the term "
                    "needs the x0 estimate implied by the depth velocity head, "
                    "and without that head there is no predicted source depth to "
                    "make consistent."
                )
            if self.warp_mode != "splat":
                raise ValueError(
                    "depth_consist_* requires warp_mode='splat': the pixel-space "
                    "intrinsics the geometry needs are only exported as "
                    "inputs['pixel_K'] when the warp block is built, so the term "
                    "would be a silent no-op."
                )
            if self.num_source_views < 2:
                raise ValueError(
                    "depth_consist_* needs at least 2 source views; with one view "
                    "there is no pair and the term is identically zero."
                )
            if not 2 <= self.depth_consist_num_views <= self.num_source_views:
                raise ValueError(
                    f"depth_consist_num_views must be in [2, {self.num_source_views}], "
                    f"got {self.depth_consist_num_views}"
                )
            if self.depth_consist_max_samples < 1:
                raise ValueError(
                    "depth_consist_max_samples must be >= 1; 0 would disable the "
                    "factor while leaving its weights set, i.e. a duplicate of the "
                    "baseline under an arm's name."
                )
            if not 0.0 < self.depth_consist_sigma_max <= 1.0:
                raise ValueError(
                    "depth_consist_sigma_max must be in (0, 1]; at 0 no sample is "
                    "ever eligible and the term never fires."
                )
        # Source RGB frames carry CLEAN latents but the shared AdaLN modulation
        # tells the whole sequence they sit at the sampled sigma.  Only the
        # sequence layout can fix this: in the channel layout a source view's
        # clean RGB and its noisy disparity share one token, so there is no
        # single correct timestep for it.
        self.per_frame_timestep = bool(per_frame_timestep) and token_layout == "sequence"

        # --- Source depth as a CLEAN condition --------------------------------
        # Source depth is the easy half of the problem: the source RGB is given
        # clean in the same frame, so it is a well-posed discriminative task, and
        # pass 1 of two-pass inference already solves it.  Pass 2 then throws that
        # away -- it rebuilds the splat from the pass-1 depth but re-noises the
        # source-depth frames from scratch, so the model is free to contradict the
        # correspondence it was just handed.  With this on, pass 2 pins those
        # frames to the pass-1 estimate at timestep 0 instead.
        #
        # Training must cover BOTH regimes or pass 1 goes out of distribution,
        # exactly as with warp_dropout_prob: on `clean_source_depth_prob` of
        # samples the source-depth frames carry the clean GT latent and are
        # excluded from the depth loss, otherwise they stay noisy and supervised.
        self.clean_source_depth = bool(clean_source_depth)
        self.clean_source_depth_prob = float(clean_source_depth_prob)
        if self.clean_source_depth:
            # Every one of these is load-bearing, and a silently-degraded version
            # of this factor would spend 40 GPU-hours measuring nothing.
            if not self.per_frame_timestep:
                raise ValueError(
                    "clean_source_depth requires per_frame_timestep=True (and hence "
                    "token_layout='sequence'): a clean frame has to be TOLD it is "
                    "clean, and a single scalar timestep cannot say that."
                )
            if not self.depth_flow_matching:
                raise ValueError(
                    "clean_source_depth requires depth_flow_matching=True; without "
                    "it there are no source-depth frames to pin."
                )
            if not (self.warp_mode == "splat" and self.warp_two_pass):
                raise ValueError(
                    "clean_source_depth requires warp_mode='splat' and "
                    "warp_two_pass=True.  Single-pass inference has no source-depth "
                    "estimate to inject, so the factor would change training only "
                    "and be unreadable at eval."
                )
            if not 0.0 < self.clean_source_depth_prob < 1.0:
                raise ValueError(
                    "clean_source_depth_prob must be strictly between 0 and 1: 0 "
                    "makes the factor a no-op, 1 leaves pass 1 out of distribution."
                )

        # --- What the RoPE frame axis encodes ---------------------------------
        # See ROPE_FRAME_AXES above for the four settings and why they exist.
        # Every one of them changes positional encoding, so checkpoints are NOT
        # interchangeable across them.
        if bool(shared_rope_frame_ids) and token_layout != "sequence":
            raise ValueError(
                "shared_rope_frame_ids only means anything with "
                "token_layout='sequence'; in the channel layout RGB and depth "
                "already share a frame position."
            )
        self.rope_frame_axis = resolve_rope_frame_axis(rope_frame_axis, shared_rope_frame_ids)
        if self.rope_frame_axis != "view_modality" and token_layout != "sequence":
            raise ValueError(
                f"rope_frame_axis={self.rope_frame_axis!r} only means anything with "
                f"token_layout='sequence'; in the channel layout there is one frame "
                f"per view, so the frame axis cannot encode modality."
            )
        # Kept as an attribute because the diagnostics print it to prove which
        # RoPE geometry a checkpoint is being evaluated under.
        self.shared_rope_frame_ids = self.rope_frame_axis == "view"
        # Cached on the module: the ids never change, and model_fn moves them to
        # the freqs table's device itself.
        self._rope_frame_ids = rope_frame_ids_for_axis(self.rope_frame_axis, self.num_views)

        # --- D3: continuous 6-DoF pose on the frame axis ----------------------
        self.rope_pose_translation_scale = resolve_rope_pose_translation_scale(
            self.rope_frame_axis, rope_pose_translation_scale
        )

        # --- D-extra: RoCE, a learned camera phase on the h/w axes -------------
        # Plan §D-extra / §12-6.  Independent of every rope_frame_axis setting:
        # RoCE rotates the h/w rotary channels and CaPE right-multiplies the frame
        # ones, so they touch disjoint dims of each head and rope_apply raises if
        # that ever stops being true.  Deliberately NOT combined with an axis in a
        # single arm, though -- two interventions in one arm is unreadable.
        self.roce_mode = resolve_roce_mode(roce)

        # --- D4: compute the camera geometry outside autocast ------------------
        # Plan §10.  One flag rather than four, because the four sites are one
        # defect -- autocast downcasting camera math that is written in fp32 -- and
        # splitting them into separate arms would spend four node-days measuring
        # pieces of a single bug fix.  The sites: relative_c2w (the relative pose),
        # ViewConditionEncoder._plucker (the ray field's pixel grid and the
        # direction rotation) and splat_sources_to_targets (the reprojection).
        # None of them is learned and none carries a gradient.
        self.camera_geometry_fp32 = bool(camera_geometry_fp32)

        # forward_preprocess is called from BOTH the training step and evaluate(),
        # and the training loop never calls model.eval() -- so self.training is
        # True inside evaluate() and cannot be used to tell the two apart.
        # evaluate() sets this flag instead.
        self._eval_inputs = False
        # 16 VAE channels for the splatted image + 1 validity mask.
        self.warp_latent_channels = 17 if self.warp_mode == "splat" else 0
        if self.warp_mode == "splat" and vae_encoding_mode != "independent":
            # The splat is built per target view and has to line up 1:1 with the
            # latent frames; a temporal VAE collapses 4 frames into 1.
            raise ValueError("warp_mode='splat' requires vae_encoding_mode='independent'")

        self.view_condition_mode = view_condition_mode
        self.view_condition_encoder = ViewConditionEncoder(
            out_channels=view_condition_out_channels,
            role_dim=view_condition_role_dim,
            hidden_channels=view_condition_hidden_channels,
            plucker_resolution=view_condition_plucker_resolution,
            pixel_scale=view_condition_pixel_scale,
            mode=view_condition_mode,
            use_role=view_condition_use_role,
            normalize_moment=view_condition_normalize_moment,
            geometry_fp32=self.camera_geometry_fp32,
        )
        accelerator.print(
            f"ViewConditionEncoder: mode={view_condition_mode} "
            f"plucker_resolution={view_condition_plucker_resolution} "
            f"pixel_scale={view_condition_pixel_scale} "
            f"plucker_channels={self.view_condition_encoder.plucker_channels} "
            f"use_role={self.view_condition_encoder.use_role} "
            f"normalize_moment={self.view_condition_encoder.normalize_moment} "
            f"out_channels={self.view_condition_encoder.out_channels}"
        )

        # Load precomputed Wan2.1 null text embedding (empty-string T5 output).
        # Using zeros is out-of-distribution; the real null embed is what the
        # base model was trained with for unconditional (CFG null) conditioning.
        null_embed_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../../../empty_text_embed_wan.pt"
        )
        if os.path.exists(null_embed_path):
            accelerator.print(f"Loading null text embedding from {null_embed_path}")
            null_embed = torch.load(null_embed_path, map_location="cpu", weights_only=True)
        else:
            accelerator.print(
                f"WARNING: {null_embed_path} not found, falling back to zeros for text context."
            )
            null_embed = torch.zeros(1, 512, 4096)
        # Store as plain attribute (not a buffer) to avoid DDP broadcasting a CPU
        # tensor via NCCL (NCCL has no CPU backend → RuntimeError).
        self._null_text_embed = null_embed.to(torch.bfloat16)

        # Cache for the VAE latent of an all-zero frame, keyed by (H, W, dtype).
        # A plain dict, so the tensors are never registered as submodules/buffers
        # and cannot leak into checkpoints or DDP synchronisation.
        self._zero_latent_cache = {}

        pretrained_head_state = None
        if init_dvd_ckpt is not None:
            from safetensors.torch import load_file

            # Snapshot the pretrained Wan2.1 output head before the DVD weights
            # overwrite it.  DVD's head is an x-prediction head specialised for
            # disparity; the flow-matching RGB branch wants the original
            # velocity head instead.
            pretrained_head_state = {
                k: v.detach().clone() for k, v in self.pipe.dit.head.state_dict().items()
            }

            state_dict = load_file(init_dvd_ckpt, device="cpu")
            # DVD checkpoints are saved with a "pipe.dit." prefix on all DiT keys;
            # strip it so keys match the bare WanModel state dict.
            prefix = "pipe.dit."
            stripped = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
            if stripped:
                accelerator.print(
                    f"Stripping '{prefix}' prefix from DVD checkpoint ({len(stripped)} / {len(state_dict)} keys)"
                )
                state_dict = stripped

            # The DVD checkpoint stores weights in LoRA format: `base_layer.*`
            # holds the FROZEN Wan2.1 weights and `lora_A`/`lora_B` hold
            # everything DVD actually learned.  Dropping the adapters therefore
            # silently reduces "initialise from DVD" to "initialise from
            # Wan2.1" -- verified by casting Wan2.1 to bf16, which reproduces
            # every base_layer tensor bit-exactly.  Merge the delta into the
            # base instead: W' = W + (alpha/r)*B@A.  This yields a dense weight,
            # so our own LoRA rank is irrelevant and no mismatch can occur.
            model_keys = set(self.pipe.dit.state_dict().keys())
            has_base_layer_in_ckpt = any("base_layer" in k for k in state_dict)
            has_base_layer_in_model = any("base_layer" in k for k in model_keys)

            if has_base_layer_in_ckpt:
                # DVD trained with `lora_rank: 512` and no explicit alpha, and
                # DiffusionTrainingModule defaults lora_alpha to lora_rank, so
                # the PEFT scaling alpha/r is exactly 1.
                lora_scale = 1.0
                remapped = {}
                merged = 0
                for k, v in state_dict.items():
                    if "lora_A" in k or "lora_B" in k:
                        continue  # consumed by the merge below
                    if k.endswith(".base_layer.weight"):
                        stem = k[: -len(".base_layer.weight")]
                        a = state_dict.get(f"{stem}.lora_A.default.weight")
                        b = state_dict.get(f"{stem}.lora_B.default.weight")
                        if a is not None and b is not None:
                            v = (
                                v.float() + lora_scale * (b.float() @ a.float())
                            ).to(v.dtype)
                            merged += 1
                    if not has_base_layer_in_model:
                        k = k.replace(".base_layer.", ".")  # FFT: strip wrapper
                    remapped[k] = v
                accelerator.print(
                    f"Remapping DVD LoRA checkpoint: merged {merged} lora_A/B "
                    f"deltas into base weights (scale={lora_scale}), "
                    f"{'stripped base_layer for FFT model' if not has_base_layer_in_model else 'kept base_layer for LoRA model'} "
                    f"({len(state_dict)} → {len(remapped)} keys)"
                )
                state_dict = remapped

            missing, unexpected = self.pipe.dit.load_state_dict(state_dict, strict=False)
            accelerator.print(
                f"Loaded init DVD checkpoint from {init_dvd_ckpt}; missing={len(missing)}, unexpected={len(unexpected)}"
            )

        if not hasattr(self.pipe.dit, "rgb_head") or self.pipe.dit.rgb_head is None:
            self.pipe.dit.enable_rgb_head(copy_from_head=True)
            if self.rgb_flow_matching and pretrained_head_state is not None:
                self.pipe.dit.rgb_head.load_state_dict(pretrained_head_state)
                accelerator.print(
                    "Initialised rgb_head from the pretrained Wan2.1 head "
                    "(velocity prediction) instead of the DVD disparity head."
                )
        self.pipe.dit.rgb_head.train()

        if self.depth_flow_matching:
            # The disparity head now regresses a velocity, whose output
            # statistics are nothing like DVD's x-prediction head (velocity is
            # roughly unit-scale, the clean latent is not).  Re-initialising it
            # from the original Wan2.1 velocity head avoids starting from a
            # systematically mis-scaled output layer.
            if self.depth_head_init == "pretrained" and pretrained_head_state is not None:
                self.pipe.dit.head.load_state_dict(pretrained_head_state)
                accelerator.print(
                    "Joint RGB-D flow: re-initialised dit.head from the pretrained "
                    "Wan2.1 velocity head (depth_head_init='pretrained')."
                )
            elif pretrained_head_state is None:
                # No DVD checkpoint was loaded, so dit.head is still the original
                # Wan2.1 velocity head -- nothing to restore.
                accelerator.print(
                    "Joint RGB-D flow: dit.head is already the pretrained Wan2.1 "
                    "velocity head (no DVD checkpoint was loaded)."
                )
            else:
                accelerator.print(
                    f"Joint RGB-D flow: keeping the DVD disparity head "
                    f"(depth_head_init={self.depth_head_init!r})."
                )
        self.depth_latent_channels = 0
        if self._needs_extra_input_channels():
            self._expand_patch_embedding(accelerator)
        if self.roce_mode != "off":
            # The camera field RoCE consumes is the encoder's RAW Plucker field, so
            # its channel count is plucker_channels (384 at pixel_scale=8), not
            # out_channels -- those coincide in concat mode and do not in add mode.
            self.pipe.dit.enable_roce(
                self.view_condition_encoder.plucker_channels, self.roce_mode
            )
            n_roce = sum(
                p.numel()
                for n, p in self.pipe.dit.named_parameters()
                if "rope_phase" in n or "roce_cam_embedding" in n
            )
            accelerator.print(
                f"RoCE enabled: mode={self.roce_mode} "
                f"cam_channels={self.view_condition_encoder.plucker_channels} "
                f"phase MLPs zero-initialised (step 0 is bit-identical to roce=off), "
                f"{n_roce / 1e6:.1f}M new parameters."
            )

    def _strip_i2v_image_input(self, accelerator):
        """Convert an I2V-structured Wan DiT into the T2V structure, in place.

        The I2V checkpoints (e.g. Wan2.1-I2V-14B-720P, key-hash 6bfcfb3b...) are
        detected by ModelManager as has_image_input=True / patch_embedding
        in_channels=36 (16 RGB latents + 4 mask + 16 image latents).  This NVS
        pipeline never provides the CLIP feature / image-latent inputs, so
        model_fn_wan_video would crash on the `if dit.has_image_input:` branch.
        Instead of feeding dummy inputs we CUT the model down to the T2V
        structure and drop the image-only weights:

          * every block's cross-attention loses k_img / v_img / norm_k_img
            (~2 x dim x dim per block of trained I2V image attention);
          * the global img_emb MLP is deleted;
          * patch_embedding keeps only the first 16 input channels (the RGB
            latent block; the 20 I2V y/mask channels are discarded).

        Everything downstream (rgb_head copy from the velocity head, the
        depth/Plucker/warp channel widening in _expand_patch_embedding, LoRA,
        freeze) then sees a plain T2V model, exactly like the 1.3B path.
        """
        dit = self.pipe.dit
        if not getattr(dit, "has_image_input", False):
            accelerator.print(
                "[strip_i2v] DiT already has has_image_input=False "
                "(T2V base); nothing to strip."
            )
            return
        n_blocks = 0
        for block in dit.blocks:
            ca = block.cross_attn
            if getattr(ca, "has_image_input", False):
                ca.has_image_input = False
                for attr in ("k_img", "v_img", "norm_k_img"):
                    if hasattr(ca, attr):
                        delattr(ca, attr)
                n_blocks += 1
        if hasattr(dit, "img_emb"):
            delattr(dit, "img_emb")
        dit.has_image_input = False
        conv = dit.patch_embedding
        if conv.in_channels != 16:
            new_conv = torch.nn.Conv3d(
                16,
                conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                bias=conv.bias is not None,
            ).to(device=conv.weight.device, dtype=conv.weight.dtype)
            with torch.no_grad():
                new_conv.weight.copy_(conv.weight[:, :16])
                if conv.bias is not None:
                    new_conv.bias.copy_(conv.bias)
            dit.patch_embedding = new_conv
            accelerator.print(
                f"[strip_i2v] patch_embedding in_channels {conv.in_channels} -> 16 "
                f"(dropped the I2V mask/image-latent channels)."
            )
        accelerator.print(
            f"[strip_i2v] removed image cross-attention from {n_blocks} blocks and "
            f"img_emb; dit.has_image_input -> False. The DiT is now structurally a "
            f"T2V model (I2V-trained trunk preserved)."
        )

    def _needs_extra_input_channels(self):
        """True when patch_embedding must be widened beyond the 16 RGB channels."""
        return (
            (self.depth_flow_matching and self.token_layout == "channel")
            or self.view_condition_mode == "concat"
            or self.warp_mode != "none"
            or self.token_layout == "sequence"
        )

    def _get_lpips(self, device):
        """Lazily build the LPIPS network; returns None if lpips is unavailable.

        Kept off the module's parameter list (plain attribute, eval mode,
        requires_grad False) so DDP never tries to sync or train it.
        """
        if "net" in self._lpips_holder:   # None here means a failed attempt
            return self._lpips_holder["net"]
        try:
            import lpips as lpips_lib
        except ImportError:
            self.accelerator.print(
                "WARNING: `lpips` is not installed, tgt_rgb_lpips will be NaN. "
                "Install with: pip install lpips"
            )
            self._lpips_holder["net"] = None
            return None
        net = lpips_lib.LPIPS(net=self.eval_lpips_net, verbose=False)
        net = net.to(device).eval()
        for param in net.parameters():
            param.requires_grad = False
        self._lpips_holder["net"] = net
        return net

    def _expand_patch_embedding(self, accelerator):
        """Widen patch_embedding so noisy depth latents can be fed in alongside RGB.

        Joint flow matching needs the depth latents as an *input*, not just a
        target.  We concatenate them on the channel axis rather than appending
        them as extra frames: the RGB and disparity of one view then live in the
        same token (perfectly pixel-aligned, which is exactly the coupling we
        want) and the token count -- hence the quadratic attention cost -- is
        unchanged.

        The new channel block is zero-initialised, so at step 0 the network is
        numerically identical to the RGB-only model and the DVD trunk
        initialisation is preserved.
        """
        conv = self.pipe.dit.patch_embedding
        old_in = conv.in_channels
        extra_depth = 0
        if self.depth_flow_matching and self.token_layout == "channel":
            extra_depth = self.pipe.dit.head.head.out_features // (
                conv.kernel_size[0] * conv.kernel_size[1] * conv.kernel_size[2]
            )
        # "concat" camera conditioning rides in on its own channel block rather
        # than being summed into the RGB latents, so the DiT gives every Plucker
        # channel an independent weight instead of forcing the whole field
        # through a 3-layer CNN bottleneck.
        extra_plucker = (
            self.view_condition_encoder.out_channels
            if self.view_condition_mode == "concat"
            else 0
        )
        extra_warp = self.warp_latent_channels
        # In "sequence" layout depth occupies its own frames rather than its own
        # channels, so the conv cannot tell the two modalities apart from the
        # content alone.  One constant channel (0 on RGB frames, 1 on depth
        # frames) gives it a learned per-modality bias for the price of one
        # column of weights, without touching the DiT itself.
        extra_modality = 1 if self.token_layout == "sequence" else 0
        new_in = old_in + extra_depth + extra_plucker + extra_warp + extra_modality
        new_conv = torch.nn.Conv3d(
            new_in,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=conv.bias is not None,
        ).to(device=conv.weight.device, dtype=conv.weight.dtype)
        with torch.no_grad():
            new_conv.weight.zero_()
            new_conv.weight[:, :old_in].copy_(conv.weight)
            if conv.bias is not None:
                new_conv.bias.copy_(conv.bias)
        self.pipe.dit.patch_embedding = new_conv
        self.depth_latent_channels = extra_depth
        accelerator.print(
            f"patch_embedding in_channels {old_in} -> {new_in} "
            f"(+{extra_depth} depth, +{extra_plucker} plucker, +{extra_warp} warp, "
            f"+{extra_modality} modality; new blocks zero-initialised)."
        )

    def set_training_param(self):
        super().set_training_param()
        for param in self.view_condition_encoder.parameters():
            param.requires_grad = True
        if hasattr(self.pipe.dit, "rgb_head") and self.pipe.dit.rgb_head is not None:
            for param in self.pipe.dit.rgb_head.parameters():
                param.requires_grad = True
        if self._needs_extra_input_channels():
            # In LoRA mode patch_embedding is frozen by default, which would
            # leave the newly added depth / Plucker / warp channels at exactly
            # zero forever and silently degrade the model back to the RGB-only,
            # camera-unconditioned baseline.
            for param in self.pipe.dit.patch_embedding.parameters():
                param.requires_grad = True
        if self.roce_mode != "off":
            # Zero-init AND frozen would make the arm a bit-exact duplicate of its
            # anchor for 4000 steps and report it as a null result.  LoRA mode
            # freezes everything it was not told about, so these have to be named.
            for name, param in self.pipe.dit.named_parameters():
                if "rope_phase" in name or "roce_cam_embedding" in name:
                    param.requires_grad = True
        if self.depth_flow_matching:
            # dit.head is the *depth* velocity head in joint mode.  rgb_head is
            # unfrozen above, so leaving this one frozen would let only one of
            # the two symmetric heads adapt -- an arbitrary asymmetry that would
            # bias any FFT-vs-LoRA comparison.
            for param in self.pipe.dit.head.parameters():
                param.requires_grad = True
        self.accelerator.print(
            f"NVS trainable parameters: {sum(param.numel() for param in self.trainable_modules())}"
        )

    def _stack_metric_depth(self, data):
        return torch.cat([data["source_depth_metric"], data["target_depth_metric"]], dim=1)

    def _encode_views(self, views, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        if self.vae_encoding_mode == "independent":
            batch_size, num_views, channels, height, width = views.shape
            flattened = rearrange(views, "b v c h w -> (b v) c 1 h w")
            latents = self.pipe.vae.encode(
                flattened,
                device=self.pipe.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
            latent_height, latent_width = latents.shape[-2:]
            latents = rearrange(latents, "(b v) c 1 h w -> b c v h w", b=batch_size, v=num_views)
            return latents, (latent_height, latent_width)

        latent_videos = views.permute(0, 2, 1, 3, 4).contiguous()
        latents = self.pipe.vae.encode(
            latent_videos,
            device=self.pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        latent_height, latent_width = latents.shape[-2:]
        return latents, (latent_height, latent_width)

    def _zero_view_latent(self, reference_views):
        """VAE latent of an all-zero frame, computed once and cached.

        In ``independent`` encoding mode every view goes through the VAE on its
        own, so the blanked target slots of the conditioning tensor always map
        to this same constant.  Recomputing it every step cost a full encode of
        ``num_target_views * batch_size`` frames for zero information.
        """
        key = (reference_views.shape[-2], reference_views.shape[-1], reference_views.dtype)
        cached = self._zero_latent_cache.get(key)
        if cached is None:
            zeros = torch.zeros(
                (1, 1) + tuple(reference_views.shape[2:]),
                device=reference_views.device,
                dtype=reference_views.dtype,
            )
            with torch.no_grad():
                cached, _ = self._encode_views(zeros)  # [1, C, 1, h, w]
            self._zero_latent_cache[key] = cached
        return cached

    @torch.no_grad()
    def _splat_sources_pixel(self, rgb_views, src_disparity, K, c2w, return_target_disparity=False):
        """Forward-splat the source images into every target view, in PIXEL space.

        `src_disparity` is [B, S, H, W] in the shared-scale convention, where
        `disp = (1/d) / s` and the camera translations have already been
        multiplied by the same `s`.  In that normalised world the depth is
        exactly `1 / disp`, so no metric depth and no separate scale estimate
        are needed -- which is also why this works at inference from the model's
        own predicted disparity.

        Returns (warped [B, T, 3, H, W], mask [B, T, 1, H, W]), both float32,
        plus the target-frame disparity [B, T, 1, H, W] when asked.
        Split out of _build_warp_block so evaluate() can score the raw splat as
        a non-learned RGB baseline without paying for the VAE encode.
        """
        num_src = self.num_source_views
        eps = 1e-3
        valid = src_disparity > eps
        depth = torch.where(
            valid,
            1.0 / src_disparity.clamp_min(eps),
            torch.zeros_like(src_disparity),
        )
        return splat_sources_to_targets(
            src_rgb=rgb_views[:, :num_src].float(),
            src_depth=depth.float(),
            src_valid=valid,
            src_K=K[:, :num_src],
            src_c2w=c2w[:, :num_src],
            tgt_K=K[:, num_src:],
            tgt_c2w=c2w[:, num_src:],
            return_target_disparity=return_target_disparity,
            geometry_fp32=self.camera_geometry_fp32,
            footprint=self.warp_splat_footprint,
        )

    def _to_view_block(self, target_block, latent_size):
        """Pad a [B, C, T, h, w] target-only block out to all num_views slots."""
        block = torch.zeros(
            target_block.shape[0],
            target_block.shape[1],
            self.num_views,
            *latent_size,
            device=target_block.device,
            dtype=target_block.dtype,
        )
        block[:, :, self.num_source_views:] = target_block
        return block

    @torch.no_grad()
    def _maybe_perturb_src_disparity(self, src_disparity):
        """Per-sample Bernoulli mixture of clean and degraded source disparity.

        Per-sample and not per-batch, for the same reason `warp_dropout_prob` is:
        a mixed batch keeps both regimes inside every gradient step.  All views of
        a sample are degraded together, because the warp block for one target view
        is built from all six sources at once -- degrading a subset would produce a
        splat that is internally inconsistent in a way inference never is.

        Returns `src_disparity` unchanged, and does no work at all, when the factor
        is off; this is the no-op path the whole c/d-round baseline reuse depends on.
        """
        if self.warp_perturb_prob <= 0.0:
            return src_disparity
        batch_size = src_disparity.shape[0]
        hit = torch.rand(batch_size, device=src_disparity.device) < self.warp_perturb_prob
        if not bool(hit.any()):
            return src_disparity
        degraded = perturb_source_disparity(src_disparity, **self.warp_perturb_kwargs)
        return torch.where(hit.reshape(batch_size, 1, 1, 1), degraded, src_disparity)

    def _build_warp_block(self, rgb_views, src_disparity, K, c2w, latent_size):
        """Splat the source views into every target view and encode the result.

        Returns (rgb_block, depth_block).  Each block is [B, 17, V, h, w]: 16
        VAE channels plus one validity mask, zero on the source slots.

        `depth_block` is None unless `depth_warp_block` is enabled, in which
        case it carries the splatted *target-frame* disparity -- the geometric
        prior for the depth frames, which otherwise receive 17 channels of
        literal zeros in the sequence layout.  Note this is 1/z measured in the
        target camera, not the source disparity dragged along: a point's
        disparity changes with the viewpoint, so carrying the source value over
        would hand the depth frames a systematically wrong target.
        """
        with torch.no_grad():
            splat = self._splat_sources_pixel(
                rgb_views, src_disparity, K, c2w,
                return_target_disparity=self.depth_warp_block,
            )
            if self.depth_warp_block:
                warped, mask, tgt_disp = splat
            else:
                (warped, mask), tgt_disp = splat, None
            warped_latents, _ = self._encode_views(warped.to(rgb_views.dtype))
            disp_latents = None
            if tgt_disp is not None:
                # The VAE expects a 3-channel image in [0, 1], which is exactly
                # how the dataset's disparity is encoded, so the splatted
                # disparity can go through the same path as the RGB splat.
                disp_latents, _ = self._encode_views(
                    tgt_disp.clamp(0.0, 1.0).expand(-1, -1, 3, -1, -1).to(rgb_views.dtype)
                )
        warped_latents = warped_latents * self.latent_scale

        batch_size, num_tgt = mask.shape[0], mask.shape[1]
        latent_mask = F.interpolate(
            mask.reshape(batch_size * num_tgt, 1, *mask.shape[-2:]).float(),
            size=latent_size,
            mode="area",
        ).reshape(batch_size, num_tgt, 1, *latent_size).permute(0, 2, 1, 3, 4)

        rgb_block = self._to_view_block(
            torch.cat([warped_latents, latent_mask.to(warped_latents.dtype)], dim=1),
            latent_size,
        )
        if disp_latents is None:
            return rgb_block, None
        disp_latents = disp_latents * self.depth_latent_scale
        depth_block = self._to_view_block(
            torch.cat([disp_latents, latent_mask.to(disp_latents.dtype)], dim=1),
            latent_size,
        )
        return rgb_block, depth_block

    def forward_preprocess(self, data):
        batch_size = data["images"].shape[0]
        num_views = data["images"].shape[1]
        height = data["images"].shape[-2]
        width = data["images"].shape[-1]

        rgb_views = data["images"].to(self.pipe.device)
        # Use normalized disparity [0,1] for VAE encoding (VAE is designed for [0,1] images).
        # Metric depth in meters would be out-of-range for the VAE.
        depth_disp = data["disparity"].to(self.pipe.device)  # [B, V, 3, H, W]

        # Independent per-view VAE encoding — bypass unit_runner to avoid
        # the (T-1) % 4 == 0 temporal-division constraint on num_views=3.
        rgb_gt, latent_size = self._encode_views(rgb_views)
        depth_latents, _ = self._encode_views(depth_disp)
        rgb_gt = rgb_gt * self.latent_scale
        depth_latents = depth_latents * self.depth_latent_scale

        # The conditioning tensor is just rgb_views with the target slots
        # blanked.  In independent mode each view is encoded on its own, so the
        # source slots come out bit-identical to rgb_gt and the target slots are
        # the constant VAE(zeros): reuse both instead of a third full encode
        # (this was 1/3 of all VAE work per step).
        if self.vae_encoding_mode == "independent":
            conditioning_rgb_latents = rgb_gt.clone()
            conditioning_rgb_latents[:, :, self.num_source_views:] = (
                self._zero_view_latent(rgb_views).to(dtype=conditioning_rgb_latents.dtype)
                * self.latent_scale
            )
        else:
            # Temporal encoding mixes views, so the shortcut does not hold.
            conditioning_rgb_views = rgb_views.clone()
            conditioning_rgb_views[:, self.num_source_views:] = 0.0
            conditioning_rgb_latents, _ = self._encode_views(conditioning_rgb_views)
            conditioning_rgb_latents = conditioning_rgb_latents * self.latent_scale

        # Use precomputed null text embedding (empty-string T5 output from Wan2.1).
        # This is in-distribution for the pretrained cross-attention weights.
        context = self._null_text_embed.to(
            dtype=self.pipe.torch_dtype, device=self.pipe.device
        ).expand(batch_size, -1, -1)

        K = data["K"].to(self.pipe.device).float()  # Ensure float32 for view_condition_encoder, [B, V(s+t), 3, 3]
        c2w = data["c2w"].to(self.pipe.device).float()
        view_roles = data["view_roles"].to(self.pipe.device)    # [B, V]

        # --- Relative camera pose: express all poses relative to first source view ---
        # The `.float()` above does NOT make this fp32 -- see relative_c2w.
        c2w_rel = relative_c2w(c2w, self.camera_geometry_fp32)   # [B, V, 4, 4]

        latent_height, latent_width = latent_size
        latent_K = intrins_resize(K.clone(), ratio_H=latent_height / height, ratio_W=latent_width / width)
        # In "pixel" mode the encoder rasterises the Plucker field on the full
        # image grid with the *original* intrinsics and folds it down with
        # PixelUnshuffle; latent_K is then unused for the rays but is still
        # exported below for downstream consumers.
        # Built once and handed to both consumers: the encoder (which scales or
        # convolves it) and, on the RoCE arm, the DiT's phase MLPs (which want it
        # raw).  In pixel mode this rasterisation is a 384x672x6 ray field per
        # view, so building it twice would be a real cost for an identical tensor.
        plucker_field = self.view_condition_encoder.plucker_field(
            latent_K,
            c2w_rel,
            latent_height=latent_height,
            latent_width=latent_width,
            pixel_K=K,
            pixel_height=height,
            pixel_width=width,
        )
        view_condition = self.view_condition_encoder(
            latent_K,
            c2w_rel,
            view_roles,
            latent_height=latent_height,
            latent_width=latent_width,
            pixel_K=K,
            pixel_height=height,
            pixel_width=width,
            plucker=plucker_field,
        ).to(dtype=conditioning_rgb_latents.dtype)

        if self.view_condition_mode == "concat":
            # The camera block is a separate input channel block, so it must NOT
            # be summed into the RGB latents as well -- doing both would corrupt
            # the clean source-view conditioning images for no benefit.
            plucker_block = view_condition
            additive_condition = torch.zeros_like(conditioning_rgb_latents)
        else:
            plucker_block = None
            additive_condition = view_condition

        warp_block = None
        depth_warp_block = None
        if self.warp_mode == "splat":
            if self._eval_inputs:
                # NEVER splat with ground-truth depth at eval time.  Source
                # depth is a prediction at inference, so a GT-built splat would
                # report a number the model cannot reproduce.  sample_rgb
                # repopulates this from its own first-pass estimate; leaving it
                # zero is exactly the regime warp_dropout_prob trains for, so a
                # single-pass eval degrades honestly instead of cheating.
                warp_block = torch.zeros(
                    batch_size,
                    self.warp_latent_channels,
                    self.num_views,
                    *latent_size,
                    device=self.pipe.device,
                    dtype=conditioning_rgb_latents.dtype,
                )
                if self.depth_warp_block:
                    depth_warp_block = torch.zeros_like(warp_block)
            else:
                warp_block, depth_warp_block = self._build_warp_block(
                    rgb_views,
                    # FACTOR e-warpnoise degrades the disparity HERE, upstream of
                    # the splat, rather than corrupting the finished latent block
                    # (which is what the diagnostic-only _corrupt_warp_block does):
                    # a degraded depth map produces a geometrically CONSISTENT
                    # wrong warp -- wrong occlusions, wrong parallax, a mask that
                    # covers the sky -- whereas perturbing the latents produces
                    # values no splat could ever emit.  Only the first is the
                    # regime inference actually lands in.
                    self._maybe_perturb_src_disparity(
                        depth_disp[:, : self.num_source_views, 0]),
                    K,
                    c2w_rel,
                    latent_size,
                )
                warp_block = warp_block.to(dtype=conditioning_rgb_latents.dtype)
                if depth_warp_block is not None:
                    depth_warp_block = depth_warp_block.to(
                        dtype=conditioning_rgb_latents.dtype
                    )
                if self.warp_dropout_prob > 0.0:
                    # Per-sample Bernoulli, not per-batch: a mixed batch keeps both
                    # regimes in every gradient step.  The RGB and depth blocks
                    # are built from the SAME splat, so they must be dropped
                    # together or the depth frames would leak a geometry cue
                    # the RGB frames were told to do without.
                    keep = (
                        torch.rand(batch_size, device=warp_block.device)
                        >= self.warp_dropout_prob
                    ).to(warp_block.dtype).reshape(batch_size, 1, 1, 1, 1)
                    warp_block = warp_block * keep
                    if depth_warp_block is not None:
                        depth_warp_block = depth_warp_block * keep

        inputs = {
            # DiT forward kwargs
            "context": context,
            "y": None,
            "clip_feature": None,
            "reference_latents": None,
            "vace_context": None,
            "vace_scale": 1,
            "motion_bucket_id": None,
            "cfg_merge": False,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            # NVS latents
            "rgb_latents": conditioning_rgb_latents + additive_condition,
            "depth_latents": depth_latents,
            # Kept separately so the flow-matching branch can replace the
            # target-view slots with noisy latents and re-add the camera
            # conditioning on top.
            "view_condition": additive_condition,
            # Only set in mode="concat": appended as its own channel block in
            # _build_flow_input instead of being summed in.
            "plucker_block": plucker_block,
            # Only set in warp_mode="splat": VAE latent of the source views
            # splatted into each target view, plus a validity mask.
            "warp_block": warp_block,
            # Only set with depth_warp_block: the same splat carrying
            # target-frame disparity instead of colour, for the depth frames.
            "depth_warp_block": depth_warp_block,
            # Pixel-space handles kept only for two-pass sampling, which has to
            # rebuild the splat from the model's own source-view disparity.
            "source_images": rgb_views[:, : self.num_source_views] if warp_block is not None else None,
            "pixel_K": K if warp_block is not None else None,
            # Ground-truth targets
            "rgb_gt": rgb_gt,
            "depth_gt": depth_latents,
            # Loss / metadata
            "loss_masks": data["loss_masks"].to(self.pipe.device),
            "view_roles": view_roles,
            "K": latent_K,
            "c2w": c2w_rel,
            # Only set with rope_frame_axis='pose': per-frame world-to-camera
            # matrices for CaPE.  _forward_model renames this to the model_fn
            # kwarg, so the training step and both inference passes cannot
            # disagree about it.
            "cape_pose": (
                cape_pose_matrices(c2w_rel, self.rope_pose_translation_scale)
                if self.rope_frame_axis == "pose"
                else None
            ),
            # Only set with roce != "off": the raw Plucker field laid out over the
            # DiT's frame axis.  The key name is model_fn's kwarg verbatim, so
            # _forward_model passes it straight through rather than renaming it.
            "roce_camera_latents": (
                roce_camera_latents(
                    plucker_field, 2 if self.token_layout == "sequence" else 1
                ).to(dtype=conditioning_rgb_latents.dtype)
                if self.roce_mode != "off"
                else None
            ),
            "sample_id": data.get("sample_id", [""] * batch_size),
            "prompt": [""] * batch_size,
        }
        return inputs

    # ------------------------------------------------------------------
    # Flow matching for the target-view RGB branch
    # ------------------------------------------------------------------

    def _shift_sigma(self, u, shift=None):
        """Wan2.1 resolution shift applied to a uniform time variable u in [0, 1]."""
        s = self.rgb_flow_shift if shift is None else float(shift)
        return s * u / (1.0 + (s - 1.0) * u)

    def _sample_sigma(self, batch_size, device, dtype):
        """Sample one noise level per batch element, shape [B, 1, 1, 1, 1]."""
        if self.rgb_flow_timestep_sampling == "logit_normal":
            u = torch.sigmoid(
                torch.randn(batch_size, device=device, dtype=torch.float32)
                * self.rgb_flow_logit_std
                + self.rgb_flow_logit_mean
            )
        else:
            u = torch.rand(batch_size, device=device, dtype=torch.float32)
        sigma = self._shift_sigma(u)
        return sigma.to(dtype).reshape(batch_size, 1, 1, 1, 1)

    def inference_sigmas(self, num_steps, device=None):
        """Descending sigma schedule of length num_steps+1, from 1.0 down to 0.0."""
        u = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float32, device=device)
        return self._shift_sigma(u, shift=self.eval_flow_shift)

    def _forward_model(self, inputs, latents, timestep):
        """Run the DiT with an explicit input latent and timestep."""
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        call_inputs = dict(inputs)
        call_inputs.pop("latents", None)
        call_inputs.pop("plucker_block", None)
        call_inputs.pop("warp_block", None)
        call_inputs.pop("depth_warp_block", None)
        call_inputs.pop("source_images", None)
        call_inputs.pop("pixel_K", None)
        # Single funnel for every DiT call -- training and both inference passes --
        # so the RoPE override cannot be applied inconsistently between them.
        if self._rope_frame_ids is not None:
            call_inputs["rope_frame_ids"] = self._rope_frame_ids
        cape_pose = call_inputs.pop("cape_pose", None)
        if cape_pose is not None:
            # Renamed here rather than in forward_preprocess so that the key the
            # pipeline sees exists in exactly one place; model_fn's signature
            # ends in **kwargs, so a typo anywhere else would be swallowed.
            call_inputs["rope_pose_matrices"] = cape_pose
        out = self.pipe.model_fn(
            **models, **call_inputs, latents=latents, timestep=timestep
        )
        if self.token_layout == "sequence" and isinstance(out, tuple):
            # Both heads are applied to every token; keep each one where its
            # modality actually lives.  The wasted head is a single 1536->64
            # Linear, i.e. far cheaper than routing tokens.
            depth_out, rgb_out = out
            v = self.num_views
            return depth_out[:, :, v:], rgb_out[:, :, :v]
        return out

    def _build_flow_input(self, inputs, noisy_rgb_latents, noisy_depth_latents=None):
        """Assemble the DiT input latent.

        RGB: source slots keep the clean conditioning latents, target slots
        carry x_t.  The camera/role conditioning is re-added to the target slots
        so every view is told where it is being rendered from.

        Depth (joint mode only): *every* slot carries x_t, because source-view
        disparity is a prediction target too -- unlike source RGB, which is the
        conditioning signal.

        Channel order must match _expand_patch_embedding exactly:
        [latent, depth (channel layout only), plucker, warp, modality flag].
        """
        model_in = inputs["rgb_latents"].clone()
        s = self.num_source_views
        model_in[:, :, s:] = (
            noisy_rgb_latents[:, :, s:].to(model_in.dtype) + inputs["view_condition"][:, :, s:]
        )
        dtype = model_in.dtype
        plucker = inputs.get("plucker_block")
        warp = inputs.get("warp_block")

        if self.token_layout == "sequence":
            # Disparity becomes its own block of frames instead of its own
            # channels, so RGB and depth no longer have to share one token, one
            # AdaLN modulation and one residual stream.  RGB frames come first
            # so the pretrained temporal prior still sees num_views coherent
            # frames at positions 0..V-1.
            if noisy_depth_latents is None:
                raise ValueError("token_layout='sequence' requires depth_flow_matching")
            depth_in = noisy_depth_latents.to(dtype).clone()
            # In "channel" layout the depth of a target view rides in the same
            # token as its RGB and so inherits the additive camera conditioning
            # for free.  Splitting the modalities apart would silently drop it,
            # so add it explicitly to keep the two layouts comparable.
            depth_in[:, :, s:] = depth_in[:, :, s:] + inputs["view_condition"][:, :, s:]
            rgb_blocks = [model_in]
            depth_blocks = [depth_in]
            if plucker is not None:
                rgb_blocks.append(plucker.to(dtype))
                depth_blocks.append(plucker.to(dtype))
            if warp is not None:
                rgb_blocks.append(warp.to(dtype))
                depth_warp = inputs.get("depth_warp_block")
                depth_blocks.append(
                    torch.zeros_like(warp, dtype=dtype)
                    if depth_warp is None
                    else depth_warp.to(dtype)
                )
            flag = torch.zeros_like(model_in[:, :1])
            rgb_blocks.append(flag)
            depth_blocks.append(flag + 1.0)
            return torch.cat(
                [torch.cat(rgb_blocks, dim=1), torch.cat(depth_blocks, dim=1)], dim=2
            )

        blocks = [model_in]
        if self.depth_flow_matching:
            if noisy_depth_latents is None:
                raise ValueError("joint flow matching requires noisy depth latents")
            blocks.append(noisy_depth_latents.to(dtype))
        if plucker is not None:
            blocks.append(plucker.to(dtype))
        if warp is not None:
            blocks.append(warp.to(dtype))
        if len(blocks) == 1:
            return model_in
        return torch.cat(blocks, dim=1)

    def _expand_timestep(self, timestep, clean_depth_mask=None):
        """Optionally hand every frame in the sequence its own noise level.

        The sequence layout feeds the DiT 2*V frames: V RGB frames followed by V
        disparity frames.  The first `num_source_views` RGB frames carry the
        CLEAN conditioning latents, yet a single scalar timestep drives the
        shared AdaLN modulation for the whole sequence, so those frames are
        told they sit at the sampled sigma and the model has to infer "actually
        I am not noisy" from the content alone.  Pinning them to timestep 0
        removes that guessing game.  Every other frame -- target RGB and, by
        default, *all* disparity frames including source disparity -- really is
        at sigma.

        `clean_depth_mask` is the clean_source_depth factor: a [B] (or [1])
        tensor, truthy where this sample's SOURCE-disparity frames also carry a
        clean latent and must therefore be pinned to 0 as well.  Frames
        `num_views .. num_views + num_source_views - 1` are the source disparity.

        Returns [B, 2*V] when enabled, otherwise `timestep` unchanged.
        """
        if not self.per_frame_timestep:
            if clean_depth_mask is not None:
                raise RuntimeError(
                    "clean_depth_mask requires per_frame_timestep; the constructor "
                    "should have rejected this configuration."
                )
            return timestep
        per_frame = timestep.reshape(-1, 1).repeat(1, 2 * self.num_views)
        per_frame[:, : self.num_source_views] = 0.0
        if clean_depth_mask is not None:
            v, s = self.num_views, self.num_source_views
            keep = (~clean_depth_mask.reshape(-1, 1).bool()).to(per_frame.dtype)
            per_frame[:, v:v + s] = per_frame[:, v:v + s] * keep
        return per_frame

    def _apply_clean_source_depth(self, depth_x_t, depth_gt, loss_masks, batch_size):
        """Replace the source-disparity frames with the CLEAN latent, in place.

        Returns (clean_mask, loss_masks).  `clean_mask` is [B] bool, or None when
        the factor is off, and is what tells `_expand_timestep` to pin those
        frames to 0.  `loss_masks` is the original tensor when the factor is off
        and a modified COPY otherwise -- never an in-place edit, because
        `inputs["loss_masks"]` is also what evaluate() scores against.

        Per-sample Bernoulli, not per-batch: both regimes have to appear in every
        gradient step, or the noisy-source-depth path that pass 1 of inference
        depends on drifts out of distribution -- the same argument as
        warp_dropout_prob.
        """
        if not self.clean_source_depth:
            return None, loss_masks
        s = self.num_source_views
        clean = torch.rand(batch_size, device=depth_x_t.device) < self.clean_source_depth_prob
        sel = clean.reshape(batch_size, 1, 1, 1, 1)
        # Clean latent, NOT (1-sigma)*gt: the frame is being handed to the model as
        # a known quantity, so it must look exactly like what pass 2 will inject.
        depth_x_t[:, :, :s] = torch.where(sel, depth_gt[:, :, :s], depth_x_t[:, :, :s])
        # A clean frame has no velocity target, so it must leave the depth loss.
        # loss_masks is [B, 2, V, 1, H, W]; slot 0 is depth.  _masked_mse divides
        # by mask.sum(), so the surviving frames renormalise back to full scale on
        # their own -- no separate loss branch.  Note this does make a clean-source
        # sample contribute fewer elements, hence slightly less weight, to the
        # batch's depth loss than a noisy-source one.
        loss_masks = loss_masks.clone()
        loss_masks[:, 0, :s] = loss_masks[:, 0, :s] * (
            (~clean).to(loss_masks.dtype).reshape(batch_size, 1, 1, 1, 1)
        )
        return clean, loss_masks

    def _decode_src_disparity_with_grad(self, latents):
        """[1, C, k, h, w] scaled depth latents -> disparity [k, H, W], WITH grad.

        The readout is `_warp_block_from_predicted_depth`'s, line for line: divide
        by `depth_latent_scale`, decode one view at a time (independent encoding
        mode means the VAE's temporal convolutions would otherwise mix unrelated
        views), average the three identical channels, clamp to [0, 1].  That is
        deliberate and load bearing -- the loss has to constrain the SAME
        disparity that pass 2 will splat with, not a differently-post-processed
        cousin of it.  The clamp does kill the gradient outside [0, 1], which is
        the correct behaviour for the same reason: a value the splat will clamp
        away cannot be made consistent.

        Gradient checkpointing is on by default and is what makes this affordable:
        training otherwise never decodes, and the peak activation cost becomes one
        view's decode rather than k of them, for one extra forward pass each.
        """
        lat = latents / self.depth_latent_scale
        views = []
        for v in range(lat.shape[2]):
            z = lat[:, :, v:v + 1].to(self.pipe.torch_dtype)
            if self.depth_consist_vae_checkpoint:
                decoded = torch.utils.checkpoint.checkpoint(
                    lambda t: self.pipe.vae.decode(t, device=self.pipe.device),
                    z,
                    use_reentrant=False,
                )
            else:
                decoded = self.pipe.vae.decode(z, device=self.pipe.device)
            views.append(decoded[:, :, 0].float().mean(dim=1).clamp(0.0, 1.0)[0])
        return torch.stack(views, dim=0)

    def _depth_consistency_loss(self, inputs, pred_depth, depth_x_t, sigma, clean_src):
        """FACTOR f-consist: are the predicted SOURCE disparities mutually consistent?

        Returns a dict whose `scale` / `repro` carry gradient and whose remaining
        entries are floats for logging.  Both losses are the mean over the samples
        actually measured, NOT over the batch, so the logged value is directly
        comparable to the measured levels in
        tools/diag_src_depth_scale_consistency.py (`pair_scale_dev` 0.1309 ->
        scale_dev here, `pair_res_after_scale` 0.1213 -> repro here).  The price is
        that the term's effective weight varies with how often a batch has an
        eligible sample; with the default sigma gate that is most steps.

        Eval never pays for this and never reports it, but not because of a gate
        here: `_forward_flow` is only reached from `forward()`, and evaluate() goes
        through `sample_rgb`.  (Worth stating because forward_preprocess's eval
        branch does NOT clear `pixel_K` -- it zeroes `warp_block` to a tensor, so
        the intrinsics are still exported and the check below would pass.)

        Three gates, in order:

          * `pixel_K is None` -- defensive only: with `warp_mode != "splat"` the
            intrinsics are never exported, and the constructor already refuses that
            combination, so this makes a future runtime change degrade to a no-op
            instead of raising a KeyError 40 minutes into a run.
          * `sigma <= depth_consist_sigma_max` -- the x0 estimate is what the term
            constrains, and near sigma=1 that estimate is the model's blurry
            unconditional guess.  Penalising its geometry is not wrong, it is
            uninformative, and the VAE decodes are the expensive part.
          * `clean_source_depth` samples are skipped -- their source depth frames
            were replaced by the CLEAN latent, so `x_t - sigma*v` is not that
            sample's prediction of anything.  (This run has the factor off, but
            the loss must not depend on that.)

        `depth_consist_max_samples` then caps the decodes per step by keeping the
        LOWEST-sigma eligible samples, which is both the compute cap and the
        sharpest x0 estimate available in the batch.
        """
        zero = pred_depth.new_zeros(())
        stats = {
            "scale": zero, "repro": zero,
            "scale_dev": float("nan"), "n_pairs": 0.0, "support": 0.0,
            "flat": float("nan"), "n_samples": 0.0, "sigma": float("nan"),
        }
        if not self.depth_consist_on:
            return stats
        pixel_K = inputs.get("pixel_K")
        if pixel_K is None:
            return stats

        num_src = self.num_source_views
        sig = sigma.reshape(-1).float()
        eligible = sig <= self.depth_consist_sigma_max
        if clean_src is not None:
            eligible = eligible & (~clean_src.reshape(-1).bool())
        cand = eligible.nonzero(as_tuple=False).reshape(-1)
        if cand.numel() == 0:
            return stats
        if cand.numel() > self.depth_consist_max_samples:
            order = torch.argsort(sig[cand])[: self.depth_consist_max_samples]
            cand = cand[order]

        # Exact under the rectified-flow parameterisation used by this module:
        # x_t = (1-s)*x0 + s*eps with the head regressing eps - x0, so
        # x_t - s*(eps - x0) = x0.  No approximation and no extra forward pass.
        x0 = depth_x_t[:, :, :num_src] - sigma * pred_depth[:, :, :num_src]
        depth_valid = inputs["loss_masks"][:, 0, :num_src, 0] > 0.5   # [B, S, H, W]
        c2w = inputs["c2w"]

        scale_terms, repro_terms = [], []
        devs, n_pairs, supports, flats, sigmas = [], [], [], [], []
        for b in cand.tolist():
            k = self.depth_consist_num_views
            views = torch.randperm(num_src, device=x0.device)[:k].tolist()
            disp = self._decode_src_disparity_with_grad(x0[b:b + 1, :, views])
            valid = depth_valid[b, views] & (disp.detach() > 1e-3)
            got = pairwise_consistency(
                disp,
                valid,
                pixel_K[b, views].float(),
                c2w[b, views].float(),
                ordered_pairs(range(k)),
                **self.depth_consist_kwargs,
            )
            if got is None:
                # No ordered pair among the sampled views cleared min_support.
                # Legitimate: BMVS source views are spread widely enough that only
                # 6-12 of 30 ordered pairs overlap on GT depth.  Counted, not
                # silently dropped -- if `n_samples` stays near zero the factor is
                # not training anything and `depth_consist_num_views` is too small.
                continue
            scale_terms.append(got["scale"])
            repro_terms.append(got["repro"])
            devs.append(got["scale_dev"])
            n_pairs.append(got["n_pairs"])
            supports.append(got["support"])
            flats.append(flatness(disp, valid))
            # The sigma of the samples actually MEASURED, not of the candidates.
            # Reported because the fork smoke off droce@4000 read |a-1| 0.052 and
            # repro 0.032 against the diagnostic's 0.1309 / 0.1213, and the
            # explanation is a sigma effect: the diagnostic measured the
            # INFERENCE-time pass-1 output (25 Euler steps from pure noise) while
            # this reads x0_hat = x_t - sigma*v, and x_t = (1-s)*x0 + s*eps hands
            # the model (1-s) of the true structure for free.  Without this column
            # the 2.5-4x gap is unattributable, and the choice between "raise the
            # weights" and "raise sigma_max" turns on exactly that attribution.
            sigmas.append(float(sig[b]))

        if not scale_terms:
            return stats
        n = len(scale_terms)
        stats["scale"] = sum(scale_terms) / n
        stats["repro"] = sum(repro_terms) / n
        stats["scale_dev"] = sum(devs) / n
        stats["n_pairs"] = sum(n_pairs) / n
        stats["support"] = sum(supports) / n
        stats["flat"] = sum(flats) / n
        stats["n_samples"] = float(n)
        stats["sigma"] = sum(sigmas) / n
        return stats

    def _forward_flow(self, inputs):
        rgb_gt = inputs["rgb_gt"]
        batch_size = rgb_gt.shape[0]
        sigma = self._sample_sigma(batch_size, rgb_gt.device, rgb_gt.dtype)

        noise = torch.randn_like(rgb_gt)
        x_t = (1.0 - sigma) * rgb_gt + sigma * noise
        # Rectified-flow velocity target: dx/dsigma = noise - x0
        velocity_target = noise - rgb_gt

        timestep = (sigma.reshape(batch_size) * self.pipe.scheduler.num_train_timesteps).to(
            dtype=self.pipe.torch_dtype
        )

        if self.depth_flow_matching:
            depth_gt = inputs["depth_gt"]
            # Independent noise, *shared* sigma: one timestep drives the whole
            # token, which is the only thing the shared AdaLN modulation can
            # express anyway.
            depth_noise = torch.randn_like(depth_gt)
            depth_x_t = (1.0 - sigma) * depth_gt + sigma * depth_noise
            depth_velocity_target = depth_noise - depth_gt
            clean_src, loss_masks = self._apply_clean_source_depth(
                depth_x_t, depth_gt, inputs["loss_masks"], batch_size
            )
            timestep = self._expand_timestep(timestep, clean_depth_mask=clean_src)
            pred = self._forward_model(
                inputs, self._build_flow_input(inputs, x_t, depth_x_t), timestep
            )
            # FACTOR f-consist.  Computed here rather than in the training loop
            # because the VAE, the pixel intrinsics and the poses are all in hand,
            # and because `depth_x_t` is local -- exporting it just so the loop
            # could rebuild x0 would put the flow parameterisation in two places.
            consist = self._depth_consistency_loss(
                inputs, pred[0], depth_x_t, sigma, clean_src
            )
            return {
                "pred": pred,
                "rgb_gt": velocity_target,
                "depth_gt": depth_velocity_target,
                "loss_masks": loss_masks,
                "weight": 1.0,
                "sigma": sigma.reshape(batch_size),
                "consist": consist,
            }

        timestep = self._expand_timestep(timestep)
        pred = self._forward_model(inputs, self._build_flow_input(inputs, x_t), timestep)
        return {
            "pred": pred,
            # The RGB head now regresses the velocity, so the "gt" handed to the
            # training loop is the velocity target, not the clean latent.
            "rgb_gt": velocity_target,
            "depth_gt": inputs["depth_gt"],
            "weight": 1.0,
            "sigma": sigma.reshape(batch_size),
        }

    @torch.no_grad()
    def sample_rgb(self, inputs, num_steps=None, generator=None, warp_corruption=None):
        """Euler ODE sampling of the target-view latents.

        Returns (pred_depth, pred_rgb_latents).  In joint mode both modalities
        are integrated with the same sigma schedule and both are returned as
        clean latents.  In RGB-only mode pred_depth is the direct x-prediction
        read out at the final (least noisy) step.

        With warp_mode="splat" and warp_two_pass this runs the ODE twice: the
        first pass sees a zeroed warp block (the regime warp_dropout_prob trains
        for) and exists only to produce a source-view disparity estimate; the
        splat is then rebuilt from that estimate and the ODE is re-run.  Source
        depth is never available at inference, so there is no way to skip the
        first pass -- but it is also the easy half of the problem, since the
        source images are given clean.

        `warp_corruption` is diagnostic-only (see `evaluate_warp_sensitivity`):
        when set, the second-pass warp block is corrupted with
        `_corrupt_warp_block` after being rebuilt from the predicted depth, to
        measure how much the model's output actually depends on the warp
        block carrying correct pixel-level correspondence.
        """
        two_pass = (
            self.warp_mode == "splat"
            and self.warp_two_pass
            and inputs.get("warp_block") is not None
        )
        if not two_pass:
            return self._sample_rgb_once(inputs, num_steps, generator)

        blind = dict(inputs)
        blind["warp_block"] = torch.zeros_like(inputs["warp_block"])
        if inputs.get("depth_warp_block") is not None:
            blind["depth_warp_block"] = torch.zeros_like(inputs["depth_warp_block"])
        pred_depth, pred_rgb = self._sample_rgb_once(blind, num_steps, generator)

        warp_block, depth_warp_block = self._warp_block_from_predicted_depth(
            inputs, pred_depth
        )
        if warp_block is None:
            return pred_depth, pred_rgb
        if warp_corruption is not None:
            warp_block = self._corrupt_warp_block(warp_block, warp_corruption)
            if depth_warp_block is not None:
                depth_warp_block = self._corrupt_warp_block(depth_warp_block, warp_corruption)
        second = dict(inputs)
        second["warp_block"] = warp_block
        second["depth_warp_block"] = depth_warp_block
        # With clean_source_depth on, the second pass is also handed the pass-1
        # source disparity as a clean latent -- the SAME estimate the splat above
        # was rebuilt from, so the two conditioning signals cannot disagree.
        # NOTE a train/inference mismatch remains: training pins these frames to
        # the VAE encoding of GT disparity, whereas here they are the model's own
        # predicted latent, which never round-tripped through the VAE.  This is the
        # same class of mismatch the splat already has (trained on GT source depth)
        # and is recorded as a threat in plan_20260905_c_sweep_6s3t §9.
        clean_src_depth = (
            pred_depth[:, :, : self.num_source_views] if self.clean_source_depth else None
        )
        return self._sample_rgb_once(
            second, num_steps, generator, clean_src_depth=clean_src_depth
        )

    @torch.no_grad()
    def _corrupt_warp_block(self, block, mode):
        """Diagnostic only -- destroy the correspondence a warp block carries
        while keeping its value statistics, so a PSNR drop can be attributed
        to the correspondence and not to values falling outside the range the
        model was trained on.

        `shuffle` permutes spatial positions independently per (sample, view):
        same per-view value distribution, zero correspondence.
        `lowfreq` average-pools and upsamples back: coarse layout survives,
        pixel-level detail does not.
        """
        B, C, V, h, w = block.shape
        if mode == "shuffle":
            flat = block.reshape(B, C, V, h * w).clone()
            for b in range(B):
                for v in range(V):
                    perm = torch.randperm(h * w, device=block.device)
                    flat[b, :, v] = flat[b, :, v][:, perm]
            return flat.reshape(B, C, V, h, w)
        if mode == "lowfreq":
            k = max(2, min(h, w) // 5)
            merged = block.reshape(B * C * V, 1, h, w)
            down = F.avg_pool2d(merged, kernel_size=k)
            up = F.interpolate(down, size=(h, w), mode="nearest")
            return up.reshape(B, C, V, h, w)
        raise ValueError(f"unknown warp_corruption mode: {mode!r}")

    @torch.no_grad()
    def _warp_block_from_predicted_depth(self, inputs, pred_depth_latents):
        """Rebuild the warp block from the model's own source-view disparity.

        Returns (None, None) when the prediction cannot be turned into a usable
        splat, in which case the caller keeps the first pass' result.
        """
        source_views = inputs.get("source_images")
        if source_views is None:
            return None, None
        num_src = self.num_source_views
        latents = pred_depth_latents[:, :, :num_src] / self.depth_latent_scale
        # Decode one view at a time: in "independent" encoding mode the frames
        # are unrelated, and the VAE's temporal convolutions would mix them.
        decoded = [
            self.pipe.vae.decode(
                latents[:, :, v:v + 1].to(self.pipe.torch_dtype), device=self.pipe.device
            )[:, :, 0]
            for v in range(num_src)
        ]
        # The disparity image is the same value replicated over 3 channels.
        disparity = torch.stack(decoded, dim=1).float().mean(dim=2).clamp(0.0, 1.0)
        rgb_block, depth_block = self._build_warp_block(
            source_views,
            disparity,
            inputs["pixel_K"],
            inputs["c2w"],
            tuple(pred_depth_latents.shape[-2:]),
        )
        return (
            rgb_block.to(dtype=pred_depth_latents.dtype),
            None if depth_block is None else depth_block.to(dtype=pred_depth_latents.dtype),
        )

    @torch.no_grad()
    def _sample_rgb_once(self, inputs, num_steps=None, generator=None, clean_src_depth=None):
        """One Euler ODE pass.

        `clean_src_depth` is the clean_source_depth factor: a
        [B, C, num_source_views, h, w] latent to hold the source-disparity frames
        at, pinned to timestep 0 and re-pinned after every Euler step so the
        integrator cannot drift them.  It is supplied by `sample_rgb` on pass 2
        from the pass-1 estimate; pass 1 has nothing to supply and so runs in the
        noisy regime that training's other half covers.
        """
        num_steps = self.eval_num_inference_steps if num_steps is None else max(1, int(num_steps))
        rgb_gt = inputs["rgb_gt"]
        sigmas = self.inference_sigmas(num_steps, device=rgb_gt.device).to(rgb_gt.dtype)

        x = torch.randn(
            rgb_gt.shape, device=rgb_gt.device, dtype=rgb_gt.dtype, generator=generator
        )
        x_depth = None
        if self.depth_flow_matching:
            depth_gt = inputs["depth_gt"]
            x_depth = torch.randn(
                depth_gt.shape, device=depth_gt.device, dtype=depth_gt.dtype, generator=generator
            )
        s = self.num_source_views
        clean_mask = None
        if clean_src_depth is not None:
            if x_depth is None:
                raise RuntimeError("clean_src_depth requires depth_flow_matching")
            clean_src_depth = clean_src_depth.to(x_depth.dtype)
            x_depth[:, :, :s] = clean_src_depth
            # One shared flag for the whole batch: at inference every sample is in
            # the clean regime, unlike training where it is a per-sample draw.
            clean_mask = torch.ones(1, dtype=torch.bool, device=x_depth.device)
        pred_depth = None
        for i in range(num_steps):
            timestep = (sigmas[i] * self.pipe.scheduler.num_train_timesteps).reshape(1).to(
                dtype=self.pipe.torch_dtype, device=rgb_gt.device
            )
            timestep = self._expand_timestep(timestep, clean_depth_mask=clean_mask)
            pred_depth, velocity = self._forward_model(
                inputs, self._build_flow_input(inputs, x, x_depth), timestep
            )
            dt = sigmas[i] - sigmas[i + 1]
            x = x - dt * velocity.to(x.dtype)
            if self.depth_flow_matching:
                x_depth = x_depth - dt * pred_depth.to(x_depth.dtype)
                if clean_src_depth is not None:
                    # The step above moved every frame, including the ones that are
                    # supposed to be held fixed; put them back.
                    x_depth[:, :, :s] = clean_src_depth
        if self.depth_flow_matching:
            return x_depth, x
        return pred_depth, x

    def forward(self, data, inputs=None, args=None):
        inputs = self.forward_preprocess(data)
        if self.rgb_flow_matching:
            result = self._forward_flow(inputs)
        else:
            models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
            result = self.pipe.training_predict(**models, **inputs)
        # Pass through loss_masks and other metadata needed by training loop.
        # setdefault, not assignment: with clean_source_depth on, _forward_flow
        # returns a mask with the clean source-depth frames already excluded, and
        # overwriting it would put those frames back into the depth loss.
        result.setdefault("loss_masks", inputs["loss_masks"])
        # setdefault, not assignment: in joint flow mode _forward_flow already
        # put the *velocity* target here and it must not be clobbered by the
        # clean latent.
        result.setdefault("depth_gt", inputs["depth_gt"])
        result.setdefault("rgb_gt", inputs["rgb_gt"])
        return result

    # ------------------------------------------------------------------
    # Evaluation: decode predictions and compute metrics
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, batch, output_dir, global_step, max_samples=2, save_vis=True,
                 vis_offset=0):
        """
        Run one forward pass (no gradient), decode predictions with the VAE,
        and compute metrics:
          - Source-view depth  : MAE on per-view normalised disparity (valid pixels)
          - Target-view RGB    : PSNR (pixel space, decoded)
          - Target-view depth  : MAE on normalised disparity (valid pixels)

        Saves ONE labelled comparison image per sample to
        output_dir/eval_step_{global_step}/b{b}_step{step}.{fmt}: 3 rows
        (rgb / depth gt / depth pred) x (num_source_views + 2 * num_target_views)
        columns (src1..srcS, then a gt/pred pair per target), plus a caption
        strip below the grid with THIS sample's own numbers -- src abs_rel/delta1
        under each source column, tgt coverage/psnr/lpips/delta1/abs_rel centered
        under each target's gt+pred pair -- so a checkpoint's visual quality and
        its scored quality are read off the same image instead of cross-
        referencing per_view_metrics.jsonl by hand.  It used to be one image per
        target view, which made the source columns -- identical across targets --
        get decoded and redrawn T times and left the reader to line up T files by
        hand.
        When the frozen eval set is scored in several chunks, every chunk resolves
        to the same eval_step_* directory, so a chunk's images would collide with
        chunk 0's `b00.. b{max_samples-1}` filenames unless disambiguated.
        `vis_offset` shifts the filename index by the caller (e.g. chunk 1 passes
        `vis_offset=max_samples` to continue the numbering as b04, b05, ...)
        instead of overwriting chunk 0's files. Pass `save_vis=False` for chunks
        that should not render at all, to skip the VAE-decode cost entirely.
        Returns a dict of scalar metric means.
        """
        import json
        import math
        import numpy as np
        from PIL import Image as PILImage

        was_training = self.training
        self.eval()

        device = self.pipe.device
        depth_eval_max = 100.0  # Metric-depth clamp max for B-metric branch
        save_dir = os.path.join(output_dir, f"eval_step_{global_step:06d}")
        os.makedirs(save_dir, exist_ok=True)

        # Truncate batch to max_samples for speed
        def _slice(x, n):
            if isinstance(x, torch.Tensor):
                return x[:n]
            if isinstance(x, list):
                return x[:n]
            return x

        batch_small = {k: _slice(v, max_samples) for k, v in batch.items()}

        # Wrap everything in autocast to match the training mixed-precision context
        # (Accelerate's autocast is not active here, so we do it manually)
        autocast_ctx = torch.autocast(
            device_type="npu", dtype=self.pipe.torch_dtype, enabled=True
        )
        with autocast_ctx:
            self._eval_inputs = True
            try:
                inputs = self.forward_preprocess(batch_small)
            finally:
                self._eval_inputs = False

            if self.rgb_flow_matching:
                # Multi-step Euler sampling of the target-view RGB latents.
                # The initial noise is drawn from a dedicated generator with a
                # fixed seed so that flow-matching eval is reproducible across
                # validation steps, runs and A/B arms; with the global RNG the
                # sampled target RGB (and hence the depth read out alongside it)
                # would drift with the training RNG state.
                gen = torch.Generator(device=self.pipe.device)
                gen.manual_seed(self.eval_sample_seed)
                pred_depth, pred_rgb = self.sample_rgb(
                    inputs, num_steps=self.eval_num_inference_steps, generator=gen
                )
            else:
                # Deterministic single-step prediction (legacy DVD behaviour)
                models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
                pred_depth, pred_rgb = self.pipe.training_predict(**models, **inputs)["pred"]

        depth_gt = inputs["depth_gt"]           # [B, 16, V, h, w]
        rgb_gt   = inputs["rgb_gt"]             # [B, 16, V, h, w]
        loss_masks = inputs["loss_masks"]        # [B, 2, V, 1, H, W]

        # Everything from here on is decoded, so step back out of the scaled
        # latent space the flow objective lives in.
        if self.latent_scale != 1.0:
            rgb_gt = rgb_gt / self.latent_scale
            pred_rgb = pred_rgb / self.latent_scale
        if self.depth_latent_scale != 1.0:
            depth_gt = depth_gt / self.depth_latent_scale
            pred_depth = pred_depth / self.depth_latent_scale

        B = pred_depth.shape[0]
        V = pred_depth.shape[2]

        # --- decode helper ---
        def _decode_view(latents_bcvhw, view_idx):
            """Decode a single view → [B,3,H,W] float32 in [0,1].

            In independent mode latents_bcvhw has shape [B,16,V,h,w] and we slice
            one view at a time.  In temporal_padding mode the latent is a compressed
            temporal sequence [B,16,T',h,w] with T' ≤ V.  For view_idx ≥ T' the
            slice would be empty, so we decode the full sequence and take the last
            available pixel frame instead.
            """
            if view_idx < latents_bcvhw.shape[2]:
                v_lat = latents_bcvhw[:, :, view_idx:view_idx+1].to(device).to(self.pipe.torch_dtype)
                decoded = self.pipe.vae.decode(v_lat, device=device)   # [B,3,1,H,W]
                return decoded[:, :, 0].float().clamp(0, 1)            # [B,3,H,W]
            # temporal_padding out-of-range: decode full sequence, use last pixel frame
            full = self.pipe.vae.decode(
                latents_bcvhw.to(device).to(self.pipe.torch_dtype), device=device
            )  # [B, 3, Vd, H, W]
            return full[:, :, -1].float().clamp(0, 1)

        # --- metric helpers ---
        def _psnr(pred_img, gt_img):
            mse = ((pred_img - gt_img) ** 2).mean().item()
            if mse < 1e-10:
                return 100.0
            return -10 * math.log10(mse)

        def _psnr_masked(pred_img, gt_img, mask_img):
            """PSNR over the pixels where mask_img == 1 only.

            For the splat baseline this separates "the geometry is wrong" from
            "there was no source pixel to copy": the full-frame number is
            dominated by the ~54% of pixels that are holes.
            """
            m = mask_img.expand_as(pred_img)
            n = m.sum()
            if n < 1:
                return float("nan")
            mse = ((((pred_img - gt_img) ** 2) * m).sum() / n).item()
            if mse < 1e-10:
                return 100.0
            return -10 * math.log10(mse)

        def _lpips(pred_img, gt_img):
            """LPIPS on [0,1] images; the net wants [-1,1].  Lower is better."""
            net = self._get_lpips(device)
            if net is None:
                return float("nan")
            with torch.no_grad():
                d = net(pred_img.float() * 2 - 1, gt_img.float() * 2 - 1)
            return float(d.mean())

        def _fit_affine(p, g):
            """Least squares  min_{a,b} ||a*p + b - g||^2  (float64, closed form)."""
            n = p.numel()
            sp, spp = p.sum(), (p * p).sum()
            sg, spg = g.sum(), (p * g).sum()
            det = n * spp - sp * sp
            if torch.abs(det) < 1e-12:
                a = g.median() / p.median().clamp_min(1e-8)
                b = torch.zeros((), dtype=torch.float64, device=p.device)
            else:
                a = (n * spg - sp * sg) / det
                b = (spp * sg - sp * spg) / det
            return a, b

        def _align_and_score(p, g, gt_v):
            """Align disparity p to GT disparity g, invert, return (abs_rel, d1, d1_clamped, abs_rel_clamped)."""
            a, b = _fit_affine(p, g)
            # Aligned disparity must stay positive; floor it at half the
            # smallest GT disparity => depth capped at 2x the farthest GT point.
            disp_floor = (g.min() * 0.5).clamp_min(1e-8)
            pred_depth_m = 1.0 / (a * p + b).clamp_min(disp_floor)

            abs_rel = ((pred_depth_m - gt_v).abs() / gt_v).mean().item()
            ratio = torch.max(pred_depth_m / gt_v, gt_v / pred_depth_m)
            delta1 = (ratio < 1.25).double().mean().item()

            pred_c = pred_depth_m.clamp(max=depth_eval_max)
            gt_c = gt_v.clamp(max=depth_eval_max)
            abs_rel_c = ((pred_c - gt_c).abs() / gt_c).mean().item()
            ratio_c = torch.max(pred_c / gt_c, gt_c / pred_c)
            delta1_c = (ratio_c < 1.25).double().mean().item()
            return abs_rel, delta1, abs_rel_c, delta1_c

        def _plane_residual(z, xs, ys):
            """Remove the best-fit plane a*x + b*y + c from z (all 1-D float64)."""
            ones = torch.ones_like(z)
            A = torch.stack([xs, ys, ones], dim=1)          # [N, 3]
            ata = A.T @ A                                    # [3, 3]
            atz = A.T @ z                                    # [3]
            try:
                coef = torch.linalg.solve(ata, atz)
            except Exception:
                coef = torch.linalg.lstsq(ata, atz.unsqueeze(1)).solution.squeeze(1)
            return z - A @ coef

        def _struct_corr(p, g, xs, ys):
            """Pearson correlation of pred/GT disparity AFTER removing a fitted plane.

            A trivial "sky far / ground near" vertical ramp -- which already
            scores delta1 ~= 0.80 on aerial datasets -- has zero residual here,
            so this metric measures only genuine scene structure and cannot be
            gamed by a low-order depth prior.  Scale/shift invariant by
            construction, so no affine alignment is needed.
            """
            pr = _plane_residual(p, xs, ys)
            gr = _plane_residual(g, xs, ys)
            pr = pr - pr.mean()
            gr = gr - gr.mean()
            denom = pr.norm() * gr.norm()
            if denom < 1e-12:
                return float("nan")
            return (pr @ gr / denom).item()

        # Views whose trivial vertical ramp already scores delta1 > 1 - this
        # value carry essentially no information about skill: the ratio below
        # becomes tiny/tiny and swings by +-10 on a rounding difference.  They
        # are reported as NaN so the nanmean aggregation drops them.
        MIN_RAMP_HEADROOM = 0.05

        def _depth_metrics(pred_lv, gt_metric_hw, gt_disp_hw, mask_hw):
            """Affine-invariant depth metrics (MiDaS / Ranftl protocol).

            pred_lv:       [1, 16, 1, h, w]  predicted disparity latent
            gt_metric_hw:  [H, W]  GT metric depth in meters (positive)
            gt_disp_hw:    [H, W]  GT normalised disparity as fed to the model
            mask_hw:       [H, W]  valid-pixel mask at full resolution

            The model is trained on an affinely-normalised disparity, so the
            prediction can only be compared to GT after fitting a scale AND a
            shift.  Aligning with a median *scale* alone (an earlier
            implementation) cannot cancel the shift, and inverting the
            disparity before alignment blows the residual up non-linearly for
            far pixels -- both make the reported numbers far worse than the
            model actually is.

            Here we solve  min_{a,b} || a * pred_disp + b - 1/gt_depth ||^2
            in *disparity* space, then invert to metric depth and report
            AbsRel / delta<1.25.

            AbsRel / delta1 alone are NOT trustworthy on datasets with a narrow
            depth spread: on BlendedMVS a plain vertical ramp scores
            delta1 ~= 0.80 and even random noise scores ~= 0.72.  We therefore
            also report, for every view:
              * delta1_ramp  -- the same metric for a trivial vertical ramp
              * delta1_skill -- (delta1 - delta1_ramp) / (1 - delta1_ramp),
                                i.e. the fraction of the remaining headroom
                                above the trivial baseline that was closed.
                                This is a PER-VIEW ratio, so the reported mean
                                is a mean-of-ratios and stays heavy-tailed even
                                with the headroom floor below.  Prefer the
                                ratio-of-means `delta1_skill_agg` added during
                                aggregation as the headline number.
              * struct_corr  -- plane-removed correlation, immune to ramps
              * disp_l1      -- raw, unaligned L1 in the model's own
                                normalised-disparity space (cross-view
                                consistency indicator)
            """
            keys = ("abs_rel", "delta1", "abs_rel_clamp100", "delta1_clamp100",
                    "disp_l1", "delta1_ramp", "delta1_skill", "struct_corr")
            nan_out = {k: float("nan") for k in keys}
            if pred_lv.shape[2] == 0:
                # temporal_padding out-of-range view: cannot decode empty latent
                return nan_out
            v_lat = pred_lv[:, :, 0:1].to(device).to(self.pipe.torch_dtype)
            pred_disp = self.pipe.vae.decode(v_lat, device=device)[0, :, 0].float().mean(0)  # [H,W]

            gt = gt_metric_hw.to(device).float()
            if pred_disp.shape != gt.shape:
                pred_disp = torch.nn.functional.interpolate(
                    pred_disp[None, None], size=gt.shape, mode="bilinear", align_corners=False
                )[0, 0]

            valid = mask_hw.bool().to(device) & torch.isfinite(gt) & (gt > 0)
            if valid.sum() < 16:
                return nan_out

            # --- raw (unaligned) L1 in the model's normalised disparity space ---
            gt_disp_norm = gt_disp_hw.to(device).float()
            disp_l1 = (pred_disp[valid] - gt_disp_norm[valid]).abs().mean().item()

            p = pred_disp[valid].double()
            gt_v = gt[valid].double()
            g = (1.0 / gt_v.clamp_min(1e-8))                   # GT metric disparity

            abs_rel, delta1, abs_rel_c, delta1_c = _align_and_score(p, g, gt_v)

            # --- trivial vertical-ramp baseline on the very same pixels ------
            H_, W_ = gt.shape
            yy = torch.linspace(0.0, 1.0, H_, device=device)[:, None].expand(H_, W_)
            xx = torch.linspace(0.0, 1.0, W_, device=device)[None, :].expand(H_, W_)
            ys = yy[valid].double()
            xs = xx[valid].double()
            _, delta1_ramp, _, _ = _align_and_score(ys, g, gt_v)
            headroom = 1.0 - delta1_ramp
            delta1_skill = (
                (delta1 - delta1_ramp) / headroom
                if headroom > MIN_RAMP_HEADROOM
                else float("nan")
            )

            struct_corr = _struct_corr(p, g, xs, ys)

            return {
                "abs_rel": abs_rel,
                "delta1": delta1,
                "abs_rel_clamp100": abs_rel_c,
                "delta1_clamp100": delta1_c,
                "disp_l1": disp_l1,
                "delta1_ramp": delta1_ramp,
                "delta1_skill": delta1_skill,
                "struct_corr": struct_corr,
            }

        disp_batch         = batch_small["disparity"].to(device)     # [B, V, 3, H, W] — GT normalised disparity
        depth_metric_batch = batch_small["depth_metric"].to(device)  # [B, V, 3, H, W] — GT metric depth (m)

        _depth_keys = ("abs_rel", "delta1", "abs_rel_clamp100", "delta1_clamp100",
                       "disp_l1", "delta1_ramp", "delta1_skill", "struct_corr")
        metrics = {f"{side}_depth_{k}": [] for side in ("src", "tgt") for k in _depth_keys}
        metrics["tgt_rgb_psnr"] = []
        metrics["tgt_rgb_lpips"] = []

        # --- VAE round-trip ceiling for DEPTH (opt-in) -----------------------
        # Target RGB has always had its ceiling (`tgt_rgb_psnr_vae` ~= 30 dB),
        # which is what makes "the model is at 13.6 dB" interpretable.  Depth had
        # none: `src_depth_abs_rel = 0.056` and `tgt_depth_abs_rel = 0.119` were
        # reported against no floor at all, so there was no way to tell how much
        # of either is the model and how much is the VAE's inability to represent
        # a disparity map in 16 latent channels through an RGB-trained decoder.
        # That gap matters for prioritisation: if 0.056 already IS the floor,
        # then "source depth is good" really means "source depth is saturated",
        # and §6's oracle-warp gain of only +1.79 dB has a much more mundane
        # explanation than the model having extracted most of the available value.
        #
        # Feeding `depth_gt` (the VAE encoding of the GT disparity) where the
        # prediction normally goes measures exactly encode -> decode -> align ->
        # score, through the identical `_depth_metrics` protocol.  Using the same
        # code path is the whole point: a ceiling computed with a different
        # alignment would not bound anything.
        #
        # Off by default because it costs one extra VAE decode per view (V=7) on
        # every eval, and it is a property of the VAE and the dataset alone --
        # completely independent of the checkpoint, so it needs measuring once,
        # not every `eval_steps`.
        if self.eval_depth_vae_ceiling:
            metrics.update({
                f"{side}_depth_vae_{k}": []
                for side in ("src", "tgt") for k in _depth_keys
            })

        # --- Non-learned target-RGB baselines -------------------------------
        # Depth has had a trivial-ramp baseline (delta1_ramp / delta1_skill)
        # from the start; target RGB has had none, so a bare "PSNR 13" carried
        # no information about whether the generative branch was adding
        # anything at all.  Two references, both computable at inference:
        #   copy  -- paste the nearest source view (by camera-centre distance).
        #            Zero geometry: the floor any NVS model must clear.
        #   splat -- forward-splat the source images with the GT source depth
        #            and fill the holes with the mean splatted colour.  This is
        #            the same geometry the warp channel already carries, scored
        #            directly; the model has to beat it to be worth its cost.
        # Both are scored against the SAME gt_px the model is scored against
        # (the VAE round-trip of the GT), otherwise the comparison is rigged in
        # the model's favour by the VAE reconstruction error.
        metrics["tgt_rgb_psnr_copy"] = []
        metrics["tgt_rgb_lpips_copy"] = []
        metrics["tgt_rgb_psnr_splat"] = []
        metrics["tgt_rgb_lpips_splat"] = []
        # Splat quality on the covered pixels only, plus the coverage itself:
        # the full-frame splat PSNR conflates "wrong colour" with "no colour".
        metrics["tgt_rgb_psnr_splat_valid"] = []
        metrics["tgt_rgb_splat_coverage"] = []
        # Same splat, but built from the model's own predicted source disparity
        # instead of GT.  This is the baseline the model is actually competing
        # against at deployment (GT source depth is never available at
        # inference either), and it isolates whether `tgt_rgb_psnr_splat_valid`
        # understates the model by handing the geometric baseline an unfair
        # answer key.
        metrics["tgt_rgb_psnr_splat_pred"] = []
        metrics["tgt_rgb_lpips_splat_pred"] = []
        metrics["tgt_rgb_psnr_splat_pred_valid"] = []
        metrics["tgt_rgb_splat_pred_coverage"] = []
        # Ceiling of the whole pipeline: the VAE round-trip against the raw
        # pixels.  Every number above is measured against the round-tripped GT,
        # so nothing can exceed this.
        metrics["tgt_rgb_psnr_vae"] = []

        # --- The model's own PSNR, split by splat coverage -------------------
        # Full-frame PSNR mixes two incomparable regions.  Where a source pixel
        # reprojects into the target, the correct answer is DETERMINED and a
        # squared-error metric is meaningful.  Where nothing reprojects, the
        # content is unobserved, any plausible completion is as good as any
        # other, and PSNR is measuring the model's luck at guessing.  On this
        # view sampling the second region is the majority (coverage ~0.45), so
        # the headline number is closer to a coverage statistic than to a
        # quality one -- which is exactly what makes "the model does not beat
        # the splat on PSNR" hard to interpret.  Splitting the two apart is the
        # only way to tell a geometry failure from an impossible task.
        #
        # The mask is built from GT source depth.  That is legitimate here
        # because it only PARTITIONS pixels for reporting; it never enters the
        # model's input (forward_preprocess zeroes the warp block under
        # _eval_inputs) and must never become a training signal.
        metrics["tgt_rgb_psnr_covered"] = []
        metrics["tgt_rgb_psnr_uncovered"] = []
        # The same partition for the two non-learned references, so the covered
        # number can be read against a floor and a ceiling instead of in a
        # vacuum.
        metrics["tgt_rgb_psnr_copy_covered"] = []
        metrics["tgt_rgb_psnr_copy_uncovered"] = []
        metrics["tgt_rgb_psnr_vae_covered"] = []

        # --- Per-view geometric difficulty ----------------------------------
        # Uniform view sampling draws sources and targets independently from all
        # ~100 views of a scene, so a single eval set mixes near-copy problems
        # with targets whose closest source looks the other way.  Logging the
        # difficulty per view is what lets any metric above be read as a
        # function of it rather than averaged over it.  Distances are in the
        # sample's normalised world (the dataset scales translations by the same
        # factor as disparity), so they are comparable across samples.
        metrics["tgt_view_min_src_angle_deg"] = []
        metrics["tgt_view_min_src_dist"] = []

        rgb_px_batch = batch_small["images"].to(device).float().clamp(0.0, 1.0)  # [B,V,3,H,W]
        # Camera centres of the relative poses, used to pick the nearest source.
        cam_centers = inputs["c2w"][:, :, :3, 3].float()                         # [B,V,3]
        # Optical axes (+z column of c2w) for the per-view difficulty measure.
        cam_axes = F.normalize(inputs["c2w"][:, :, :3, 2].float(), dim=-1)       # [B,V,3]
        # One record per scored target view, written next to the images.  The
        # aggregation below collapses every list to a mean, which is what made
        # this diagnostic impossible to run on past checkpoints without a full
        # re-inference: the per-view values existed and were thrown away.
        per_view_rows = []
        splat_px = splat_mask = None
        splat_px_pred = splat_mask_pred = None
        if self.num_source_views > 0 and self.num_target_views > 0:
            splat_px, splat_mask = self._splat_sources_pixel(
                rgb_px_batch,
                disp_batch[:, : self.num_source_views, 0].float(),
                batch_small["K"].to(device).float(),
                inputs["c2w"],
            )

            # Decode the model's own source-view disparity prediction to pixel
            # space and splat with THAT instead of GT.  pred_depth has already
            # been divided by depth_latent_scale above, so it is directly
            # decodable, exactly like the per-view decode _depth_metrics does
            # for the src_depth_* metrics -- this is the same prediction,
            # just also turned into a splat rather than only scored against GT.
            num_src = self.num_source_views
            pred_src_disp_px = torch.stack(
                [
                    self.pipe.vae.decode(
                        pred_depth[:, :, sv:sv + 1].to(device).to(self.pipe.torch_dtype),
                        device=device,
                    )[:, :, 0].float().mean(dim=1)
                    for sv in range(num_src)
                ],
                dim=1,
            ).clamp(0.0, 1.0)  # [B, num_src, H, W]
            splat_px_pred, splat_mask_pred = self._splat_sources_pixel(
                rgb_px_batch,
                pred_src_disp_px,
                batch_small["K"].to(device).float(),
                inputs["c2w"],
            )

        # --- Visualisation tile builders -------------------------------------
        # Hoisted out of both loops.  They used to be redefined per target view,
        # and the source columns rebuilt with them, which cost
        # num_source_views * num_target_views VAE decodes to redraw identical
        # tiles; the grid is now assembled once per sample.
        def _to_np(t):
            return (t[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        def _pred_depth_to_np(latents_bcvhw, view_idx):
            """Decode pred depth latent → colorized HxWx3 (plasma), no clamp."""
            import matplotlib.cm as cm
            if view_idx < latents_bcvhw.shape[2]:
                v_lat = latents_bcvhw[:, :, view_idx:view_idx+1].to(device).to(self.pipe.torch_dtype)
                raw = self.pipe.vae.decode(v_lat, device=device)
                raw = raw[0, :, 0].float().mean(0).cpu().numpy()  # [H,W]
            else:
                # temporal_padding: decode full sequence, use last pixel frame
                full = self.pipe.vae.decode(
                    latents_bcvhw[0:1].to(device).to(self.pipe.torch_dtype), device=device
                )
                raw = full[0, :, -1].float().mean(0).cpu().numpy()  # [H,W]
            if self.eval_vis_depth_norm == "per_view":
                # LEGACY: stretch each tile to its own 2%/98% range.  This
                # is NOT comparable to the GT row below (which uses the
                # raw shared-scale value), and it makes every prediction
                # span the full dark-purple->yellow ramp regardless of
                # how flat or mis-scaled it actually is.
                lo = np.percentile(raw, 2)
                hi = np.percentile(raw, 98)
                if hi - lo > 1e-6:
                    raw = (raw - lo) / (hi - lo)
            return (cm.plasma(np.clip(raw, 0, 1))[:, :, :3] * 255).astype(np.uint8)

        def _gt_depth_to_np(disp_3hw):
            """Visualize GT disparity from batch [0,1] → plasma.

            Under disparity_norm_mode="shared_scale" this [0,1] value is
            already a common, cross-view-consistent scale, so the
            prediction must be rendered with the SAME fixed mapping (see
            eval_vis_depth_norm) for the two rows to be comparable.
            """
            import matplotlib.cm as cm
            gray = disp_3hw[0].cpu().numpy()
            return (cm.plasma(gray)[:, :, :3] * 255).astype(np.uint8)

        for b in range(B):
            # Source columns are the same for every target view, so build them
            # once here rather than inside the target loop.
            src_rgb_nps = src_gt_depth_nps = src_pred_depth_nps = None
            tgt_tiles = []
            tgt_captions = []
            if save_vis:
                src_rgb_nps = [_to_np(_decode_view(rgb_gt[b:b+1], sv))
                               for sv in range(self.num_source_views)]
                src_gt_depth_nps = [_gt_depth_to_np(disp_batch[b, sv])
                                    for sv in range(self.num_source_views)]
                src_pred_depth_nps = [_pred_depth_to_np(pred_depth[b:b+1], sv)
                                      for sv in range(self.num_source_views)]

            # Source view depth metrics (views 0..num_source_views-1)
            # Kept per sv, mirroring tgt_depth_rows below, so the vis caption
            # strip can show THIS sample's source depth quality, not the
            # eval-set mean.
            src_depth_rows = []
            for sv in range(self.num_source_views):
                mask_hw = loss_masks[b, 0, sv, 0]  # [H, W]
                m = _depth_metrics(
                    pred_depth[b:b+1, :, sv:sv+1],
                    depth_metric_batch[b, sv, 0],   # GT metric depth in meters
                    disp_batch[b, sv, 0],           # GT normalised disparity
                    mask_hw,
                )
                src_depth_rows.append(m)
                for k in _depth_keys:
                    metrics[f"src_depth_{k}"].append(m[k])
                if self.eval_depth_vae_ceiling:
                    mv = _depth_metrics(
                        depth_gt[b:b+1, :, sv:sv+1],   # GT disparity, VAE round-tripped
                        depth_metric_batch[b, sv, 0],
                        disp_batch[b, sv, 0],
                        mask_hw,
                    )
                    for k in _depth_keys:
                        metrics[f"src_depth_vae_{k}"].append(mv[k])

            # Target view depth metrics (views num_source_views..)
            # Kept per tv as well, so the per-view dump can carry depth and RGB
            # for the same target view on one row.
            tgt_depth_rows = []
            for tv in range(self.num_target_views):
                vi = self.num_source_views + tv
                mask_hw = loss_masks[b, 0, vi, 0]  # [H, W]
                m = _depth_metrics(
                    pred_depth[b:b+1, :, vi:vi+1],
                    depth_metric_batch[b, vi, 0],   # GT metric depth in meters
                    disp_batch[b, vi, 0],           # GT normalised disparity
                    mask_hw,
                )
                tgt_depth_rows.append(m)
                for k in _depth_keys:
                    metrics[f"tgt_depth_{k}"].append(m[k])
                if self.eval_depth_vae_ceiling:
                    mv = _depth_metrics(
                        depth_gt[b:b+1, :, vi:vi+1],   # GT disparity, VAE round-tripped
                        depth_metric_batch[b, vi, 0],
                        disp_batch[b, vi, 0],
                        mask_hw,
                    )
                    for k in _depth_keys:
                        metrics[f"tgt_depth_vae_{k}"].append(mv[k])
                    tgt_depth_rows[-1].update(
                        {f"vae_{k}": mv[k] for k in _depth_keys}
                    )

            # Target view RGB PSNR (decode to pixel space)
            for tv in range(self.num_target_views):
                vi = self.num_source_views + tv
                pred_px = _decode_view(pred_rgb[b:b+1], vi)    # [1,3,H,W]
                gt_px   = _decode_view(rgb_gt[b:b+1],   vi)
                metrics["tgt_rgb_psnr"].append(_psnr(pred_px, gt_px))
                metrics["tgt_rgb_lpips"].append(_lpips(pred_px, gt_px))

                # --- non-learned baselines on the very same reference --------
                raw_px = rgb_px_batch[b, vi][None]             # [1,3,H,W]
                metrics["tgt_rgb_psnr_vae"].append(_psnr(gt_px, raw_px))

                # Nearest source view by camera-centre distance.  Rotation is
                # ignored on purpose: this is meant to be the dumbest defensible
                # choice, not a tuned retrieval baseline.
                d = (cam_centers[b, : self.num_source_views] - cam_centers[b, vi]).norm(dim=-1)
                nearest = int(torch.argmin(d).item())
                copy_px = rgb_px_batch[b, nearest][None]       # [1,3,H,W]
                metrics["tgt_rgb_psnr_copy"].append(_psnr(copy_px, gt_px))
                metrics["tgt_rgb_lpips_copy"].append(_lpips(copy_px, gt_px))

                # How hard is this particular target?  Angle to the best-aligned
                # source, not to the nearest one: `d` above ranks by centre
                # distance, which can pick a camera sitting next to the target
                # but pointing away.
                cos = (cam_axes[b, : self.num_source_views] * cam_axes[b, vi]).sum(-1)
                min_angle = float(
                    torch.rad2deg(torch.acos(cos.clamp(-1.0, 1.0))).min()
                )
                metrics["tgt_view_min_src_angle_deg"].append(min_angle)
                metrics["tgt_view_min_src_dist"].append(float(d.min()))

                row = {
                    "sample_id": inputs["sample_id"][b],
                    "target_index": tv,
                    "view_index": vi,
                    "tgt_rgb_psnr": metrics["tgt_rgb_psnr"][-1],
                    "tgt_rgb_lpips": metrics["tgt_rgb_lpips"][-1],
                    "tgt_rgb_psnr_copy": metrics["tgt_rgb_psnr_copy"][-1],
                    "tgt_rgb_psnr_vae": metrics["tgt_rgb_psnr_vae"][-1],
                    "tgt_view_min_src_angle_deg": min_angle,
                    "tgt_view_min_src_dist": float(d.min()),
                }
                row.update(
                    {f"tgt_depth_{k}": tgt_depth_rows[tv][k] for k in _depth_keys}
                )
                if self.eval_depth_vae_ceiling:
                    # Same view, same mask, same alignment -- so the per-view row
                    # carries a paired (model, ceiling) couple and the headroom can
                    # be stratified without re-running anything.
                    row.update(
                        {f"tgt_depth_vae_{k}": tgt_depth_rows[tv][f"vae_{k}"]
                         for k in _depth_keys}
                    )

                if splat_px is not None:
                    w = splat_px[b, tv]                        # [3,H,W]
                    m = splat_mask[b, tv]                      # [1,H,W]
                    coverage = float(m.mean())
                    # Holes get the mean splatted colour, not black: black would
                    # punish the baseline for something no honest deployment
                    # would do, and would make the number a coverage statistic
                    # rather than a geometry one.
                    denom = m.sum().clamp_min(1.0)
                    fill = (w * m).sum(dim=(1, 2)) / denom     # [3]
                    filled = (w * m + fill[:, None, None] * (1.0 - m))[None]
                    metrics["tgt_rgb_splat_coverage"].append(coverage)
                    metrics["tgt_rgb_psnr_splat"].append(_psnr(filled, gt_px))
                    metrics["tgt_rgb_lpips_splat"].append(_lpips(filled, gt_px))
                    metrics["tgt_rgb_psnr_splat_valid"].append(
                        _psnr_masked(w[None], gt_px, m[None])
                    )

                    # The whole point: the model's error where the answer is
                    # determined by the source views, versus where it is not.
                    hole = 1.0 - m
                    metrics["tgt_rgb_psnr_covered"].append(
                        _psnr_masked(pred_px, gt_px, m[None])
                    )
                    metrics["tgt_rgb_psnr_uncovered"].append(
                        _psnr_masked(pred_px, gt_px, hole[None])
                    )
                    metrics["tgt_rgb_psnr_copy_covered"].append(
                        _psnr_masked(copy_px, gt_px, m[None])
                    )
                    metrics["tgt_rgb_psnr_copy_uncovered"].append(
                        _psnr_masked(copy_px, gt_px, hole[None])
                    )
                    metrics["tgt_rgb_psnr_vae_covered"].append(
                        _psnr_masked(gt_px, raw_px, m[None])
                    )
                    for k in (
                        "tgt_rgb_splat_coverage", "tgt_rgb_psnr_splat",
                        "tgt_rgb_lpips_splat", "tgt_rgb_psnr_splat_valid",
                        "tgt_rgb_psnr_covered", "tgt_rgb_psnr_uncovered",
                        "tgt_rgb_psnr_copy_covered", "tgt_rgb_psnr_copy_uncovered",
                        "tgt_rgb_psnr_vae_covered",
                    ):
                        row[k] = metrics[k][-1]

                if splat_px_pred is not None:
                    wp = splat_px_pred[b, tv]                  # [3,H,W]
                    mp = splat_mask_pred[b, tv]                # [1,H,W]
                    denom_p = mp.sum().clamp_min(1.0)
                    fill_p = (wp * mp).sum(dim=(1, 2)) / denom_p
                    filled_p = (wp * mp + fill_p[:, None, None] * (1.0 - mp))[None]
                    metrics["tgt_rgb_splat_pred_coverage"].append(float(mp.mean()))
                    metrics["tgt_rgb_psnr_splat_pred"].append(_psnr(filled_p, gt_px))
                    metrics["tgt_rgb_lpips_splat_pred"].append(_lpips(filled_p, gt_px))
                    metrics["tgt_rgb_psnr_splat_pred_valid"].append(
                        _psnr_masked(wp[None], gt_px, mp[None])
                    )
                    for k in (
                        "tgt_rgb_splat_pred_coverage", "tgt_rgb_psnr_splat_pred",
                        "tgt_rgb_lpips_splat_pred", "tgt_rgb_psnr_splat_pred_valid",
                    ):
                        row[k] = metrics[k][-1]

                per_view_rows.append(row)

                if save_vis:
                    # Two columns per target: GT and prediction.  Collected here
                    # and drawn after the loop, so all T targets land in one image.
                    tgt_tiles.append((
                        _to_np(gt_px),
                        _to_np(pred_px),
                        _gt_depth_to_np(disp_batch[b, vi]),
                        _pred_depth_to_np(pred_depth[b:b+1], vi),
                    ))
                    # Same `row` that goes into per_view_metrics.jsonl, so the
                    # caption under a tile and the JSONL line for it never drift
                    # apart.  .get with a NaN fallback rather than indexing:
                    # tgt_rgb_splat_coverage is only set when num_source_views > 0
                    # and num_target_views > 0 (always true here, but this
                    # function should not crash on a config where it is not).
                    tgt_captions.append([
                        f"cov {row.get('tgt_rgb_splat_coverage', float('nan')):.3f}",
                        f"psnr {row.get('tgt_rgb_psnr', float('nan')):.2f} dB",
                        f"lpips {row.get('tgt_rgb_lpips', float('nan')):.3f}",
                        f"delta1 {row.get('tgt_depth_delta1', float('nan')):.3f}",
                        f"abs_rel {row.get('tgt_depth_abs_rel', float('nan')):.3f}",
                    ])

            if save_vis and (src_rgb_nps or tgt_tiles):
                src_captions = [
                    [f"abs_rel {m['abs_rel']:.3f}", f"delta1 {m['delta1']:.3f}"]
                    for m in src_depth_rows
                ]
                img, warning = _compose_eval_grid(
                    src_rgb_nps, src_gt_depth_nps, src_pred_depth_nps, tgt_tiles,
                    downscale=self.eval_vis_downscale,
                    src_captions=src_captions,
                    tgt_captions=tgt_captions,
                )
                if warning and not getattr(self, "_warned_grid_labels", False):
                    print(f"[eval] {warning}")
                    self._warned_grid_labels = True

                path = os.path.join(
                    save_dir,
                    f"b{b + vis_offset:02d}_step{global_step:06d}.{self.eval_vis_format}")
                if self.eval_vis_format == "jpg":
                    img.save(path, quality=90, optimize=True)
                else:
                    img.save(path, optimize=True)

        # Per-view dump.  Append, because the frozen eval set is scored in
        # several chunks that all resolve to the same eval_step_* directory; each
        # row carries sample_id + target_index so downstream analysis can
        # deduplicate if a step is ever evaluated twice.
        if per_view_rows:
            with open(
                os.path.join(save_dir, "per_view_metrics.jsonl"), "a", encoding="utf-8"
            ) as handle:
                for row in per_view_rows:
                    handle.write(json.dumps(row, default=float) + "\n")

        # Aggregate means (NaN-safe: temporal_padding can yield undecodable views)
        mean_metrics = {}
        for k, v in metrics.items():
            if not v:
                continue
            arr = np.asarray(v, dtype=np.float64)
            if np.all(np.isnan(arr)):
                continue
            mean_metrics[k] = float(np.nanmean(arr))

        # Ratio-of-means skill.  `delta1_skill` is averaged per view, and a
        # single view with a near-perfect ramp dominates that mean; on a 12-view
        # target set this alone moved the reported figure from +0.01 to -0.51.
        # Computing the ratio from the aggregated delta1 / ramp is stable
        # because the denominator is the eval-set-wide headroom.
        for side in ("src", "tgt"):
            d1 = mean_metrics.get(f"{side}_depth_delta1")
            ramp = mean_metrics.get(f"{side}_depth_delta1_ramp")
            if d1 is None or ramp is None or 1.0 - ramp <= 1e-3:
                continue
            mean_metrics[f"{side}_depth_delta1_skill_agg"] = (d1 - ramp) / (1.0 - ramp)

        if was_training:
            self.train()
        return mean_metrics

    @torch.no_grad()
    def evaluate_warp_sensitivity(self, batch, max_samples=4):
        """Diagnostic only -- not used by training or by `evaluate()`.

        Answers whether the model's covered-pixel accuracy depends on the
        warp block carrying correct, pixel-precise correspondence, or on
        source-depth accuracy at all.  Four variants of the SAME batch, SAME
        sampling seed, scored against the SAME GT-splat covered/uncovered
        partition:
          baseline -- the model's normal eval-time warp block: two-pass,
                      built from its OWN predicted source depth.
          shuffle  -- baseline warp block with spatial positions permuted per
                      (sample, view). Same values, zero correspondence.
          lowfreq  -- baseline warp block low-pass filtered. Coarse layout
                      kept, per-pixel detail destroyed.
          oracle   -- warp block rebuilt from GT source depth (never
                      available at real inference; upper bound on what a
                      perfect source-depth predictor could buy).
        If shuffle/lowfreq barely hurt covered PSNR and oracle barely helps
        it, the model is not using pixel-level correspondence from the warp
        channel -- regardless of how accurate that correspondence is.
        """
        import math

        was_training = self.training
        self.eval()
        device = self.pipe.device

        def _slice(x, n):
            if isinstance(x, torch.Tensor):
                return x[:n]
            if isinstance(x, list):
                return x[:n]
            return x

        batch_small = {k: _slice(v, max_samples) for k, v in batch.items()}

        def _decode_view(latents_bcvhw, view_idx):
            v_lat = latents_bcvhw[:, :, view_idx:view_idx + 1].to(device).to(self.pipe.torch_dtype)
            decoded = self.pipe.vae.decode(v_lat, device=device)
            return decoded[:, :, 0].float().clamp(0, 1)

        def _psnr_masked(pred_img, gt_img, mask_img):
            m = mask_img.expand_as(pred_img)
            n = m.sum()
            if n < 1:
                return float("nan")
            mse = ((((pred_img - gt_img) ** 2) * m).sum() / n).item()
            if mse < 1e-10:
                return 100.0
            return -10 * math.log10(mse)

        num_src = self.num_source_views
        autocast_ctx = torch.autocast(
            device_type="npu", dtype=self.pipe.torch_dtype, enabled=True
        )
        with autocast_ctx:
            self._eval_inputs = True
            try:
                inputs_eval = self.forward_preprocess(batch_small)
            finally:
                self._eval_inputs = False

            variants = {}
            for mode in ("baseline", "shuffle", "lowfreq"):
                gen = torch.Generator(device=device)
                gen.manual_seed(self.eval_sample_seed)
                corruption = None if mode == "baseline" else mode
                _, pred_rgb = self.sample_rgb(
                    inputs_eval, num_steps=self.eval_num_inference_steps,
                    generator=gen, warp_corruption=corruption,
                )
                variants[mode] = pred_rgb

            # Oracle: rebuild the warp block from GT source depth instead of
            # the model's own prediction, exactly the way forward_preprocess
            # builds it during TRAINING (self._eval_inputs is already False
            # here). warp_dropout_prob must be forced to 0 or a random subset
            # of samples would get a zeroed block instead of the oracle one.
            # warp_perturb_prob must be forced to 0 for the SAME reason: this arm
            # trains on a deliberately degraded splat, and leaving the factor on
            # here would make the "oracle" upper bound a degraded oracle -- a
            # number that looks sane, moves in the right direction, and silently
            # understates the ceiling of the very axis the arm is testing.
            saved_dropout = self.warp_dropout_prob
            saved_perturb = self.warp_perturb_prob
            self.warp_dropout_prob = 0.0
            self.warp_perturb_prob = 0.0
            try:
                inputs_oracle = self.forward_preprocess(batch_small)
            finally:
                self.warp_dropout_prob = saved_dropout
                self.warp_perturb_prob = saved_perturb
            gen = torch.Generator(device=device)
            gen.manual_seed(self.eval_sample_seed)
            _, pred_rgb_oracle = self._sample_rgb_once(
                inputs_oracle, num_steps=self.eval_num_inference_steps, generator=gen
            )
            variants["oracle"] = pred_rgb_oracle

        rgb_gt = inputs_eval["rgb_gt"]
        if self.latent_scale != 1.0:
            rgb_gt = rgb_gt / self.latent_scale
            for k in variants:
                variants[k] = variants[k] / self.latent_scale

        B = rgb_gt.shape[0]
        rgb_px_batch = batch_small["images"].to(device).float().clamp(0.0, 1.0)
        disp_batch = batch_small["disparity"].to(device)
        _, splat_mask = self._splat_sources_pixel(
            rgb_px_batch, disp_batch[:, :num_src, 0].float(),
            batch_small["K"].to(device).float(), inputs_eval["c2w"],
        )

        rows = []
        for b in range(B):
            for tv in range(self.num_target_views):
                vi = num_src + tv
                gt_px = _decode_view(rgb_gt[b:b + 1], vi)
                m = splat_mask[b, tv]
                hole = 1.0 - m
                row = {
                    "sample_id": inputs_eval["sample_id"][b],
                    "view_index": vi,
                    "tgt_rgb_splat_coverage": float(m.mean()),
                }
                for mode, pred_rgb in variants.items():
                    pred_px = _decode_view(pred_rgb[b:b + 1], vi)
                    row[f"covered_{mode}"] = _psnr_masked(pred_px, gt_px, m[None])
                    row[f"uncovered_{mode}"] = _psnr_masked(pred_px, gt_px, hole[None])
                rows.append(row)

        if was_training:
            self.train()
        return rows

    @torch.no_grad()
    def _corrupt_pose_c2w(self, c2w, mode):
        """Diagnostic only -- replace the TARGET views' camera poses with wrong
        ones, leaving the source views (and therefore the reference view that
        `relative_c2w` divides by) bit-identical.

        Operates on the RAW c2w, before `relative_c2w`, so the corrupted poses
        go through exactly the same pipeline as the true ones and the result is
        a *legal* pose field -- just one describing different cameras.  That is
        the whole point: a corruption that produced out-of-distribution VALUES
        would be indistinguishable from one that produced wrong GEOMETRY.

        `swap_tgt` cyclically shifts the target poses among themselves.  Every
        target then asks for a view that some other target in the same sample
        wants, so the marginal distribution of poses, baselines and the
        source-target angle spread is unchanged; only the assignment is wrong.
        `src0` points every target at source view 0, i.e. "do not move" -- a
        pose the model has certainly seen, and the one whose correct answer is
        already sitting in the conditioning images.

        c2w: [B, V, 4, 4].
        """
        num_src = self.num_source_views
        num_tgt = c2w.shape[1] - num_src
        out = c2w.clone()
        if mode == "swap_tgt":
            if num_tgt < 2:
                raise ValueError(
                    "swap_tgt needs >=2 target views to be a derangement; got "
                    f"{num_tgt}.  With one target a cyclic shift is the identity, "
                    "and the diagnostic would silently measure nothing."
                )
            out[:, num_src:] = torch.roll(out[:, num_src:], shifts=1, dims=1)
            return out
        if mode == "src0":
            out[:, num_src:] = out[:, 0:1].expand_as(out[:, num_src:])
            return out
        raise ValueError(f"unknown pose corruption mode: {mode!r}")

    @torch.no_grad()
    def evaluate_pose_sensitivity(self, batch, max_samples=4):
        """Diagnostic only -- not used by training or by `evaluate()`.

        The companion of `evaluate_warp_sensitivity`, for the OTHER geometric
        input.  Two paths carry camera geometry into this model: the explicit
        forward splat (`warp_block`, pixel correspondence) and the abstract
        Plucker ray field (`plucker_block`, "where is this camera").  §6 of
        report_20260904_p0a_coverage_stratification.md measured the first one
        (shuffle -3.37 dB).  The second has never been measured, and a
        weight-space analysis cannot settle it -- see
        report_20260910_pose_conditioning_magnitude.md, where a magnitude
        argument gave the wrong answer in both directions.

        Six variants of the SAME batch, SAME sampling seed, SAME GT-splat
        partition:
          baseline     -- normal eval-time inputs.
          swap_tgt     -- Plucker field rasterised from target poses cyclically
                          shifted among themselves.  Valid poses, wrong ones.
                          THE DECISIVE VARIANT: if the model reads the pose
                          field at all it must now render the wrong viewpoint.
          src0         -- every target pose replaced by source view 0's.
          pose_shuffle -- baseline Plucker block with latent positions permuted
                          per (sample, view).  Mirrors the warp `shuffle`, but
                          note it is a coarser corruption in "pixel" mode,
                          where PixelUnshuffle has already folded each 8x8
                          patch into the channel axis: the intra-patch geometry
                          survives, the patch layout does not.
          warp_shuffle -- warp block shuffled, pose field intact.  Present so
                          the two paths are read on ONE batch, seed and
                          checkpoint instead of across reports.
          both_shuffle -- both, to see whether they are redundant.
        The warp block is rebuilt inside `sample_rgb` from `inputs["c2w"]`,
        which is left TRUE in every pose variant, so `swap_tgt` and `src0`
        corrupt the abstract path only -- the splat still lands correctly.
        """
        import math

        if self.view_condition_mode != "concat":
            # In "add" mode the pose field is summed into rgb_latents, so
            # swapping it means swapping a tensor that also carries the
            # conditioning images through a second VAE encode.  Refuse rather
            # than quietly change two things at once.
            raise NotImplementedError(
                "evaluate_pose_sensitivity assumes view_condition_mode='concat'; "
                f"got {self.view_condition_mode!r}."
            )

        was_training = self.training
        self.eval()
        device = self.pipe.device

        def _slice(x, n):
            if isinstance(x, (torch.Tensor, list)):
                return x[:n]
            return x

        batch_small = {k: _slice(v, max_samples) for k, v in batch.items()}

        def _decode_view(latents_bcvhw, view_idx):
            v_lat = latents_bcvhw[:, :, view_idx:view_idx + 1].to(device).to(self.pipe.torch_dtype)
            decoded = self.pipe.vae.decode(v_lat, device=device)
            return decoded[:, :, 0].float().clamp(0, 1)

        def _psnr_masked(pred_img, gt_img, mask_img):
            m = mask_img.expand_as(pred_img)
            n = m.sum()
            if n < 1:
                return float("nan")
            mse = ((((pred_img - gt_img) ** 2) * m).sum() / n).item()
            if mse < 1e-10:
                return 100.0
            return -10 * math.log10(mse)

        def _lpips(pred_img, gt_img):
            """LPIPS on [0,1] images; the net wants [-1,1].  Lower is better.

            Full-frame only, and deliberately so: LPIPS is a CNN over patches,
            so there is no honest way to restrict it to a splat-coverage mask
            the way `_psnr_masked` does.  It is reported because PSNR alone
            cannot adjudicate this diagnostic -- PSNR's optimum is the
            conditional MEAN, so a corruption that makes the model blurrier
            can RAISE it (report_20260907_c_sweep §9-1b), and LPIPS has ~1.4x
            PSNR's discrimination power on this sweep.
            """
            net = self._get_lpips(device)
            if net is None:
                return float("nan")
            with torch.no_grad():
                d = net(pred_img.float() * 2 - 1, gt_img.float() * 2 - 1)
            return float(d.mean())

        # (name, pose corruption, plucker corruption, warp corruption)
        variant_spec = (
            ("baseline", None, None, None),
            ("swap_tgt", "swap_tgt", None, None),
            ("src0", "src0", None, None),
            ("pose_shuffle", None, "shuffle", None),
            ("warp_shuffle", None, None, "shuffle"),
            ("both_shuffle", None, "shuffle", "shuffle"),
        )

        num_src = self.num_source_views
        autocast_ctx = torch.autocast(
            device_type="npu", dtype=self.pipe.torch_dtype, enabled=True
        )
        with autocast_ctx:
            self._eval_inputs = True
            try:
                inputs_eval = self.forward_preprocess(batch_small)
                # One extra preprocess per pose mode, cached: it is a full VAE
                # encode, and only the pose-derived tensors are kept from it.
                alt_pose = {}
                for mode in {s[1] for s in variant_spec if s[1] is not None}:
                    batch_pose = dict(batch_small)
                    batch_pose["c2w"] = self._corrupt_pose_c2w(
                        batch_small["c2w"].to(device).float(), mode
                    )
                    alt = self.forward_preprocess(batch_pose)
                    alt_pose[mode] = {
                        k: alt[k]
                        for k in ("plucker_block", "cape_pose", "roce_camera_latents")
                        if alt.get(k) is not None
                    }
            finally:
                self._eval_inputs = False

            variants = {}
            for name, pose_mode, plucker_mode, warp_mode in variant_spec:
                inputs = dict(inputs_eval)
                if pose_mode is not None:
                    inputs.update(alt_pose[pose_mode])
                if plucker_mode is not None:
                    inputs["plucker_block"] = self._corrupt_warp_block(
                        inputs["plucker_block"], plucker_mode
                    )
                gen = torch.Generator(device=device)
                gen.manual_seed(self.eval_sample_seed)
                _, pred_rgb = self.sample_rgb(
                    inputs, num_steps=self.eval_num_inference_steps,
                    generator=gen, warp_corruption=warp_mode,
                )
                variants[name] = pred_rgb

        rgb_gt = inputs_eval["rgb_gt"]
        if self.latent_scale != 1.0:
            rgb_gt = rgb_gt / self.latent_scale
            for k in variants:
                variants[k] = variants[k] / self.latent_scale

        rgb_px_batch = batch_small["images"].to(device).float().clamp(0.0, 1.0)
        disp_batch = batch_small["disparity"].to(device)
        _, splat_mask = self._splat_sources_pixel(
            rgb_px_batch, disp_batch[:, :num_src, 0].float(),
            batch_small["K"].to(device).float(), inputs_eval["c2w"],
        )

        rows = []
        for b in range(rgb_gt.shape[0]):
            for tv in range(self.num_target_views):
                vi = num_src + tv
                gt_px = _decode_view(rgb_gt[b:b + 1], vi)
                m = splat_mask[b, tv]
                hole = 1.0 - m
                ones = torch.ones_like(m)
                row = {
                    "sample_id": inputs_eval["sample_id"][b],
                    "view_index": vi,
                    "tgt_rgb_splat_coverage": float(m.mean()),
                }
                for name, pred_rgb in variants.items():
                    pred_px = _decode_view(pred_rgb[b:b + 1], vi)
                    # The pose field conditions the WHOLE frame, not just the
                    # part a splat happens to reach, so `all_` is the headline
                    # here -- unlike the warp diagnostic, where `covered_` is.
                    row[f"all_{name}"] = _psnr_masked(pred_px, gt_px, ones[None])
                    row[f"covered_{name}"] = _psnr_masked(pred_px, gt_px, m[None])
                    row[f"uncovered_{name}"] = _psnr_masked(pred_px, gt_px, hole[None])
                    row[f"lpips_{name}"] = _lpips(pred_px, gt_px)
                rows.append(row)

        if was_training:
            self.train()
        return rows

    @torch.no_grad()
    def evaluate_posterior_mean(self, batch, num_samples=4, max_samples=4):
        """Diagnostic only -- separates model error from the SAMPLING penalty.

        `evaluate()` scores one stochastic draw from a generative model with
        PSNR, a metric minimised by the posterior MEAN.  If the posterior were
        perfectly calibrated and the GT an independent draw from it, then
        E||sample - gt||^2 = 2 E||mean - gt||^2 exactly, i.e. reporting a sample
        costs 10*log10(2) = 3.01 dB that no change to the model can recover.
        Some fraction of the 16 dB between tgt_rgb_psnr (13.6) and its VAE
        ceiling (30.0) is therefore an artefact of what is being measured, and
        prioritising architecture work without knowing that fraction risks
        chasing a gap that is partly definitional.

        Draws `num_samples` independent samples of the same batch (identical
        conditioning, different seeds) and reports, per target view:
          psnr_mean_k    -- PSNR of the average of the first k samples, k=1..N.
                            k=1 is exactly what evaluate() reports; the curve
                            shows how much of the gap is sampling variance.
          psnr_single    -- mean of the N individual per-sample PSNRs, a
                            lower-variance estimate of the k=1 number.
          lpips_single / lpips_mean -- averaging is expected to HURT LPIPS. That
                            is the trade being measured, not a bug: it shows
                            whether the model is a sharp-but-misaligned sampler
                            or a blurry regressor.
          std_covered / std_uncovered -- per-pixel standard deviation across the
                            N samples, split by the GT-splat partition.

        The std split is the sharpest thing here.  A model that has understood
        which pixels the source views determine should be markedly LESS uncertain
        on covered pixels than on holes.  If the two are equal, its uncertainty
        is not conditioned on evidence availability at all -- which would explain
        directly why the covered-vs-uncovered margin does not grow with coverage
        (report_20260904_p0a §3.2 and the difference-in-differences table), and
        would locate the defect in how evidence is fused rather than in how
        accurate the warp block is (which §6 already showed the model does read).
        """
        import math

        was_training = self.training
        self.eval()
        device = self.pipe.device

        def _slice(x, n):
            if isinstance(x, (torch.Tensor, list)):
                return x[:n]
            return x

        batch_small = {k: _slice(v, max_samples) for k, v in batch.items()}

        def _decode_view(latents_bcvhw, view_idx):
            v_lat = latents_bcvhw[:, :, view_idx:view_idx + 1].to(device).to(self.pipe.torch_dtype)
            return self.pipe.vae.decode(v_lat, device=device)[:, :, 0].float().clamp(0, 1)

        def _psnr_masked(pred_img, gt_img, mask_img):
            m = mask_img.expand_as(pred_img)
            n = m.sum()
            if n < 1:
                return float("nan")
            mse = ((((pred_img - gt_img) ** 2) * m).sum() / n).item()
            return 100.0 if mse < 1e-10 else -10 * math.log10(mse)

        def _masked_mean(x, mask_img):
            m = mask_img.expand_as(x)
            n = m.sum()
            return float("nan") if n < 1 else float((x * m).sum() / n)

        def _lpips(pred_img, gt_img):
            net = self._get_lpips(device)
            if net is None:
                return float("nan")
            return float(net(pred_img.float() * 2 - 1, gt_img.float() * 2 - 1).mean())

        num_src = self.num_source_views
        with torch.autocast(device_type="npu", dtype=self.pipe.torch_dtype, enabled=True):
            self._eval_inputs = True
            try:
                inputs_eval = self.forward_preprocess(batch_small)
            finally:
                self._eval_inputs = False

            # Seed i = eval_sample_seed + i, so sample 0 reproduces evaluate()
            # exactly and the k=1 column is directly comparable to train.log.
            draws = []
            for i in range(num_samples):
                gen = torch.Generator(device=device)
                gen.manual_seed(self.eval_sample_seed + i)
                _, pred_rgb = self.sample_rgb(
                    inputs_eval, num_steps=self.eval_num_inference_steps, generator=gen
                )
                draws.append(pred_rgb)

        rgb_gt = inputs_eval["rgb_gt"]
        if self.latent_scale != 1.0:
            rgb_gt = rgb_gt / self.latent_scale
            draws = [d / self.latent_scale for d in draws]

        B = rgb_gt.shape[0]
        rgb_px_batch = batch_small["images"].to(device).float().clamp(0.0, 1.0)
        disp_batch = batch_small["disparity"].to(device)
        _, splat_mask = self._splat_sources_pixel(
            rgb_px_batch, disp_batch[:, :num_src, 0].float(),
            batch_small["K"].to(device).float(), inputs_eval["c2w"],
        )

        rows = []
        for b in range(B):
            for tv in range(self.num_target_views):
                vi = num_src + tv
                gt_px = _decode_view(rgb_gt[b:b + 1], vi)
                m = splat_mask[b, tv][None]                  # [1,1,H,W]
                hole = 1.0 - m
                # Averaging happens in PIXEL space, not latent space: the decoder
                # is non-linear, so the decode of a latent mean is not the mean of
                # the decodes, and only the latter is the posterior mean of the
                # quantity PSNR is computed on.
                preds = [_decode_view(d[b:b + 1], vi) for d in draws]
                stack = torch.cat(preds, dim=0)              # [N,3,H,W]
                std = stack.std(dim=0, unbiased=False)[None]  # [1,3,H,W]

                row = {
                    "sample_id": inputs_eval["sample_id"][b],
                    "view_index": vi,
                    "tgt_rgb_splat_coverage": float(m.mean()),
                    "psnr_single": float(
                        sum(_psnr_masked(p, gt_px, torch.ones_like(m)) for p in preds)
                        / len(preds)
                    ),
                    "lpips_single": float(sum(_lpips(p, gt_px) for p in preds) / len(preds)),
                    "std_covered": _masked_mean(std, m),
                    "std_uncovered": _masked_mean(std, hole),
                    "std_all": float(std.mean()),
                }
                running = torch.zeros_like(preds[0])
                for k, p in enumerate(preds, start=1):
                    running = running + p
                    avg = running / k
                    row[f"psnr_mean_{k}"] = _psnr_masked(avg, gt_px, torch.ones_like(m))
                    row[f"covered_mean_{k}"] = _psnr_masked(avg, gt_px, m)
                    row[f"uncovered_mean_{k}"] = _psnr_masked(avg, gt_px, hole)
                row["lpips_mean"] = _lpips(running / len(preds), gt_px)
                rows.append(row)

        if was_training:
            self.train()
        return rows