"""Forward splatting of source views into target views.

Used to hand the DiT an explicit photometric correspondence signal instead of
asking 30 layers of self-attention to discover wide-baseline stereo from 102
training scenes.

Forward splatting (source -> target) is deliberate: the alternative, backward
warping, needs *target* depth, which is only available as ground truth at
training time.  A model trained on a backward warp built from GT target depth
would rely on geometry it can never have at inference.  Splatting only needs
SOURCE depth, and source depth is the easy half of the problem -- the source
images are given clean as conditioning.
"""

import torch

# Lives next to `ray_condition`, which is where two of the four autocast defects
# this guards against actually are.  Imported rather than duplicated so all four
# call sites are provably using the same rule.
from diffsynth.models.wan_video_camera_controller import fp32_geometry


def _pixel_rays_to_world(depth, valid, K, c2w):
    """Unproject a pixel grid to world coordinates.

    depth/valid: [N, H, W];  K: [N, 3, 3];  c2w: [N, 4, 4].
    Returns world points [N, H*W, 3] in the same units as ``c2w``'s translation.
    """
    num, height, width = depth.shape
    device = depth.device
    row, col = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    # +0.5 puts the sample at the pixel centre, which is the convention the
    # matching floor() on the target side assumes.
    pix = torch.stack([col + 0.5, row + 0.5, torch.ones_like(col)], dim=0)
    pix = pix.reshape(3, height * width)

    cam = torch.linalg.inv(K.float()) @ pix              # [N, 3, HW]
    cam = cam * depth.reshape(num, 1, height * width).float()
    rot = c2w.float()[:, :3, :3]
    trans = c2w.float()[:, :3, 3:4]
    world = rot @ cam + trans                            # [N, 3, HW]
    return world.transpose(1, 2).contiguous()            # [N, HW, 3]


@torch.no_grad()
def splat_sources_to_targets(
    src_rgb,
    src_depth,
    src_valid,
    src_K,
    src_c2w,
    tgt_K,
    tgt_c2w,
    z_near=1e-3,
    z_tolerance=0.02,
    return_target_disparity=False,
    geometry_fp32=False,
):
    """Splat every source pixel into every target view with a shared z-buffer.

    `geometry_fp32` takes the whole body out of autocast.  Nothing in here is
    learned and nothing carries a gradient, so the only cost is bandwidth on
    tensors that are small next to the VAE encode this feeds; see `fp32_geometry`
    for why the existing `.float()` calls do not already achieve this.  It is off
    by default because turning it on changes `warp_block`, i.e. a model input.

    See `_splat_sources_to_targets` for the argument and return shapes.
    """
    with fp32_geometry(src_rgb.device, geometry_fp32):
        return _splat_sources_to_targets(
            src_rgb, src_depth, src_valid, src_K, src_c2w, tgt_K, tgt_c2w,
            z_near=z_near, z_tolerance=z_tolerance,
            return_target_disparity=return_target_disparity,
        )


@torch.no_grad()
def _splat_sources_to_targets(
    src_rgb,
    src_depth,
    src_valid,
    src_K,
    src_c2w,
    tgt_K,
    tgt_c2w,
    z_near=1e-3,
    z_tolerance=0.02,
    return_target_disparity=False,
):
    """The splat itself.  Call `splat_sources_to_targets`, which adds the dtype
    context; this is split out only so that context wraps every line below.

    Args:
        src_rgb:   [B, S, 3, H, W]
        src_depth: [B, S, H, W]   positive, same world scale as the poses
        src_valid: [B, S, H, W]   bool
        src_K, src_c2w: [B, S, 3, 3] / [B, S, 4, 4]
        tgt_K, tgt_c2w: [B, T, 3, 3] / [B, T, 4, 4]
        z_tolerance: a splat is kept when its depth is within this relative
            margin of the front-most depth landing on the same target pixel.
            Pure equality would drop the front surface itself whenever two
            source views disagree by a rounding error.
        return_target_disparity: also return 1/z of every kept splat measured in
            the TARGET camera frame.  This is not the source disparity carried
            along -- a point's disparity changes with the viewpoint -- so it can
            only be produced here, inside the loop that already has z.

    Returns:
        warped: [B, T, 3, H, W]  zeros where nothing landed
        mask:   [B, T, 1, H, W]  1 where at least one source pixel landed
        tgt_disp: [B, T, 1, H, W], only when return_target_disparity is set
    """
    batch, num_src, _, height, width = src_rgb.shape
    num_tgt = tgt_K.shape[1]
    device = src_rgb.device
    num_pix = height * width

    world = _pixel_rays_to_world(
        src_depth.reshape(batch * num_src, height, width),
        src_valid.reshape(batch * num_src, height, width),
        src_K.reshape(batch * num_src, 3, 3),
        src_c2w.reshape(batch * num_src, 4, 4),
    ).reshape(batch, num_src, num_pix, 3)

    colors = src_rgb.float().reshape(batch, num_src, 3, num_pix).transpose(2, 3)
    valid = src_valid.reshape(batch, num_src, num_pix)

    warped = torch.zeros(batch, num_tgt, num_pix, 3, device=device, dtype=torch.float32)
    mask = torch.zeros(batch, num_tgt, num_pix, device=device, dtype=torch.float32)
    tgt_disp = (
        torch.zeros(batch, num_tgt, num_pix, device=device, dtype=torch.float32)
        if return_target_disparity
        else None
    )

    for t in range(num_tgt):
        w2c = torch.linalg.inv(tgt_c2w[:, t].float())        # [B, 4, 4]
        K_t = tgt_K[:, t].float()

        # Two passes over the source views: the z-buffer has to be complete
        # before any colour is written, otherwise a later source view could
        # overwrite a nearer surface that an earlier one already deposited.
        projected = []
        zbuf = torch.full((batch, num_pix), float("inf"), device=device)
        for s in range(num_src):
            cam = world[:, s] @ w2c[:, :3, :3].transpose(1, 2) + w2c[:, :3, 3].unsqueeze(1)
            z = cam[..., 2]
            uvw = cam @ K_t.transpose(1, 2)
            col = torch.where(z > z_near, uvw[..., 0] / z.clamp_min(z_near), torch.full_like(z, -1.0))
            row = torch.where(z > z_near, uvw[..., 1] / z.clamp_min(z_near), torch.full_like(z, -1.0))
            xi = col.floor().long()
            yi = row.floor().long()
            in_view = (
                (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
                & (z > z_near) & valid[:, s]
            )
            flat = torch.where(in_view, yi * width + xi, torch.zeros_like(xi))
            zbuf.scatter_reduce_(
                1, flat, torch.where(in_view, z, torch.full_like(z, float("inf"))),
                reduce="amin", include_self=True,
            )
            projected.append((flat, z, in_view))

        for s in range(num_src):
            flat, z, in_view = projected[s]
            front = zbuf.gather(1, flat)
            keep = in_view & (z <= front * (1.0 + z_tolerance) + 1e-6)
            if not bool(keep.any()):
                continue
            bidx = torch.arange(batch, device=device).unsqueeze(1).expand_as(flat)[keep]
            pidx = flat[keep]
            warped[bidx, t, pidx] = colors[:, s][keep]
            mask[bidx, t, pidx] = 1.0
            if tgt_disp is not None:
                tgt_disp[bidx, t, pidx] = 1.0 / z[keep].clamp_min(z_near)

    warped = warped.transpose(2, 3).reshape(batch, num_tgt, 3, height, width)
    mask = mask.reshape(batch, num_tgt, 1, height, width)
    if tgt_disp is None:
        return warped, mask
    return warped, mask, tgt_disp.reshape(batch, num_tgt, 1, height, width)
