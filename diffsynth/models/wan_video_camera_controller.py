import contextlib

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
import os
from typing_extensions import Literal


def fp32_geometry(device, enabled):
    """Take camera geometry out of autocast (ported from zhu_dvd/DVD, D4 fix).

    Camera math written with `.float()` on the operands is STILL bf16 under
    torch.autocast: the *op* picks the dtype, and `matmul` is on autocast's
    lower-precision list while `linalg.inv` is on its fp32 list.  Measured on
    real BlendedMVS poses: relative camera centres off by a median 0.37% of the
    sample baseline (p90 2.91%, max 75.3%), ray-condition pixel grids quantised
    (bf16 holds integers exactly only to 256), ray directions ~0.08 deg off.
    Off by default everywhere: enabling it changes model inputs, so it is a
    sweep factor (`camera_geometry_fp32`), not a silent bug fix.
    """
    if not enabled:
        return contextlib.nullcontext()
    device_type = device.type if hasattr(device, "type") else str(device)
    return torch.autocast(device_type=device_type, enabled=False)


def relative_c2w(c2w, geometry_fp32=False):
    """Express every camera-to-world matrix relative to view 0.  [B, V, 4, 4].

    Off path (the historical behaviour): invert view 0, left-multiply.  Under
    bf16 autocast the translation column evaluates R_ref^T t - R_ref^T t_ref --
    two O(distance-to-origin) terms canceling into an O(baseline) one, which
    bf16 cannot represent.

    fp32 path fixes it twice over: (1) translations are subtracted BEFORE the
    rotation, so the cancellation never happens at all; (2) the whole block runs
    outside autocast so the returned dtype is fp32 (ray_condition sizes its
    pixel grid with c2w.dtype; a bf16 dtype silently quantises the grid).
    """
    if not geometry_fp32:
        c2w_ref_inv = torch.linalg.inv(c2w[:, 0:1])
        return c2w_ref_inv @ c2w
    with fp32_geometry(c2w.device, True):
        c2w = c2w.float()
        r_ref_t = c2w[:, 0:1, :3, :3].transpose(-1, -2)      # [B, 1, 3, 3]
        # Subtract first, rotate second: the difference is O(baseline) before
        # it ever reaches a matmul, so there is nothing left to cancel.
        d = c2w[..., :3, 3] - c2w[:, 0:1, :3, 3]             # [B, V, 3]
        rel = torch.zeros_like(c2w)
        rel[..., :3, :3] = r_ref_t @ c2w[..., :3, :3]
        rel[..., :3, 3] = (r_ref_t @ d.unsqueeze(-1)).squeeze(-1)
        rel[..., 3, 3] = 1.0
        return rel


class SimpleAdapter(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, num_residual_blocks=1):
        super(SimpleAdapter, self).__init__()

        # Pixel Unshuffle: reduce spatial dimensions by a factor of 8
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=8)

        # Convolution: reduce spatial dimensions by a factor
        #  of 2 (without overlap)
        self.conv = nn.Conv2d(in_dim * 64, out_dim,
                              kernel_size=kernel_size, stride=stride, padding=0)

        # Residual blocks for feature extraction
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(out_dim) for _ in range(num_residual_blocks)]
        )

    def forward(self, x):
        # Reshape to merge the frame dimension into batch
        bs, c, f, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(bs * f, c, h, w)

        # Pixel Unshuffle operation
        x_unshuffled = self.pixel_unshuffle(x)

        # Convolution operation
        x_conv = self.conv(x_unshuffled)

        # Feature extraction with residual blocks
        out = self.residual_blocks(x_conv)

        # Reshape to restore original bf dimension
        out = out.view(bs, f, out.size(1), out.size(2), out.size(3))

        # Permute dimensions to reorder (if needed), e.g., swap channels and feature frames
        out = out.permute(0, 2, 1, 3, 4)

        return out

    def process_camera_coordinates(
        self,
        direction: Literal["Left", "Right", "Up", "Down", "LeftUp", "LeftDown", "RightUp", "RightDown"],
        length: int,
        height: int,
        width: int,
        speed: float = 1/54,
        origin=(0, 0.532139961, 0.946026558, 0.5, 0.5,
                0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0)
    ):
        if origin is None:
            origin = (0, 0.532139961, 0.946026558, 0.5, 0.5,
                      0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0)
        print(
            f"Generating camera coordinates with direction: {direction}, length: {length}, speed: {speed}, origin: {origin}")
        coordinates = generate_camera_coordinates(
            direction, length, speed, origin)
        print(f"Generated {len(coordinates)} camera coordinates.")
        plucker_embedding = process_pose_file(coordinates, width, height)
        print(
            f"Processed camera coordinates into plucker embedding with shape: {plucker_embedding.shape}")
        return plucker_embedding


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return out


class Camera(object):
    """Copied from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    """

    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)


def get_relative_pose(cam_params):
    """Copied from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    """
    abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
    abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
    cam_to_origin = 0
    target_cam_c2w = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, -cam_to_origin],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    abs2rel = target_cam_c2w @ abs_w2cs[0]
    ret_poses = [target_cam_c2w, ] + \
        [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
    ret_poses = np.array(ret_poses, dtype=np.float32)
    return ret_poses


def custom_meshgrid(*args):
    # torch>=2.0.0 only
    return torch.meshgrid(*args, indexing='ij')


def ray_condition(K, c2w, H, W, device):
    """Copied from https://github.com/hehao13/CameraCtrl/blob/main/inference.py

    All geometry is computed in float32 with autocast disabled, whatever the
    input dtype: under bf16 autocast the ``directions @ R`` matmul silently
    drops to bf16, and a bf16 pixel grid is only exact up to integer 256
    (8-bit mantissa) -- at 384x672 that misplaces ~292 column coordinates by
    up to ~3px.  The field is a pure function of the camera parameters (no
    gradient), so the fp32 result is simply cast back to the input dtype.
    """
    # c2w: B, V, 4, 4
    # K: B, V, 4
    out_dtype = c2w.dtype
    K = K.float()
    c2w = c2w.float()
    # Disable autocast for THIS tensor's device type (npu/cuda/cpu) -- a
    # hardcoded backend name would not disable a different active context.
    device_type = torch.device(device).type

    B = K.shape[0]

    with torch.autocast(device_type=device_type, enabled=False):
        j, i = custom_meshgrid(
            torch.linspace(0, H - 1, H, device=device, dtype=torch.float32),
            torch.linspace(0, W - 1, W, device=device, dtype=torch.float32),
        )
        i = i.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]
        j = j.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]

        fx, fy, cx, cy = K.chunk(4, dim=-1)  # B,V, 1

        zs = torch.ones_like(i)  # [B, HxW]
        xs = (i - cx) / fx * zs
        ys = (j - cy) / fy * zs
        zs = zs.expand_as(ys)

        directions = torch.stack((xs, ys, zs), dim=-1)  # B, V, HW, 3
        directions = directions / \
            directions.norm(dim=-1, keepdim=True)  # B, V, HW, 3

        rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)  # B, V, 3, HW
        rays_o = c2w[..., :3, 3]  # B, V, 3
        rays_o = rays_o[:, :, None].expand_as(rays_d)  # B, V, 3, HW
        # c2w @ dirctions
        rays_dxo = torch.linalg.cross(rays_o, rays_d)
        plucker = torch.cat([rays_dxo, rays_d], dim=-1)
        plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)  # B, V, H, W, 6
    # plucker = plucker.permute(0, 1, 4, 2, 3)
    return plucker.to(out_dtype)


def process_pose_file(cam_params, width=672, height=384, original_pose_width=1280, original_pose_height=720, device='cpu', return_poses=False):
    if return_poses:
        return cam_params
    else:
        cam_params = [Camera(cam_param) for cam_param in cam_params]

        sample_wh_ratio = width / height
        # Assuming placeholder ratios, change as needed
        pose_wh_ratio = original_pose_width / original_pose_height

        if pose_wh_ratio > sample_wh_ratio:
            resized_ori_w = height * pose_wh_ratio
            for cam_param in cam_params:
                cam_param.fx = resized_ori_w * cam_param.fx / width
        else:
            resized_ori_h = width / pose_wh_ratio
            for cam_param in cam_params:
                cam_param.fy = resized_ori_h * cam_param.fy / height

        intrinsic = np.asarray([[cam_param.fx * width,
                                cam_param.fy * height,
                                cam_param.cx * width,
                                cam_param.cy * height]
                                for cam_param in cam_params], dtype=np.float32)

        K = torch.as_tensor(intrinsic)[None]  # [1, 1, 4]
        # Assuming this function is defined elsewhere
        c2ws = get_relative_pose(cam_params)
        c2ws = torch.as_tensor(c2ws)[None]  # [1, n_frame, 4, 4]
        plucker_embedding = ray_condition(K, c2ws, height, width, device=device)[
            0].permute(0, 3, 1, 2).contiguous()  # V, 6, H, W
        plucker_embedding = plucker_embedding[None]
        plucker_embedding = rearrange(
            plucker_embedding, "b f c h w -> b f h w c")[0]
        return plucker_embedding


def generate_camera_coordinates(
    direction: Literal["Left", "Right", "Up", "Down", "LeftUp", "LeftDown", "RightUp", "RightDown"],
    length: int,
    speed: float = 1/54,
    origin=(0, 0.532139961, 0.946026558, 0.5, 0.5,
            0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0)
):
    coordinates = [list(origin)]
    while len(coordinates) < length:
        coor = coordinates[-1].copy()
        if "Left" in direction:
            coor[9] += speed
        if "Right" in direction:
            coor[9] -= speed
        if "Up" in direction:
            coor[13] += speed
        if "Down" in direction:
            coor[13] -= speed
        coordinates.append(coor)
    return coordinates
