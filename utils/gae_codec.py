"""Frozen GAE-64 codec wrapper (DA3-GIANT encoder + GAECodec) for rgbd_video_gen.

Encode: images [B,V,3,H,W] in [0,1] -> posterior-mean latents [B,V,64,h,w]
Decode: latents -> RGB pixels / metric depth + rays via the frozen DPT head.
Device: npu -> cuda -> cpu, first available wins.
"""
import os
import sys

import numpy as np
import torch

_DEFAULT_GAE_ROOT = "/mnt/sfs_turbo/t00845730/GAE-GeometricAutoEncoder-main"
_DEFAULT_DA3_SRC = "/mnt/sfs_turbo/t00845730/zhangyiheng/Depth-Anything-3/src"
_DEFAULT_CODEC_CKPT = os.path.join(_DEFAULT_GAE_ROOT, "ckpts/gae_64.pt")
_DEFAULT_CODEC_CFG = os.path.join(_DEFAULT_GAE_ROOT, "configs/gae_64.yaml")


def _pick_device(device=None):
    if device is not None:
        return torch.device(device)
    try:
        import torch_npu  # noqa: F401

        if torch.npu.is_available():
            # 多卡 DDP 下各 rank 由 accelerate set_device；跟随当前卡而不是写死 npu:0
            return torch.device(f"npu:{torch.npu.current_device()}")
    except ImportError:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


class GAECodecWrapper:
    def __init__(self, gae_root=_DEFAULT_GAE_ROOT, codec_cfg=_DEFAULT_CODEC_CFG,
                 codec_ckpt=_DEFAULT_CODEC_CKPT, device=None):
        # 项目 utils 是命名空间包且先于 gae 被导入，会占住 sys.modules，使 GAE
        # src/utils（常规包）里的 config_utils / dpt_helpers 不可达。把该目录并入
        # 命名空间包的 __path__，两边子模块即可共存（同名文件项目优先）。
        import utils as _utils_ns
        _gae_utils_dir = os.path.join(gae_root, "src", "utils")
        if os.path.isdir(_gae_utils_dir) and _gae_utils_dir not in _utils_ns.__path__:
            _utils_ns.__path__.append(_gae_utils_dir)
        if gae_root not in sys.path:
            sys.path.insert(0, gae_root)
        # GAE 的 DA3 backbone 依赖 depth_anything_3（pip 未装，用本地源码）
        if os.path.isdir(_DEFAULT_DA3_SRC) and _DEFAULT_DA3_SRC not in sys.path:
            sys.path.append(_DEFAULT_DA3_SRC)
        from gae import GAE

        self.device = _pick_device(device)
        # 懒加载前清一下碎片缓存，降低大 backbone .to(device) 时的 OOM 概率
        try:
            if self.device.type == "npu":
                torch.npu.empty_cache()
            elif self.device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass
        cwd = os.getcwd()
        try:
            os.chdir(gae_root)  # codec config uses relative stats_dir
            self.model = GAE.from_configs(
                codec_cfg=codec_cfg, codec_ckpt=codec_ckpt, flow_cfg=None,
                device=self.device,
            )
        finally:
            os.chdir(cwd)
        self.latent_channels = self.model.codec.latent_dim

    def encode(self, images):
        """images: [B,V,3,H,W] float in [0,1], numpy or tensor -> latents [B,V,C,h,w] (torch, fp32)."""
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        images = images.to(device=self.device, dtype=torch.float32)
        with torch.inference_mode():
            return self.model.encode(images)

    def decode_rgb(self, latents, num_views=None):
        """latents: [B*V,C,h,w] (or [B,V,C,h,w]) -> RGB [B*V,3,H,W] in [0,1]."""
        if latents.ndim == 5:
            latents = latents.flatten(0, 1)
        v = num_views or 1
        with torch.inference_mode():
            return self.model.decode_rgb(latents.to(self.device), num_views=v)

    def decode_rgb_with_grad(self, latents, num_views=1):
        """Raw latents [N,C,h,w] -> RGB [N,3,H,W], DIFFERENTIABLE w.r.t. latents.

        Training-path twin of decode_rgb: skips the inference_mode-decorated
        GAE.decode_rgb so the RGB-aux loss can backprop into the DiT's latent
        prediction. Codec weights stay frozen (load_codec sets
        requires_grad_(False)); only the input carries a graph. The ImageNet
        denormalization mirrors GAE._denormalize_rgb.
        """
        rgb = self.model.codec.decode_rgb(latents.to(self.device), num_views=num_views)
        mean = getattr(self.model.backbone, "encoder_mean", None)
        std = getattr(self.model.backbone, "encoder_std", None)
        if mean is not None and std is not None:
            mean = mean.to(device=rgb.device, dtype=rgb.dtype).view(1, -1, 1, 1)
            std = std.to(device=rgb.device, dtype=rgb.dtype).view(1, -1, 1, 1)
            rgb = (rgb * std + mean).clamp(0.0, 1.0)
        return rgb

    def decode_depth(self, latents, height, width):
        """latents: [B*V,C,h,w] -> dict(depth/ray/...) at (height,width); depth is metric meters."""
        if latents.ndim == 5:
            latents = latents.flatten(0, 1)
        with torch.inference_mode():
            out = self.model.decode_geometry(latents.to(self.device), height, width)
        return {k: v for k, v in out.items()}

    def standardize(self, z, mean, std):
        return (z - mean.to(z)) / (std.to(z) + 1e-6)

    def destandardize(self, z, mean, std):
        return z * (std.to(z) + 1e-6) + mean.to(z)


def load_latent_stats(path):
    """-> (mean, std) float32 tensors of shape [C]."""
    stats = torch.load(path, map_location="cpu", weights_only=False)
    mean = torch.as_tensor(stats.get("mean", stats.get("latent_mean"))).float()
    std = torch.as_tensor(stats.get("std", stats.get("latent_std"))).float()
    return mean, std
