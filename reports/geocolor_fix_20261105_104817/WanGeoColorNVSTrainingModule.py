"""WanGeoColorNVSTrainingModule — geometry/color split of the NVS latent.

GAE-inspired hypothesis: a scene's GEOMETRY should live in the DA3 (GAE-64)
latent while its COLOR should live in the Wan VAE latent.  The DiT therefore
consumes TWO noisy streams for the target views — the geometry latent and the
color latent of the same (noisy) cameras — and regresses TWO velocities through
two output heads:

    input   = concat(x_t_geo [64], x_t_color [16])  -> Conv3d(80, dim)
    outputs = dit.head -> v_geo (64)    dit.rgb_head -> v_color (16)

Both streams share one sigma (one timestep per clip, source frames pinned to 0
via the per-frame AdaLN) and get independent flow-matching targets; the FM loss
is the masked MSE over the concatenated velocities (both streams are
standardized to unit variance, so the sum weights them equally).

Spatial grid: the DiT runs on the Wan latent's native grid (H//8 x W//8, even)
because bicubic-round-tripping the color latent down to the GAE grid caps the
Wan-VAE decode at ~25 dB PSNR vs ~42 dB at native resolution (16 re10k frames,
probe 2026-11-01).  The GAE latent is upsampled 27x48 -> 48x84 on the way in
and sampled back down before the GAE decode at eval.

Color latents come from the offline Wan cache (utils/cache_wan_latents.py);
RGB is decoded through the frozen Wan VAE at eval only.
"""
import os

import torch
import torch.nn.functional as F

from diffsynth.models.view_condition import ViewConditionEncoder
from diffsynth.models.wan_video_dit import Head
from utils.gae_codec import load_latent_stats
from utils.projection import intrins_resize

from .WanGAENVSTrainingModule import WanGAENVSTrainingModule, cape_pose_matrices


def _even_ceil(x):
    return -(-x // 8) + (-(-x // 8) % 2)


class WanGeoColorNVSTrainingModule(WanGAENVSTrainingModule):
    def __init__(
        self,
        *args,
        wan_latent_cache_root="/mnt/sfs_turbo/t00845730/wan_latent_cache",
        wan_latent_dataset_name="re10k_ovf8_wan",
        wan_latent_stats_path=None,
        color_loss_weight=1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # The parent built the plumbing for a SINGLE 64ch stream (patch_embedding
        # Conv3d(64), head out_dim=64, rgb_head removed).  Widen the input to
        # 64+16 and add the color head back.  The GEOMETRY slice (channels
        # 0:geo_c) trains from scratch exactly as in the parent, but the COLOR
        # path is seeded from the pretrained Wan2.1 projections the parent
        # snapshotted into _pretrained_wan_projections: the color latent IS the
        # space the pretrained DiT was trained on, so patch-embedding channels
        # [geo_c:geo_c+16] copy the pretrained 16ch conv block (same
        # trunk-preserving trick as the parent's _expand_patch_embedding) and
        # rgb_head IS the pretrained velocity head verbatim.
        dit = self.pipe.dit
        old_conv = dit.patch_embedding
        old_head = dit.head
        pre = getattr(self, "_pretrained_wan_projections", None)
        if pre is None:
            raise RuntimeError(
                "geo-color color-stream seeding requires the pretrained Wan "
                "projections snapshotted by WanGAENVSTrainingModule into "
                "_pretrained_wan_projections"
            )
        geo_c = self.gae_latent_channels
        color_c = 16
        self.color_channels = color_c
        dit.patch_embedding = torch.nn.Conv3d(
            geo_c + color_c,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        ).to(device=old_conv.weight.device, dtype=old_conv.weight.dtype)
        torch.nn.init.kaiming_normal_(
            dit.patch_embedding.weight, nonlinearity="linear")
        with torch.no_grad():
            pre_w = pre["patch_embedding"]["weight"]
            if tuple(pre_w.shape) != (
                old_conv.out_channels, color_c, *old_conv.kernel_size
            ):
                raise RuntimeError(
                    f"pretrained patch_embedding weight {tuple(pre_w.shape)} "
                    f"does not match the color slice "
                    f"({old_conv.out_channels}, {color_c}, {old_conv.kernel_size})"
                )
            dit.patch_embedding.weight[:, geo_c:].copy_(pre_w)
            pre_b = pre["patch_embedding"].get("bias")
            if dit.patch_embedding.bias is not None:
                if pre_b is not None:
                    dit.patch_embedding.bias.copy_(pre_b)
                else:
                    torch.nn.init.zeros_(dit.patch_embedding.bias)
        dit.head = Head(
            dit.dim, geo_c, old_head.patch_size, eps=1e-6
        ).to(device=old_conv.weight.device, dtype=old_conv.weight.dtype)
        dit.rgb_head = Head(
            dit.dim, color_c, old_head.patch_size, eps=1e-6
        ).to(device=old_conv.weight.device, dtype=old_conv.weight.dtype)
        dit.rgb_head.load_state_dict(pre["head"])
        self.accelerator.print(
            f"[geo-color] patch_embedding -> Conv3d({geo_c + color_c}, "
            f"{old_conv.out_channels}, ...) (geo slice kaiming-fresh, color "
            f"slice [{geo_c}:{geo_c + color_c}] seeded from pretrained Wan "
            f"conv), head -> Head(out_dim={geo_c}) (fresh), rgb_head -> "
            f"Head(out_dim={color_c}) (SEEDED from pretrained Wan velocity head)"
        )

        # Second camera conditioner for the color stream: same plucker field
        # geometry, but 16 output channels so it can be added to the Wan latent
        # exactly like the 64ch encoder is added to the GAE latent.  Built in
        # the same default dtype as the parent's encoder (fp32 -- the plucker
        # input is fp32 and eval runs outside autocast); accelerator.prepare
        # moves it to the rank device.
        #
        # Per-stream plucker resolution: the Wan color grid (48x84 for 378x672)
        # has no integer pixel_scale relationship to the pixel grid (47.25
        # rows -> padded 48), so plucker_resolution="pixel" can only ever work
        # for the geo stream.  Allow an override (config keys
        # view_condition_plucker_resolution_rgb / view_condition_pixel_scale_rgb,
        # both defaulting to the geo stream's values) so pixel-mode runs can
        # keep the color conditioner at native latent resolution.
        src = self.view_condition_encoder
        _args_cfg = kwargs.get("args")
        if _args_cfg is not None:
            rgb_resolution = _args_cfg.get(
                "view_condition_plucker_resolution_rgb", src.plucker_resolution)
            rgb_pixel_scale = _args_cfg.get(
                "view_condition_pixel_scale_rgb", src.pixel_scale)
        else:
            rgb_resolution = src.plucker_resolution
            rgb_pixel_scale = src.pixel_scale
        self.view_condition_encoder_rgb = ViewConditionEncoder(
            out_channels=color_c,
            role_dim=src.role_embedding.embedding_dim,
            hidden_channels=src.backbone[0].out_channels,
            plucker_resolution=rgb_resolution,
            pixel_scale=rgb_pixel_scale,
            mode=src.mode,
            use_role=src.use_role,
            normalize_moment=src.normalize_moment,
            geometry_fp32=self.camera_geometry_fp32,
        )
        self.accelerator.print(
            f"[geo-color] second ViewConditionEncoder(out_channels={color_c}, "
            f"plucker_resolution={rgb_resolution}, pixel_scale={rgb_pixel_scale}"
            ") for the Wan-latent (color) stream"
        )

        # --- Wan color latent cache -----------------------------------------
        self.wan_latent_cache_root = wan_latent_cache_root
        self.wan_latent_dataset_name = wan_latent_dataset_name
        if wan_latent_stats_path is None:
            wan_latent_stats_path = os.path.join(
                wan_latent_cache_root, wan_latent_dataset_name, "latent_stats.pt")
        wan_mean, wan_std = load_latent_stats(wan_latent_stats_path)
        self.wan_latent_mean = wan_mean.to(self.pipe.device)
        self.wan_latent_std = wan_std.to(self.pipe.device)
        self._wan_file_cache = {}
        # Flow matching assumes Var(x0) ~ 1; the per-channel standardization
        # delivers that, so the color term enters the joint MSE with weight 1.
        # A configured weight w scales the color channels by sqrt(w) in BOTH the
        # prediction and the target, which weights the squared term by exactly w
        # without touching anything downstream (the trainer's masked MSE).
        self.color_loss_weight = float(color_loss_weight)
        if self.color_loss_weight <= 0:
            raise ValueError(
                "color_loss_weight must be > 0 (the color stream IS the "
                "experiment); use the GAE module instead to disable it")
        e2 = float((wan_mean ** 2 + wan_std ** 2).mean())
        self.accelerator.print(
            f"[geo-color] wan latent cache: {wan_latent_cache_root}/"
            f"{wan_latent_dataset_name}  E[x^2]={e2:.4f}  "
            f"color_loss_weight={self.color_loss_weight:.3f}"
        )

        # Grids: geo true grid = resolution//14 (GAE/DA3); wan native = the
        # grid the VAE conv stack actually produces ((H+4)//8: 378x672 -> 47x84,
        # VERIFIED by probe -- it is NOT ceil(H/8)=48); DiT grid = wan native
        # rounded up to even (patchify (1,2,2) needs even dims).
        # Decode MUST map the DiT grid back to the wan native grid before the
        # VAE: decoding the even (48,84) grid leaves the image vertically
        # stretched by 48/47, which caps the color round-trip at ~22.6 dB vs
        # 34.4 dB (measured 2026-11-01 on the re10k overfit scenes).  Bicubic
        # 47->48 on the way in and 48->47 on the way out are near-inverses.
        res_h, res_w = (int(x) for x in kwargs["args"].resolution)
        self.geo_true_grid = (res_h // 14, res_w // 14)
        self.wan_native_grid = ((res_h + 4) // 8, (res_w + 4) // 8)
        self.dit_latent_hw = (_even_ceil(res_h), _even_ceil(res_w))
        self._geo_true_grid = self.geo_true_grid   # parent contract (loss crop)
        self.accelerator.print(
            f"[geo-color] grids: gae {self.geo_true_grid} -> dit "
            f"{self.dit_latent_hw} (wan native {self.wan_native_grid})"
        )

    # -- color cache ----------------------------------------------------------

    def _load_cached_wan_latents(self, data):
        """[B, 16, V, h, w] standardized color latents on the DiT (even) grid.

        Same fail-loud policy as the GAE cache loader: a missing scene/view is
        an error, never a silent zero fill.
        """
        device, dtype = self.pipe.device, self.pipe.torch_dtype
        cache_dir = os.path.join(
            self.wan_latent_cache_root, self.wan_latent_dataset_name)
        out = []
        for b in range(len(data["scene_id"])):
            sid = data["scene_id"][b]
            fp = os.path.join(cache_dir, f"{sid}.safetensor")
            bucket = self._wan_file_cache.get(fp)
            if bucket is None:
                from safetensors.torch import load_file

                if not os.path.exists(fp):
                    raise FileNotFoundError(
                        f"Wan latent cache miss: {fp} (run utils/cache_wan_latents.py)")
                bucket = load_file(fp)
                if len(self._wan_file_cache) > 64:
                    self._wan_file_cache.clear()
                self._wan_file_cache[fp] = bucket
            view_ids = list(data["source_view_ids"][b]) + list(
                data["target_view_ids"][b])
            missing = [vid for vid in view_ids if f"v{vid}" not in bucket]
            if missing:
                raise KeyError(
                    f"scene {sid}: views {missing[:4]} not in wan cache "
                    f"({len(bucket)} cached)")
            z = torch.stack([bucket[f"v{vid}"] for vid in view_ids]).to(
                device=device, dtype=torch.float32)
            out.append(z)
        z = torch.stack(out)                       # [B, V, 16, h, w] native grid
        z = z.permute(0, 2, 1, 3, 4).contiguous()  # [B, 16, V, h, w]
        mean = self.wan_latent_mean.to(device=z.device, dtype=torch.float32)
        std = self.wan_latent_std.to(device=z.device, dtype=torch.float32)
        z = (z - mean.reshape(1, -1, 1, 1, 1)) / (std.reshape(1, -1, 1, 1, 1) + 1e-6)
        dh, dw = self.dit_latent_hw
        if tuple(z.shape[-2:]) != (dh, dw):
            z = F.interpolate(
                z.flatten(0, 1), size=(dh, dw), mode="bicubic", align_corners=False
            ).reshape(z.shape[0], z.shape[1], z.shape[2], dh, dw)
        return z.to(dtype)

    # -- input assembly -------------------------------------------------------

    def forward_preprocess(self, data):
        batch_size = data["images"].shape[0]
        height = data["images"].shape[-2]
        width = data["images"].shape[-1]

        # Geometry stream: cached GAE latents on their TRUE grid (27x48),
        # standardized per-channel, then bicubic-upsampled onto the DiT grid.
        z_small = self._load_cached_latents(data)    # [B, 64, V, h, w], EVEN grid
        h0, w0 = self._gae_true_grid
        z_small = z_small[:, :, :, :h0, :w0]
        dh, dw = self.dit_latent_hw
        z_geo = F.interpolate(
            z_small.flatten(0, 1), size=(dh, dw), mode="bicubic", align_corners=False
        ).reshape(z_small.shape[0], z_small.shape[1], z_small.shape[2], dh, dw)

        # Color stream: cached Wan latents, standardized, already on the DiT grid.
        z_color = self._load_cached_wan_latents(data)  # [B, 16, V, dh, dw]

        context = self._null_text_embed.to(
            dtype=self.pipe.torch_dtype, device=self.pipe.device
        ).expand(batch_size, -1, -1)

        K = data["K"].to(self.pipe.device).float()
        c2w = data["c2w"].to(self.pipe.device).float()
        view_roles = data["view_roles"].to(self.pipe.device)

        c2w_rel = self._relative_c2w(c2w)

        latent_K = intrins_resize(
            K.clone(), ratio_H=dh / height, ratio_W=dw / width
        )
        plucker_field = self.view_condition_encoder.plucker_field(
            latent_K,
            c2w_rel,
            latent_height=dh,
            latent_width=dw,
            pixel_K=K,
            pixel_height=height,
            pixel_width=width,
        )
        enc_kwargs = dict(
            latent_height=dh,
            latent_width=dw,
            pixel_K=K,
            pixel_height=height,
            pixel_width=width,
            plucker=plucker_field,
        )
        vc_geo = self.view_condition_encoder(
            latent_K, c2w_rel, view_roles, **enc_kwargs
        ).to(dtype=z_geo.dtype)                       # [B, 64, V, dh, dw]
        vc_color = self.view_condition_encoder_rgb(
            latent_K, c2w_rel, view_roles, **enc_kwargs
        ).to(dtype=z_color.dtype)                     # [B, 16, V, dh, dw]

        inputs = {
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
            # streams (clean + camera-conditioned; target slots get re-noised
            # in _build_flow_input, sources stay clean)
            "geo_gt": z_geo,
            "color_gt": z_color,
            "vc_geo": vc_geo,
            "vc_color": vc_color,
            "rgb_latents": z_geo + vc_geo,     # parent's key: the geo stream in
            "color_conditioning": z_color + vc_color,
            # targets / metadata
            "loss_masks": data["loss_masks"].to(self.pipe.device),
            "view_roles": view_roles,
            "images": data["images"].to(
                device=self.pipe.device, dtype=torch.float32
            ),
            "K": latent_K,
            "c2w": c2w_rel,
            "cape_pose": (
                cape_pose_matrices(c2w_rel, self.rope_pose_translation_scale)
                if self.rope_frame_axis == "pose" else None
            ),
            "roce_camera_latents": None,
            "sample_id": data.get("sample_id", [""] * batch_size),
            "prompt": [""] * batch_size,
        }
        return inputs

    # -- flow matching (two latent modalities, one sigma) ----------------------

    def _build_flow_input(self, inputs, noisy_geo, noisy_color=None):
        """[B, 80, V, h, w]: source slots clean+conditioned, target slots noisy.

        The camera conditioning is re-added to the target slots of BOTH streams
        so every view knows where it is being rendered from (mirrors the parent).
        """
        if noisy_color is None:
            raise ValueError("geo/color flow matching requires both noisy streams")
        s = self.num_source_views
        geo = inputs["rgb_latents"].clone()
        geo[:, :, s:] = noisy_geo[:, :, s:].to(geo.dtype) + inputs["vc_geo"][:, :, s:]
        color = inputs["color_conditioning"].clone()
        color[:, :, s:] = (
            noisy_color[:, :, s:].to(color.dtype) + inputs["vc_color"][:, :, s:]
        )
        return torch.cat([geo, color], dim=1)

    def _forward_flow(self, inputs):
        geo_gt = inputs["geo_gt"]            # [B, 64, V, h, w] standardized
        color_gt = inputs["color_gt"]        # [B, 16, V, h, w] standardized
        batch_size = geo_gt.shape[0]
        sigma = self._sample_sigma(batch_size, geo_gt.device, geo_gt.dtype)

        noise_geo = torch.randn_like(geo_gt)
        noise_color = torch.randn_like(color_gt)
        x_t_geo = (1.0 - sigma) * geo_gt + sigma * noise_geo
        x_t_color = (1.0 - sigma) * color_gt + sigma * noise_color
        v_target_geo = noise_geo - geo_gt
        v_target_color = noise_color - color_gt

        w = self.color_loss_weight ** 0.5
        if w != 1.0:
            # see the color_loss_weight comment in __init__
            x_t_color = x_t_color * w
            v_target_color = v_target_color * w

        timestep = (
            sigma.reshape(batch_size) * self.pipe.scheduler.num_train_timesteps
        ).to(dtype=self.pipe.torch_dtype)
        timestep = self._expand_timestep(timestep)   # [B, V], source frames at 0

        pred_geo, pred_color = self._forward_model(
            inputs, self._build_flow_input(inputs, x_t_geo, x_t_color), timestep
        )
        # per-stream MSE (detached, logging only) -- the trainer recomputes the
        # joint loss from the concatenated tensors below, identical math.
        from examples.wanvideo.model_training.train_with_accelerate_rgbd_nvs import _masked_mse, _resize_mask

        mask = _resize_mask(
            inputs["loss_masks"][:, 1], pred_geo.shape[-2:]).to(pred_geo.dtype)
        mask = mask.permute(0, 2, 1, 3, 4)           # [B, 1, V, h, w]
        stream_losses = {
            "geo_mse": _masked_mse(pred_geo, v_target_geo, mask).detach(),
            "color_mse": _masked_mse(pred_color, v_target_color, mask).detach(),
        }
        return {
            "pred": torch.cat([pred_geo, pred_color], dim=1),
            "rgb_gt": torch.cat([v_target_geo, v_target_color], dim=1),
            "loss_masks": inputs["loss_masks"],
            "weight": 1.0,
            "sigma": sigma.reshape(batch_size),
            "rgb_aux": None,
            "stream_losses": stream_losses,
        }

    def forward(self, data, inputs=None, args=None):
        inputs = self.forward_preprocess(data)
        result = self._forward_flow(inputs)
        result.setdefault("loss_masks", inputs["loss_masks"])
        return result

    # -- sampling -------------------------------------------------------------

    @torch.no_grad()
    def sample_latents(self, inputs=None, num_steps=None, generator=None):
        """Euler-integrate BOTH target streams; source slots stay clean."""
        if inputs is None:
            raise ValueError("geo/color sampling needs forward_preprocess inputs")
        num_steps = (
            self.eval_num_inference_steps if num_steps is None else max(1, int(num_steps))
        )
        z_geo = inputs["geo_gt"]
        z_color = inputs["color_gt"]
        s = self.num_source_views
        sigmas = self.inference_sigmas(num_steps, device=z_geo.device).to(z_geo.dtype)

        xg = torch.randn(z_geo.shape, device=z_geo.device, dtype=z_geo.dtype,
                         generator=generator)
        xc = torch.randn(z_color.shape, device=z_color.device, dtype=z_color.dtype,
                         generator=generator)
        xg[:, :, :s] = z_geo[:, :, :s]
        xc[:, :, :s] = z_color[:, :, :s]
        w = self.color_loss_weight ** 0.5
        for i in range(num_steps):
            timestep = (
                sigmas[i] * self.pipe.scheduler.num_train_timesteps
            ).reshape(1).to(dtype=self.pipe.torch_dtype, device=z_geo.device)
            timestep = self._expand_timestep(timestep)
            vg, vc = self._forward_model(
                inputs, self._build_flow_input(inputs, xg, xc), timestep
            )
            dt = sigmas[i] - sigmas[i + 1]
            xg[:, :, s:] = xg[:, :, s:] - dt * vg[:, :, s:].to(xg.dtype)
            xc[:, :, s:] = xc[:, :, s:] - dt * (vc[:, :, s:] / w).to(xc.dtype)
            xg[:, :, :s] = z_geo[:, :, :s]
            xc[:, :, :s] = z_color[:, :, :s]
        return xg, xc

    # -- eval-side decode ------------------------------------------------------

    def _decode_color_views(self, z_standardized, height, width):
        """[B, 16, V, dh, dw] standardized -> rgb [B, V, 3, H, W] via the Wan VAE."""
        mean = self.wan_latent_mean.to(z_standardized.device).reshape(1, -1, 1, 1)
        std = self.wan_latent_std.to(z_standardized.device).reshape(1, -1, 1, 1)
        h_nat, w_nat = self.wan_native_grid
        B, V = z_standardized.shape[0], z_standardized.shape[2]
        rgbs = []
        for b in range(B):
            for v in range(V):
                z = z_standardized[b, :, v].unsqueeze(0).float()  # [1,16,dh,dw]
                if tuple(z.shape[-2:]) != (h_nat, w_nat):
                    z = F.interpolate(z, size=(h_nat, w_nat), mode="bicubic",
                                      align_corners=False)
                lat = (z * (std + 1e-6) + mean).to(torch.bfloat16)  # [1,16,47,84]
                dec = self.pipe.vae.decode(
                    lat.unsqueeze(2), device=str(z_standardized.device))
                img = ((dec.clamp(-1, 1) + 1) / 2)[0, :, 0].float()  # [3,H',W']
                img = img[:, :height, :width]
                if img.shape[1] < height or img.shape[2] < width:
                    img = F.pad(img, (0, max(0, width - img.shape[2]),
                                      0, max(0, height - img.shape[1])),
                                mode="replicate")
                rgbs.append(img.cpu())
        return torch.stack(rgbs).reshape(B, V, 3, height, width)

    def _decode_geo_depth(self, z_standardized, height, width):
        """[B, 64, V, dh, dw] standardized -> depth [B, V, H, W] via the GAE codec."""
        h0, w0 = self._geo_true_grid
        mean = self.gae_latent_mean.to(z_standardized.device).reshape(1, -1, 1, 1)
        std = self.gae_latent_std.to(z_standardized.device).reshape(1, -1, 1, 1)
        wrapper = self._get_gae()
        B, V = z_standardized.shape[0], z_standardized.shape[2]
        depths = []
        for b in range(B):
            for v in range(V):
                z = z_standardized[b, :, v].unsqueeze(0).float()
                if tuple(z.shape[-2:]) != (h0, w0):
                    z = F.interpolate(z, size=(h0, w0), mode="bicubic",
                                      align_corners=False)
                lat = z * (std + 1e-6) + mean             # [1,64,27,48] raw
                geo = wrapper.decode_depth(lat, height, width)
                d = geo["depth"]
                while d.ndim > 2:
                    d = d[0]
                depths.append(d.float().cpu())
        return torch.stack(depths).reshape(B, V, height, width)

    # -- evaluation ------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, batch, output_dir, global_step, max_samples=2, save_vis=True,
                 vis_offset=0):
        """Sample both streams, decode geo->depth / color->RGB, score both."""
        import math

        import numpy as np
        from PIL import Image

        was_training = self.training
        self.eval()
        device = self.pipe.device
        save_dir = os.path.join(output_dir, f"eval_step_{global_step:06d}")
        os.makedirs(save_dir, exist_ok=True)

        def _slice(x, n):
            if isinstance(x, torch.Tensor):
                return x[:n]
            if isinstance(x, list):
                return x[:n]
            return x

        batch_small = {k: _slice(v, max_samples) for k, v in batch.items()}
        self._eval_inputs = True
        try:
            inputs = self.forward_preprocess(batch_small)
        finally:
            self._eval_inputs = False

        gen = torch.Generator(device=device)
        gen.manual_seed(self.eval_sample_seed)
        pred_geo, pred_color = self.sample_latents(inputs=inputs, generator=gen)

        height = batch_small["images"].shape[-2]
        width = batch_small["images"].shape[-1]
        pred_rgb = self._decode_color_views(pred_color, height, width)
        pred_depth = self._decode_geo_depth(pred_geo, height, width)

        gt_rgb = batch_small["images"].to(device).float().clamp(0, 1)
        gt_depth = batch_small["depth_metric"][:, :, 0].to(device).float()
        valid = (gt_depth > 1e-3).float()

        lpips_net = self._get_lpips(device)
        metrics = {
            "tgt_rgb_psnr": [], "tgt_rgb_lpips": [],
            "tgt_depth_abs_rel": [], "tgt_depth_abs_rel_aligned": [],
            "tgt_depth_delta1": [], "src_depth_abs_rel": [],
        }
        B = pred_rgb.shape[0]
        for b in range(B):
            for v in range(self.num_source_views, self.num_views):
                pr, gr = pred_rgb[b, v].to(device), gt_rgb[b, v]
                mse = float(((pr - gr) ** 2).mean())
                metrics["tgt_rgb_psnr"].append(
                    100.0 if mse < 1e-10 else -10 * math.log10(mse))
                if lpips_net is not None:
                    with torch.no_grad():
                        metrics["tgt_rgb_lpips"].append(float(lpips_net(
                            pr[None].float() * 2 - 1, gr[None].float() * 2 - 1)))
                pd, gd = pred_depth[b, v].to(device), gt_depth[b, v]
                m = valid[b, v].bool() & torch.isfinite(gd)
                if m.sum() >= 16:
                    p, g = pd[m].float(), gd[m].float()
                    metrics["tgt_depth_abs_rel"].append(float(
                        ((p - g).abs() / g).mean()))
                    scale = float(g.median() / p.median().clamp_min(1e-6))
                    pa = p * scale
                    metrics["tgt_depth_abs_rel_aligned"].append(float(
                        ((pa - g).abs() / g).mean()))
                    metrics["tgt_depth_delta1"].append(float(
                        (torch.maximum(pa / g, g / pa) < 1.25).float().mean()))
            for v in range(self.num_source_views):
                pd = pred_depth[b, v].to(device)
                gd = gt_depth[b, v]
                m = valid[b, v].bool() & torch.isfinite(gd)
                if m.sum() >= 16:
                    p, g = pd[m].float(), gd[m].float()
                    metrics["src_depth_abs_rel"].append(float(
                        ((p - g).abs() / g).mean()))

        if save_vis:
            for b in range(B):
                rows = []
                for v in range(self.num_views):
                    img = (pred_rgb[b, v].clamp(0, 1).permute(1, 2, 0).numpy() * 255)
                    rows.append(img.astype(np.uint8))
                strip = np.concatenate(rows, axis=1)
                Image.fromarray(strip).save(os.path.join(
                    save_dir, f"b{b + vis_offset:02d}_pred_rgb.jpg"))

        out = {}
        for k, vals in metrics.items():
            vals = [v for v in vals if v == v]
            out[k] = float(np.mean(vals)) if vals else float("nan")
        if was_training:
            self.train()
        return out

    # -- parent overrides ------------------------------------------------------

    def set_training_param(self):
        super().set_training_param()
        # dit.head / patch_embedding are unfrozen by the parent override; the
        # color head and the second conditioner are new here.
        for module in (self.pipe.dit.rgb_head, self.view_condition_encoder_rgb):
            for param in module.parameters():
                param.requires_grad = True
