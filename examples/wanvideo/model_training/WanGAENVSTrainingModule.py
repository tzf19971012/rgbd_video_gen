"""WanGAENVSTrainingModule — GAE-latent NVS on the Wan DiT (Phase 2).

Replaces the generated state of the RGB-D NVS system with the geometry-native
GAE latent: the DiT generates standardized GAE-64 latents (one per view, no
temporal compression), conditioned on clean source-view latents as evidence
tokens pinned at timestep 0, metric Plucker ray fields, and a null text embed.

Differences vs WanRGBDNVSTrainingModule:
  * latents come from the offline cache (utils/cache_gae_latents.py), not the
    Wan VAE — depth is no longer a prediction target, it is a DECODE product
    (codec feature branch -> frozen DA3 DPT head) used at eval only;
  * patch_embedding is a fresh Conv3d(64, ...) and dit.head a fresh Head with
    out_dim=64 — the 16-channel Wan projections cannot be reused;
  * single output head regressing the rectified-flow velocity of the latent;
  * per-frame timestep with the source frames pinned to 0 (GAE's "evidence,
    not state" conditioning, in the state layout the Wan DiT natively supports).
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

from diffsynth.models.wan_video_dit import Head
from utils.projection import intrins_resize

from .WanRGBDNVSTrainingModule import (
    WanRGBDNVSTrainingModule,
    cape_pose_matrices,
)
from utils.gae_codec import GAECodecWrapper, load_latent_stats


class WanGAENVSTrainingModule(WanRGBDNVSTrainingModule):
    def __init__(
        self,
        *args,
        gae_cache_root="/mnt/sfs_turbo/t00845730/gae_latent_cache",
        gae_dataset_name="bmvs",
        gae_stats_path=None,
        gae_latent_channels=64,
        gae_rope_frame_axis="view",
        gae_auto_flow_shift=True,
        gae_codec_device=None,
        gae_rgb_loss_weight=0.0,
        gae_rgb_t_max=0.4,
        gae_rgb_max_views=1,
        **kwargs,
    ):
        # Force the configuration this module is written for before the parent
        # validates it: no depth branch, no warp, channel layout, camera
        # conditioning summed into a 64-channel latent.
        kwargs["token_layout"] = "channel"
        kwargs["warp_mode"] = "none"
        kwargs["depth_flow_matching"] = False
        kwargs["rgb_flow_matching"] = True
        kwargs["view_condition_out_channels"] = gae_latent_channels
        kwargs["clean_source_depth"] = False
        kwargs["per_frame_timestep"] = False
        super().__init__(*args, **kwargs)

        if gae_rope_frame_axis not in ("view", "constant"):
            raise ValueError(
                f"gae_rope_frame_axis must be 'view' or 'constant', "
                f"got {gae_rope_frame_axis!r}"
            )
        self.gae_cache_root = gae_cache_root
        self.gae_dataset_name = gae_dataset_name
        self.gae_latent_channels = int(gae_latent_channels)
        self.gae_rope_frame_axis = gae_rope_frame_axis
        self._gae_holder = {}      # lazy codec (plain dict: never a submodule)
        self._gae_codec_device = gae_codec_device
        self._latent_file_cache = {}
        self._gae_true_grid = None  # (h, w) of the cached latents before pad-to-even

        # GAE-style RGB auxiliary loss (GAE src/stage2/transport/flow.py
        # compute_rgb_aux_loss, adapted to velocity prediction): decode the
        # x-prediction of the target views through the frozen codec RGBHead and
        # regress it to the GT pixels. Without a decode-side objective the
        # latent MSE alone lets the target slots collapse to a gray conditional
        # mean -- PSNR keeps improving while LPIPS and every generated pixel
        # degrade (observed on ovf_gae_bmvs_lora steps 200-800).
        self.gae_rgb_loss_weight = float(gae_rgb_loss_weight)
        self.gae_rgb_t_max = float(gae_rgb_t_max)
        self.gae_rgb_max_views = int(gae_rgb_max_views)
        if self.gae_rgb_loss_weight > 0:
            self.accelerator.print(
                f"[gae] rgb aux loss ENABLED: weight={self.gae_rgb_loss_weight} "
                f"t_max={self.gae_rgb_t_max} max_views={self.gae_rgb_max_views} "
                "(x_pred -> frozen RGBHead -> L1 on target views)"
            )

        # Per-channel standardization stats of the cache (GAE Eq. 15). Plain
        # attributes, moved to the pipe device on first use.
        if gae_stats_path is None:
            gae_stats_path = os.path.join(
                gae_cache_root, gae_dataset_name, "latent_stats.pt")
        mean, std = load_latent_stats(gae_stats_path)
        self.gae_latent_mean = mean.to(self.pipe.device)
        self.gae_latent_std = std.to(self.pipe.device)

        # The parent kept the 16-channel Wan projections (nothing needed
        # widening under the forced config). Both must be replaced: the latent
        # space has 64 channels in AND out.
        dit = self.pipe.dit
        old_conv = dit.patch_embedding
        dit.patch_embedding = torch.nn.Conv3d(
            self.gae_latent_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        ).to(device=old_conv.weight.device, dtype=old_conv.weight.dtype)
        torch.nn.init.kaiming_normal_(
            dit.patch_embedding.weight, nonlinearity="linear")
        if dit.patch_embedding.bias is not None:
            torch.nn.init.zeros_(dit.patch_embedding.bias)

        dit.head = Head(
            dit.dim, self.gae_latent_channels, dit.head.patch_size, eps=1e-6
        ).to(device=old_conv.weight.device, dtype=old_conv.weight.dtype)
        dit.rgb_head = None
        self.accelerator.print(
            f"[gae] patch_embedding -> Conv3d({self.gae_latent_channels}, "
            f"{old_conv.out_channels}, ...) (fresh), head -> Head(out_dim="
            f"{self.gae_latent_channels}) (fresh), rgb_head removed."
        )

        # Frame-axis RoPE ids for the V per-view latent frames. "view" hands
        # frame v the id v (matching the parent's measured-best c-sweep arm);
        # "constant" removes the label entirely (view identity still arrives
        # through the Plucker field).
        v = self.num_views
        if gae_rope_frame_axis == "view":
            self._rope_frame_ids = torch.arange(v)
        else:
            self._rope_frame_ids = torch.zeros(v, dtype=torch.long)

        # GAE's resolution-aware time shift: sqrt(per-sample latent numel / 4096)
        # over ONE view's latent (C x H x W), matching GAE official
        # (train_flow.py: shift_dim = prod((in_ch, latent_h, latent_w)); the
        # timestep is shared across all views in a clip and must NOT multiply
        # the shift). Override by setting rgb_flow_shift in the config together
        # with gae_auto_flow_shift=False.
        if gae_auto_flow_shift:
            try:
                res_h, res_w = (int(x) for x in kwargs["args"].resolution)
                h, w = res_h // 14, res_w // 14
                numel = h * w * self.gae_latent_channels
                self.rgb_flow_shift = float(max(numel, 4096) / 4096) ** 0.5
                # The parent snapped eval_flow_shift to the CONFIG rgb_flow_shift
                # before this override ran; keep the ODE schedule consistent with
                # the training shift unless the config pinned eval explicitly.
                self.eval_flow_shift = self.rgb_flow_shift
                self.accelerator.print(
                    f"[gae] auto flow shift = {self.rgb_flow_shift:.3f} "
                    f"(V={v}, latent {h}x{w}x{self.gae_latent_channels})"
                )
            except Exception as exc:
                self.accelerator.print(
                    f"[gae] auto flow shift failed ({exc}); keeping "
                    f"rgb_flow_shift={self.rgb_flow_shift}"
                )

    # -- parent overrides ----------------------------------------------------

    def set_training_param(self):
        super().set_training_param()
        # The fresh projections must train in BOTH LoRA and full-ft modes; the
        # parent only unfreezes patch_embedding when its own widening ran.
        for module in (self.pipe.dit.patch_embedding, self.pipe.dit.head):
            for param in module.parameters():
                param.requires_grad = True

    def _needs_extra_input_channels(self):
        return False

    def _expand_timestep(self, timestep, clean_depth_mask=None):
        """[B] -> [B, V] with the source evidence frames pinned to 0."""
        per_frame = timestep.reshape(-1, 1).repeat(1, self.num_views)
        per_frame[:, : self.num_source_views] = 0.0
        return per_frame

    # -- latent cache --------------------------------------------------------

    def _load_cached_latents(self, data):
        """[B, V, C, h, w] standardized latents, dataset view order.

        Raises when a scene or view is missing from the cache: a silent zero
        fill would train the model on garbage for exactly the scenes the cache
        skipped (the holey bmvs ones), which is the worst possible failure
        mode. Run the cache script until it reports 0 missing first.
        """
        device, dtype = self.pipe.device, self.pipe.torch_dtype
        cache_dir = os.path.join(self.gae_cache_root, self.gae_dataset_name)
        out = []
        for b in range(len(data["scene_id"])):
            sid = data["scene_id"][b]
            fp = os.path.join(cache_dir, f"{sid}.safetensor")
            bucket = self._latent_file_cache.get(fp)
            if bucket is None:
                from safetensors.torch import load_file

                if not os.path.exists(fp):
                    raise FileNotFoundError(
                        f"GAE latent cache miss: {fp} (scene not cached?)")
                bucket = load_file(fp)
                if len(self._latent_file_cache) > 64:
                    self._latent_file_cache.clear()
                self._latent_file_cache[fp] = bucket
            view_ids = list(data["source_view_ids"][b]) + list(
                data["target_view_ids"][b])
            missing = [vid for vid in view_ids if f"v{vid}" not in bucket]
            if missing:
                raise KeyError(
                    f"scene {sid}: views {missing[:4]} not in cache "
                    f"({len(bucket)} cached)")
            z = torch.stack([bucket[f"v{vid}"] for vid in view_ids]).to(
                device=device, dtype=torch.float32)
            out.append(z)
        z = torch.stack(out)         # [B, V, C, h, w] (dataset view order)
        # The flow code (_build_flow_input / sample_latents / the trainer's
        # masks) slices VIEWS on dim 2: [B, C, V, h, w].
        z = z.permute(0, 2, 1, 3, 4).contiguous()
        # .to(z.device): the stats were pinned at __init__ time, when the pipe
        # may still sit on CPU -- accelerator.prepare moves the module (but not
        # these plain attributes) afterwards.
        mean = self.gae_latent_mean.to(device=z.device, dtype=torch.float32)
        std = self.gae_latent_std.to(device=z.device, dtype=torch.float32)
        z = (z - mean.reshape(1, -1, 1, 1, 1)) / (
            std.reshape(1, -1, 1, 1, 1) + 1e-6
        )
        z = z.to(dtype)
        # The DiT patchifies with (1, 2, 2): an ODD cached grid (e.g. 378/14=27)
        # loses its last row in unpatchify and pred/gt shapes diverge.  Pad to
        # even, remember the true grid, and crop model outputs back to it.
        h, w = z.shape[-2:]
        self._gae_true_grid = (h, w)
        if h % 2 or w % 2:
            z = F.pad(z, (0, w % 2, 0, h % 2))
        return z

    # -- input assembly ------------------------------------------------------

    def forward_preprocess(self, data):
        batch_size = data["images"].shape[0]
        num_views = data["images"].shape[1]
        height = data["images"].shape[-2]
        width = data["images"].shape[-1]

        z = self._load_cached_latents(data)          # [B, C, V, h, w], standardized, EVEN grid
        # The ray field must sit on the TRUE cached grid (height//14); z may be
        # padded by one row/col for the DiT's (1,2,2) patchify.
        latent_height, latent_width = height // 14, width // 14

        context = self._null_text_embed.to(
            dtype=self.pipe.torch_dtype, device=self.pipe.device
        ).expand(batch_size, -1, -1)

        K = data["K"].to(self.pipe.device).float()
        # With disparity_norm_mode != "shared_scale" the dataset leaves c2w
        # untouched, so these translations are METRIC — exactly what the ray
        # field needs for scale-aware conditioning.
        c2w = data["c2w"].to(self.pipe.device).float()
        view_roles = data["view_roles"].to(self.pipe.device)

        c2w_rel = self._relative_c2w(c2w)

        latent_K = intrins_resize(
            K.clone(), ratio_H=latent_height / height, ratio_W=latent_width / width
        )
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
        ).to(dtype=z.dtype)                          # [B, C, V, h, w] ("add")
        if view_condition.shape[-2:] != z.shape[-2:]:
            view_condition = F.pad(
                view_condition, (0, z.shape[-1] - view_condition.shape[-1],
                                 0, z.shape[-2] - view_condition.shape[-2])
            )

        conditioning = z + view_condition            # source evidence rides in
        inputs = {
            # DiT forward kwargs (everything else is swallowed by **kwargs)
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
            # GAE latents
            "rgb_latents": conditioning,
            "view_condition": view_condition,
            "gae_latents": z,        # clean, standardized, unconditioned
            # Targets / metadata
            "rgb_gt": z,
            "loss_masks": data["loss_masks"].to(self.pipe.device),
            "view_roles": view_roles,
            "images": data["images"].to(
                device=self.pipe.device, dtype=torch.float32
            ),  # GT pixels for the rgb aux loss (swallowed by model_fn **kwargs)
            "K": latent_K,
            "c2w": c2w_rel,
            "cape_pose": (
                cape_pose_matrices(c2w_rel, self.rope_pose_translation_scale)
                if self.rope_frame_axis == "pose"
                else None
            ),
            "roce_camera_latents": None,
            "sample_id": data.get("sample_id", [""] * batch_size),
            "prompt": [""] * batch_size,
        }
        return inputs

    def _relative_c2w(self, c2w):
        from .WanRGBDNVSTrainingModule import relative_c2w

        return relative_c2w(c2w, self.camera_geometry_fp32)

    # -- flow matching (single latent modality) -------------------------------

    def _forward_flow(self, inputs):
        rgb_gt = inputs["rgb_gt"]                    # [B, C, V, h, w]
        batch_size = rgb_gt.shape[0]
        sigma = self._sample_sigma(batch_size, rgb_gt.device, rgb_gt.dtype)

        noise = torch.randn_like(rgb_gt)
        x_t = (1.0 - sigma) * rgb_gt + sigma * noise
        velocity_target = noise - rgb_gt

        timestep = (
            sigma.reshape(batch_size) * self.pipe.scheduler.num_train_timesteps
        ).to(dtype=self.pipe.torch_dtype)
        timestep = self._expand_timestep(timestep)   # [B, V], source frames at 0

        pred = self._forward_model(
            inputs, self._build_flow_input(inputs, x_t), timestep
        )
        # The padded patchify row/col carry no target: crop pred AND target
        # back to the true cached grid so the loss compares like with like.
        h0, w0 = self._gae_true_grid
        pred = pred[:, :, :, :h0, :w0]
        velocity_target = velocity_target[:, :, :, :h0, :w0]
        rgb_aux = None
        if self.gae_rgb_loss_weight > 0 and self.training:
            rgb_aux = self._rgb_aux_loss(inputs, x_t, pred, sigma)
        return {
            "pred": pred,
            "rgb_gt": velocity_target,
            "loss_masks": inputs["loss_masks"],
            "weight": 1.0,
            "sigma": sigma.reshape(batch_size),
            "rgb_aux": rgb_aux,
        }

    def _rgb_aux_loss(self, inputs, x_t, v_pred, sigma):
        """L1 decode loss on the target-view x-prediction (GAE compute_rgb_aux_loss).

        velocity -> x conversion: x_t = z + sigma*(noise - z), so
        x_pred = x_t - sigma * v_pred. Only batches with sigma < gae_rgb_t_max
        contribute (high-noise x_pred is garbage; its decode gradient is noise),
        and at most gae_rgb_max_views target views per batch element are decoded
        to cap the RGBHead cost. Gradient flows: L1 -> RGBHead (frozen) ->
        x_pred -> DiT. Returns None on gated-off steps so the trainer can skip
        the term entirely.
        """
        h0, w0 = self._gae_true_grid
        x_t = x_t[:, :, :, :h0, :w0]
        x_pred = x_t - sigma * v_pred.to(x_t.dtype)
        s = self.num_source_views
        x_pred = x_pred[:, :, s:]                    # target views only
        low = sigma.reshape(-1) < self.gae_rgb_t_max
        if not bool(low.any()):
            return None
        xp = x_pred[low]                             # [Bk, C, V_tgt, h, w]
        n_tgt = xp.shape[2]
        k = min(max(1, self.gae_rgb_max_views), n_tgt)
        if k < n_tgt:
            idx = torch.randperm(n_tgt, device=xp.device)[:k]
            xp = xp[:, :, idx]
        mean = self.gae_latent_mean.to(
            device=xp.device, dtype=xp.dtype).reshape(1, -1, 1, 1)
        std = self.gae_latent_std.to(
            device=xp.device, dtype=xp.dtype).reshape(1, -1, 1, 1)
        B, C, K = xp.shape[:3]
        z_raw = (
            xp.permute(0, 2, 1, 3, 4).reshape(B * K, C, *xp.shape[-2:]) * std + mean
        ).float()                                    # destandardize, [Bk*K, C, h, w]
        rgb_pred = self._get_gae().decode_rgb_with_grad(z_raw, num_views=1)
        rgb_pred = rgb_pred.reshape(B, K, *rgb_pred.shape[1:]).float()
        tgt = inputs["images"][low][:, s:].to(torch.float32)   # [Bk, V_tgt, 3, H, W]
        if k < n_tgt:
            tgt = tgt[:, idx]
        return F.l1_loss(rgb_pred, tgt)

    def forward(self, data, inputs=None, args=None):
        inputs = self.forward_preprocess(data)
        result = self._forward_flow(inputs)
        result.setdefault("loss_masks", inputs["loss_masks"])
        result.setdefault("rgb_gt", inputs["rgb_gt"])
        return result

    # -- sampling -------------------------------------------------------------

    @torch.no_grad()
    def sample_latents(self, inputs, num_steps=None, generator=None):
        """Euler-integrate the target-view slots; source slots stay clean."""
        num_steps = (
            self.eval_num_inference_steps if num_steps is None else max(1, int(num_steps))
        )
        z = inputs["gae_latents"]                    # [B, C, V, h, w]
        s = self.num_source_views
        sigmas = self.inference_sigmas(num_steps, device=z.device).to(z.dtype)

        x = torch.randn(z.shape, device=z.device, dtype=z.dtype, generator=generator)
        x[:, :, :s] = z[:, :, :s]
        for i in range(num_steps):
            timestep = (
                sigmas[i] * self.pipe.scheduler.num_train_timesteps
            ).reshape(1).to(dtype=self.pipe.torch_dtype, device=z.device)
            timestep = self._expand_timestep(timestep)
            velocity = self._forward_model(
                inputs, self._build_flow_input(inputs, x), timestep
            )
            dt = sigmas[i] - sigmas[i + 1]
            x[:, :, s:] = x[:, :, s:] - dt * velocity[:, :, s:].to(x.dtype)
            x[:, :, :s] = z[:, :, :s]                # re-pin the evidence
        h0, w0 = self._gae_true_grid
        return x[:, :, :, :h0, :w0]

    # -- eval-side codec ------------------------------------------------------

    def _get_gae(self):
        if "wrapper" not in self._gae_holder:
            # 未显式配置 gae_codec_device 时跟随本 rank 的训练卡，避免全部 rank
            # 的 codec 都落到物理 npu:0 上（多卡下会把卡 0 打爆，OOM 于懒加载）。
            _dev = self._gae_codec_device or str(self.accelerator.device)
            self._gae_holder["wrapper"] = GAECodecWrapper(device=torch.device(_dev))
            self.accelerator.print(
                f"[gae] codec loaded on {self._gae_holder['wrapper'].device} "
                "for decode (eval + rgb aux loss)"
            )
        return self._gae_holder["wrapper"]

    def _decode_views(self, z_standardized, height, width):
        """[B, C, V, h, w] standardized -> rgb [B,V,3,H,W], depth [B,V,H,W]."""
        wrapper = self._get_gae()
        mean = self.gae_latent_mean.to(z_standardized.device).reshape(1, -1, 1, 1)
        std = self.gae_latent_std.to(z_standardized.device).reshape(1, -1, 1, 1)
        B, C, V = z_standardized.shape[:3]
        rgbs, depths = [], []
        for b in range(B):
            for v in range(V):
                flat = z_standardized[b, :, v].unsqueeze(0)
                lat = flat * (std + 1e-6) + mean    # destandardize
                rgbs.append(wrapper.decode_rgb(lat.float(), num_views=1)[0].cpu())
                geo = wrapper.decode_depth(lat.float(), height, width)
                d = geo["depth"]
                while d.ndim > 2:
                    d = d[0]
                depths.append(d.float().cpu())
        rgb = torch.stack(rgbs).reshape(B, V, 3, height, width)
        depth = torch.stack(depths).reshape(B, V, height, width)
        return rgb, depth

    # -- evaluation ------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, batch, output_dir, global_step, max_samples=2, save_vis=True,
                 vis_offset=0):
        """Sample target latents, decode through the codec, score RGB + metric depth."""
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
        pred_z = self.sample_latents(
            inputs, num_steps=self.eval_num_inference_steps, generator=gen
        )

        height = batch_small["images"].shape[-2]
        width = batch_small["images"].shape[-1]
        pred_rgb, pred_depth = self._decode_views(pred_z, height, width)

        gt_rgb = batch_small["images"].to(device).float().clamp(0, 1)
        gt_depth = batch_small["depth_metric"][:, :, 0].to(device).float()
        valid = (gt_depth > 1e-3).float()
        tgt = slice(self.num_source_views, self.num_views)

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
