"""WanGeoColorTokenNVSTrainingModule — token-level geo/color merge (v3).

v1 (WanGeoColorNVSTrainingModule) merged the two modalities in the INPUT
CHANNEL dimension: cat([x_t_geo(64), x_t_color(16)], dim=1) -> Conv3d(80, dim),
which forces one conv to entangle two different latent spaces and forces the
geometry latent onto the color grid (bicubic 27x48 -> 48x84).

v3 merges at the TOKEN level instead.  Each modality keeps its native latent
rate and gets its OWN patch embedding — the projection that aligns both to the
same 1536-d token space:

    tokens_geo   = Conv3d(64, 1536, (1,2,2))  on the GAE grid  28x48  -> 14x24  = 336 tok/view
    tokens_color = Conv3d(16, 1536, (1,2,2))  on the Wan grid  48x84  -> 24x42  = 1008 tok/view
    tokens       = per-frame concat [tokens_geo | tokens_color]          -> 1344 tok/view

The merged frame-major sequence goes through the UNCHANGED pretrained DiT
blocks (full cross-modal self-attention; RoPE built per modality block so each
token keeps its own lattice position), and the two velocity heads read back
only their own modality's token slice:

    dit.head    (v_geo,   64ch) <- tokens_geo    -> unpatchify (f, 14, 24) -> 28x48
    dit.rgb_head(v_color, 16ch) <- tokens_color  -> unpatchify (f, 24, 42) -> 48x84

NOTE: patchifying both streams on the SAME grid and SUMMING tokens would be
mathematically identical to the v1 80-channel conv (a conv is linear in its
input channels, so the concat conv IS the sum of the two slice convs).  The
genuine architectural deltas here are (a) per-modality projections feeding a
shared attention space instead of a shared input conv, and (b) the geometry
stream no longer leaves its native latent grid — no 27x48 -> 48x84 bicubic.

Seeding follows v1: the color conv + rgb_head copy the pretrained Wan2.1
projections verbatim (the color latent IS the Wan training space); the geo
conv + geo head are fresh kaiming.  Sequence cost: 1344 vs 1008 tokens/view
(1.33x sequence, ~1.78x attention).
"""
import torch

from einops import rearrange

from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d

from .WanGeoColorNVSTrainingModule import WanGeoColorNVSTrainingModule


class WanGeoColorTokenNVSTrainingModule(WanGeoColorNVSTrainingModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Replace the v1 80-channel merged conv with TWO per-modality patch
        # embeddings.  The parent's head/rgb_head construction is already
        # exactly what v3 needs (fresh geo Head(64), rgb_head SEEDED from the
        # pretrained Wan velocity head) -- only the input projections change.
        dit = self.pipe.dit
        old_conv = dit.patch_embedding           # the v1 Conv3d(80, dim)
        pre = self._pretrained_wan_projections
        geo_c = self.gae_latent_channels         # 64
        color_c = self.color_channels            # 16
        dim = old_conv.out_channels

        # Geometry projection: fresh kaiming (the geo space is not the Wan
        # space; nothing to inherit).
        dit.patch_embedding = torch.nn.Conv3d(
            geo_c, dim,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        ).to(device=old_conv.weight.device, dtype=old_conv.weight.dtype)
        torch.nn.init.kaiming_normal_(
            dit.patch_embedding.weight, nonlinearity="linear")
        if dit.patch_embedding.bias is not None:
            torch.nn.init.zeros_(dit.patch_embedding.bias)

        # Color projection: the pretrained Wan patch embedding verbatim, so at
        # step 0 the color path projects exactly like pretrained T2V.
        dit.color_patch_embedding = torch.nn.Conv3d(
            color_c, dim,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        ).to(device=old_conv.weight.device, dtype=old_conv.weight.dtype)
        pre_w = pre["patch_embedding"]["weight"]
        if tuple(pre_w.shape) != (dim, color_c, *old_conv.kernel_size):
            raise RuntimeError(
                f"pretrained patch_embedding weight {tuple(pre_w.shape)} does "
                f"not match the color projection "
                f"({dim}, {color_c}, {old_conv.kernel_size})"
            )
        with torch.no_grad():
            dit.color_patch_embedding.weight.copy_(pre_w)
            if dit.color_patch_embedding.bias is not None:
                pre_b = pre["patch_embedding"].get("bias")
                if pre_b is not None:
                    dit.color_patch_embedding.bias.copy_(pre_b)
                else:
                    torch.nn.init.zeros_(dit.color_patch_embedding.bias)

        # Grids: the geometry stream works on its OWN padded native grid
        # (loader output, e.g. 28x48 = (27,48) padded even), the color stream
        # on the Wan/DiT grid (48x84).  Token counts per view after (1,2,2):
        self.geo_work_grid = tuple(self._gae_loader_grid())
        self.color_work_grid = tuple(self.dit_latent_hw)
        gh, gw = self.geo_work_grid
        ch, cw = self.color_work_grid
        self.geo_tokens_per_view = (gh // 2) * (gw // 2)
        self.color_tokens_per_view = (ch // 2) * (cw // 2)
        self.accelerator.print(
            f"[geo-color-token] patch_embedding -> Conv3d({geo_c}, {dim}) "
            f"(fresh) on geo grid {self.geo_work_grid} "
            f"({self.geo_tokens_per_view} tok/view), color_patch_embedding -> "
            f"Conv3d({color_c}, {dim}) (SEEDED from pretrained Wan conv) on "
            f"color grid {self.color_work_grid} "
            f"({self.color_tokens_per_view} tok/view); merged sequence "
            f"{self.geo_tokens_per_view + self.color_tokens_per_view} "
            f"tok/view, per-modality heads, full cross-modal attention."
        )

    def _gae_loader_grid(self):
        """The (h, w) the GAE cache loader emits: true grid padded to even.

        Derived once from the resolution config so forward_preprocess does not
        have to run the loader to know its own shapes.
        """
        h0, w0 = self.geo_true_grid
        return (h0 + h0 % 2, w0 + w0 % 2)

    # -- input assembly -------------------------------------------------------

    def forward_preprocess(self, data):
        """Same contract as v1, but the geo stream stays on its native grid.

        geo  [B, 64, V, 28, 48]  (loader output: true 27x48 padded even)
        color[B, 16, V, 48, 84]  (unchanged)
        Each stream gets its own latent_K / plucker field / ViewConditionEncoder
        pass at its own grid; vc_geo is computed on the TRUE 27x48 grid and
        zero-padded to 28x48 (mirrors the parent GAE module's pad handling).
        """
        batch_size = data["images"].shape[0]
        height = data["images"].shape[-2]
        width = data["images"].shape[-1]

        z_geo = self._load_cached_latents(data)    # [B, 64, V, 28, 48], standardized
        z_color = self._load_cached_wan_latents(data)  # [B, 16, V, dh, dw]

        context = self._null_text_embed.to(
            dtype=self.pipe.torch_dtype, device=self.pipe.device
        ).expand(batch_size, -1, -1)

        K = data["K"].to(self.pipe.device).float()
        c2w = data["c2w"].to(self.pipe.device).float()
        view_roles = data["view_roles"].to(self.pipe.device)
        c2w_rel = self._relative_c2w(c2w)

        h0, w0 = self._gae_true_grid
        gh, gw = self.geo_work_grid
        dh, dw = self.color_work_grid

        # Per-grid intrinsics + ray fields (cheap, deterministic geometry).
        latent_K_geo = self.view_condition_encoder_latent_K(
            K, h0, w0, height, width)
        latent_K_color = self.view_condition_encoder_latent_K(
            K, dh, dw, height, width)
        plucker_geo = self.view_condition_encoder.plucker_field(
            latent_K_geo, c2w_rel,
            latent_height=h0, latent_width=w0,
            pixel_K=K, pixel_height=height, pixel_width=width,
        )
        plucker_color = self.view_condition_encoder.plucker_field(
            latent_K_color, c2w_rel,
            latent_height=dh, latent_width=dw,
            pixel_K=K, pixel_height=height, pixel_width=width,
        )
        enc_geo = dict(
            latent_height=h0, latent_width=w0,
            pixel_K=K, pixel_height=height, pixel_width=width,
            plucker=plucker_geo,
        )
        enc_color = dict(
            latent_height=dh, latent_width=dw,
            pixel_K=K, pixel_height=height, pixel_width=width,
            plucker=plucker_color,
        )
        vc_geo = self.view_condition_encoder(
            latent_K_geo, c2w_rel, view_roles, **enc_geo
        ).to(dtype=z_geo.dtype)                    # [B, 64, V, 27, 48]
        if tuple(vc_geo.shape[-2:]) != (gh, gw):
            vc_geo = F_pad_to(vc_geo, (gh, gw))
        vc_color = self.view_condition_encoder_rgb(
            latent_K_color, c2w_rel, view_roles, **enc_color
        ).to(dtype=z_color.dtype)                  # [B, 16, V, dh, dw]

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
            "geo_gt": z_geo,
            "color_gt": z_color,
            "vc_geo": vc_geo,
            "vc_color": vc_color,
            "rgb_latents": z_geo + vc_geo,     # v1's key: the geo stream in
            "color_conditioning": z_color + vc_color,
            "loss_masks": data["loss_masks"].to(self.pipe.device),
            "view_roles": view_roles,
            "images": data["images"].to(
                device=self.pipe.device, dtype=torch.float32
            ),
            "K": latent_K_color,
            "c2w": c2w_rel,
            "cape_pose": None,
            "roce_camera_latents": None,
            "sample_id": data.get("sample_id", [""] * batch_size),
            "prompt": [""] * batch_size,
        }
        return inputs

    def view_condition_encoder_latent_K(self, K, lh, lw, height, width):
        from utils.projection import intrins_resize

        return intrins_resize(
            K.clone(), ratio_H=lh / height, ratio_W=lw / width)

    # -- flow matching ----------------------------------------------------------

    def _build_flow_input(self, inputs, noisy_geo, noisy_color=None):
        """(geo_in, color_in): source slots clean+conditioned, targets noisy.

        v1 returned the channel-wise concat; v3 keeps the streams separate all
        the way into the per-modality patch embeddings.
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
        return geo, color

    def _forward_model(self, inputs, flow_input, timestep):
        """Per-modality patchify -> merged token sequence -> DiT -> two heads.

        Replicates model_fn_wan_video's exact timestep/RoPE/checkpointing
        plumbing for the subset of features this module uses (per-frame
        timestep, view-axis frame RoPE, no CaPE/RoCE/VACE/tea-cache/seq-parallel
        -- every one of those paths would be a silent no-op here anyway).
        """
        geo_in, color_in = flow_input
        dit = self.pipe.dit

        # Timestep embedding, identical to model_fn_wan_video for 2-D timestep.
        t = dit.time_embedding(
            sinusoidal_embedding_1d(dit.freq_dim, timestep.reshape(-1)))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
        t = t.unflatten(0, timestep.shape)            # (B, V, dim)
        t_mod = t_mod.unflatten(0, timestep.shape)    # (B, V, 6, dim)
        context = dit.text_embedding(inputs["context"])

        b, _, f = geo_in.shape[:3]
        gh2, gw2 = self.geo_work_grid[0] // 2, self.geo_work_grid[1] // 2
        ch2, cw2 = self.color_work_grid[0] // 2, self.color_work_grid[1] // 2

        # Separate patchify: each modality projects to the shared 1536-d token
        # space on its own lattice.
        xg = dit.patch_embedding(geo_in)              # [B, dim, V, gh2, gw2]
        xc = dit.color_patch_embedding(color_in)      # [B, dim, V, ch2, cw2]
        xg = rearrange(xg, "b c f h w -> b f (h w) c")
        xc = rearrange(xc, "b c f h w -> b f (h w) c")
        n_g = xg.shape[2]
        n_c = xc.shape[2]
        x = torch.cat([xg, xc], dim=2)                # per-frame [geo | color]
        x = x.reshape(b, f * x.shape[2], x.shape[-1])

        # RoPE per modality block: the frame axis uses the shared view ids,
        # the h/w axes use each block's own lattice sizes.  Both blocks are
        # built frame-major, then interleaved PER FRAME to match the token
        # layout [frame v: geo(n_g) | color(n_c)] -- concatenating the flat
        # blocks instead would hand every token the positions of the wrong
        # frame, a pure silent failure.
        ids = self._rope_frame_ids.to(dit.freqs[0].device)
        frame_freqs = dit.freqs[0][ids]               # [V, d_frame]
        freqs_g = _axial_freqs(dit, frame_freqs, f, gh2, gw2)  # [V*n_g, 1, d]
        freqs_c = _axial_freqs(dit, frame_freqs, f, ch2, cw2)  # [V*n_c, 1, d]
        d = freqs_g.shape[-1]
        freqs = torch.cat(
            [
                freqs_g.reshape(f, n_g, d),
                freqs_c.reshape(f, n_c, d),
            ],
            dim=1,                                    # per-frame [geo | color]
        ).reshape(f * (n_g + n_c), 1, d)
        freqs = freqs.to(device=x.device)

        use_gc = inputs.get("use_gradient_checkpointing", False)
        use_gc_off = inputs.get("use_gradient_checkpointing_offload", False)

        def create_custom_forward(module):
            def custom_forward(*fargs):
                return module(*fargs)
            return custom_forward

        for block in dit.blocks:
            if use_gc_off:
                with torch.autograd.graph.save_on_cpu():
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs,
                        None, None, None,
                        use_reentrant=False,
                    )
            elif use_gc:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod, freqs,
                    None, None, None,
                    use_reentrant=False,
                )
            else:
                x = block(x, context, t_mod, freqs, None, None, None)

        # Each head reads only its own modality's token slice (frame-major, so
        # Head's (b, frames, n/frames, d) view stays valid after the split).
        hs = x.view(b, f, n_g + n_c, -1)
        hs_g = hs[:, :, :n_g].reshape(b, f * n_g, -1)
        hs_c = hs[:, :, n_g:].reshape(b, f * n_c, -1)

        pred_geo = dit.head(hs_g, t)                  # [B, V*n_g, 64*4]
        pred_geo = dit.unpatchify(pred_geo, (f, gh2, gw2))
        pred_color = dit.rgb_head(hs_c, t)            # [B, V*n_c, 16*4]
        pred_color = dit.unpatchify(pred_color, (f, ch2, cw2))
        return pred_geo, pred_color

    def _forward_flow(self, inputs):
        geo_gt = inputs["geo_gt"]            # [B, 64, V, 28, 48] standardized
        color_gt = inputs["color_gt"]        # [B, 16, V, 48, 84] standardized
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
            x_t_color = x_t_color * w
            v_target_color = v_target_color * w

        timestep = (
            sigma.reshape(batch_size) * self.pipe.scheduler.num_train_timesteps
        ).to(dtype=self.pipe.torch_dtype)
        timestep = self._expand_timestep(timestep)   # [B, V], source frames at 0

        pred_geo, pred_color = self._forward_model(
            inputs,
            self._build_flow_input(inputs, x_t_geo, x_t_color),
            timestep,
        )
        # The padded geo row/col carries no target: crop pred AND target back
        # to the true cached grid before the loss (parent GAE convention).
        h0, w0 = self._gae_true_grid
        pred_geo = pred_geo[:, :, :, :h0, :w0]
        v_target_geo = v_target_geo[:, :, :, :h0, :w0]

        from examples.wanvideo.model_training.train_with_accelerate_rgbd_nvs import (
            _masked_mse,
            _resize_mask,
        )

        # One mask PER GRID: the streams no longer share a spatial grid
        # (geo 27x48 vs color 48x84), so the v1 single-mask shortcut is gone.
        def _stream_mask(shape):
            m = _resize_mask(inputs["loss_masks"][:, 1], shape).to(pred_geo.dtype)
            return m.permute(0, 2, 1, 3, 4)          # [B, 1, V, h, w]

        geo_mse = _masked_mse(pred_geo, v_target_geo, _stream_mask(pred_geo.shape[-2:]))
        color_mse = _masked_mse(pred_color, v_target_color, _stream_mask(pred_color.shape[-2:]))
        # The trainer recomputes the joint loss from the per-stream tensors
        # below.  The streams cannot be channel-concatenated any more (their
        # grids differ), so "pred"/"rgb_gt" carry LISTS ordered [geo, color].
        stream_losses = {
            "geo_mse": geo_mse.detach(),
            "color_mse": color_mse.detach(),
        }
        return {
            "pred": [pred_geo, pred_color],
            "rgb_gt": [v_target_geo, v_target_color],
            "loss_masks": inputs["loss_masks"],
            "weight": 1.0,
            "sigma": sigma.reshape(batch_size),
            "rgb_aux": None,
            "stream_losses": stream_losses,
        }

    # -- sampling ---------------------------------------------------------------

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
        # Crop the geo stream to the true grid before anything downstream
        # (GAE decode) sees a padded latent.
        h0, w0 = self._gae_true_grid
        xg = xg[:, :, :, :h0, :w0]
        return xg, xc

    # -- parent overrides ---------------------------------------------------------

    def set_training_param(self):
        super().set_training_param()
        # color_patch_embedding is seeded from the pretrained Wan conv (like
        # rgb_head): trainable at base LR, so it must be unfrozen here.
        for param in self.pipe.dit.color_patch_embedding.parameters():
            param.requires_grad = True


def _axial_freqs(dit, frame_freqs, f, h, w):
    """[f*h*w, d] RoPE angle table for one modality block (frame-major)."""
    d_frame = frame_freqs.shape[-1]
    return torch.cat(
        [
            frame_freqs.view(f, 1, 1, d_frame).expand(f, h, w, -1),
            dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(f * h * w, 1, -1)


def F_pad_to(x, hw):
    import torch.nn.functional as F

    return F.pad(x, (0, hw[1] - x.shape[-1], 0, hw[0] - x.shape[-2]))
