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
        gae_prediction="v",
        gae_rope_frame_axis="view",
        gae_auto_flow_shift=True,
        gae_codec_device=None,
        gae_rgb_loss_weight=0.0,
        gae_rgb_t_max=0.4,
        gae_rgb_max_views=1,
        gae_patch_mode="native",
        v_loss_t_eps=0.05,
        camera_drop_prob=0.0,
        ref_drop_prob=0.0,
        eval_cfg_scale=1.0,
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

        # Parameterization of the flow head.  GAE stage2 ships 'x' (the head
        # predicts the clean latent; flow.py:329 converts it to a velocity via
        # convert_x_to_v -- clamp t at t_eps=0.05 -- and trains with UNIFORM
        # v-space MSE, so its weighting profile is identical to a plain v-head
        # under uniform v-MSE).  The arms therefore differ in what the network
        # regresses (x_pred vs v), not in the loss weighting:
        #   'v' + v_loss_t_eps=1.0 -> w=1  : profile-identical to GAE shipped
        #   'v' + v_loss_t_eps=0.05 -> w=1/sigma^2 : extra low-sigma emphasis
        #   'x'                            : GAE-faithful parameterization
        if gae_prediction not in ("v", "x"):
            raise ValueError(
                f"gae_prediction must be 'v' or 'x', got {gae_prediction!r}"
            )
        self.gae_prediction = gae_prediction

        if gae_rope_frame_axis not in ("view", "constant"):
            raise ValueError(
                f"gae_rope_frame_axis must be 'view' or 'constant', "
                f"got {gae_rope_frame_axis!r}"
            )
        # Tokenization resolution experiment (patchify hypothesis, plan A/B):
        #   native    - Wan (1,2,2) patchify on the cached 27x48 grid
        #               (336 tokens/view; each token blends 2x2 latent pixels);
        #   patch1    - fresh (1,1,1) patch_embedding + Head + dit.patch_size,
        #               1296 tokens/view, one token per latent pixel, no padding;
        #   upsample2 - nearest-2x the latent to 54x96 BEFORE the (1,2,2)
        #               patchify.  Mathematically identical to patch1: each
        #               2x2 patch holds 4 identical values, so the conv output
        #               is the (1,1,1) result times a constant.  The codec
        #               never sees the upsampled grid -- sample_latents /
        #               _rgb_aux_loss avg_pool back before decode, and
        #               nearest-up -> avg_pool is an EXACT identity (verified:
        #               max|err|=0 latent, PSNR=inf decoded).
        if gae_patch_mode not in ("native", "patch1", "upsample2"):
            raise ValueError(
                f"gae_patch_mode must be 'native' | 'patch1' | 'upsample2', "
                f"got {gae_patch_mode!r}"
            )
        self.gae_patch_mode = gae_patch_mode
        self._gae_upsample = 2 if gae_patch_mode == "upsample2" else 1
        self._gae_codec_grid = None  # (h, w) of the cached latents (pre-upsample)
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

        # Velocity-MSE weighting w(t) = 1 / max(t, v_loss_t_eps)^2, the exact
        # equivalent of the GAE/RAEv2 shipped recipe (prediction='x' with
        # v = (x_t - x_pred) / max(t, eps) before the MSE, see
        # GAE src/stage2/transport/flow.py convert_x_to_v): uniform weight on
        # the velocity error equals a 1/t^2 weight on the x-space error, which
        # is what keeps the low-noise (detail-forming) regime trained.  Without
        # it the effective x-space weight is t^2 and the low-sigma region is
        # doubly suppressed (few samples after shift warp AND ~zero loss
        # weight) -- the model collapses to the conditional mean and every
        # sample comes out blurry.
        self.v_loss_t_eps = float(v_loss_t_eps)

        # CFG condition dropout (GAE flow_gae128 protocol: camera_drop 0.2,
        # ref_drop_prob 0.2).  Zeroing both probabilities restores the old
        # always-conditioned behaviour.
        self.camera_drop_prob = float(camera_drop_prob)
        self.ref_drop_prob = float(ref_drop_prob)
        self.eval_cfg_scale = float(eval_cfg_scale)
        if self.camera_drop_prob > 0 or self.ref_drop_prob > 0:
            self.accelerator.print(
                f"[gae] CFG condition dropout ENABLED: camera_drop={self.camera_drop_prob} "
                f"ref_drop={self.ref_drop_prob} eval_cfg_scale={self.eval_cfg_scale}"
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
        # widening under the forced config: this module overrides
        # _needs_extra_input_channels() to False, so concat-mode plucker
        # channels are added by the fresh conv below, not by the parent).
        # Both must be replaced: the latent space has 64 channels in AND out.
        dit = self.pipe.dit
        old_conv = dit.patch_embedding
        old_head = dit.head
        # Snapshot the pretrained Wan2.1 projections before replacing them.
        # Plain-dict tensors, so they never register as submodules (no DDP /
        # checkpoint leakage).  The geo-color subclass seeds its COLOR stream
        # (patch-embedding slice [64:80] + rgb_head) from this snapshot; the
        # geometry stream stays fresh.
        self._pretrained_wan_projections = {
            "patch_embedding": {
                k: v.detach().clone() for k, v in old_conv.state_dict().items()
            },
            "head": {
                k: v.detach().clone() for k, v in old_head.state_dict().items()
            },
        }
        # patch1 arm: 1x1x1 patchify, one token per latent pixel (1296/view vs
        # 336 at (1,2,2)).  The Head's Linear and WanModel.unpatchify both read
        # patch_size, so the frozen dit.patch_size must move with it (the
        # training path unpatchifies through model_fn_wan_video).
        if self.gae_patch_mode == "patch1":
            conv_kernel = conv_stride = (1, 1, 1)
            head_patch = (1, 1, 1)
        else:
            conv_kernel, conv_stride = old_conv.kernel_size, old_conv.stride
            head_patch = old_head.patch_size
        # Concat-mode camera conditioning rides in as its OWN input channel
        # block appended after the latent (parent _build_flow_input, channel
        # layout), so the fresh conv must carry it: 64 latent + plucker block
        # (294 at pixel_scale=7 / 384 at 8).  This module forces
        # _needs_extra_input_channels()=False, so the parent's widening never
        # ran -- if the extra block is not re-accounted here the first forward
        # feeds 64+plucker channels into a 64-channel conv and dies.
        # Zero-init the plucker slice so step 0 is numerically the
        # camera-free model (matches the parent's widening contract).
        concat_plucker_ch = (
            self.view_condition_encoder.out_channels
            if self.view_condition_mode == "concat"
            else 0
        )
        dit.patch_embedding = torch.nn.Conv3d(
            self.gae_latent_channels + concat_plucker_ch,
            old_conv.out_channels,
            kernel_size=conv_kernel,
            stride=conv_stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        ).to(device=old_conv.weight.device, dtype=old_conv.weight.dtype)
        torch.nn.init.kaiming_normal_(
            dit.patch_embedding.weight, nonlinearity="linear")
        if concat_plucker_ch:
            with torch.no_grad():
                dit.patch_embedding.weight[
                    :, self.gae_latent_channels:
                ].zero_()
        if dit.patch_embedding.bias is not None:
            torch.nn.init.zeros_(dit.patch_embedding.bias)

        dit.head = Head(
            dit.dim, self.gae_latent_channels, head_patch, eps=1e-6
        ).to(device=old_conv.weight.device, dtype=old_conv.weight.dtype)
        dit.patch_size = head_patch
        dit.rgb_head = None
        self.accelerator.print(
            f"[gae] patch_mode={self.gae_patch_mode}: patch_embedding -> "
            f"Conv3d({self.gae_latent_channels}+{concat_plucker_ch}, "
            f"{old_conv.out_channels}, "
            f"kernel={conv_kernel}) (fresh, plucker block zero-init), "
            f"head -> Head(out_dim="
            f"{self.gae_latent_channels}, patch={head_patch}) (fresh), "
            f"dit.patch_size={dit.patch_size}, rgb_head removed."
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
        # Tokenization arms (see __init__):
        #   upsample2: nearest-2x the standardized latent to the DiT working
        #     grid 54x96 BEFORE patchify.  Standardize-then-upsample is safe
        #     (per-channel affine commutes with nearest duplication) and the
        #     resulting grid is always even, so the odd-grid pad below never
        #     fires.  _gae_codec_grid remembers the TRUE cached grid:
        #     sample_latents / _rgb_aux_loss avg_pool back to it before any
        #     codec call.
        #   patch1: (1,1,1) patchify accepts the odd cached grid as-is, so the
        #     pad is skipped entirely.
        #   native: original behaviour -- pad odd grids to even for (1,2,2).
        h, w = z.shape[-2:]
        self._gae_codec_grid = (h, w)
        if self._gae_upsample > 1:
            B, C, V = z.shape[:3]
            z = z.permute(0, 2, 1, 3, 4).reshape(B * V, C, h, w)
            z = F.interpolate(z, scale_factor=self._gae_upsample, mode="nearest")
            z = z.reshape(B, V, C, h * self._gae_upsample, w * self._gae_upsample)
            z = z.permute(0, 2, 1, 3, 4).contiguous()
            h, w = h * self._gae_upsample, w * self._gae_upsample
        self._gae_true_grid = (h, w)
        if self.gae_patch_mode != "patch1" and (h % 2 or w % 2):
            z = F.pad(z, (0, w % 2, 0, h % 2))
        return z

    # -- input assembly ------------------------------------------------------

    def forward_preprocess(self, data):
        batch_size = data["images"].shape[0]
        num_views = data["images"].shape[1]
        height = data["images"].shape[-2]
        width = data["images"].shape[-1]

        z = self._load_cached_latents(data)          # [B, C, V, h, w], standardized, EVEN grid
        # The ray field must sit on the TRUE working grid -- the grid z actually
        # carries, recorded pre-pad by _load_cached_latents.  For the GAE cache
        # that is height//14 (x2 in upsample2); for a Wan-VAE cache (pure-color)
        # it is the wan native grid -- deriving it as height//14 here would
        # zero-pad the conditioning field into the wrong corner.
        assert self._gae_true_grid is not None, "latent loader must record the true grid"
        latent_height, latent_width = self._gae_true_grid

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

        if self.view_condition_mode == "concat":
            # Parent's concat path (WanRGBDNVSTrainingModule.forward_preprocess):
            # the camera block rides as its OWN input channel block --
            # _build_flow_input appends inputs["plucker_block"] after the
            # latents (channel order matches the widened patch_embedding
            # [latent, plucker]).  Nothing is summed into the latents.
            conditioning = z
            plucker_block = view_condition
            additive_condition = torch.zeros_like(z)
        else:
            conditioning = z + view_condition    # source evidence rides in
            plucker_block = None
            additive_condition = view_condition

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
            "view_condition": additive_condition,
            # Only set in mode="concat": appended as its own channel block in
            # _build_flow_input instead of being summed in.
            "plucker_block": plucker_block,
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

        timestep = (
            sigma.reshape(batch_size) * self.pipe.scheduler.num_train_timesteps
        ).to(dtype=self.pipe.torch_dtype)
        timestep = self._expand_timestep(timestep)   # [B, V], source frames at 0

        # CFG condition dropout (per batch element, Bernoulli).  Sampled OUTSIDE
        # the checkpointed DiT call so the masks/noise are fixed inputs.
        cam_drop = ref_drop = None
        if self.training and (self.camera_drop_prob > 0 or self.ref_drop_prob > 0):
            dev = rgb_gt.device
            if self.camera_drop_prob > 0:
                cam_drop = torch.rand(batch_size, device=dev) < self.camera_drop_prob
            if self.ref_drop_prob > 0:
                ref_drop = torch.rand(batch_size, device=dev) < self.ref_drop_prob

        pred = self._forward_model(
            inputs,
            self._build_flow_input(
                inputs, x_t, cam_drop=cam_drop, ref_drop=ref_drop),
            timestep
        )
        # The padded patchify row/col carry no target: crop pred AND target
        # back to the true cached grid so the loss compares like with like.
        h0, w0 = self._gae_true_grid
        pred = pred[:, :, :, :h0, :w0]

        if self.gae_prediction == "x":
            # GAE-original parameterization, faithful to flow.py:329-336: the
            # head outputs x_pred, which is converted to a velocity via
            # convert_x_to_v (paper Eq. 11, clamp t at v_loss_t_eps=0.05) and
            # regressed to v_target with UNIFORM v-space MSE.
            # NOTE on what this arm actually tests: in x-space the convert
            # error is x_err/t, so the weighting profile is IDENTICAL to the
            # plain v-head under uniform v-MSE (patch1's v_loss_t_eps=1.0).
            # The arm varies the PARAMETERIZATION (what the network regresses),
            # not the loss weighting.  x_pred stays available for the aux loss.
            flow_target = (noise - rgb_gt)[:, :, :, :h0, :w0]
            x_pred = pred
            t_safe = sigma.clamp_min(self.v_loss_t_eps).reshape(
                batch_size, 1, 1, 1, 1).to(pred.dtype)
            pred = (x_t[:, :, :, :h0, :w0] - pred) / t_safe
            v_weight = torch.ones(
                batch_size, device=pred.device, dtype=pred.dtype)
        else:
            flow_target = (noise - rgb_gt)[:, :, :, :h0, :w0]
            # Per-element velocity-MSE weight 1/max(sigma, eps)^2: in x-space
            # this profile is x_err^2/sigma^4, i.e. MORE low-sigma emphasis
            # than GAE's shipped convert+uniform-v recipe (x_err^2/sigma^2).
            v_weight = sigma.clamp_min(self.v_loss_t_eps).square().reshape(batch_size)
            v_weight = (1.0 / v_weight).to(pred.dtype)
            x_pred = x_t[:, :, :, :h0, :w0] - sigma.reshape(
                batch_size, 1, 1, 1, 1).to(pred.dtype) * pred

        rgb_aux = None
        if self.gae_rgb_loss_weight > 0 and self.training:
            rgb_aux = self._rgb_aux_loss(inputs, x_pred, sigma)
        return {
            "pred": pred,
            "rgb_gt": flow_target,
            "loss_masks": inputs["loss_masks"],
            "weight": v_weight,
            "sigma": sigma.reshape(batch_size),
            "rgb_aux": rgb_aux,
        }

    def _rgb_aux_loss(self, inputs, x_pred, sigma):
        """L1 decode loss on the target-view x-prediction (GAE compute_rgb_aux_loss).

        ``x_pred`` arrives already in x-parameterization: 'x' mode passes the
        head output through; 'v' mode converts first (x_t = z + sigma*(noise-z),
        so x_pred = x_t - sigma * v_pred). Only batches with sigma <
        gae_rgb_t_max contribute (high-noise x_pred is garbage; its decode
        gradient is noise), and at most gae_rgb_max_views target views per
        batch element are decoded to cap the RGBHead cost. Gradient flows:
        L1 -> RGBHead (frozen) -> x_pred -> DiT. Returns None on gated-off
        steps so the trainer can skip the term entirely.
        """
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
        # upsample2 arm: collapse the 2x working grid back to the codec grid
        # (exact identity on constant 2x2 blocks) before the frozen RGBHead.
        if self._gae_upsample > 1:
            z_raw = F.avg_pool2d(z_raw, self._gae_upsample)
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
    def sample_latents(self, inputs, num_steps=None, generator=None, cfg_scale=None):
        """Euler-integrate the target-view slots; source slots stay clean.

        cfg_scale > 1 enables classifier-free guidance: each step runs an
        additional unconditional pass (bare x_t, zero camera conditioning,
        pure-noise evidence slots) and extrapolates
        v = v_uncond + cfg * (v_cond - v_uncond).

        In 'x' parameterization the head output is the clean-latent prediction;
        it is converted to a velocity via GAE convert_x_to_v (paper Eq. 11):
        v = (x_t - x_pred) / max(sigma, t_eps).
        """
        num_steps = (
            self.eval_num_inference_steps if num_steps is None else max(1, int(num_steps))
        )
        cfg = self.eval_cfg_scale if cfg_scale is None else float(cfg_scale)
        z = inputs["gae_latents"]                    # [B, C, V, h, w]
        s = self.num_source_views
        sigmas = self.inference_sigmas(num_steps, device=z.device).to(z.dtype)

        def _to_velocity(out, x_cur, sigma_val):
            if self.gae_prediction != "x":
                return out
            sig = max(float(sigma_val), self.v_loss_t_eps)
            return (x_cur - out) / sig

        x = torch.randn(z.shape, device=z.device, dtype=z.dtype, generator=generator)
        x[:, :, :s] = z[:, :, :s]
        for i in range(num_steps):
            timestep = (
                sigmas[i] * self.pipe.scheduler.num_train_timesteps
            ).reshape(1).to(dtype=self.pipe.torch_dtype, device=z.device)
            timestep = self._expand_timestep(timestep)
            velocity = _to_velocity(
                self._forward_model(
                    inputs, self._build_flow_input(inputs, x), timestep
                )[:, :, s:],
                x[:, :, s:], float(sigmas[i]),
            )
            if cfg > 1.0:
                # Unconditional branch: strip the additive camera/role
                # conditioning everywhere and replace the clean evidence slots
                # with pure noise.  _build_flow_input takes source slots from
                # rgb_latents, so handing it pure noise there yields a
                # fully-unconditional input; only the target-slot velocity is
                # consumed.
                uncond_inputs = dict(inputs)
                uncond_inputs["rgb_latents"] = torch.randn(
                    z.shape, device=z.device, dtype=x.dtype)
                uncond_inputs["view_condition"] = torch.zeros_like(
                    inputs["view_condition"])
                if inputs.get("plucker_block") is not None:
                    # concat mode: the camera block is an input channel block;
                    # the unconditional branch must strip it too.
                    uncond_inputs["plucker_block"] = torch.zeros_like(
                        inputs["plucker_block"])
                v_uncond = _to_velocity(
                    self._forward_model(
                        uncond_inputs,
                        self._build_flow_input(uncond_inputs, x),
                        timestep,
                    )[:, :, s:],
                    x[:, :, s:], float(sigmas[i]),
                )
                velocity = v_uncond + cfg * (velocity - v_uncond)
            dt = sigmas[i] - sigmas[i + 1]
            x[:, :, s:] = x[:, :, s:] - dt * velocity.to(x.dtype)
            x[:, :, :s] = z[:, :, :s]                # re-pin the evidence
        h0, w0 = self._gae_true_grid
        x = x[:, :, :, :h0, :w0]
        # Fold the upsampled working grid back onto the codec's grid BEFORE
        # anything downstream (eval decode, rgb aux) sees a latent.  avg_pool
        # of a nearest-upsampled tensor is an exact identity, so a model that
        # emits constant 2x2 blocks loses nothing.
        if self._gae_upsample > 1:
            B, C, V = x.shape[:3]
            x = x.permute(0, 2, 1, 3, 4).reshape(B * V, C, h0, w0)
            x = F.avg_pool2d(x, self._gae_upsample)
            ch, cw = self._gae_codec_grid
            x = x.reshape(B, V, C, ch, cw).permute(0, 2, 1, 3, 4).contiguous()
        return x

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

    # -- pose-sensitivity diagnostic (GAE backend) -----------------------------

    @torch.no_grad()
    def evaluate_pose_sensitivity(self, batch, max_samples=4):
        """GAE override of the parent's RGBD diagnostic.

        Differences vs the parent version, everything else mirrored:
          * sampling goes through `sample_latents` (Euler on the GAE latent,
            cfg_scale pinned to 1.0 so the diagnostic measures pure
            conditioning, not guidance) and decoding through the frozen codec
            (`_decode_views`), never the Wan VAE -- these latents are
            standardized 64-channel GAE codes, VAE-decoding them is garbage;
          * add mode is SUPPORTED, and the RGBD-era refusal does not apply:
            the parent refused because its add-mode pose swap would force a
            second VAE encode of the conditioning images.  Here the clean
            latent comes from the cache (`gae_latents`), so a pose swap is
            the exact linear re-add  rgb_latents = gae_latents + vc_alt;
            source slots are bit-identical because `_corrupt_pose_c2w` only
            touches target rows, and the role embedding / source Plucker
            field therefore do not move either;
          * warp_shuffle / both_shuffle are DROPPED: this module pins
            warp_mode="none", so there is no warp path to shuffle and the
            variants would silently measure nothing.
        Returns rows keyed  {all,covered,uncovered,lpips}_{variant}.
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
            net = self._get_lpips(device)
            if net is None:
                return float("nan")
            with torch.no_grad():
                d = net(pred_img.float() * 2 - 1, gt_img.float() * 2 - 1)
            return float(d.mean())

        # (name, pose corruption, plucker corruption); warp variants dropped.
        variant_spec = (
            ("baseline", None, None),
            ("swap_tgt", "swap_tgt", None),
            ("src0", "src0", None),
            ("pose_shuffle", None, "shuffle"),
        )

        num_src = self.num_source_views
        height, width = batch_small["images"].shape[-2:]
        autocast_ctx = torch.autocast(
            device_type="npu", dtype=self.pipe.torch_dtype, enabled=True
        )
        with autocast_ctx:
            self._eval_inputs = True
            try:
                inputs_eval = self.forward_preprocess(batch_small)
                # One extra preprocess per pose mode; only pose-derived tensors
                # are kept.  In add mode the pose rides inside rgb_latents as
                # z + vc, so the swap needs vc_alt AND the re-added latents.
                keep_concat = ("plucker_block", "cape_pose", "roce_camera_latents")
                alt_pose = {}
                for mode in {s[1] for s in variant_spec if s[1] is not None}:
                    batch_pose = dict(batch_small)
                    batch_pose["c2w"] = self._corrupt_pose_c2w(
                        batch_small["c2w"].to(device).float(), mode
                    )
                    alt = self.forward_preprocess(batch_pose)
                    keys = keep_concat + (
                        ("view_condition",) if self.view_condition_mode != "concat" else ()
                    )
                    alt_pose[mode] = {k: alt[k] for k in keys if alt.get(k) is not None}
            finally:
                self._eval_inputs = False

            variants = {}
            for name, pose_mode, plucker_mode in variant_spec:
                inputs = dict(inputs_eval)
                if pose_mode is not None:
                    alt = alt_pose[pose_mode]
                    if self.view_condition_mode == "concat":
                        inputs.update(alt)
                    else:
                        vc_alt = alt["view_condition"]
                        inputs["view_condition"] = vc_alt
                        # Exact pose swap on the additive path; source slots
                        # unchanged (vc_alt == vc_true there).
                        inputs["rgb_latents"] = (
                            inputs["gae_latents"] + vc_alt.to(inputs["gae_latents"].dtype)
                        )
                if plucker_mode is not None:
                    blk = inputs.get("plucker_block")
                    if blk is None:
                        blk = inputs["view_condition"]
                        inputs["view_condition"] = self._corrupt_warp_block(
                            blk, plucker_mode
                        )
                    else:
                        inputs["plucker_block"] = self._corrupt_warp_block(
                            blk, plucker_mode
                        )
                gen = torch.Generator(device=device)
                gen.manual_seed(self.eval_sample_seed)
                variants[name] = self.sample_latents(
                    inputs, num_steps=self.eval_num_inference_steps,
                    generator=gen, cfg_scale=1.0,
                )

        def _decode_all(lat_bcvhw):
            rgb, _ = self._decode_views(lat_bcvhw.float(), height, width)
            return rgb.to(device)  # [B, V, 3, H, W]; codec returns on CPU

        gt_px_all = _decode_all(inputs_eval["gae_latents"])
        pred_px = {name: _decode_all(lat) for name, lat in variants.items()}

        rgb_px_batch = batch_small["images"].to(device).float().clamp(0.0, 1.0)
        disp_batch = batch_small["disparity"].to(device)
        _, splat_mask = self._splat_sources_pixel(
            rgb_px_batch, disp_batch[:, :num_src, 0].float(),
            batch_small["K"].to(device).float(), inputs_eval["c2w"],
        )

        rows = []
        for b in range(gt_px_all.shape[0]):
            for tv in range(self.num_target_views):
                vi = num_src + tv
                gt_px = gt_px_all[b, vi]
                m = splat_mask[b, tv]
                while m.ndim > 2:
                    m = m[0]          # [1,H,W] -> [H,W]; _psnr_masked re-adds
                hole = 1.0 - m
                ones = torch.ones_like(m)
                row = {
                    "sample_id": inputs_eval["sample_id"][b],
                    "view_index": vi,
                    "tgt_rgb_splat_coverage": float(m.mean()),
                }
                for name in pred_px:
                    pred_img = pred_px[name][b, vi]
                    row[f"all_{name}"] = _psnr_masked(pred_img, gt_px, ones[None])
                    row[f"covered_{name}"] = _psnr_masked(pred_img, gt_px, m[None])
                    row[f"uncovered_{name}"] = _psnr_masked(pred_img, gt_px, hole[None])
                    row[f"lpips_{name}"] = _lpips(pred_img, gt_px)
                rows.append(row)

        if was_training:
            self.train()
        return rows

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
        want_depth = pred_depth is not None     # pure-color decode has no depth

        gt_rgb = batch_small["images"].to(device).float().clamp(0, 1)
        tgt = slice(self.num_source_views, self.num_views)

        lpips_net = self._get_lpips(device)
        metrics = {"tgt_rgb_psnr": [], "tgt_rgb_lpips": []}
        if want_depth:
            metrics.update({
                "tgt_depth_abs_rel": [], "tgt_depth_abs_rel_aligned": [],
                "tgt_depth_delta1": [], "src_depth_abs_rel": [],
            })
            gt_depth = batch_small["depth_metric"][:, :, 0].to(device).float()
            valid = (gt_depth > 1e-3).float()
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
                if not want_depth:
                    continue
                pd, gd = pred_depth[b, v].to(device), gt_depth[b, v]
                m = valid[b, v].bool() & torch.isfinite(gd)
                if m.sum() >= 16:
                    p, g = pd[m].float(), gd[m].float()
                    metrics["tgt_depth_abs_rel"].append(float(
                        ((p - g).abs() / g).mean()))
                    scale = float(g.median() / p.median().clamp_min(1e-6))
                    pa = p * scale
                    metrics["tgt_depth_abs_rel_aligned"].append(float(
                        ((pa - g) / g).abs().mean()))
                    metrics["tgt_depth_delta1"].append(float(
                        (torch.maximum(pa / g, g / pa) < 1.25).float().mean()))
            if not want_depth:
                continue
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
