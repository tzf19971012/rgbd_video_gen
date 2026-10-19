import torch
import torch.nn as nn

from .wan_video_camera_controller import fp32_geometry, ray_condition


class ViewConditionEncoder(nn.Module):
    """Encode per-view camera geometry (Plucker rays) plus a view-role embedding
    into an additive latent-space conditioning tensor.

    ``plucker_resolution`` selects where the Plucker field is rasterised:

    * ``"latent"`` (legacy): rays are generated directly on the latent grid
      (e.g. 48x84) from intrinsics rescaled by ``latent / pixel``.  This
      point-samples a single ray per VAE block and discards the other
      ``pixel_scale**2 - 1`` sub-positions.
    * ``"pixel"``: rays are generated on the full image grid (e.g. 384x672)
      from the original intrinsics, then folded onto the latent grid with
      ``nn.PixelUnshuffle`` -- a lossless space-to-depth reshape that moves each
      ``r x r`` block into the channel dimension.  A 1x1 convolution then
      *learns* how to pool the ``pixel_scale**2`` sub-positions instead of
      discarding them, and the half-pixel convention baked into
      ``ray_condition`` is applied at pixel rather than latent granularity.

    Both modes are output-identical at initialisation (the last conv is
    zero-initialised), so they are directly A/B comparable.

    Scene-scale note (relevant to ``disparity_norm_mode``): only the moment
    component ``o x d`` of the Plucker embedding carries world scale -- the
    direction component is unit-norm -- so the conditioning magnitude is
    proportional to the camera baseline.  ``disparity_norm_mode="shared_scale"``
    multiplies ``c2w[:3, 3]`` by the same factor it applies to the disparities,
    placing baselines and depths in one normalised unit system across samples.
    Under ``"per_view"`` the world scale is untouched and the moment magnitude
    varies by orders of magnitude between samples.  This is independent of
    ``plucker_resolution``: both modes consume the same ``c2w``.
    """

    def __init__(
        self,
        out_channels=16,
        role_dim=8,
        hidden_channels=64,
        num_roles=3,
        plucker_resolution="latent",
        pixel_scale=8,
        mode="add",
        use_role=True,
        normalize_moment=False,
        geometry_fp32=False,
    ):
        super().__init__()
        if plucker_resolution not in ("latent", "pixel"):
            raise ValueError(
                f"plucker_resolution must be 'latent' or 'pixel', got {plucker_resolution!r}"
            )
        if mode not in ("add", "concat"):
            raise ValueError(f"mode must be 'add' or 'concat', got {mode!r}")
        if int(pixel_scale) < 1:
            raise ValueError(f"pixel_scale must be >= 1, got {pixel_scale!r}")
        self.plucker_resolution = plucker_resolution
        self.pixel_scale = int(pixel_scale)
        self.mode = mode
        self.normalize_moment = bool(normalize_moment)
        # Rasterise the ray field outside autocast.  Two separate defects hide in
        # the bf16 path, and neither touches a learned parameter, so this flag is
        # purely about the geometry:
        #   * `ray_condition` sizes its pixel grid with
        #     `torch.linspace(0, W-1, W, dtype=c2w.dtype)`.  bf16 holds integers
        #     exactly only to 256, so at 384x672 the grid collapses -- 425 of 672
        #     distinct column coordinates, 292 of them wrong, up to 3 px off, with
        #     the step size degrading to 2 and then 4 towards the right edge.
        #   * `directions @ R^T` is downcast by autocast even though both operands
        #     are fp32, costing ~0.08 deg of ray direction, i.e. ~1 px median and
        #     ~2.4 px at p99 at this focal length.
        # Off by default: the field is a model input, so this is a sweep factor.
        self.geometry_fp32 = bool(geometry_fp32)
        # The role is only an explicit input in "add" mode.  In "concat" mode it
        # is redundant *and* harmful: source slots are identifiable by the fact
        # that they carry a clean image latent while target slots do not, and a
        # spatially constant role channel is exactly the shortcut that made the
        # additive encoder collapse into a role indicator.
        self.use_role = bool(use_role) and mode == "add"
        self.role_embedding = nn.Embedding(num_roles, role_dim) if self.use_role else None
        role_channels = role_dim if self.use_role else 0

        if plucker_resolution == "pixel":
            self.unshuffle = nn.PixelUnshuffle(self.pixel_scale)
            self.plucker_channels = 6 * self.pixel_scale ** 2
        else:
            self.unshuffle = None
            self.plucker_channels = 6

        if mode == "concat":
            # No bottleneck at all: the Plucker block is handed to the DiT's own
            # patch_embedding, which gives every channel its own weight.  The
            # only parameter here is a per-channel gain so the block can be
            # rescaled without a conv that could collapse it again.  It is
            # initialised to 1 (not 0) because this block is the *sole* signal
            # identifying a target view -- zero-init would reproduce the very
            # starvation this mode exists to fix; the newly added
            # patch_embedding channels are zero-init instead, which already
            # makes step 0 numerically identical to the RGB-only model.
            self.backbone = None
            self.out_channels = self.plucker_channels
            self.channel_scale = nn.Parameter(torch.ones(self.plucker_channels))
            return

        self.out_channels = out_channels
        self.channel_scale = None
        if plucker_resolution == "pixel":
            # 1x1 conv: mixes the pixel_scale**2 sub-positions that PixelUnshuffle
            # folded into the channel axis.  Spatial context is still supplied by
            # the following 3x3 conv, exactly as in the latent branch.
            first_conv = nn.Conv2d(
                self.plucker_channels + role_channels, hidden_channels, kernel_size=1
            )
        else:
            first_conv = nn.Conv2d(6 + role_channels, hidden_channels, kernel_size=3, padding=1)

        self.backbone = nn.Sequential(
            first_conv,
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )
        nn.init.zeros_(self.backbone[-1].weight)
        nn.init.zeros_(self.backbone[-1].bias)

    def _plucker(self, K, c2w, height, width):
        """-> [B, V, 6, height, width]; channels 0:3 are the moment o x d, 3:6 the direction d."""
        fx = K[..., 0, 0]
        fy = K[..., 1, 1]
        cx = K[..., 0, 2]
        cy = K[..., 1, 2]
        ray_K = torch.stack([fx, fy, cx, cy], dim=-1)
        # The Plucker field is a pure function of the (non-differentiable) camera
        # parameters, so no autograd graph is needed.  This matters in pixel mode
        # where the intermediates are pixel_scale**2 times larger.
        with torch.no_grad(), fp32_geometry(K.device, self.geometry_fp32):
            if self.geometry_fp32:
                # `ray_condition` derives its pixel grid's dtype from c2w, so the
                # cast has to happen before the call, not inside it.
                ray_K = ray_K.float()
                c2w = c2w.float()
            plucker = ray_condition(ray_K, c2w, height, width, device=K.device)
            plucker = plucker.permute(0, 1, 4, 2, 3).contiguous()
            if self.normalize_moment:
                # Only the moment o x d carries world scale (the direction is
                # unit-norm), so its magnitude is proportional to the camera
                # baseline and varies by orders of magnitude between scenes even
                # under disparity_norm_mode="shared_scale".  Left raw it either
                # dwarfs or vanishes against the direction component depending on
                # the scene.  Normalising per view puts the two components on a
                # comparable footing without touching their spatial pattern.
                moment = plucker[:, :, :3]
                rms = moment.pow(2).mean(dim=(2, 3, 4), keepdim=True).sqrt().clamp_min(1e-6)
                plucker = torch.cat([moment / rms, plucker[:, :, 3:]], dim=2)
        return plucker

    def plucker_field(
        self,
        K,
        c2w,
        latent_height,
        latent_width,
        pixel_K=None,
        pixel_height=None,
        pixel_width=None,
    ):
        """-> [B, V, plucker_channels, latent_height, latent_width], no learned parts.

        The raw geometry, before `channel_scale` (concat mode) or the backbone
        (add mode).  Split out of `forward` so RoCE can consume the same field
        without rasterising it a second time -- in pixel mode that is a
        384x672x6 ray field per view -- and without picking up a gradient path
        into either learned part, which would make the RoCE arm change two things
        at once.  It carries no autograd graph at all (`_plucker` is under
        no_grad and PixelUnshuffle is a reshape).
        """
        if K.ndim != 4 or c2w.ndim != 4:
            raise ValueError("K and c2w must have shape [B, V, ...].")
        batch_size, num_views = K.shape[0], K.shape[1]

        if self.plucker_resolution == "pixel":
            if pixel_K is None or pixel_height is None or pixel_width is None:
                raise ValueError(
                    "plucker_resolution='pixel' requires pixel_K, pixel_height and pixel_width."
                )
            r = self.pixel_scale
            if pixel_height != latent_height * r or pixel_width != latent_width * r:
                raise ValueError(
                    f"pixel grid {pixel_height}x{pixel_width} is not {r}x the latent grid "
                    f"{latent_height}x{latent_width}; adjust pixel_scale."
                )
            plucker = self._plucker(pixel_K, c2w, pixel_height, pixel_width)
            plucker = self.unshuffle(
                plucker.reshape(batch_size * num_views, 6, pixel_height, pixel_width)
            )
            plucker = plucker.reshape(
                batch_size, num_views, self.plucker_channels, latent_height, latent_width
            )
        else:
            plucker = self._plucker(K, c2w, latent_height, latent_width)
        return plucker

    def forward(
        self,
        K,
        c2w,
        view_roles,
        latent_height,
        latent_width,
        pixel_K=None,
        pixel_height=None,
        pixel_width=None,
        plucker=None,
    ):
        """`plucker` reuses a field already built by `plucker_field`; None builds it.

        Passing it in is bit-identical to letting this method build it -- the field
        is a pure function of the camera parameters -- and only exists so the RoCE
        arm does not pay for the rasterisation twice.
        """
        if K.ndim != 4 or c2w.ndim != 4:
            raise ValueError("K and c2w must have shape [B, V, ...].")
        batch_size, num_views = K.shape[0], K.shape[1]
        if plucker is None:
            plucker = self.plucker_field(
                K, c2w, latent_height, latent_width,
                pixel_K=pixel_K, pixel_height=pixel_height, pixel_width=pixel_width,
            )

        if self.mode == "concat":
            out = plucker * self.channel_scale.view(1, 1, -1, 1, 1).to(plucker.dtype)
            return out.permute(0, 2, 1, 3, 4).contiguous()

        if self.use_role:
            if view_roles.ndim == 1:
                view_roles = view_roles.unsqueeze(0).expand(batch_size, -1)
            if view_roles.ndim != 2:
                raise ValueError("view_roles must have shape [B, V] or [V].")

            role_ids = view_roles.clamp_min(0).clamp_max(self.role_embedding.num_embeddings - 1)
            role_emb = self.role_embedding(role_ids)
            # The role embedding is spatially constant, so it is broadcast straight
            # onto the latent grid and never participates in the unshuffle.
            role_emb = role_emb.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, -1, latent_height, latent_width
            )
            x = torch.cat([plucker, role_emb.to(plucker.dtype)], dim=2)
        else:
            x = plucker

        x = x.reshape(batch_size * num_views, x.shape[2], latent_height, latent_width)
        x = self.backbone(x)
        x = x.reshape(batch_size, num_views, self.out_channels, latent_height, latent_width)
        return x.permute(0, 2, 1, 3, 4).contiguous()