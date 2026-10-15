"""Epipolar attention bias for the pose-known NVS DiT.

Motivation: the pose-probe (09-20) showed the model is pose-blind outside the
source cone -- target-view tokens gather information from anywhere in the
source views, so "copy the nearest source" is a legal solution.  The soft
epipolar bias changes the cost: for a target-view token `a` attending to a
source-view token `b`, the attention logit is penalised by the Sampson
distance of `b` to the epipolar line of `a`, which is EXACT geometry derived
from the known intrinsics K and relative poses c2w.  Correct correspondences
sit on the line (distance ~0, no penalty); anything off the line pays -gamma*d.

Design constraints (deliberate):
  - NO new weights except one zero-initialised learnable scalar `gamma` per
    model (created by WanModel.enable_epipolar).  gamma=0 makes the forward
    BIT-IDENTICAL to the model without the bias, so Wan2.1 pretraining is
    untouched and loading is a strict superset.
  - The bias is an additive attention mask, which only the SDPA backend
    supports; flash_attention() falls back to SDPA whenever a bias is passed.
  - Token layout is the DiT's patchify order: view-major
    (view 0's (th x tw) tokens, then view 1's, ...), patch 1x2x2 over the
    latent grid (48 x 84) -> 24 x 42 = 1008 tokens per view, 6 views = 6048.
"""

import torch


def _homogeneous_token_centers(latent_height, latent_width, patch, device):
    """Unit-square homogeneous coords of token centers, [3, th*tw]."""
    th = latent_height // patch
    tw = latent_width // patch
    # Token centers in latent-pixel coords, then normalised to [0, 1].
    u = ((torch.arange(tw, device=device) + 0.5) * patch) / latent_width
    v = ((torch.arange(th, device=device) + 0.5) * patch) / latent_height
    uu, vv = torch.meshgrid(u, v, indexing="xy")  # [th, tw]
    ones = torch.ones_like(uu)
    return torch.stack([uu.reshape(-1), vv.reshape(-1), ones.reshape(-1)], dim=0)


def _skew(t):
    """Skew-symmetric matrix [t]x of a translation vector t [3] or [3,1]."""
    t = t.reshape(3)
    return torch.stack(
        [
            torch.stack([torch.zeros_like(t[0]), -t[2], t[1]]),
            torch.stack([t[2], torch.zeros_like(t[0]), -t[0]]),
            torch.stack([-t[1], t[0], torch.zeros_like(t[0])]),
        ]
    )


def epipolar_distance_matrix(
    latent_K,
    c2w_rel,
    latent_height,
    latent_width,
    patch=2,
    dist_cap=2.0,
):
    """Sampson point-to-epipolar-line distance for every cross-view token pair.

    Args:
        latent_K: [B, V, 3, 3] intrinsics AT LATENT RESOLUTION (the module's
            `latent_K`, already scaled by intrins_resize).
        c2w_rel: [B, V, 4, 4] scale-normalised camera-to-world poses (the
            module's `c2w_rel`, first source view as reference).
        latent_height, latent_width: latent grid size (e.g. 48, 84).
        patch: DiT patch size along h/w (2 for Wan2.1 1x2x2).
        dist_cap: Sampson distances are clamped to this (degenerate points near
            the epipole would otherwise blow the bias up).

    Returns:
        D: [B, L, L] float32, L = V * (latent_height//patch) * (latent_width//patch),
        view-major token order matching patchify.  D[b, tok_query, tok_key] is
        the Sampson distance of the KEY token to the epipolar line of the
        QUERY token (same [query, key] convention as attention logits).
        Same-view blocks are exactly zero.
    """
    device = latent_K.device

    # Unit-square homogeneous token centers, [3, N].
    uv = _homogeneous_token_centers(latent_height, latent_width, patch, device)

    # Geometry in fp32 with autocast disabled: under bf16 autocast every `@`
    # below silently drops to bf16, which measurably rots R_rel/E/F and hence
    # the Sampson distances (same mixed-precision hazard as the relative-pose
    # computation in forward_preprocess).  Pure function of the cameras, no
    # gradient needed.
    device_type = torch.device(device).type
    with torch.autocast(device_type=device_type, enabled=False):
        return _epipolar_distance_matrix_fp32(
            latent_K.float(), c2w_rel.float(), uv.float(),
            latent_height, latent_width, patch, dist_cap)


def _epipolar_distance_matrix_fp32(latent_K, c2w_rel, uv,
                                   latent_height, latent_width, patch, dist_cap):
    B, V = latent_K.shape[0], latent_K.shape[1]
    device = latent_K.device
    th = latent_height // patch
    tw = latent_width // patch
    N = th * tw
    L = V * N

    # Normalised intrinsics per view: divide the first two ROWS of each 3x3
    # matrix by the latent grid size so that the homogeneous coords live on
    # the unit square.  NB: the row index lives on dim -2, so the slice is
    # K[:, :, 0, :] -- K[:, 0, :] would select VIEW 0 instead.
    K = latent_K.clone()
    K[:, :, 0, :] = K[:, :, 0, :] / latent_width
    K[:, :, 1, :] = K[:, :, 1, :] / latent_height

    R = c2w_rel[:, :, :3, :3]  # [B, V, 3, 3]
    C = c2w_rel[:, :, :3, 3]  # [B, V, 3]  (already disparity-scale-normalised)

    D = torch.zeros(B, L, L, dtype=torch.float32, device=device)

    for b in range(B):
        Kinv = torch.linalg.inv(K[b])  # [V, 3, 3]

        for i in range(V):
            for j in range(V):
                if i == j:
                    continue
                # Relative pose i -> j in normalised cam coords.
                # c2w stores the cam2world rotation R and camera centre C, so
                # X_cam = R^T (X - C) and
                #   x_j = R_rel x_i + t_rel,  with
                #   R_rel = R_j^T R_i,  t_rel = R_j^T (C_i - C_j)
                R_rel = R[b, j].transpose(0, 1) @ R[b, i]
                t_rel = R[b, j].transpose(0, 1) @ (C[b, i] - C[b, j])
                E = _skew(t_rel) @ R_rel
                # Fundamental matrix acting on PIXEL-homogeneous coords:
                #   u_j^T F u_i = 0  for true correspondences,
                #   F = K_j^{-T} E K_i^{-1}
                F = Kinv[j].transpose(0, 1) @ E @ Kinv[i]

                # NOTE: F already contains the K^{-1} factors, so the bilinear
                # form must be evaluated on the raw pixel-homogeneous token
                # coords `uv` -- applying it to camera coords would fold K^{-1}
                # in twice and destroy the distance scale.
                # Constraint: u_j^T F u_i = 0 for query u_j (view j) and key
                # u_i (view i) -- F acts on the KEY side.
                Fu_key = F @ uv  # [3, N] -- epipolar line of each KEY (view i)
                Ft_query = F.transpose(0, 1) @ uv  # [3, N] -- per QUERY (view j)

                # Sampson distance (first-order geometric correction):
                #   d^2 = (u_q^T F u_k)^2 / (|F u_k|^2_{xy} + |F^T u_q|^2_{xy})
                num = (Fu_key.transpose(0, 1) @ uv) ** 2  # [N_key, N_query]
                den = (Fu_key[0:2] ** 2).sum(dim=0).unsqueeze(1)  # [N_key, 1]
                den = den + (Ft_query[0:2] ** 2).sum(dim=0).unsqueeze(0)  # [N, N]
                d2 = num / den.clamp_min(1e-12)
                d = torch.sqrt(d2.clamp_min(0.0)).clamp(max=dist_cap)
                # d is [key, query]; the attention convention is [query, key].
                d = d.transpose(0, 1)

                # query = view j tokens (rows), key = view i tokens (cols).
                D[b, j * N : (j + 1) * N, i * N : (i + 1) * N] = d

    return D
