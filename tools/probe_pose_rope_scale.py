"""D3 pre-flight: the numbers that decide WHICH pose-conditioned RoPE to build.

reports/plan_20260906_rope_geometry.md §D3 listed three blockers and got two of
them wrong, so this probe settles all three on real BlendedMVS val samples
before any model code is written.  Zero GPU, zero training.

  1. SCALE.  §D3 feared "camera translation is in per-sample units, so the
     effective frequency content differs per scene".  It is already fixed:
     kubric4d_rgbd_nvs_dataset.py:587-589 multiplies c2w[:3,3] by the SAME
     shared disparity scale that normalises the disparity maps, so the (image,
     disparity, pose) triple is one consistently rescaled world and parallax is
     preserved.  What is NOT known is whether the resulting ||t|| is O(1) and
     how wide its cross-sample spread is -- that is what sets the frequency
     ladder.  Measured here.

  2. NORM.  §D3 said EscherNet's 6-DoF CaPE "constructs a rotation acting on a
     4-vector".  It does not.  The reference implementation
     (kxhit/EscherNet 6DoF/diffusers/models/attention_processor.py:36) is

         def cape_embed(f, P):
             f = rearrange(f, '... (d k) -> ... d k', k=4)
             return rearrange(f @ P, '... d k -> ... (d k)', k=4)

     with P = pose for the key and P^-T for the query (dataset.py:169-171 builds
     `pose_inv = inv(P).transpose(-1,-2)`).  An SE(3) matrix is NOT orthogonal,
     so CaPE does NOT preserve ||q||, ||k||, and it inflates the attention logit
     by an amount that grows with the baseline.  Wan's pretrained blocks have
     only ever seen unit-modulus rotations on these dims.  Measured here, as the
     ratio of the frame-slice logit std to the RoPE baseline.

  3. FRAME.  CaPE's exactness identity is q'.k' = q^T (P_k P_q^-1)^T k.  That is
     invariant to the choice of world frame ONLY if P is world-to-camera:
     w2c -> w2c G^-1 under x_world -> G x_world, and w2c_k c2w_q is unchanged.
     Feeding c2w instead conjugates the relative pose by G, i.e. silently makes
     the encoding depend on which view happens to be view 0 -- which D2 permutes
     and which plan §10 shows is computed wrong in bf16.  Asserted numerically
     here on real poses, both ways, because getting it backwards is a textbook
     §6-2 silent failure: the model trains, converges and reports numbers.

It also measures the ORTHOGONAL alternative that the CaPE reading suggests --
factorise the SE(3) action into (a) R (+1) on 4-dim groups, which is exactly the
t=0 case of CaPE and therefore genuinely orthogonal, and (b) a rotary phase
w_p . c on complex pairs, whose logit depends on w_p . (c_i - c_j) exactly.
Both halves are norm-preserving, continuous, exactly relative, and both collapse
to the identity as their scale goes to 0 -- so they admit the zero-init warm
start that ReDirector's RoCE (arXiv 2511.19827) relies on and CaPE cannot offer.
The price is that the translation half is only frame-EQUIvariant: w_p . (c_i-c_j)
changes under a global rotation.  That is measured too, so the tradeoff against
CaPE's exact invariance is a number rather than an argument.

Usage:
  python tools/probe_pose_rope_scale.py --num-samples 24
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from examples.dataset.kubric4d_rgbd_nvs_dataset import Kubric4DRGBDNVSDataset  # noqa: E402

_DATA_ROOT = os.environ.get("DVD_DATA_ROOT", "")
DEFAULT_MANIFEST = os.path.join(_DATA_ROOT, "datasets/BlendedMVS/manifest.jsonl")
DEFAULT_KUBRIC_ROOT = os.path.join(_DATA_ROOT, "datasets/BlendedMVS/BlendedMVS")
DEFAULT_SPLIT_PATH = "configs/bmvs_split_v1.json"

# Wan 14B: dim 1536, 12 heads -> head_dim 128; precompute_freqs_cis_3d splits it
# as (dim - 2*(dim//3), dim//3, dim//3) = (44, 42, 42) REAL dims, and rope_apply
# pairs consecutive reals into complex, so the frame axis is real dims 0..43.
HEAD_DIM = 128
FRAME_DIMS = HEAD_DIM - 2 * (HEAD_DIM // 3)   # 44


def quantiles(x, qs=(0.02, 0.10, 0.50, 0.90, 0.98)):
    x = torch.as_tensor(x, dtype=torch.float64).flatten()
    return {q: float(torch.quantile(x, q)) for q in qs}


def fmt_q(d):
    return "  ".join(f"p{int(q * 100):02d}={v:.4f}" for q, v in d.items())


def build_dataset(args):
    return Kubric4DRGBDNVSDataset(
        kubric4d_root=args.kubric4d_root,
        manifest_path=args.manifest_path,
        num_source_views=args.num_source_views,
        num_target_views=args.num_target_views,
        random_permute_source_views=True,
        deterministic_view_seed=args.eval_view_seed,
        min_view_valid_fraction=args.min_view_valid_fraction,
        samples_per_scene=args.samples_per_scene,
        split_path=args.split_path,
        split="val",
        disparity_norm_mode="shared_scale",
        disparity_norm_quantile=args.disparity_norm_quantile,
        disparity_norm_headroom=args.disparity_norm_headroom,
        height=args.height,
        width=args.width,
    )


def cape_embed(f, P):
    """kxhit/EscherNet 6DoF attention_processor.py:36, verbatim in torch.

    f: [..., d] with d % 4 == 0.  P: [..., 4, 4] broadcasting over f's batch.
    Row-vector convention, so this is f_group @ P.
    """
    g = f.reshape(*f.shape[:-1], f.shape[-1] // 4, 4)
    g = torch.einsum("...gk,...kl->...gl", g, P)
    return g.reshape(*f.shape)


def scale_translation(P, lam):
    """(R, t) -> (R, lam*t).

    This is a GROUP AUTOMORPHISM of SE(3): the relative pose of the scaled poses
    is exactly the scaled relative pose, because the composed translation is
    linear in the individual translations.  So CaPE's exactness identity and its
    world-frame invariance both survive verbatim for any lam, and lam is the only
    knob that shrinks the logit inflation without breaking either.  lam = 0 is
    rotation-only, which is orthogonal; lam = 1 is CaPE as published.

    Contrast the OTHER obvious gate, P(lam) = exp(lam * log P): it does reach the
    identity at lam=0, but exp(lam log P_k) exp(-lam log P_q) is not a function
    of P_k P_q^-1, so it breaks exactness everywhere except lam in {0, 1}.
    """
    Q = P.clone()
    Q[..., :3, 3] = Q[..., :3, 3] * lam
    return Q


def rot_only(P):
    """The lam=0 case of CaPE: R (+) 1.  Orthogonal by construction."""
    return scale_translation(P, 0.0)


def random_se3(generator, device, dtype=torch.float64):
    A = torch.randn(3, 3, generator=generator, device=device, dtype=dtype)
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diagonal(R)).unsqueeze(0)
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    G = torch.eye(4, device=device, dtype=dtype)
    G[:3, :3] = Q
    G[:3, 3] = torch.randn(3, generator=generator, device=device, dtype=dtype) * 3.0
    return G


def logit_stats(transform_q, transform_k, poses, generator, n_draw, dims=FRAME_DIMS):
    """std of the frame-slice contribution to one attention logit.

    q, k are drawn N(0, 1) per channel, which is what RMSNorm hands rope_apply
    up to the learned per-channel weight (norm_q/norm_k run BEFORE rope_apply,
    wan_video_dit.py:169-173).  The baseline to compare against is sqrt(dims):
    for unit-variance q, k the sum of `dims` products has variance `dims`, and
    an orthogonal transform leaves that untouched.
    """
    V = poses.shape[0]
    out = []
    for i in range(V):
        for j in range(V):
            q = torch.randn(n_draw, dims, generator=generator, dtype=torch.float64)
            k = torch.randn(n_draw, dims, generator=generator, dtype=torch.float64)
            qi = transform_q(q, poses[i])
            kj = transform_k(k, poses[j])
            out.append((qi * kj).sum(-1))
    return torch.cat(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest-path", default=DEFAULT_MANIFEST)
    p.add_argument("--kubric4d-root", default=DEFAULT_KUBRIC_ROOT)
    p.add_argument("--split-path", default=DEFAULT_SPLIT_PATH)
    p.add_argument("--num-source-views", type=int, default=6)
    p.add_argument("--num-target-views", type=int, default=3)
    p.add_argument("--eval-view-seed", type=int, default=1234)
    p.add_argument("--min-view-valid-fraction", type=float, default=0.5)
    p.add_argument("--samples-per-scene", type=int, default=10)
    p.add_argument("--disparity-norm-quantile", type=float, default=0.98)
    p.add_argument("--disparity-norm-headroom", type=float, default=1.5)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num-samples", type=int, default=24)
    p.add_argument("--num-draws", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ds = build_dataset(args)
    n = min(args.num_samples, len(ds))
    # STRIDE, do not take ds[0..n-1].  samples_per_scene=10 means the first
    # twelve indices are one or two scenes, and BlendedMVS scene geometry is
    # wildly heterogeneous -- a near-parallel drone sweep has ~2.6 deg median
    # relative rotation while an orbit has ~88 deg.  Reading the first n indices
    # reported 3.28 deg as "the" median and would have killed the rotation-only
    # arm on the strength of two scenes.
    step = max(1, len(ds) // n)
    indices = list(range(0, len(ds), step))[:n]
    print(f"dataset: {len(ds)} val samples, probing {n} with stride {step} "
          f"(samples_per_scene={args.samples_per_scene}, so this is ~one per scene)")

    gen = torch.Generator().manual_seed(args.seed)

    all_pair_dist = []        # ||c_i - c_j|| in the normalised world, all pairs
    all_pair_angle = []       # relative rotation angle, all pairs, degrees
    min_angle = []            # per-sample closest pair of orientations
    all_scale = []            # depth_norm_scale itself, for the record
    per_sample_cape = []      # frame-slice logit std under CaPE / sqrt(44)
    per_sample_rot = []       # same, rotation-only (should be exactly 1.0)
    per_sample_cape_max = []  # worst view pair in the sample
    inv_err_cape = []         # |logit(P) - logit(G P)| under a random global SE(3)
    inv_err_cape_c2w = []
    inv_err_rotphase = []
    all_w2c = []              # kept for the lambda sweep below

    for idx in indices:
        item = ds[idx]
        c2w = item["c2w"].double()                 # [V, 4, 4], translation already * s
        all_scale.append(float(item["depth_norm_scale"]))
        centers = c2w[:, :3, 3]
        V = centers.shape[0]
        d = torch.cdist(centers, centers)
        upper = torch.triu(torch.ones(V, V), diagonal=1) > 0
        all_pair_dist.append(d[upper])

        # Relative rotation angle.  This is what the lambda=0 (rotation-only,
        # exactly orthogonal) arm has to work with: if two views share an
        # orientation they become POSITIONALLY IDENTICAL on the frame axis, and
        # unlike a wrong ids length that collision does not raise anywhere.
        # Rrel[a, b] = R_a^T R_b, so the trace is the standard axis-angle cosine.
        Rrel = torch.einsum("aji,bjk->abik", c2w[:, :3, :3], c2w[:, :3, :3])
        cos = ((torch.diagonal(Rrel, dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1, 1)
        ang = torch.rad2deg(torch.acos(cos))
        all_pair_angle.append(ang[upper])
        min_angle.append(float(ang[upper].min()))

        w2c = torch.linalg.inv(c2w)

        # --- 2. CaPE norm inflation on the 44-dim frame slice -----------------
        # query carries P^-T, key carries P (EscherNet dataset.py:169-171).
        w2c_invT = torch.linalg.inv(w2c).transpose(-1, -2)
        cape_q = lambda x, P: cape_embed(x, P)   # noqa: E731  (P pre-inverted)
        cape_k = lambda x, P: cape_embed(x, P)   # noqa: E731

        base = FRAME_DIMS ** 0.5
        stats = []
        for i in range(V):
            for j in range(V):
                q = torch.randn(args.num_draws, FRAME_DIMS, generator=gen, dtype=torch.float64)
                k = torch.randn(args.num_draws, FRAME_DIMS, generator=gen, dtype=torch.float64)
                lg = (cape_embed(q, w2c_invT[i]) * cape_embed(k, w2c[j])).sum(-1)
                stats.append(float(lg.std()) / base)
        per_sample_cape.append(sum(stats) / len(stats))
        per_sample_cape_max.append(max(stats))

        rot_stats = []
        # Zero the translation of the POSE, then invert-transpose.  Zeroing
        # P^-T's [:3, 3] instead is a no-op -- P^-T carries -(R^T t)^T in its
        # BOTTOM ROW, not its last column -- and silently leaves the translation
        # in, which is how this row first read 3.20x instead of 1.00x.
        w2c_rot = rot_only(w2c)
        Rq = torch.linalg.inv(w2c_rot).transpose(-1, -2)
        Rk = w2c_rot
        for i in range(V):
            for j in range(V):
                q = torch.randn(args.num_draws, FRAME_DIMS, generator=gen, dtype=torch.float64)
                k = torch.randn(args.num_draws, FRAME_DIMS, generator=gen, dtype=torch.float64)
                lg = (cape_embed(q, Rq[i]) * cape_embed(k, Rk[j])).sum(-1)
                rot_stats.append(float(lg.std()) / base)
        per_sample_rot.append(sum(rot_stats) / len(rot_stats))

        # --- 3. frame invariance ---------------------------------------------
        G = random_se3(gen, centers.device)
        q = torch.randn(args.num_draws, FRAME_DIMS, generator=gen, dtype=torch.float64)
        k = torch.randn(args.num_draws, FRAME_DIMS, generator=gen, dtype=torch.float64)

        def cape_logit(P):
            Pq = torch.linalg.inv(P).transpose(-1, -2)
            acc = []
            for i in range(V):
                for j in range(V):
                    acc.append((cape_embed(q, Pq[i]) * cape_embed(k, P[j])).sum(-1))
            return torch.stack(acc)

        # x_world -> G x_world  =>  w2c -> w2c G^-1,  c2w -> G c2w
        Ginv = torch.linalg.inv(G)
        ref = cape_logit(w2c)
        moved = cape_logit(w2c @ Ginv)
        inv_err_cape.append(float((moved - ref).abs().max() / ref.abs().max()))

        ref_c = cape_logit(c2w)
        moved_c = cape_logit(G @ c2w)
        inv_err_cape_c2w.append(float((moved_c - ref_c).abs().max() / ref_c.abs().max()))

        # rotary phase on the camera centre: phase_v = w . c_v, so the logit
        # depends on w . (c_i - c_j) exactly -- translation invariant, but only
        # EQUIvariant under a global rotation.
        omega = torch.randn(FRAME_DIMS // 2, 3, generator=gen, dtype=torch.float64)

        def rotphase_logit(cs):
            phase = cs @ omega.T                      # [V, 22]
            zq = torch.view_as_complex(q.reshape(-1, FRAME_DIMS // 2, 2).contiguous())
            zk = torch.view_as_complex(k.reshape(-1, FRAME_DIMS // 2, 2).contiguous())
            acc = []
            for i in range(V):
                for j in range(V):
                    rel = torch.polar(torch.ones_like(phase[i]), phase[i] - phase[j])
                    acc.append((zq * zk.conj() * rel).real.sum(-1))
            return torch.stack(acc)

        ref_r = rotphase_logit(centers)
        moved_r = rotphase_logit((G[:3, :3] @ centers.T).T + G[:3, 3])
        inv_err_rotphase.append(float((moved_r - ref_r).abs().max() / ref_r.abs().max()))

        all_w2c.append(w2c)

    pair = torch.cat(all_pair_dist)
    print()
    print("--- 1. pose scale in the disparity-normalised world -------------------")
    print(f"depth_norm_scale (1/length):        {fmt_q(quantiles(all_scale))}")
    print(f"||c_i - c_j||, all {pair.numel()} pairs:      {fmt_q(quantiles(pair))}")
    print(f"                                    mean={float(pair.mean()):.4f}  "
          f"max={float(pair.max()):.4f}")
    ang = torch.cat(all_pair_angle)
    print(f"relative rotation angle (deg):      {fmt_q(quantiles(ang))}")
    print(f"  closest pair of orientations per sample: "
          f"{fmt_q(quantiles(min_angle))}")
    med = float(torch.quantile(pair.double(), 0.5))
    print(f"  => a pair rotating pi rad over the MEDIAN baseline needs "
          f"|omega| = {torch.pi / med:.3f} rad/unit")
    print(f"  => spread p98/p02 = "
          f"{quantiles(pair)[0.98] / max(quantiles(pair)[0.02], 1e-9):.1f}x, "
          f"i.e. the ladder must cover that many octaves")

    print()
    print("--- 2. frame-slice logit inflation vs RoPE (1.0 = no inflation) -------")
    print(f"CaPE 6-DoF, mean over view pairs:   {fmt_q(quantiles(per_sample_cape))}")
    print(f"CaPE 6-DoF, worst view pair:        {fmt_q(quantiles(per_sample_cape_max))}")
    print(f"rotation-only (t=0), orthogonal:    {fmt_q(quantiles(per_sample_rot))}")

    print()
    print("--- 3. invariance to the choice of world frame (0 = exactly invariant) -")
    print(f"CaPE on w2c   (relative max err):   {fmt_q(quantiles(inv_err_cape))}")
    print(f"CaPE on c2w   (relative max err):   {fmt_q(quantiles(inv_err_cape_c2w))}")
    print(f"rotary phase on camera centre:      {fmt_q(quantiles(inv_err_rotphase))}")

    print()
    print("--- 4. translation scale lambda vs logit inflation ---------------------")
    print("  f = CaPE block's own inflation; total = whole 128-dim head logit std")
    print("  relative to sqrt(128), when the CaPE block is `dims` of the 44 frame")
    print("  dims and the other (128 - dims) keep their orthogonal RoPE.")
    print(f"  {'lambda':>7}  {'f_mean':>7}  {'f_worst':>7}  "
          f"{'total@24':>9}  {'total@44':>9}")
    for lam in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
        f_mean, f_worst = [], []
        # Same draws for every lambda, so the Monte-Carlo noise on the std
        # estimate is common-mode and the lambda trend -- and the fact that
        # lambda=0 must land on exactly 1.000 -- are both readable.
        sweep_gen = torch.Generator().manual_seed(args.seed + 991)
        for w2c in all_w2c:
            P = scale_translation(w2c, lam)
            Pq = torch.linalg.inv(P).transpose(-1, -2)
            V = P.shape[0]
            s = []
            for i in range(V):
                for j in range(V):
                    q = torch.randn(args.num_draws, FRAME_DIMS, generator=sweep_gen, dtype=torch.float64)
                    k = torch.randn(args.num_draws, FRAME_DIMS, generator=sweep_gen, dtype=torch.float64)
                    lg = (cape_embed(q, Pq[i]) * cape_embed(k, P[j])).sum(-1)
                    s.append(float(lg.std()) / FRAME_DIMS ** 0.5)
            f_mean.append(sum(s) / len(s))
            f_worst.append(max(s))
        fm = sum(f_mean) / len(f_mean)
        fw = sum(f_worst) / len(f_worst)
        tot = lambda d, f: ((HEAD_DIM - d + d * f * f) / HEAD_DIM) ** 0.5  # noqa: E731
        print(f"  {lam:>7.2f}  {fm:>7.3f}  {fw:>7.3f}  "
              f"{tot(24, fw):>9.3f}  {tot(44, fw):>9.3f}")


if __name__ == "__main__":
    main()
