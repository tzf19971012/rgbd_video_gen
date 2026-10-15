"""Cross-view consistency utilities for the f-consist factor.

Reconstructed support module: the training module imports
`flatness / ordered_pairs / pairwise_consistency` at module level, but the file
was never committed to the branch.  Semantics follow the docstrings in
WanRGBDNVSTrainingModule._depth_consistency_loss and the measured diagnostic
levels it references (pair_scale_dev ~0.13 -> scale_dev, pair_res_after_scale
~0.12 -> repro, constant-disparity field scores zero on everything).

The factor is OFF in every shipped config (depth_consist_*_weight 0.0), so this
code is never exercised in training; it only has to exist, be differentiable
and behave sanely if switched on.
"""

import torch


def ordered_pairs(n):
    """All ordered index pairs (i, j), i != j, over range(n)."""
    return [(i, j) for i in range(n) for j in range(n) if i != j]


def flatness(disp, valid):
    """Degenerate-solution monitor: 0.0 for a constant disparity field.

    Coefficient of variation of the valid disparities (std/mean); a constant
    field has zero spread, a real scene does not.
    """
    with torch.no_grad():
        v = disp[valid]
        if v.numel() < 2:
            return 0.0
        mean = v.mean()
        if float(mean.abs()) < 1e-8:
            return 0.0
        return float((v.std() / mean.abs()).cpu())


def _fit_affine(src, dst, trim_quantile):
    """Robust trimmed least-squares fit dst ~ a*src + b.  Differentiable.

    The first pass uses ALL pixels (mask decisions are detached everywhere), a
    second pass refits keeping only pixels whose |residual| is within
    `trim_quantile` of the residual distribution.
    """
    def _closed_form(s, d):
        s_mean = s.mean()
        d_mean = d.mean()
        s_c = s - s_mean
        var = (s_c * s_c).sum().clamp_min(1e-12)
        a = (s_c * (d - d_mean)).sum() / var
        b = d_mean - a * s_mean
        return a, b

    a, b = _closed_form(src, dst)
    if trim_quantile is not None and 0.0 < trim_quantile < 1.0:
        with torch.no_grad():
            res = (dst - (a * src + b)).abs()
            keep = res <= torch.quantile(res.reshape(-1), trim_quantile)
        if keep.sum() >= 2:
            a, b = _closed_form(src[keep], dst[keep])
    return a, b


def pairwise_consistency(
    disp,
    valid,
    K,
    c2w,
    pairs,
    min_support=0.02,
    ratio_clip=4.0,
    trim_quantile=0.9,
):
    """Mutual consistency of k predicted source disparity maps.

    disp:  [k, H, W] disparity (WITH gradient), shared-scale convention, >0 valid
    valid: [k, H, W] bool
    K:     [k, 3, 3] (or [k, 4, 4]) intrinsics -- accepted for signature
           compatibility with the diagnostic tooling; the scale/repro terms are
           pure per-pair affine fits and do not use them.
    c2w:   [k, 4, 4] camera-to-world -- likewise unused by the affine terms.
    pairs: iterable of ordered (i, j) index pairs.

    Returns None when NO ordered pair clears `min_support` (legitimate for
    widely-spread source views), otherwise a dict:

        scale     mean over pairs of (a - 1)^2          (loss term, grad)
        repro     mean over pairs of the scale-aligned
                  residual, normalised by the pair's mean
                  disparity                             (loss term, grad)
        scale_dev mean |a - 1|                          (float, logging)
        n_pairs   number of pairs measured              (float, logging)
        support   mean overlap fraction of the measured pairs (float)
    """
    k = disp.shape[0]
    scale_terms, repro_terms, devs, supports = [], [], [], []
    for i, j in pairs:
        if not (0 <= i < k and 0 <= j < k) or i == j:
            continue
        overlap = valid[i] & valid[j]
        support = overlap.float().mean()
        if float(support) < float(min_support):
            continue
        src = disp[i][overlap].float()
        dst = disp[j][overlap].float()
        if src.numel() < 2:
            continue
        # Ratio pre-clip: guard the fit against pathological disparity ratios
        # (sky vs near-field) by clamping the raw ratio range of the data.
        with torch.no_grad():
            src_lo = torch.quantile(src.detach(), 0.02).clamp_min(1e-6)
            src_hi = torch.quantile(src.detach(), 0.98).clamp_min(1e-6)
            span = (src_hi / src_lo).clamp_min(1.0)
            if float(span) > float(ratio_clip):
                continue
        a, b = _fit_affine(src, dst, trim_quantile)
        aligned = a * src + b
        norm = dst.mean().abs().clamp_min(1e-6)
        scale_terms.append((a - 1.0).pow(2))
        repro_terms.append((dst - aligned).abs().mean() / norm)
        devs.append(float((a - 1.0).abs()))
        supports.append(float(support))
    if not scale_terms:
        return None
    n = len(scale_terms)
    return {
        "scale": sum(scale_terms) / n,
        "repro": sum(repro_terms) / n,
        "scale_dev": sum(devs) / n,
        "n_pairs": float(n),
        "support": sum(supports) / n,
    }
