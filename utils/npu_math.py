"""NPU-compatible replacements for linalg ops torch_npu routes to CPU.

torch_npu has no kernel for `torch.linalg.inv` / `torch.linalg.solve`
(aten::linalg_inv_ex / aten::_linalg_solve_ex), so every call silently
round-trips host<->device and serialises the stream.  The helpers here
re-express the same math with primitives the NPU does support (transpose,
matmul, elementwise).  All functions are exact for their intended inputs.
"""

import torch


def inv_rigid(c2w):
    """Inverse of rigid [.., 4, 4] transforms: [R|t]^-1 = [R^T | -R^T t].

    Intended for SE(3) camera poses (c2w / w2c / relative poses).  For inputs
    with tiny orthonormality drift the error passes through at the same
    magnitude `torch.linalg.inv` would produce.  Uses only matmul/transpose,
    so no CPU fallback.
    """
    r = c2w[..., :3, :3]
    t = c2w[..., :3, 3]
    r_inv = r.transpose(-1, -2)
    out = torch.zeros_like(c2w)
    out[..., :3, :3] = r_inv
    out[..., :3, 3] = -(r_inv @ t.unsqueeze(-1)).squeeze(-1)
    out[..., 3, 3] = 1.0
    return out


def inv3x3(m):
    """Closed-form inverse of [.., 3, 3] matrices via the adjugate.

    Intended for intrinsics / calibration matrices (well-conditioned, positive
    determinant).  No pivoting -- guard the determinant upstream if the input
    can be singular.  Uses only elementwise ops, so no CPU fallback.
    """
    a, b, c = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    d, e, f = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    g, h, i = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    adj00 = e * i - f * h
    adj01 = c * h - b * i
    adj02 = b * f - c * e
    det = a * adj00 + d * adj01 + g * adj02
    out = torch.empty_like(m)
    out[..., 0, 0] = adj00
    out[..., 0, 1] = adj01
    out[..., 0, 2] = adj02
    out[..., 1, 0] = f * g - d * i
    out[..., 1, 1] = a * i - c * g
    out[..., 1, 2] = c * d - a * f
    out[..., 2, 0] = d * h - e * g
    out[..., 2, 1] = b * g - a * h
    out[..., 2, 2] = a * e - b * d
    return out / det.unsqueeze(-1).unsqueeze(-1)
