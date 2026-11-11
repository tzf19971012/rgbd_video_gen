"""Plücker Flip PE with Normalize-Gate-Inject — faithful port of the GAE
stage-2 camera injection (GAE-GeometricAutoEncoder src/stage2/models/camera.py,
class PluckerFlipPE_V13, Apache-2.0) onto the Wan DiT attention layout.

Faithfulness notes:
  * feat_q = (d, m̂, log‖m‖), feat_k = (m̂, d, log‖m‖) — the d↔m̂ FLIP between
    the Q and K feature orderings is the mechanism's namesake.
  * PE = RMSNorm(E(feat)) * sigmoid-gate(MLP(log‖m‖)) * α, added to q/k POST-RoPE
    (GAE PluckerAttention.forward applies it after its 2D RoPE).
  * Output layers zero-init ("zero" init mode) → step 0 is bit-identical to the
    camera-free model; the first MLP layer is kaiming so gradients are alive.
  * log_scale_aug: training-time additive perturbation of the gate's log-scale
    input (GAE default prob 0.3, range [-1.2, 1.6]) — metric-scale robustness.
  * cam_residual (GAE's optional V-residual) is NOT ported: the released
    flow_gae128 recipe does not enable it.
"""
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Match GAE RMSNorm; accumulate in fp32 for bf16 stability.
        xf = x.float()
        normalized = xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight


class PluckerFlipPE(nn.Module):
    """Per-attention-block additive Plücker positional encoding on Q/K.

    One instance per attention block (GAE instantiates PluckerAttention -- and
    therefore its `self.plucker_pe` -- inside every block), each with its own
    E_q/E_k MLPs, RMSNorms, alpha scales and scale gate.
    """

    def __init__(
        self,
        dim: int,
        plucker_init: str = "zero",
        plucker_init_scale: float = 0.01,
        plucker_mlp_hidden: int = 64,
        plucker_scale: float = 1.0,
        log_scale_aug_prob: float = 0.3,
        log_scale_aug_range: tuple = (-1.2, 1.6),
    ):
        super().__init__()
        self.dim = dim
        self.use_mlp = plucker_mlp_hidden > 0
        self.use_scale = plucker_scale > 0
        self.log_scale_aug_prob = float(log_scale_aug_prob)
        self.log_scale_aug_range = tuple(float(x) for x in log_scale_aug_range)
        if self.log_scale_aug_prob < 0.0 or self.log_scale_aug_prob > 1.0:
            raise ValueError(
                f"log_scale_aug_prob must be in [0, 1], got {log_scale_aug_prob}")
        in_dim = 7  # (d(3), m̂(3), log_s(1))

        if self.use_mlp:
            self.eq = nn.Sequential(
                nn.Linear(in_dim, plucker_mlp_hidden, bias=False),
                nn.GELU(),
                nn.Linear(plucker_mlp_hidden, dim, bias=False),
            )
            self.ek = nn.Sequential(
                nn.Linear(in_dim, plucker_mlp_hidden, bias=False),
                nn.GELU(),
                nn.Linear(plucker_mlp_hidden, dim, bias=False),
            )
        else:
            self.eq = nn.Linear(in_dim, dim, bias=False)
            self.ek = nn.Linear(in_dim, dim, bias=False)

        self.norm_pe_q = RMSNorm(dim)
        self.norm_pe_k = RMSNorm(dim)

        sg_hidden = max(dim // 4, 1)
        self.scale_gate = nn.Sequential(
            nn.Linear(1, sg_hidden),
            nn.SiLU(),
            nn.Linear(sg_hidden, dim),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.scale_gate[0].bias)
        nn.init.zeros_(self.scale_gate[2].bias)

        if self.use_scale:
            self.alpha_q = nn.Parameter(torch.tensor(float(plucker_scale)))
            self.alpha_k = nn.Parameter(torch.tensor(float(plucker_scale)))

        self._init_weights(plucker_init, plucker_init_scale)

    def _gate_log_scale(self, log_scale: torch.Tensor) -> torch.Tensor:
        """Training-time augmentation of the gate input only (not the PE feats)."""
        if (
            not self.training
            or self.log_scale_aug_prob <= 0.0
            or torch.rand((), device=log_scale.device) >= self.log_scale_aug_prob
        ):
            return log_scale
        lo, hi = self.log_scale_aug_range
        return log_scale + torch.empty_like(log_scale).uniform_(lo, hi)

    def _init_weights(self, mode: str, scale: float):
        if self.use_mlp:
            # Zero only the OUTPUT layers; zeroing both would kill gradients
            # (h = GELU(0) = 0 -> dL/dW1 = 0).
            qk_output_layers = [self.eq[2], self.ek[2]]
            qk_input_layers = [self.eq[0], self.ek[0]]
        else:
            qk_output_layers = [self.eq, self.ek]
            qk_input_layers = []
        for m in qk_output_layers:
            if mode == "zero":
                nn.init.zeros_(m.weight)
            else:
                nn.init.normal_(m.weight, 0.0, scale)
        for m in qk_input_layers:
            if mode == "zero":
                nn.init.kaiming_uniform_(m.weight, a=5 ** 0.5)
            else:
                nn.init.normal_(m.weight, 0.0, scale)

    @staticmethod
    def decompose_plucker(plucker_6d: torch.Tensor):
        """(d, m) -> feat_q=(d, m̂, log_s), feat_k=(m̂, d, log_s), log_scale."""
        d = plucker_6d[..., :3]
        m = plucker_6d[..., 3:]
        m_norm = m.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        m_hat = m / m_norm
        log_scale = torch.log(m_norm)
        feat_q = torch.cat([d, m_hat, log_scale], dim=-1)
        feat_k = torch.cat([m_hat, d, log_scale], dim=-1)
        return feat_q, feat_k, log_scale

    def apply_to_qk(self, q: torch.Tensor, k: torch.Tensor, plucker_6d: torch.Tensor):
        """q, k: (B, S, H*D) POST-RoPE; plucker_6d: (B, S, 6). Additive PE."""
        feat_q, feat_k, log_scale = self.decompose_plucker(plucker_6d)

        pe_q = self.norm_pe_q(self.eq(feat_q.to(q.dtype)))
        pe_k = self.norm_pe_k(self.ek(feat_k.to(k.dtype)))

        gate = self.scale_gate(self._gate_log_scale(log_scale).to(q.dtype))
        pe_q = gate * pe_q
        pe_k = gate * pe_k

        if self.use_scale:
            pe_q = self.alpha_q * pe_q
            pe_k = self.alpha_k * pe_k

        return q + pe_q, k + pe_k
