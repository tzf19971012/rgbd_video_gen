import copy
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.lora import LoRALinearLayer
from einops import rearrange

from .utils import hash_state_dict_keys
from .wan_video_camera_controller import SimpleAdapter

try:
    import flash_attn_interface

    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn

    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn

    SAGE_ATTN_AVAILABLE = True
    print(f"========= Using sage attention, please note that this is for inference speed up only!==========")
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    compatibility_mode=False,
):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        if isinstance(x, tuple):
            x = x[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(
            10000,
            -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(
                dim // 2
            ),
        ),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def num_frame_rope_dims(head_dim: int) -> int:
    """REAL dims of a head that precompute_freqs_cis_3d hands to the frame axis.

    Mirrors that function's split exactly (it is `dim - 2*(dim//3)`), and since
    rope_apply pairs consecutive reals into one complex channel, these are dims
    0..n-1 of every head -- the frame block comes FIRST in the cat below.
    head_dim 128 -> 44.
    """
    return head_dim - 2 * (head_dim // 3)


# RoCE (ReDirector) modes.  "off" is spelled out rather than left as None so a
# config, a log line and an arm name can all use the same three tokens.
ROCE_MODES = ("off", "qk", "qk_vo")


def num_hw_rope_phase_dims(head_dim: int) -> int:
    """COMPLEX channels of a head that the h and w axes own TOGETHER.

    This is the width RoCE's per-block phase MLP must have per head.  `rope_apply`
    LEFT-pads the phase tensor with zeros, so this number decides where the
    learned phase field lands: exactly right and it covers h+w and nothing else,
    one too many and the whole field slides onto the frame block -- which is where
    CaPE lives (`num_frame_rope_dims`) and where an extra rotation destroys the
    exact relative-pose identity.  Nothing about that failure raises, hence the
    check in rope_apply.

    ReDirector spells this `2 * head_dim // 3 // 2`, which is correct for
    head_dim 128 (42 = 21 + 21) but wrong whenever head_dim//3 is odd: at
    head_dim 64 it says 21 while h and w own 10 complex channels each.  Derived
    from precompute_freqs_cis_3d's real split here instead, so the tiny test DiTs
    (head_dim 24) exercise the same code path as the 1.3B model.
    """
    return 2 * ((head_dim // 3) // 2)


def cape_apply(x, pose, num_heads, frames, frame_dims):
    """EscherNet 6-DoF CaPE on the frame-axis slice of every head.

    kxhit/EscherNet, 6DoF/diffusers/models/attention_processor.py:36:

        def cape_embed(f, P):
            f = rearrange(f, '... (d k) -> ... d k', k=4)
            return rearrange(f @ P, '... d k -> ... (d k)', k=4)

    Row-vector convention, so this is `group @ P` on each consecutive group of 4
    real dims.  The caller passes P for the key and P^-T for the query, which is
    what makes the logit exactly

        q'.k' = q^T (P_k P_q^-1)^T k

    i.e. a function of the RELATIVE pose alone.  See
    reports/plan_20260906_rope_geometry.md §12 for the two facts that are easy to
    get wrong and silent when wrong: P must be WORLD-TO-CAMERA (with c2w the
    identity becomes a conjugation by the world frame, measured relative error
    1.77 on real poses), and this transform is NOT orthogonal, so it inflates the
    attention logit -- x2.60 on the full head at the published translation scale,
    x1.14 at the 0.2 this project uses (tools/probe_pose_rope_scale.py §4).

    `pose` is [B, frames, 4, 4], NOT per token: x is viewed as
    [B, frames, tokens_per_frame, ...] instead of materialising one 4x4 per token,
    which at 18 frames x 1560 latent pixels would be 28080 matrices per sample per
    call for no reason.  That view is only valid because patchify flattens (f,h,w)
    frame-major.
    """
    dtype = x.dtype
    b, s, _ = x.shape
    if s % frames != 0:
        raise ValueError(
            f"cape_apply needs the token count {s} to be divisible by the frame "
            f"count {frames}; sequence parallelism splits the token axis and "
            f"breaks the frame-major view this relies on."
        )
    if frame_dims % 4 != 0:
        raise ValueError(f"CaPE needs 4-dim groups, but the frame axis owns {frame_dims} dims")
    # Only the frame slice goes through fp32; the other 84 of 128 dims stay in
    # their input dtype.  rope_apply's comment applies here too -- q/k are
    # [B, 28080, 1536], i.e. 86 MB in bf16, and this runs 2x per block x 40 blocks.
    x = x.reshape(b, frames, s // frames, num_heads, -1)
    g = x[..., :frame_dims].float().reshape(
        b, frames, s // frames, num_heads, frame_dims // 4, 4
    )
    g = torch.einsum("bftngk,bfkl->bftngl", g, pose.float())
    g = g.reshape(b, frames, s // frames, num_heads, frame_dims).to(dtype)
    return torch.cat([g, x[..., frame_dims:]], dim=-1).reshape(b, s, -1)


def rope_apply(x, freqs, num_heads, phase=None):
    """Rotary embedding, optionally with ReDirector's learned camera phase (RoCE).

    `freqs` is the fixed 3D axial table, `[s, 1, head_dim//2]` complex, or None to
    apply the phase alone (that is the v / output path of RoCE's GTA-style value
    rotation, which has no fixed positional part).

    `phase` is `[b, s, num_heads * num_hw_rope_phase_dims(head_dim)]` REAL radians,
    left-zero-padded onto the frame block here so it rotates the h/w channels only.
    Being a rotation it preserves |q| and |k| exactly -- the opposite of cape_apply,
    which is an SE(3) matrix and inflates the logit.  The two therefore compose
    safely and commute: they act on disjoint channels.
    """
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    head_dim = x.shape[-1]
    if phase is not None:
        phase = rearrange(phase, "b s (n d) -> b s n d", n=num_heads)
        frame_channels = head_dim // 2 - phase.shape[-1]
        # The one silent failure in this function: a phase that is too wide pushes
        # the h/w field onto the frame block, where CaPE's exact relative-pose
        # identity lives and where nothing would ever complain.
        if frame_channels != num_frame_rope_dims(head_dim) // 2:
            raise ValueError(
                f"RoCE phase is {phase.shape[-1]} complex channels per head, but "
                f"head_dim {head_dim} gives h+w only "
                f"{num_hw_rope_phase_dims(head_dim)}; the left zero-pad would land "
                f"{frame_channels} on the frame block instead of "
                f"{num_frame_rope_dims(head_dim) // 2}"
            )
        phase = torch.nn.functional.pad(phase, (frame_channels, 0), value=0.0).float()
        # exp(1j*0) == 1+0j, so the padded frame channels are multiplied by exactly
        # the identity -- untouched by construction, not by omission.
        phase_shift = torch.polar(torch.ones_like(phase), phase)
        freqs = phase_shift if freqs is None else freqs.unsqueeze(0) * phase_shift
    elif freqs is None:
        return rearrange(x, "b s n d -> b s (n d)")
    # float32 is the standard precision for RoPE.  The original float64 cast
    # quadrupled the memory traffic of this op (q/k are [B, 7056, 1536], i.e.
    # 347 MB in bf16 but 1.39 GB in fp64) and it runs 60x per forward pass,
    # which is pure bandwidth waste for accuracy that is never observable.
    x_out = torch.view_as_complex(
        x.to(torch.float32).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2)
    )
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        # print(f"x device: {x.device}, weight device: {self.weight.device}")
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, q, k, v):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)
        # RoCE (ReDirector, arm droce): created by enable_roce(), absent otherwise,
        # so a plain Wan2.1 checkpoint keeps its exact state-dict key set.
        self.rope_phase_qk = None
        self.rope_phase_vo = None

    def enable_roce(self, mode: str = "qk_vo"):
        """Add ReDirector's two zero-init phase MLPs to this block.

        `Sequential(SiLU, Linear(dim, num_heads * hw_channels, bias=False))` with
        the Linear zero-initialised, exactly as ReDirector's
        model_utils.adjust_to_ReDirector -- so at step 0 every phase is 0,
        exp(1j*0) == 1, and the model is BIT-IDENTICAL to the arm without RoCE.
        That is what makes droce readable against its anchor: the intervention can
        only grow away from the anchor, it cannot start somewhere else.

        mode "qk" rotates q and k only; "qk_vo" additionally applies -phase to v
        and +phase after o(), the GTA-style value rotation that ReDirector's own
        ablation needs to keep geometric consistency from dropping.
        """
        if mode not in ROCE_MODES or mode == "off":
            raise ValueError(f"roce mode must be one of {ROCE_MODES[1:]}, got {mode!r}")
        width = self.num_heads * num_hw_rope_phase_dims(self.head_dim)

        def _phase_mlp():
            mlp = nn.Sequential(nn.SiLU(), nn.Linear(self.dim, width, bias=False))
            nn.init.zeros_(mlp[-1].weight)
            return mlp

        self.rope_phase_qk = _phase_mlp()
        self.rope_phase_vo = _phase_mlp() if mode == "qk_vo" else None
        return self

    def forward(self, x, freqs, cape=None, roce_cam_emb=None):
        phase_qk = phase_vo = None
        if roce_cam_emb is not None:
            if self.rope_phase_qk is None:
                raise ValueError(
                    "a RoCE camera embedding reached a block whose phase MLPs were "
                    "never created; call WanModel.enable_roce() before the forward"
                )
            phase_qk = self.rope_phase_qk(roce_cam_emb)
            if self.rope_phase_vo is not None:
                phase_vo = self.rope_phase_vo(roce_cam_emb)
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads, phase_qk)
        k = rope_apply(k, freqs, self.num_heads, phase_qk)
        if phase_vo is not None:
            # GTA-style value rotation: v is un-rotated going in and re-rotated
            # coming out, so the attention OUTPUT is equivariant to the camera
            # rather than only the logits being invariant to it.
            v = rope_apply(v, None, self.num_heads, -phase_vo)
        if cape is not None:
            # Pose-conditioned frame axis (plan §12 / arm d3).  Safe to compose
            # with rope_apply ONLY because the caller pins the frame RoPE ids to
            # all-zero, making freqs' frame block exactly 1+0j -- otherwise both
            # act on dims 0..43 and, since a 2x2 rotation and a 4x4 SE(3) matrix
            # do not commute, the exact relative-pose identity is destroyed.
            # model_fn_wan_video raises rather than allowing that combination.
            pose_q, pose_k, frames, frame_dims = cape
            q = cape_apply(q, pose_q, self.num_heads, frames, frame_dims)
            k = cape_apply(k, pose_k, self.num_heads, frames, frame_dims)
        x = self.attn(q, k, v)
        x = self.o(x)
        if phase_vo is not None:
            x = rope_apply(x, None, self.num_heads, phase_vo)
        return x


class CrossAttention(nn.Module):
    def __init__(
        self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        return self.o(x)


class SelfAttentionSeparate(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, rank=64):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        if rank > 0:
            # LoraLinear
            self.q_zl_before = LoRALinearLayer(dim, dim, rank=rank)
            self.k_zl_before = LoRALinearLayer(dim, dim, rank=rank)
            self.v_zl_before = LoRALinearLayer(dim, dim, rank=rank)

            self.q_zl_after = LoRALinearLayer(dim, dim, rank=rank)
            self.k_zl_after = LoRALinearLayer(dim, dim, rank=rank)
            self.v_zl_after = LoRALinearLayer(dim, dim, rank=rank)
        else:
            # Normal Linear
            self.q_zl_before = nn.Linear(dim, dim)
            self.k_zl_before = nn.Linear(dim, dim)
            self.v_zl_before = nn.Linear(dim, dim)

            self.q_zl_after = nn.Linear(dim, dim)
            self.k_zl_after = nn.Linear(dim, dim)
            self.v_zl_after = nn.Linear(dim, dim)

        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)
        self.zero_init_linear()

    def zero_init_linear(self):
        layers_to_handle = [
            self.q_zl_before,
            self.k_zl_before,
            self.v_zl_before,
            self.q_zl_after,
            self.k_zl_after,
            self.v_zl_after,
        ]
        for _layer in layers_to_handle:
            if isinstance(_layer, nn.Linear):
                nn.init.zeros_(_layer.weight)
                if _layer.bias is not None:
                    nn.init.zeros_(_layer.bias)

    def forward(self, x, freqs, camera_pose_embedding=None):
        if camera_pose_embedding is not None:

            q = self.norm_q(self.q(x))
            k = self.norm_k(self.k(x))
            v = self.v(x)
        # TODO uncomment
        # ----------------------------------------------------------------
        else:
            q = self.norm_q(
                self.q(x + self.q_zl_before(camera_pose_embedding))
                + self.q_zl_after(camera_pose_embedding)
            )
            k = self.norm_k(
                self.k(x + self.k_zl_before(camera_pose_embedding))
                + self.k_zl_after(camera_pose_embedding)
            )
            v = self.v(x + self.v_zl_before(camera_pose_embedding)) + self.v_zl_after(
                camera_pose_embedding
            )
        # --------------------------------------------------------------------

        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = self.attn(q, k, v)
        return self.o(x)


class CrossAttentionSeparate(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        eps: float = 1e-6,
        has_image_input: bool = False,
        rank=64,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        if rank > 0:
            # LoraLinear
            self.q_zl_before = LoRALinearLayer(dim, dim, rank=rank)
            self.q_zl_after = LoRALinearLayer(dim, dim, rank=rank)
        else:
            # Normal linear
            self.q_zl_before = nn.Linear(dim, dim, bias=False)
            self.q_zl_after = nn.Linear(dim, dim, bias=False)

        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)
        self.zero_init_linear()

    def zero_init_linear(self):
        layers_to_handle = [
            self.q_zl_before,
            self.q_zl_after,
        ]
        for _layer in layers_to_handle:
            if isinstance(_layer, nn.Linear):
                nn.init.zeros_(_layer.weight)
                if _layer.bias is not None:
                    nn.init.zeros_(_layer.bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor, camera_pose_embedding=None):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        # q = self.norm_q(self.q(x))
        # k = self.norm_k(self.k(ctx))
        # v = self.v(ctx)
        # TODO uncomment
        # -------------------------------------------------
        q = self.norm_q(
            self.q(x + self.q_zl_before(camera_pose_embedding))
            + self.q_zl_after(camera_pose_embedding)
        )
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        # -------------------------------------------------

        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        return self.o(x)


class GateModule(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual


class DiTBlock(nn.Module):
    def __init__(
        self,
        has_image_input: bool,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input
        )
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    # @torch.compile(mode='max-autotune')
    def forward(
        self,
        x,
        context,
        t_mod,
        freqs,
        camera_pose_embedding=None,
        cape=None,
        roce_cam_emb=None,
    ):
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        modulation = self.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        if t_mod.dim() == 4:
            # Per-frame timestep: t_mod is (B, F, 6, dim), so the factors come
            # out as (B, F, 1, dim) and broadcast over the spatial tokens as
            # soon as x is *viewed* as (B, F, h*w, dim).  Expanding the factors
            # to (B, F*h*w, dim) instead would materialise six full-size
            # tensors in every one of the 30 blocks, so always reshape x.
            factors = (modulation.unsqueeze(0) + t_mod).chunk(6, dim=2)
            frames = t_mod.shape[1]
        else:
            factors = (modulation + t_mod).chunk(6, dim=1)
            frames = 0
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = factors

        def _by_frame(y):
            b, n, d = y.shape
            return y.view(b, frames, n // frames, d)

        def _modulate(y, shift, scale):
            if not frames:
                return modulate(y, shift, scale)
            b, n, d = y.shape
            return modulate(_by_frame(y), shift, scale).reshape(b, n, d)

        def _gate(y, gate, residual):
            if not frames:
                return self.gate(y, gate, residual)
            b, n, d = residual.shape
            return y + (gate * _by_frame(residual)).reshape(b, n, d)

        input_x = _modulate(self.norm1(x), shift_msa, scale_msa)
        x = _gate(
            x, gate_msa, self.self_attn(input_x, freqs, cape, roce_cam_emb)
        )
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = _modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = _gate(x, gate_mlp, self.ffn(input_x))
        return x


class CameraDiTBlock(nn.Module):
    def __init__(
        self,
        has_image_input: bool,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6,
        camera_lora_rank=64,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttentionSeparate(
            dim, num_heads, eps, rank=camera_lora_rank
        )
        self.cross_attn = CrossAttentionSeparate(
            dim, num_heads, eps, has_image_input=has_image_input, rank=camera_lora_rank
        )
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(self, x, context, t_mod, freqs, camera_pose_embedding=None):
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=1)
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        # x = self.gate(x, gate_msa, self.self_attn(input_x, freqs))
        x = self.gate(
            x, gate_msa, self.self_attn(input_x, freqs, camera_pose_embedding)
        )

        x = x + self.cross_attn(self.norm3(x), context, camera_pose_embedding)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(
        self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float
    ):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        # `modulation` is (1, 2, dim), so t_mod must carry a broadcastable
        # modulation axis.  Callers pass the raw time embedding (B, dim), which
        # only happens to broadcast when B == 1 -- for B > 1 it aligns against
        # the size-2 axis and blows up.  Insert the missing axis here so a
        # per-sample timestep (as used by flow matching) works at any batch size.
        # A (B, F, dim) embedding means one timestep per frame; the modulation
        # then broadcasts over the spatial tokens of each frame.
        frames = 0
        modulation = self.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        if t_mod.dim() == 2:
            t_mod = t_mod.unsqueeze(1)
        else:
            frames = t_mod.shape[1]
            t_mod = t_mod.unsqueeze(2)
            modulation = modulation.unsqueeze(0)
        shift, scale = (modulation + t_mod).chunk(2, dim=-2)
        if frames:
            b, n, d = x.shape
            y = self.norm(x).view(b, frames, n // frames, d)
            return self.head((y * (1 + scale) + shift).reshape(b, n, d))
        x = self.head(self.norm(x) * (1 + scale) + shift)
        return x


class WanModel(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        add_rgb_head: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(
                approximate="tanh"), nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList(
            [
                DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)
                for _ in range(num_layers)
            ]
        )
        self.head = Head(dim, out_dim, patch_size, eps)
        self.rgb_head = None
        if add_rgb_head:
            self.enable_rgb_head(copy_from_head=True)
        head_dim = dim // num_heads
        # Kept as attributes so callers (model_fn_wan_video building CaPE pose
        # matrices) can size the frame-axis slice without re-deriving dim/num_heads.
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(
                1280, dim, has_pos_emb=has_image_pos_emb
            )  # clip_feature_dim = 1280
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(
                16, dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv

        # RoCE, created by enable_roce() and absent otherwise (arm droce).
        self.roce_mode = "off"
        self.roce_cam_embedding = None

        self.control_adapter = None
        self.add_control_adapter = add_control_adapter
        if add_control_adapter:
            self.control_adapter = SimpleAdapter(
                in_dim_control_adapter,
                dim,
                kernel_size=patch_size[1:],
                stride=patch_size[1:],
            )
        else:
            self.control_adapter = None

    def enable_rgb_head(self, copy_from_head: bool = True):
        if copy_from_head or self.rgb_head is None:
            self.rgb_head = copy.deepcopy(self.head)
        return self.rgb_head

    def enable_roce(self, cam_channels: int, mode: str = "qk_vo"):
        """Camera-conditioned RoPE phase on the h/w axes (ReDirector's RoCE).

        `cam_channels` is the channel count of the Plucker field the caller will
        hand to `roce_camera_tokens`.  The conv that turns that field into one
        token feature per patch is deliberately the SAME geometry as
        patch_embedding (kernel == stride == patch_size), because that -- and
        nothing else -- is what guarantees the phase of token i describes the same
        patch that x[i] came from.  Getting that wrong would be a pure silent
        failure: the model would train happily on a shuffled camera field.

        Parameter cost at the 1.3B config (dim 1536, 12 heads, 30 blocks): 2 x
        1536 x 504 x 30 = 46.4M in the phase MLPs plus cam_channels x 1536 x 4 in
        the conv.  Only the phase MLPs matter for behaviour; they are zero-init,
        so this call does not change a single output value.
        """
        if mode not in ROCE_MODES:
            raise ValueError(f"roce mode must be one of {ROCE_MODES}, got {mode!r}")
        if mode == "off":
            return self
        self.roce_mode = mode
        self.roce_cam_embedding = nn.Conv3d(
            cam_channels,
            self.dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        ).to(device=self.patch_embedding.weight.device,
             dtype=self.patch_embedding.weight.dtype)
        for block in self.blocks:
            block.self_attn.enable_roce(mode)
        return self

    def roce_camera_tokens(self, camera_latents: torch.Tensor):
        """[B, C, F, h, w] Plucker field -> [B, F*h'*w', dim], in patchify's order.

        The rearrange is character-for-character the one in `patchify`; keep them
        identical or the phase field silently transposes against the tokens.
        """
        if self.roce_cam_embedding is None:
            raise ValueError("enable_roce() has not been called on this model")
        y = self.roce_cam_embedding(camera_latents)
        return rearrange(y, "b c f h w -> b (f h w) c").contiguous()

    def patchify(
        self, x: torch.Tensor, control_camera_latents_input: torch.Tensor = None
    ):
        x = self.patch_embedding(x)
        if (
            self.control_adapter is not None
            and control_camera_latents_input is not None
        ):
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            # x = x[0].unsqueeze(0)
        grid_size = x.shape[2:]
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=grid_size[0],
            h=grid_size[1],
            w=grid_size[2],
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2],
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **kwargs,
    ):
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)

        if self.has_image_input:
            # print(f"x,y shape: {x.shape}, {y.shape if y is not None else 'None'}")
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)

        x, (f, h, w) = self.patchify(x)

        # NOTE: this is a second, independent construction of the same RoPE
        # frequency table as `model_fn_wan_video` in
        # diffsynth/pipelines/wan_video_new_determine.py, and it deliberately has
        # NO `rope_frame_ids` override: every pipeline in this repo calls
        # `model_fn_wan_video` rather than WanModel.forward, so adding the flag
        # here would create a path that silently ignores it.  If you ever route
        # training through this method, port the override before trusting the
        # shared_rope_frame_ids factor.
        freqs = (
            torch.cat(
                [
                    self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            )
            .reshape(f * h * w, 1, -1)
            .to(x.device)
        )

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        for block in self.blocks:
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x,
                            context,
                            t_mod,
                            freqs,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x,
                        context,
                        t_mod,
                        freqs,
                        use_reentrant=False,
                    )
            else:
                x = block(x, context, t_mod, freqs)

        hidden_states = x
        depth = self.head(hidden_states, t)
        depth = self.unpatchify(depth, (f, h, w))

        if hasattr(self, "rgb_head") and self.rgb_head is not None:
            rgb = self.rgb_head(hidden_states, t)
            rgb = self.unpatchify(rgb, (f, h, w))

            return depth, rgb

        return depth

    @staticmethod
    def state_dict_converter():
        return WanModelStateDictConverter()


class WanModelStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        rename_dict = {
            "blocks.0.attn1.norm_k.weight": "blocks.0.self_attn.norm_k.weight",
            "blocks.0.attn1.norm_q.weight": "blocks.0.self_attn.norm_q.weight",
            "blocks.0.attn1.to_k.bias": "blocks.0.self_attn.k.bias",
            "blocks.0.attn1.to_k.weight": "blocks.0.self_attn.k.weight",
            "blocks.0.attn1.to_out.0.bias": "blocks.0.self_attn.o.bias",
            "blocks.0.attn1.to_out.0.weight": "blocks.0.self_attn.o.weight",
            "blocks.0.attn1.to_q.bias": "blocks.0.self_attn.q.bias",
            "blocks.0.attn1.to_q.weight": "blocks.0.self_attn.q.weight",
            "blocks.0.attn1.to_v.bias": "blocks.0.self_attn.v.bias",
            "blocks.0.attn1.to_v.weight": "blocks.0.self_attn.v.weight",
            "blocks.0.attn2.norm_k.weight": "blocks.0.cross_attn.norm_k.weight",
            "blocks.0.attn2.norm_q.weight": "blocks.0.cross_attn.norm_q.weight",
            "blocks.0.attn2.to_k.bias": "blocks.0.cross_attn.k.bias",
            "blocks.0.attn2.to_k.weight": "blocks.0.cross_attn.k.weight",
            "blocks.0.attn2.to_out.0.bias": "blocks.0.cross_attn.o.bias",
            "blocks.0.attn2.to_out.0.weight": "blocks.0.cross_attn.o.weight",
            "blocks.0.attn2.to_q.bias": "blocks.0.cross_attn.q.bias",
            "blocks.0.attn2.to_q.weight": "blocks.0.cross_attn.q.weight",
            "blocks.0.attn2.to_v.bias": "blocks.0.cross_attn.v.bias",
            "blocks.0.attn2.to_v.weight": "blocks.0.cross_attn.v.weight",
            "blocks.0.ffn.net.0.proj.bias": "blocks.0.ffn.0.bias",
            "blocks.0.ffn.net.0.proj.weight": "blocks.0.ffn.0.weight",
            "blocks.0.ffn.net.2.bias": "blocks.0.ffn.2.bias",
            "blocks.0.ffn.net.2.weight": "blocks.0.ffn.2.weight",
            "blocks.0.norm2.bias": "blocks.0.norm3.bias",
            "blocks.0.norm2.weight": "blocks.0.norm3.weight",
            "blocks.0.scale_shift_table": "blocks.0.modulation",
            "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
            "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
            "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
            "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
            "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
            "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
            "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
            "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
            "condition_embedder.time_proj.bias": "time_projection.1.bias",
            "condition_embedder.time_proj.weight": "time_projection.1.weight",
            "patch_embedding.bias": "patch_embedding.bias",
            "patch_embedding.weight": "patch_embedding.weight",
            "scale_shift_table": "head.modulation",
            "proj_out.bias": "head.head.bias",
            "proj_out.weight": "head.head.weight",
        }
        state_dict_ = {}
        print(
            f"hash_state_dict_keys(state_dict): {hash_state_dict_keys(state_dict)}")
        for name, param in state_dict.items():
            if name in rename_dict:
                state_dict_[rename_dict[name]] = param
            else:
                name_ = ".".join(name.split(
                    ".")[:1] + ["0"] + name.split(".")[2:])
                if name_ in rename_dict:
                    name_ = rename_dict[name_]
                    name_ = ".".join(
                        name_.split(".")[:1]
                        + [name.split(".")[1]]
                        + name_.split(".")[2:]
                    )
                    state_dict_[name_] = param
        if hash_state_dict_keys(state_dict) == "cb104773c6c2cb6df4f9529ad5c60d0b":
            config = {
                "model_type": "t2v",
                "patch_size": (1, 2, 2),
                "text_len": 512,
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "window_size": (-1, -1),
                "qk_norm": True,
                "cross_attn_norm": True,
                "eps": 1e-6,
            }
        else:
            config = {}
        return state_dict_, config

    def from_civitai(self, state_dict):
        state_dict = {
            name: param
            for name, param in state_dict.items()
            if not name.startswith("vace")
        }
        print(
            f"hash_state_dict_keys(state_dict): {hash_state_dict_keys(state_dict)} from civitai"
        )

        if hash_state_dict_keys(state_dict) == "9269f8db9040a9d860eaca435be61814":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6,
            }
        elif hash_state_dict_keys(state_dict) == "aafcfd9672c3a2456dc46e1cb6e52c70":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
            }
        elif hash_state_dict_keys(state_dict) == "6bfcfb3b342cb286ce886889d519a77e":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
            }
        elif hash_state_dict_keys(state_dict) == "6d6ccde6845b95ad9114ab993d917893":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6,
            }
        elif hash_state_dict_keys(state_dict) == "6bfcfb3b342cb286ce886889d519a77e":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
            }
        elif hash_state_dict_keys(state_dict) == "349723183fc063b2bfc10bb2835cf677":
            # 1.3B PAI control
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6,
            }
        elif hash_state_dict_keys(state_dict) == "efa44cddf936c70abd0ea28b6cbe946c":
            # 14B PAI control
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
            }
        elif hash_state_dict_keys(state_dict) == "3ef3b1f8e1dab83d5b71fd7b617f859f":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
                "has_image_pos_emb": True,
            }
        elif hash_state_dict_keys(state_dict) == "70ddad9d3a133785da5ea371aae09504":
            # 1.3B PAI control v1.1
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6,
                "has_ref_conv": True,
            }
        elif hash_state_dict_keys(state_dict) == "26bde73488a92e64cc20b0a7485b9e5b":
            # 14B PAI control v1.1
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
                "has_ref_conv": True,
            }
        elif hash_state_dict_keys(state_dict) == "ac6a5aa74f4a0aab6f64eb9a72f19901":
            # 1.3B PAI control-camera v1.1
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 32,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6,
                "has_ref_conv": False,
                "add_control_adapter": True,
                "in_dim_control_adapter": 24,
            }
        elif hash_state_dict_keys(state_dict) == "b61c605c2adbd23124d152ed28e049ae":
            # 14B PAI control-camera v1.1
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 32,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
                "has_ref_conv": False,
                "add_control_adapter": True,
                "in_dim_control_adapter": 24,
            }
        else:
            config = {}
        return state_dict, config
