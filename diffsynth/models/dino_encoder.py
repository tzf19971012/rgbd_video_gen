import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class Dnv2Config:
    """Minimal config container for the local Dnv2 checkpoint."""

    def __init__(self, cfg: dict):
        self.hidden_size = cfg.get("hidden_size", 768)
        self.num_hidden_layers = cfg.get("num_hidden_layers", 12)
        self.num_attention_heads = cfg.get("num_attention_heads", 12)
        self.intermediate_size = cfg.get("intermediate_size", 3072)
        self.hidden_act = cfg.get("hidden_act", "gelu")
        self.hidden_dropout_prob = cfg.get("hidden_dropout_prob", 0.0)
        self.attention_probs_dropout_prob = cfg.get("attention_probs_dropout_prob", 0.0)
        self.layer_norm_eps = cfg.get("layer_norm_eps", 1e-6)
        self.num_channels = cfg.get("num_channels", 3)
        self.image_size = cfg.get("image_size", 518)
        self.patch_size = cfg.get("patch_size", 14)
        self.num_patches = (self.image_size // self.patch_size) ** 2


class Dnv2PatchEmbeddings(nn.Module):
    def __init__(self, config: Dnv2Config):
        super().__init__()
        self.projection = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )


class Dnv2Embeddings(nn.Module):
    def __init__(self, config: Dnv2Config):
        super().__init__()
        self.patch_embeddings = Dnv2PatchEmbeddings(config)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.mask_token = nn.Parameter(torch.zeros(1, config.hidden_size))
        num_positions = config.num_patches + 1
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, num_positions, config.hidden_size)
        )

    def interpolate_pos_encoding(self, embeddings, height, width):
        """Interpolate 1D position embeddings to arbitrary HxW."""
        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1
        if num_patches == num_positions and height == width:
            return self.position_embeddings
        class_pos_embed = self.position_embeddings[:, 0:1]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]
        h0 = height // self.patch_embeddings.projection.kernel_size[0]
        w0 = width // self.patch_embeddings.projection.kernel_size[1]
        # (1, N, D) -> (1, D, N) -> (1, D, h, w)
        patch_pos_embed = (
            patch_pos_embed.transpose(1, 2)
            .view(1, dim, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)))
        )
        patch_pos_embed = F.interpolate(
            patch_pos_embed, size=(h0, w0), mode="bilinear", align_corners=False
        )
        patch_pos_embed = patch_pos_embed.view(1, dim, -1).transpose(1, 2)
        return torch.cat([class_pos_embed, patch_pos_embed], dim=1)

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = True):
        B, C, H, W = pixel_values.shape
        embeddings = self.patch_embeddings.projection(pixel_values)
        embeddings = embeddings.flatten(2).transpose(1, 2)  # (B, N, D)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        embeddings = torch.cat([cls_tokens, embeddings], dim=1)

        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, H, W)
        else:
            embeddings = embeddings + self.position_embeddings
        return embeddings


class Dnv2SelfAttention(nn.Module):
    def __init__(self, config: Dnv2Config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        B, L, _ = x.shape
        x = x.view(B, L, self.num_attention_heads, self.attention_head_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        q = self.transpose_for_scores(self.query(hidden_states))
        k = self.transpose_for_scores(self.key(hidden_states))
        v = self.transpose_for_scores(self.value(hidden_states))
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.attention_head_size)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(out.shape[0], out.shape[1], self.all_head_size)
        return out


class Dnv2SelfOutput(nn.Module):
    def __init__(self, config: Dnv2Config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class Dnv2Attention(nn.Module):
    def __init__(self, config: Dnv2Config):
        super().__init__()
        self.attention = Dnv2SelfAttention(config)
        self.output = Dnv2SelfOutput(config)

    def forward(self, hidden_states):
        attn_output = self.attention(hidden_states)
        attn_output = self.output(attn_output)
        return attn_output


class Dnv2MLP(nn.Module):
    def __init__(self, config: Dnv2Config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class Dnv2LayerScale(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        # The original checkpoint uses the unusual parameter name "gmma".
        self.gmma = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, hidden_states):
        return self.gmma * hidden_states


class Dnv2Layer(nn.Module):
    def __init__(self, config: Dnv2Config):
        super().__init__()
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = Dnv2Attention(config)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = Dnv2MLP(config)
        self.layerscale1 = Dnv2LayerScale(config.hidden_size)
        self.layerscale2 = Dnv2LayerScale(config.hidden_size)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.layernorm_before(hidden_states)
        hidden_states = self.attention(hidden_states)
        hidden_states = self.layerscale1(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layernorm_after(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.layerscale2(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Dnv2Encoder(nn.Module):
    def __init__(self, config: Dnv2Config):
        super().__init__()
        self.blocks = nn.ModuleList([Dnv2Layer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states):
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return hidden_states


class Dnv2Model(nn.Module):
    def __init__(self, config: Dnv2Config):
        super().__init__()
        self.config = config
        self.embeddings = Dnv2Embeddings(config)
        self.encoder = Dnv2Encoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = True, return_dict: bool = True):
        embeddings = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        encoder_outputs = self.encoder(embeddings)
        sequence_output = self.layernorm(encoder_outputs)
        if not return_dict:
            return (sequence_output,)
        return type("Dnv2Output", (), {"last_hidden_state": sequence_output})()


class DINOV2Encoder(nn.Module):
    """
    Frozen DINOv2 / Dnv2 feature extractor.

    Input:  (N, 3, H, W) tensor in [0, 1] range.
    Output: (N, h*w, D) patch tokens, where h = H//14, w = W//14.
            The CLS token is dropped.
    """

    def __init__(self, model_name="facebook/dinov2-base"):
        super().__init__()
        self.model_name = model_name

        # Local Dnv2 checkpoint (e.g. ./dino_v2)
        if os.path.isdir(model_name) and os.path.exists(os.path.join(model_name, "config.json")):
            with open(os.path.join(model_name, "config.json"), "r") as f:
                cfg = json.load(f)
            config = Dnv2Config(cfg)
            self.model = Dnv2Model(config)
            from safetensors.torch import load_file
            state_dict = load_file(os.path.join(model_name, "model.safetensors"))
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"DINOv2 warning: missing keys {missing}")
            if unexpected:
                print(f"DINOv2 warning: unexpected keys {unexpected}")
        else:
            from transformers import Dinov2Model
            self.model = Dinov2Model.from_pretrained(model_name)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, images: torch.Tensor):
        # DINOv2 expects fp32 inputs
        images = images.float()
        outputs = self.model(
            images,
            interpolate_pos_encoding=True,
            return_dict=True,
        )
        # last_hidden_state: (N, 1 + h*w, D)
        return outputs.last_hidden_state[:, 1:].contiguous()
