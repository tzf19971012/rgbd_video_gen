"""WanPureColorNVSTrainingModule — 纯 Wan-latent 单流 NVS 对照（判决实验）。

问题：geo-color 里 color 分支的模糊来自哪里？
  (a) 与 geo 流共享 trunk 的容量/梯度竞争，还是
  (b) "6 source 视图 -> 3 target 视角渲染"这个任务本身在当前数据/步数下就收敛慢。

本模块把 (a) 完全移除——单一 16ch Wan VAE latent 流：
  * latent 来自离线 Wan cache（re10k_ovf8_wan，与 geo-color 的 color 缓存同源）；
  * patch_embedding / head 直接 load 回 Wan2.1 预训练权重（16ch in/out 与
    预训练逐位同构，load 而非 fresh init——这是与 GAE 模块的本质差异，
    "完全照搬 Wan" 就是不动这两块）；
  * 损失 = 均匀 v-MSE（Wan / GAE-shipped 剖面，v_loss_t_eps=1.0 即 w=1）；
  * eval 用冻结 Wan VAE 解码 RGB（不产深度，深度指标自动关闭）。

判决：若它锐利 -> geo-color 的糊来自共享 trunk 竞争；若它同糊 ->
瓶颈是任务本身（数据/步数/低 sigma 采样），两线配置除通道数与投影
初始化外逐位一致。
"""
import torch

from .WanGAENVSTrainingModule import WanGAENVSTrainingModule


class WanPureColorNVSTrainingModule(WanGAENVSTrainingModule):
    def __init__(
        self,
        *args,
        gae_latent_channels=16,
        keep_pretrained_projections=True,
        **kwargs,
    ):
        # 与 GAE 模块相同的强制配置（单流、channel 布局、无深度/无 warp），
        # 但 latent 通道 = 16（Wan VAE）。concat 模式下相机条件是独立通道块，
        # 编码器 out_channels 由配置给出（老 run: 384），不能强制成 16。
        concat_mode = kwargs.get("view_condition_mode", "add") == "concat"
        kwargs["token_layout"] = "channel"
        kwargs["warp_mode"] = "none"
        kwargs["depth_flow_matching"] = False
        kwargs["rgb_flow_matching"] = True
        if not concat_mode:
            kwargs["view_condition_out_channels"] = int(gae_latent_channels)
        kwargs["clean_source_depth"] = False
        kwargs["per_frame_timestep"] = False
        super().__init__(
            *args, gae_latent_channels=int(gae_latent_channels), **kwargs
        )

        # GAE 父类把预训练投影换成了 fresh init（16ch 时形状相同）。纯 Wan
        # 模式把它们 load 回预训练权重：in_channels=16 与 out_dim=16 时逐位
        # 同构，这是"照搬 Wan"的字面含义。
        dit = self.pipe.dit
        pre = getattr(self, "_pretrained_wan_projections", None)
        if keep_pretrained_projections and pre is not None:
            missing = dit.patch_embedding.load_state_dict(
                pre["patch_embedding"], strict=True)
            missing_h = dit.head.load_state_dict(pre["head"], strict=True)
            self.accelerator.print(
                "[pure-color] patch_embedding / head RESTORED to pretrained "
                f"Wan2.1 weights (16ch in/out, load_state_dict {missing} {missing_h})"
            )
        elif keep_pretrained_projections:
            raise RuntimeError(
                "pure-color needs _pretrained_wan_projections (snapshotted by "
                "the GAE parent) to restore the Wan projections"
            )
        self.keep_pretrained_projections = bool(keep_pretrained_projections)

        # concat 模式（老 run 机制）：相机条件作为独立通道块拼在 latent 之后，
        # 参考父类 _expand_patch_embedding——预训练 latent 块原样保留，新增
        # plucker 块 zero-init（step 0 时网络输出与纯 Wan 严格一致）。
        if concat_mode:
            plucker_ch = self.view_condition_encoder.out_channels
            conv = dit.patch_embedding
            old_in = conv.in_channels
            new_conv = torch.nn.Conv3d(
                old_in + plucker_ch,
                conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                bias=conv.bias is not None,
            ).to(device=conv.weight.device, dtype=conv.weight.dtype)
            with torch.no_grad():
                new_conv.weight.zero_()
                new_conv.weight[:, :old_in].copy_(conv.weight)
                if new_conv.bias is not None and conv.bias is not None:
                    new_conv.bias.copy_(conv.bias)
            dit.patch_embedding = new_conv
            self.accelerator.print(
                f"[pure-color] concat: patch_embedding widened {old_in} -> "
                f"{old_in + plucker_ch} (+{plucker_ch} plucker block, "
                "zero-initialised; pretrained latent block preserved)"
            )

    # -- eval decode：Wan VAE（无深度） ---------------------------------------

    def _decode_views(self, z_standardized, height, width):
        """[B, 16, V, h, w] standardized -> (rgb [B,V,3,H,W], None)。

        latent 网格在 sample_latents 里已裁回 Wan 原生 (47,84)（缓存即该网格），
        解码链路与 geo-color 的 _decode_color_views 完全一致（同一冻结 VAE）。
        """
        mean = self.gae_latent_mean.to(z_standardized.device).reshape(1, -1, 1, 1)
        std = self.gae_latent_std.to(z_standardized.device).reshape(1, -1, 1, 1)
        B, C, V = z_standardized.shape[:3]
        rgbs = []
        for b in range(B):
            for v in range(V):
                lat = z_standardized[b, :, v].unsqueeze(0)
                lat = lat * (std + 1e-6) + mean          # destandardize
                lat = lat.to(torch.bfloat16)             # [1,16,h,w]
                dec = self.pipe.vae.decode(
                    lat.unsqueeze(2), device=str(z_standardized.device))
                img = ((dec.clamp(-1, 1) + 1) / 2)[0, :, 0].float()  # [3,H',W']
                img = img[:, :height, :width]
                if img.shape[1] < height or img.shape[2] < width:
                    img = torch.nn.functional.pad(
                        img, (0, max(0, width - img.shape[2]),
                              0, max(0, height - img.shape[1])),
                        mode="replicate")
                rgbs.append(img.cpu())
        rgb = torch.stack(rgbs).reshape(B, V, 3, height, width)
        return rgb, None
