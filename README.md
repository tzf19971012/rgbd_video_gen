# Car_rgbd_video_gen

基于 Wan2.1-T2V-1.3B DiT + GAE-64 几何 latent 的多视角 NVS / 相机可控视频生成。
主干 LoRA 微调,6 源视角 → 3 目标视角,Plucker 射线场做相机条件注入。

## 整体 Pipeline

```
RealEstate10K (8 场景 overfit / 全量 7884 场景)
        │  离线缓存
        ▼
GAE-64 latent cache (utils/cache_gae_latents.py, 每视角 64x27x48, 标准化)
        │
        ▼
┌─────────────────────────── DiT 输入组装 (channel layout) ───────────────────────────┐
│  source 6 views: clean latent, timestep 钉 0 (GAE "evidence, not state")            │
│  target 3 views: x_t = (1-σ)z + σ·ε, timestep = σ                                  │
│  相机条件 Plucker ray field:                                                       │
│    add 模式(旧)   6ch@latent → 3 层 CNN 瓶颈 → 加进 latent                          │
│    concat 模式(新) pixel 栅格生成 → PixelUnshuffle → 独立通道块进 patch_embedding  │
│      · patch1 + pixel_scale=14 → 1176ch/latent 像素 (27x48 原生网格)                │
│      · upsample2 + pixel_scale=7 → 294ch (54x96 工作网格)                           │
│    CFG 训练端: camera_drop 0.2 / ref_drop 0.2 (concat 下 strip plucker 块)          │
└──────────────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
Wan2.1-1.3B DiT + LoRA r128 (q,k,v,o,ffn.0,ffn.2)
  patch_embedding 重铸 Conv3d(64+plucker_ch → 1536), head out_dim=64
  RoPE frame axis = view id
        │
        ▼
训练目标: rectified-flow v-MSE (v_loss_t_eps=1.0, GAE x-pred 等价加权)
  σ ~ logit_normal(mean=-1.5, std=1.0), flow shift 4.5 (单视角算法, train/eval 自洽)
  可选 RGB aux loss: x_pred → 冻结 GAE RGBHead → 目标视角 L1 (aux0 臂 = 关闭)
        │
        ▼
Eval: 25 步 ODE (shift 4.5), PSNR / LPIPS(alex) / depth(abs_rel, delta1)
诊断: diag_pose_sensitivity(swap_tgt 等 4-6 变体, 按覆盖度分层)
      traj 轨迹外推可视化 (infer_re10k_bev_trajectory / make_traj_videos)
```

### 训练入口与配置

| 臂 | 配置 | 说明 |
|---|---|---|
| 单流 aux0 patch1 (anchor) | `gae_ovf_re10k_aux0_patch1.yaml` | 8 场景, LoRA, aux 关闭 |
| patch1 concatpixel + drop | `gae_ovf_re10k_patch1_concatpixel_drop.yaml` | concat@14 + camera_drop 0.2 + eval_cfg 1.5 |
| 全量 re10k patch1 (集群) | `gae_re10k_patch1.yaml` + `train_script/yundao_train_gae_patch1_re10k.sh` | 7884 场景, 支持 `YUNDAO_RESUME_CKPT` 续训 |
| 双流 geocolor v1/v3 | `geo_color_ovf_re10k_lora*.yaml` | geo 64ch + color 16ch, v3 = token 级合并 |
| purecolor | `gae_ovf_re10k_purecolor_lora.yaml` | Wan-VAE latent 单流对照 |

单卡续训(如 patch1):`CKPT=...checkpoint-step-N STEP=N NPUS=<卡> bash scripts/launch_local_1npu_patch1_resume.sh`

## 已解决的关键问题

1. **生成发糊(灰 mush)根因 — flow shift 虚高**
   `gae_auto_flow_shift` 把 V=9 乘进 shift(13.5),σ 分布塌进高噪声段(98% step σ>0.79),
   模型学到"从近纯噪声回归条件均值"——PSNR 最高也最糊。
   修复:shift 降回 4.5(GAE 官方单视角)+ eval 25 步 + logit_mean=-1.5。
   详见 `reports/20261031_gae_re10k_rgb_aux_blur_shift_分析报告.md`。
2. **纯 latent MSE 下目标塌缩成灰色条件均值**
   加 RGB aux loss(冻结 RGBHead 解码 L1);后续消融证明 aux5 净负收益已停,
   aux0(关闭)+ patch1 到 5800 步创四项最好 —— aux 的价值依赖 σ 门控标定。
3. **patchify 低通 — (1,2,2) 每 token 揉 4 个 latent 像素**
   B1(patch1, (1,1,1)) / B2(upsample2) 双臂判决:patch1 不糊,token 1296/view。
4. **集群 eval LPIPS 全 NaN** — pod 环境缺 lpips 包,pip install 后恢复。
5. **concat 模式两处工程坑**
   patch_embedding 重铸写死 64 进 → concat 下运行时必崩,已按 plucker 通道数加宽
   (plucker 切片零初始化,step 0 数值等价 camera-free);训练端 camera_drop 原本
   只 strip 加法条件,concat 下是静默 no-op,已改为同时 strip plucker 通道块。
6. **断点续训工具链** — CheckpointManager 状态合并恢复、单卡 bs1×accum8(有效 batch
   8 不变)、ROMA 多机拓扑校验(EI0015 同物理卡防护)、`YUNDAO_RESUME_CKPT` 精确恢复。

## 已面临的问题

1. **相机控制能力弱(核心问题)**
   诊断史:显式 warp 路径 shuffle 掉 3.37 dB(模型确实读几何),而抽象 Plucker
   路径从未被证实被读——add 注入(6ch 过 CNN 瓶颈、仅输入端加一次)最易被 LoRA
   无视。当前对策:concat@pixel(每通道独立权重)+ camera_drop/pose-CFG;
   判据 = `tools/diag_pose_sensitivity` 的 swap_tgt 恶化量。若仍弱,升级路线:
   RoCE(逐 block 相机 token,代码已就位)或 CaPE(位姿进 RoPE)。
2. **数据规模的混杂**
   8 场景 overfit 下相机控制"测不准"(模型可靠记忆混过位姿信号);全量 re10k
   cache 已建,集群 run(7884 场景)训到 3300+ 步待续。需要 overfit vs 全量的
   swap_tgt 对比来归因"注入 vs 数据"。
3. **PSNR 与观感背离**
   PSNR 最优解 = 条件均值 = 糊;判定一律以 LPIPS + 轨迹可视化优先,
   PSNR 只作参考(见 c-sweep 报告 §9)。
4. **评估统计功效有限** — 冻结 eval 仅 8 样本 × 3 目标视角;swap_tgt 差值 <0.2 dB
   级别的结论需扩样本再下。
5. **pose-CFG 未验证** — drop 0.2 训出的无条件分支质量决定 eval_cfg 1.5 是放大
   控制还是放大伪影,待 concatpixel_drop 臂出 ckpt 后测量。
6. **双流臂的色流主导风险** — geocolor v3 中 geo token 仅占 336/1344,
   需盯 geo_mse 相对下降速度。
