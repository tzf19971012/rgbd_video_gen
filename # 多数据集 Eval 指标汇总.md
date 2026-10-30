# 多数据集 Eval 指标汇总

基于 `logs/multi4_nvs_lora/checkpoint-step-*/eval_multi/multi_dataset_metrics.json` 统计。

---

## RGB PSNR（越高越好）

```
```plaintext
┌───────────────────────────┬──────────┬──────────────────────┐
│ 数据集                    │ 最佳值   │ 最佳 Step            │
├───────────────────────────┼──────────┼──────────────────────┤
│ blendedmvs                │ 15.4450  │ step-30000           │
│ gtasfm                    │ 14.5417  │ step-21500           │
│ gtav                      │ 14.8828  │ step-21500           │
│ matrixcity_chunk00        │ 18.0680  │ step-27000           │
│ matrixcity_chunk01        │ 17.4367  │ step-22000           │
│ tartanair                 │ 15.5140  │ step-21500           │
└───────────────────────────┴──────────┴──────────────────────┘

```
```

## RGB LPIPS（越低越好）

```
```plaintext
┌───────────────────────────┬──────────┬──────────────────────┐
│ 数据集                    │ 最佳值   │ 最佳 Step            │
├───────────────────────────┼──────────┼──────────────────────┤
│ blendedmvs                │ 0.4484   │ step-30000           │
│ gtasfm                    │ 0.4186   │ step-20000           │
│ gtav                      │ 0.3630   │ step-27000           │
│ matrixcity_chunk00        │ 0.3015   │ step-27000           │
│ matrixcity_chunk01        │ 0.3195   │ step-30000           │
│ tartanair                 │ 0.4591   │ step-21500           │
└───────────────────────────┴──────────┴──────────────────────┘

```
```

## Depth AbsRel（越低越好）

```
```plaintext
┌───────────────────────────┬──────────┬──────────────────────┐
│ 数据集                    │ 最佳值   │ 最佳 Step            │
├───────────────────────────┼──────────┼──────────────────────┤
│ blendedmvs                │ 0.0606   │ step-30000           │
│ gtasfm                    │ 0.1712   │ step-16500           │
│ gtav                      │ 0.4683   │ step-23000           │
│ matrixcity_chunk00        │ 0.1468   │ step-29000           │
│ matrixcity_chunk01        │ 0.1961   │ step-19500           │
│ tartanair                 │ 3.5395   │ step-26000           │
└───────────────────────────┴──────────┴──────────────────────┘

```
```

> 
> 注：TartanAir 的 Depth AbsRel 值偏高是因为其深度尺度范围与其他数据集不同，该指标跨数据集不可直接对比。

## Depth δ₁（越高越好）

```
```plaintext
┌───────────────────────────┬──────────┬──────────────────────┐
│ 数据集                    │ 最佳值   │ 最佳 Step            │
├───────────────────────────┼──────────┼──────────────────────┤
│ blendedmvs                │ 0.9507   │ step-30000           │
│ gtasfm                    │ 0.8071   │ step-20000           │
│ gtav                      │ 0.6590   │ step-21500           │
│ matrixcity_chunk00        │ 0.8399   │ step-27000           │
│ matrixcity_chunk01        │ 0.8187   │ step-31000           │
│ tartanair                 │ 0.5732   │ step-20000           │
└───────────────────────────┴──────────┴──────────────────────┘

```
```

---

# 实验配置汇总

基于基座模型 [Wan-AI/Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) 的 RGB-D Novel View Synthesis (NVS) 实验。

---

## 硬件与训练框架

| 平台 | 分布式方式 | 进程数 | 硬件 |
| --- | --- | --- | --- |
| CUDA | accelerate launch | 1 / 4 / 8 | GPU |
| Ascend NPU | torchrun | 8 | NPU (HCCL_BUFFSIZE=200) |

混合精度统一使用 **BF16**。

---

## 共同架构参数

| 参数 | 值 |
| --- | --- |
| 训练范式 | Joint RGB-D Rectified Flow Matching（速度回归） |
| 分辨率 | 384 × 672 |
| 源视图 / 目标视图 | 6 / 3 |
| VAE 编码 | independent（逐视图独立） |
| Token 布局 | sequence（RGB 与 Depth 分开为独立 token） |
| Latent 缩放 RGB / Depth | 1.94 / 1.68 |
| Warp 模式 | splat, two-pass, dropout=0.5 |
| 视差归一化 | shared_scale, quantile=0.98, headroom=1.5 |
| 视角条件 | Plucker 坐标，pixel 分辨率，concat, scale=8 |
| Flow shift | 5.0, logit_normal 采样 |
| Depth 头初始化 | pretrained（从 Wan2.1 预训练权重） |
| 损失 | RGB MSE + Depth MSE, 权重各 1.0 |
| Depth 一致性损失 | 关闭 |
| Batch size / Grad accum | 1 / 4 |
| DataLoader workers | 4 |
| 优化器 warmup | 100 steps |
| 梯度裁剪 | 开启 |
| Gradient checkpointing | 开启 |
| Seed | 42 |

---

## 实验一览

### 实验 1: multi4_nvs_lora（基线 LoRA）

| 字段 | 值 |
| --- | --- |
| 配置文件 | `train_config/normal_config/multi4_nvs_lora.yaml` |
| 训练方式 | LoRA, rank=128, 模块 =`q,k,v,o,ffn.0,ffn.2` |
| 学习率 | 1e-4,**constant**（无衰减） |
| 最大步数 | 500k |
| 数据集 | BlendedMVS / GTAV_540(train) / MatrixCity_chunk00 / TartanAir |
| 视角顺序 | random |
| 输出路径 | `logs/multi4_nvs_lora` |
| 启动脚本 | `train_multi4_nvs_lora.sh` / `yundao_all_data_nvs_lora.sh`（NPU） |
| 指标 | 见上方表格 |

---

### 实验 2: multi4_nvs_lora512（高秩 LoRA）

| 字段 | 值 |
| --- | --- |
| 配置文件 | `train_config/normal_config/multi4_nvs_lora512.yaml` |
| 相对实验 1 的差异 | LoRA rank**128 → 512** |
| 初始化 | 从`multi4_nvs_lora/checkpoint-step-2500` 加载权重（不加载优化器） |
| 输出路径 | `logs/multi4_nvs_lora512` |
| 其余 | 与实验 1 一致 |

---

### 实验 3: multi4_nvs_lora_lrdecay（LoRA + 余弦衰减）

| 字段 | 值 |
| --- | --- |
| 配置文件 | `train_config/normal_config/multi4_nvs_lora_lrdecay.yaml` |
| 相对实验 1 的差异 | constant LR →**cosine decay over 50k steps, lr_min=1e-6** |
| 初始化 | 从`multi4_nvs_lora/checkpoint-step-2500` 加载权重，global_step=2500 |
| 目的 | 缓解 constant LR 下训练后期 loss 震荡 |
| 输出路径 | `logs/multi4_nvs_lora_lrdecay` |
| 其余 | 与实验 1 一致 |

---

### 实验 4: multi4_nvs_fullft（全量微调）

| 字段 | 值 |
| --- | --- |
| 配置文件 | `train_config/normal_config/multi4_nvs_fullft.yaml` |
| 训练方式 | **全量微调**（train_all=true） |
| 学习率 | **2e-5**, constant |
| 最大 checkpoint | 3（全量权重占用较大） |
| 输出路径 | `logs/multi4_nvs_fullft` |
| 启动脚本 | `train_multi4_nvs_fullft.sh` / `yundao_all_data_nvs_fullft.sh`（NPU） |
| 其余 | 与实验 1 一致 |

---

### 实验 5: multi4_nvs_fullft_lrdecay（全量微调 + 余弦衰减）

| 字段 | 值 |
| --- | --- |
| 配置文件 | `train_config/normal_config/multi4_nvs_fullft_lrdecay.yaml` |
| 相对实验 4 的差异 | constant LR →**cosine decay over 50k steps, lr_min=1e-6** |
| 初始化 | 从`multi4_nvs_fullft/checkpoint-step-2000` 加载权重，global_step=2000 |
| 输出路径 | `logs/multi4_nvs_fullft`（与实验 4 共享目录） |
| 其余 | 与实验 4 一致 |

---

### 实验 6: dl3dv_lora_video（DL3DV 领域微调）

| 字段 | 值 |
| --- | --- |
| 配置文件 | `train_config/normal_config/dl3dv_lora_video.yaml` |
| 数据集 | **DL3DV_WAI** 单数据集（`/efs-mi-east4-2/datasets/DL3DV_WAI`） |
| 视角顺序 | **video**（按时间顺序采样） |
| 场景划分 | `split_exclude_eval200.json`，排除 200 个 eval 场景 |
| 初始化 | 从`multi4_nvs_lora/checkpoint-step-32000` 加载权重 |
| 最大步数 | 300k |
| 输出路径 | `logs/dl3dv_lora_from32000` |
| 其余 | LoRA rank=128, 与实验 1 一致 |

---

## 推理配置

| 配置文件 | 数据集 | 用途 |
| --- | --- | --- |
| `train_config/normal_config/gtasfm_infer.yaml` | GTA-SfM | 基于 multi4_nvs_lora checkpoint 推理 |
| `train_config/normal_config/matrixcity_infer.yaml` | MatrixCity chunk01 | 基于 multi4_nvs_lora checkpoint 推理 |

---

## 实验演进关系

```
```plaintext
multi4_nvs_lora (LoRA rank=128, constant LR)   ← 基线
    ├── step-2500 → multi4_nvs_lora512 (rank=512)      ← 高秩探索
    ├── step-2500 → multi4_nvs_lora_lrdecay (cosine LR) ← 减缓震荡
    └── step-32000 → dl3dv_lora_video (video order)     ← 领域迁移

multi4_nvs_fullft (full fine-tune, constant LR)
    └── step-2000 → multi4_nvs_fullft_lrdecay (cosine LR)

```
```

---

## 训练脚本速查

| 脚本 | 对应配置 | 平台 |
| --- | --- | --- |
| `train_script/train_multi4_nvs_lora.sh` | multi4_nvs_lora | 单卡 /accelerate |
| `train_script/train_multi4_nvs_lora512.sh` | multi4_nvs_lora512 | 单卡 /accelerate |
| `train_script/train_multi4_nvs_fullft.sh` | multi4_nvs_fullft | 单卡 /accelerate |
| `train_script/train_dl3dv_lora.sh` | dl3dv_lora_video | 单卡 /accelerate |
| `train_script/yundao_all_data_nvs_lora.sh` | multi4_nvs_lora | NPU 8 卡 /torchrun |
| `train_script/yundao_all_data_nvs_lora512.sh` | multi4_nvs_lora512 | NPU 8 卡 /torchrun |
| `train_script/yundao_all_data_nvs_lora128_lrdecay.sh` | multi4_nvs_lora_lrdecay | NPU 8 卡 /torchrun |
| `train_script/yundao_all_data_nvs_fullft.sh` | multi4_nvs_fullft | NPU 8 卡 /torchrun |

