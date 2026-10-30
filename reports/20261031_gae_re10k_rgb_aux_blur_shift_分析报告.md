# GAE re10k rgb_aux 训练分析与降 shift 复训方案

日期:2026-10-31
分析对象:`logs/gae_re10k_rgb_aux20261030_022403/train.log`(step 3010 → 12090+,集群在跑,尚未停)
代码基线:本仓库 `ee0da02`(旧仓 `rgbd_video_gen` 的全量快照)

---

## 1. 实验背景

- **任务**:GAE-64 latent 后端 + Wan2.1-1.3B DiT LoRA(r=128)的 NVS 训练;
  6 源视角 → 3 目标视角,条件注入 = Plucker ray map("add" 模式)。
- **数据**:re10k 全量(cache 7884 场景 × 16 views),resolution 378×672(latent 27×48)。
- **损失**:latent flow-matching MSE + RGB aux loss(权重 1.0,σ<0.9 门控,x_pred →
  冻结 GAE RGBHead → 目标视角 L1)。加入 aux 的动机:纯 latent MSE 下 target latent
  塌缩成灰色条件均值(bmvs arm A:PSNR 升、LPIPS 0.645→0.791 崩)。
- **本 run 链路**:20261029_140212 训到 step 3000 → 本 run 从 checkpoint-step-3000 续训。

## 2. 训练 / Eval 趋势(step 3010–12090)

按 1000 步分箱(均值):

| step | loss | rgb_aux | 备注 |
|------|------|---------|------|
| 3000 | 0.427 | 0.073 | |
| 5000 | 0.391 | 0.070 | |
| 7000 | 0.370 | 0.067 | |
| 9000 | 0.353 | 0.063 | |
| 11000 | 0.357 | 0.065 | 轻微平台/回弹 |
| 12000 | 0.350 | 0.062 | 恢复下降 |

冻结 eval(8 样本,每 1000 步,全部单调改善):

| step | PSNR | delta1 | abs_rel | abs_rel_aligned |
|------|------|--------|---------|-----------------|
| 4000 | 15.21 | 0.775 | 0.190 | 0.204 |
| 8000 | 16.92 | 0.841 | 0.155 | 0.156 |
| 12000 | 17.72 | 0.872 | 0.149 | 0.138 |

`src_depth_abs_rel=0.0883` 恒定 = 输入端 DA3 深度,不随训练变,正常。
**可视化结论:指标全线变好,但生成图持续发糊(灰 mush),PSNR 与观感背离。**

## 3. 模糊根因:auto flow shift 把整个 σ 体系压进高噪声段

### 3.1 shift 怎么来的

`gae_auto_flow_shift=true` 时 shift = sqrt(V·h·w·C / 4096) = sqrt(9×27×48×64/4096)
= **13.5**(log 第一行/第二十行可证;GAE 官方 train_flow.py 只按**单视角**
latent 算,即 sqrt(27×48×64/4096) = 4.5,当前代码 `WanGAENVSTrainingModule.py:146`
已改回单视角,但产生本 run 的版本把 V=9 乘了进去)。

### 3.2 训练端:σ 分布塌缩(Monte-Carlo 模拟 2e5 样本,logit_normal(0,1))

| 训练 shift | σ_mean | p10 | p50 | p90 | P(σ<0.6) |
|---|---|---|---|---|---|
| **13.5(本 run)** | **0.903** | 0.79 | 0.93 | 0.98 | **1.4%** |
| 4.5(GAE 官方) | 0.78 | 0.55 | 0.82 | 0.94 | 13.6% |

log 里 sigma_mean≈0.90±0.03 与理论 0.903 吻合:模型 98% 的 step 在 σ>0.79 上训练,
学到的本质是"从近纯噪声直接回归条件均值"——**这正是 PSNR 最高、也最糊的解**
(回归均值最小化 MSE)。旧 c-sweep(`tools/analyze_c_shift_scan.py`)已量化:
训练 shift 5→1 会在 pinned sampler 上掉 1.30–1.86 dB PSNR —— 高 PSNR 与糊同源,
PSNR 在该设定下是会骗人的指标。

### 3.3 Eval 端:同一 shift + 仅 4 步 ODE,低噪声细化只分到 1 步

eval schedule 与训练共用 shift(eval_flow_shift 被 auto 吸成 13.5),
`eval_num_inference_steps=4` 时 σ 序列为:

- **shift 13.5**:`[1.0, 0.976, 0.931, 0.818, 0.0]` — 前 3 步泡在 σ>0.8(v 预测基本
  不变),最后一步从 0.818 一把 Euler 到 0,所有高频细节都由模型从未训练过的
  区间外推 → mush。
- shift 0.5 对照:`[1.0, 0.60, 0.333, 0.143, 0.0]` — 低噪声段有 3 步细化。

### 3.4 RGB aux 为什么救不回来

- σ<0.9 门控只覆盖 **~34%** step(理论 34.1%,与 sigma_mean≈0.9 互证);
- 触发时 σ 仍高达 0.85+,x_pred = x_t − σ·v_pred 仍是均值为主,L1 信号弱,
  且每步只解码 1 个 target view;
- 根本限制:aux 是回归侧正则,**补不回 DiT 从未训练过的低 σ 细化能力**。
  本 run rgb_aux 0.073→0.062 方向正确,但只能缓解、无法根治。

## 4. tgt_rgb_lpips=NaN 根因(已定位)

集群 train.log 第 136 行:`WARNING: lpips is not installed, tgt_rgb_lpips will be NaN`
—— **集群 pod 的 conda 环境缺 lpips 包**,`_get_lpips` 返回 None → 指标列表为空 → NaN。
非 NPU 数值问题:本机 `.venv` 已装 lpips 0.1.4(alexnet 权重已缓存),
在 910B3 上实测前向正常(随机图像对 0.214,相同图 0.0)。
集群侧需在 pod 环境里 `pip install lpips`;本地新 run 不受影响。

## 5. 本次变更(新仓库 `rgbd_video_gen_gae`)

新配置 `train_config/normal_config/gae_re10k_lora_rgb_aux_shift45.yaml`,
与 `gae_re10k_lora_rgb_aux.yaml` 的差异(其余协议全部保持):

| 项 | 旧 | 新 | 理由 |
|---|---|---|---|
| gae_auto_flow_shift | true(→13.5) | **false** | 摘掉 V 因子 |
| rgb_flow_shift | 5.0(被 auto 覆盖) | **4.5** | GAE 官方单视角算法,σ_mean 0.90→0.78,P(σ<0.6) 1.4%→13.6% |
| gae_rgb_t_max | 0.9(触发 ~34%) | **0.6**(触发 ~14%) | σ 分布下移后重新标定;aux 集中在低 σ 段,x_pred 最可信处监督,解码开销也低于旧 run |
| eval_flow_shift | null(=13.5) | **null(=4.5,随训练)** | 监控 eval 用模型自己的工作点;跨 arm 公平对比用 eval_script pinned sampler(shift 0.5 / 25 步)单独扫 |
| eval_num_inference_steps | 4 | **25** | 低 σ 段细化步数从 1 → ~10 |
| 运行环境 | 集群 1 进程(bs16) | **本机 3 卡 DDP(bs16×3=48)** | 本地快速迭代 |

其余不变:LoRA r=128 / lr 1e-4 / bf16 / grad-ckpt / seed 42 / max_steps 100000 /
validate 1000 / **从 step 0 全新训练**(shift 变了,训练分布不同,不续训旧 ckpt,
保证 A/B 干净)。

## 6. 预期与验收

- **PSNR 大概率比旧 run 低几个 dB**:旧数字含均值回归虚高,属预期内回落;
- **LPIPS 应显著低于旧 arm**(旧 arm A 参考值 0.79),可视化明显变清晰;
- 若 5k 步后 LPIPS 不降反升 / 依旧灰 mush,则下一步把 shift 进一步降到 1–2,
  或把 timestep 采样改 uniform / logit_mean 调负,继续压低 σ 覆盖。
- 验收看三点:`train.log` 里 sigma_mean≈0.78(而非 0.90)、
  eval 行 `tgt_rgb_lpips` 有数值(非 nan)、`eval_step_*` 可视化清晰度。

## 7. 监控

```bash
tail -f rgbd_video_gen_gae/logs/gae_re10k_rgb_aux_shift45_*/train.log
# 可视化
ls rgbd_video_gen_gae/logs/gae_re10k_rgb_aux_shift45_*/eval_step_*/
# tensorboard
tensorboard --logdir rgbd_video_gen_gae/logs/gae_re10k_rgb_aux_shift45_*/tensorboard
```
