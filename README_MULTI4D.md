# 多数据集 RGB-D NVS 训练指南

基于 Wan2.1-T2V-1.3B 的 RGB-D 新视角合成（NVS）训练，支持 **BlendedMVS / GTA-SfM / MatrixCity / TartanAir** 多数据集联合训练。
数据加载全部走 `manifest.jsonl`（不遍历数据集目录，初始化毫秒级）。训练脚本：`examples/wanvideo/model_training/train_with_accelerate_rgbd_nvs.py`。

## 1. 环境准备

```bash
cd DVD-main/DVD-main
source env_npu.sh            # CANN 环境（含 libhccl）
source .venv/bin/activate    # torch_npu venv
export ASCEND_RT_VISIBLE_DEVICES=0    # 选一颗空闲 NPU（用 npu-smi info 查看）
```

> 本机 `.venv` 是 `--system-site-packages` 最小环境、没有 `accelerate` 入口点，
> 启动脚本统一用 `.venv/bin/python -u -m ...` 直接跑（与本机正在运行的作业同模式）。
> 多进程训练：`USE_ACCELERATE=1 ACCELERATE_CONFIG=train_config/accelerate_config/accelerate_npu_1gpu.yaml bash <脚本>`。

## 2. 数据集 manifest 制作

### 2.1 已有 manifest

如果数据集目录下已经有 `manifest.jsonl`（例如 `sample_data` 里四个数据集都有），**什么都不用做**，
yaml 里指过去就行（见第 3 节）。

### 2.2 从原始数据生成

新平台/完整版数据集没有 manifest 时，用下面的工具生成（输出格式已与参考 manifest 逐视图数值验证）：

```bash
# TartanAir: 扫 <env>/<难度>/<P###>/，读 pose_left.txt + image_left/ + depth_clean|depth_left/
python utils/prepare_multi4d_manifests.py tartanair \
    --data_root /path/to/TartanAir --out_manifest /path/to/TartanAir/manifest.jsonl \
    --stride 10            # 每 10 帧取 1；默认 1 = 全量

# MatrixCity: 读每个 scene 的 transforms.json + rgb/ + depth_npy/（没有 npy 时加 --convert-exr 现场转 depth/*.exr）
python utils/prepare_multi4d_manifests.py matrixcity \
    --data_root /path/to/MatrixCity --out_manifest /path/to/MatrixCity/manifest.jsonl

# GTA-SfM: 读 train/test/*.hdf5（内含 BMP 图、深度、K、pose），
# 同时把 rgb/深度解包到 <data_root>/extracted/<场景>/{rgb,depth}/
python utils/prepare_multi4d_manifests.py gta_sfm \
    --data_root /path/to/GTA-SfM --out_manifest /path/to/GTA-SfM/manifest.jsonl \
    --stride 5             # 示例数据用的 stride；hdf5 里 10000 是"无深度"哨兵，自动置 0 无效标记

# BlendedMVS: 仓库自带工具（读 cam.txt、顺便 pfm→npy）
python utils/prepare_blendedmvs.py --data_root /path/to/BlendedMVS \
    --out_manifest /path/to/BlendedMVS/manifest.jsonl --min_views 5
```

### 2.3 manifest 格式

一行一个场景，`rgb_path`/`depth_path` 相对 `kubric4d_root`（或绝对路径）：

```json
{"scene_id": "abandonedfactory/Easy/P000", "time_id": "t0",
 "views": [{"view_id": "000000",
            "rgb_path": "abandonedfactory/Easy/P000/image_left/000000_left.png",
            "depth_path": "abandonedfactory/Easy/P000/depth_clean/000000_left_depth.npy",
            "K": [[320,0,320],[0,320,240],[0,0,1]],
            "c2w": [[...4x4...]],
            "depth_scale": 1.0}]}
```

要求：views 按时序排列；`c2w` 平移单位与 depth 一致（米）；每个场景 view 数 ≥ `num_source_views + num_target_views`（默认 6+3=9，不足自动丢弃）。

## 3. 训练配置（yaml）

配置在 `train_config/normal_config/` 下。**切换平台/机器时只需改文件顶部的路径段**：

```yaml
wan_model_root: "/path/to/models"   # Wan2.1 权重根目录（其下有 Wan-AI/Wan2.1-T2V-1.3B/）
skip_download: true                 # 权重已在本地，跳过 ModelScope 下载
sample_data_root: "/path/to/sample_data"
output_path: "logs/multi4_nvs_fullft"
```

数据段是三个**对齐的列表**（单数据集也可写标量）：

```yaml
kubric4d_root:  [${sample_data_root}/BlendedMVS, ${sample_data_root}/GTA-SfM, ...]
manifest_path:  [manifest.jsonl, manifest.jsonl, ...]
scene_filter:   [null, train, null, null]    # 可选：按 scene_id 子串过滤（GTA 只留 train 场景）
```

启动脚本支持 `key=value` 命令行覆盖（不改 yaml 就能调参），如 `batch_size=4 log_step=10`。

## 4. LoRA 微调

配置：`train_config/normal_config/multi4_nvs_lora.yaml`（基于 `bmvs_c_base.yaml`）
开关：`train_all: false` + `lora_base_model: "dit"` + `lora_rank: 128` + `lr: 1e-4`，可训练参数 **177.8M**。

```bash
bash train_script/train_multi4_nvs_lora.sh
# 或带覆盖：
bash train_script/train_multi4_nvs_lora.sh train_config/normal_config/multi4_nvs_lora.yaml \
    batch_size=4 log_step=10 output_path=logs/my_lora_run
```

## 5. 全量微调（full fine-tune）

配置：`train_config/normal_config/multi4_nvs_fullft.yaml`
与 LoRA 版唯一区别：`train_all: true` + `lora_base_model: null` + `lr: 2e-5`，
放开整个 DiT + NVS 附加头，可训练参数 **1.42B**（checkpoint 每个 ~5.7GB，注意磁盘）。

```bash
bash train_script/train_multi4_nvs_fullft.sh
```

**batch size 参考**（Ascend 910 64GB HBM 单卡实测，6 源 + 3 目标视图、384x672、梯度检查点开）：

| batch_size | 峰值显存 | 结论 |
|---|---|---|
| 1 | 14.2 GB | 很宽裕 |
| 2 | 14.2 GB | ✅ |
| 4 | 21.5 GB | ✅ 推荐 |
| 8 | — | ❌ OOM |

多卡按每卡 bs=4 配（如 8 卡 × 4 × accum 2 = 全局 64）。LoRA 版优化器状态只算 177M 参数，bs 还可以更大。

## 6. 训练监控

- 日志：`<output_path>/train.log`（loss/rgb/depth 每 `log_step` 步一行，含 sigma 分桶诊断）
- TensorBoard：`<output_path>/tensorboard`
- checkpoint：`<output_path>/checkpoint-step-<N>/`，按 `max_checkpoints` 自动裁剪；
  `resume=true training_state_dir=<ckpt>` 可断点续训

## 7. 相关文件清单

| 用途 | 文件 |
|---|---|
| LoRA 微调 | `train_config/normal_config/multi4_nvs_lora.yaml` + `train_script/train_multi4_nvs_lora.sh` |
| 全量微调 | `train_config/normal_config/multi4_nvs_fullft.yaml` + `train_script/train_multi4_nvs_fullft.sh` |
| manifest 生成 | `utils/prepare_multi4d_manifests.py`（GTA-SfM/MatrixCity/TartanAir）、`utils/prepare_blendedmvs.py`（BlendedMVS） |
| 多数据集加载 | `examples/dataset/kubric4d_rgbd_nvs_dataset.py`（`kubric4d_root`/`manifest_path`/`scene_filter` 支持列表） |
| 权重路径 | yaml 顶部 `wan_model_root` + `skip_download` |
