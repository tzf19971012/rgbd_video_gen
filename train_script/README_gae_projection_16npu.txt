云道16卡 GAE projection 续训入口

在2个节点上执行相同入口，每节点8张NPU：
  bash /mnt/sfs_turbo/t00845730/rgbd_video_gen_gae/train_script/yundao_resume_gae_projection_16npu.sh

默认配置：
  每卡batch_size=8，gradient_accumulation_steps=1，全局batch=128。
  总训练步数30000；从8600开始，额外21400个optimizer updates。
  主学习率3e-5，新输入/输出层学习率1.5e-4，沿用原实验。
  使用全部7884个RE10K训练场景和修正c2w的训练manifest；不混入测试集。
  六源图、三目标图，patch1 concatpixel，projection_condition=gae。
  每200步验证/保存，单独输出，不覆盖原本机训练目录。

checkpoint：
  原始来源 logs/projection_gae_3npu_resume_3card_20261110_110818/checkpoint-step-8600。
  已通过硬链接固定保存到 reports/projection_gae_yundao_16npu_from8600/checkpoint-step-8600，
  防止原训练max_checkpoints=1清理该断点。配置指向这个相同状态的固定副本。
  恢复模型、优化器及调度器；3卡切换16卡后重新设置各rank随机种子，不是逐位精确重放。
  checkpoint权重不进入Git，云道节点需挂载相同SFS目录。

配置文件：
  train_config/normal_config/gae_projection_re10k_yundao_16npu_from8600.yaml
训练使用原projection代码快照：
  reports/projection_ablation_20261109_113836/resume_3card_20261110_110818/code/train.py
输出：
  logs/projection_gae_yundao_16npu_from8600_to30000
每节点控制台：输出目录/yundao_node_<rank>.log

如镜像已有正确的torch/torch_npu环境，可在两节点都设置：
  export YUNDAO_PYTHON_BIN=python
其他环境选择沿用参考脚本YUNDAO_CONDA_ENV/YUNDAO_PYTHON_BIN逻辑。
若OOM，重新提交时可覆盖（本脚本不自动改变BS）：
  bash train_script/yundao_resume_gae_projection_16npu.sh batch_size=4
或 batch_size=2。不会自动替换恢复点；若要接续云道新断点，需要另行明确配置。

检查：bash语法、实际登录shell命令展开、两个节点dry-run、完整状态文件和
GAE projection adapter形状检查均通过。尚未在云道进行真实16卡启动/显存测试。
