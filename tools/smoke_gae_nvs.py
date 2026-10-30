"""CPU smoke test for the GAE-latent NVS training stack (Phase 2).

One batch through forward_preprocess -> _forward_flow -> backward, on CPU,
against the real bmvs cache. No NPU needed; the eval codec stays unloaded.
"""
import os
import sys

sys.path.insert(0, "/mnt/sfs_turbo/t00845730/rgbd_video_gen")
os.chdir("/mnt/sfs_turbo/t00845730/rgbd_video_gen")

import torch
from accelerate import Accelerator
from omegaconf import OmegaConf

cfg = OmegaConf.load("train_config/normal_config/gae_nvs_lora.yaml")
cfg.output_path = "/tmp/gae_smoke"
cfg.mixed_precision = "no"
cfg.dataloader_num_workers = 0
cfg.num_train_samples = 4
cfg.batch_size = 1
cfg.use_gradient_checkpointing = False

accelerator = Accelerator(cpu=True)
print("[smoke] building module (loads Wan2.1-1.3B DiT on CPU)...", flush=True)


from examples.wanvideo.model_training.WanGAENVSTrainingModule import (
    WanGAENVSTrainingModule,
)

model = WanGAENVSTrainingModule(
    args=cfg,
    accelerator=accelerator,
    model_id_with_origin_paths=cfg.model_id_with_origin_paths,
    trainable_models=cfg.trainable_models,
    lora_rank=cfg.lora_rank,
    lora_base_model=cfg.lora_base_model,
    lora_target_modules=cfg.lora_target_modules,
    local_model_path=cfg.wan_model_root,
    skip_download=True,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    num_source_views=cfg.num_source_views,
    num_target_views=cfg.num_target_views,
    vae_encoding_mode=cfg.vae_encoding_mode,
    init_dvd_ckpt=None,
    view_condition_plucker_resolution="latent",
    view_condition_pixel_scale=8,
    view_condition_mode="add",
    view_condition_use_role=True,
    camera_geometry_fp32=False,
    rgb_flow_shift=cfg.rgb_flow_shift,
    rgb_flow_timestep_sampling="logit_normal",
    eval_num_inference_steps=4,
    eval_lpips_net="alex",
    gae_cache_root=cfg.gae_cache_root,
    gae_dataset_name=cfg.gae_dataset_name,
    gae_stats_path=cfg.gae_stats_path,
    gae_rope_frame_axis=cfg.gae_rope_frame_axis,
    gae_auto_flow_shift=True,
)
model.set_training_param()
n_train = sum(p.numel() for p in model.trainable_modules())
print(f"[smoke] module ready; trainable params {n_train/1e6:.1f}M", flush=True)

from examples.dataset import Kubric4DRGBDNVSDataset
from examples.wanvideo.model_training.train_with_accelerate_rgbd_nvs import (
    _masked_mse,
    _resize_mask,
    custom_collate_fn,
)
from torch.utils.data import DataLoader, Subset

ds = Kubric4DRGBDNVSDataset(
    kubric4d_root=cfg.kubric4d_root,
    manifest_path=cfg.manifest_path,
    resolution=tuple(cfg.resolution),
    num_source_views=cfg.num_source_views,
    num_target_views=cfg.num_target_views,
    random_permute_source_views=True,
    disparity_norm_mode="metric",
    min_view_valid_fraction=0.25,
    samples_per_scene=1,
)
loader = DataLoader(Subset(ds, [0, 1]), batch_size=1, shuffle=False,
                    collate_fn=custom_collate_fn)
batch = next(iter(loader))
print(f"[smoke] batch: images {tuple(batch['images'].shape)}, "
      f"scene {batch['scene_id'][0]}, "
      f"src views {batch['source_view_ids'][0]}, "
      f"tgt views {batch['target_view_ids'][0]}", flush=True)

model.train()
res = model(batch)
pred, vel = res["pred"], res["rgb_gt"]
print(f"[smoke] pred {tuple(pred.shape)} (expect [1,64,9,27,36]), "
      f"velocity target {tuple(vel.shape)}, sigma {float(res['sigma'][0]):.3f}",
      flush=True)
assert pred.shape[1] == 64 and pred.shape[2] == 9

mask = _resize_mask(res["loss_masks"][:, 1], pred.shape[-2:]).to(pred.dtype)
mask = mask.permute(0, 2, 1, 3, 4)
loss = _masked_mse(pred, vel, mask)
print(f"[smoke] loss = {float(loss):.6f}", flush=True)
loss.backward()
grads = sum(int(p.grad is not None and p.grad.abs().sum() > 0)
            for p in model.trainable_modules())
print(f"[smoke] backward OK, {grads} trainable tensors carry grad", flush=True)

# timestep shape contract for the per-frame path
t = model._expand_timestep(torch.tensor([500.0]))
print(f"[smoke] per-frame timestep {t.shape}: {t[0].tolist()}", flush=True)
print("[smoke] PASS")
