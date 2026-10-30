"""Standalone GAE-NVS checkpoint eval with an inference-step sweep.

Rebuilds the frozen in-training eval set byte-identically (same dataset
clipping, num_train_samples truncation, eval_view_seed and indices as
train_with_accelerate_gae_nvs.py), loads a checkpoint, then scores and saves
view strips for each --steps value on identical batches.

Also decodes the GT cached latents straight through the codec (no sampling)
as a sanity check of the cache -> standardize -> destandardize -> decode path.
"""
import argparse
import copy
import math
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from omegaconf import OmegaConf
from safetensors.torch import load_file

try:
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
except ImportError:
    pass

from accelerate import Accelerator
from accelerate.utils import set_seed

from examples.dataset import Kubric4DRGBDNVSDataset  # noqa: F401
from examples.wanvideo.model_training.WanGAENVSTrainingModule import (
    WanGAENVSTrainingModule,
)
from examples.wanvideo.model_training.train_with_accelerate_gae_nvs import (
    build_dataset,
    restrict_views_to_cache,
)
from examples.wanvideo.model_training.train_with_accelerate_rgbd_nvs import (
    custom_collate_fn,
    load_dotenv,
)


def save_strip(rgb, path):
    rows = []
    for v in range(rgb.shape[0]):
        img = rgb[v].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
        rows.append(img.astype(np.uint8))
    Image.fromarray(np.concatenate(rows, axis=1)).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--steps", type=str, default="4,25",
                        help="comma-separated inference step counts")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    accelerator = Accelerator(mixed_precision=cfg.mixed_precision)
    set_seed(args.seed)
    load_dotenv()
    os.makedirs(args.output_dir, exist_ok=True)

    model = WanGAENVSTrainingModule(
        args=cfg,
        accelerator=accelerator,
        model_id_with_origin_paths=cfg.model_id_with_origin_paths,
        trainable_models=cfg.trainable_models,
        lora_rank=cfg.lora_rank,
        lora_base_model=cfg.lora_base_model,
        lora_target_modules=cfg.lora_target_modules,
        local_model_path=cfg.get("wan_model_root", "./models"),
        skip_download=cfg.get("skip_download", False),
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        num_source_views=cfg.num_source_views,
        num_target_views=cfg.num_target_views,
        vae_encoding_mode=cfg.vae_encoding_mode,
        init_dvd_ckpt=None,
        view_condition_plucker_resolution=cfg.get(
            "view_condition_plucker_resolution", "latent"),
        view_condition_pixel_scale=cfg.get("view_condition_pixel_scale", 8),
        view_condition_mode=cfg.get("view_condition_mode", "add"),
        view_condition_use_role=cfg.get("view_condition_use_role", True),
        camera_geometry_fp32=cfg.get("camera_geometry_fp32", False),
        rgb_flow_shift=cfg.get("rgb_flow_shift", 5.0),
        rgb_flow_timestep_sampling=cfg.get(
            "rgb_flow_timestep_sampling", "logit_normal"),
        rgb_flow_logit_mean=cfg.get("rgb_flow_logit_mean", 0.0),
        rgb_flow_logit_std=cfg.get("rgb_flow_logit_std", 1.0),
        eval_num_inference_steps=cfg.get("eval_num_inference_steps", 4),
        eval_flow_shift=cfg.get("eval_flow_shift", None),
        eval_sample_seed=cfg.get("eval_sample_seed", 1234),
        eval_lpips_net=cfg.get("eval_lpips_net", "alex"),
        gae_cache_root=cfg.get("gae_cache_root"),
        gae_dataset_name=cfg.get("gae_dataset_name", "bmvs"),
        gae_stats_path=cfg.get("gae_stats_path", None),
        gae_rope_frame_axis=cfg.get("gae_rope_frame_axis", "view"),
        gae_auto_flow_shift=cfg.get("gae_auto_flow_shift", True),
        gae_codec_device=cfg.get("gae_codec_device", None),
    )
    model.set_training_param()

    weights = os.path.join(args.checkpoint, "model.safetensors")
    state = load_file(weights, device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    accelerator.print(
        f"Loaded {weights}: {len(state)} tensors, missing={len(missing)}, "
        f"unexpected={len(unexpected)}")
    if missing:
        accelerator.print(f"  first missing: {missing[:5]}")
    if unexpected:
        accelerator.print(f"  first unexpected: {unexpected[:5]}")

    model.to(accelerator.device)
    model.eval()

    # -- frozen eval set, identical to the trainer --------------------------
    train_dataset = build_dataset(cfg, split=cfg.get("train_split", None))
    num_views = int(cfg.num_source_views) + int(cfg.num_target_views)
    restrict_views_to_cache(
        train_dataset,
        cfg.get("gae_cache_root"),
        cfg.get("gae_dataset_name", "bmvs"),
        num_views,
        accelerator,
    )
    if cfg.get("num_train_samples", None) is not None:
        train_dataset.records = train_dataset.records[: int(cfg.num_train_samples)]

    eval_max_samples = int(cfg.get("eval_max_samples", 2))
    eval_batch_size = max(1, min(eval_max_samples, int(cfg.get("eval_batch_size", 4))))
    eval_dataset = copy.copy(train_dataset)
    eval_dataset.deterministic_view_seed = int(cfg.get("eval_view_seed", 1234))
    samples_per_scene = getattr(eval_dataset, "samples_per_scene", 1)
    num_scenes = len(eval_dataset.records)
    scene_stride = max(1, num_scenes // eval_max_samples)
    eval_indices = [
        scene_idx * samples_per_scene
        for scene_idx in range(0, num_scenes, scene_stride)
    ][:eval_max_samples]
    loader = DataLoader(
        Subset(eval_dataset, eval_indices),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=custom_collate_fn,
        drop_last=False,
    )
    accelerator.print(f"Frozen eval set: {len(eval_indices)} samples")
    batches = list(loader)

    # -- GT-latent decode sanity check (first batch, once) ------------------
    accelerator.print("\n=== GT cached-latent decode sanity check ===")
    batch = batches[0]
    with torch.no_grad():
        inputs = model.forward_preprocess(batch)
        height = batch["images"].shape[-2]
        width = batch["images"].shape[-1]
        h0, w0 = model._gae_true_grid
        dec_rgb, _ = model._decode_views(
            inputs["gae_latents"][:, :, :, :h0, :w0], height, width)
        gt = batch["images"].to(dec_rgb.device).float().clamp(0, 1)
        for b in range(dec_rgb.shape[0]):
            save_strip(dec_rgb[b], os.path.join(
                args.output_dir, f"gt_latent_decode_b{b:02d}.jpg"))
            for v in range(model.num_source_views, model.num_views):
                mse = float(((dec_rgb[b, v] - gt[b, v]) ** 2).mean())
                psnr = 100.0 if mse < 1e-10 else -10 * math.log10(mse)
                accelerator.print(
                    f"  b{b} tgt view{v}: decoded-GT-latent psnr={psnr:.2f}")
        del inputs, dec_rgb

    # -- step sweep ----------------------------------------------------------
    step_grid = [int(s) for s in args.steps.split(",") if s.strip()]
    for num_steps in step_grid:
        model.eval_num_inference_steps = num_steps
        cell_dir = os.path.join(args.output_dir, f"steps{num_steps}")
        os.makedirs(cell_dir, exist_ok=True)
        accelerator.print(f"\n=== inference steps = {num_steps} ===")
        all_metrics = {}
        for i, batch in enumerate(batches):
            m = model.evaluate(batch, output_dir=cell_dir, global_step=i,
                               max_samples=eval_batch_size, save_vis=True)
            accelerator.print(
                f"[batch {i}] " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()))
            for k, v in m.items():
                all_metrics.setdefault(k, []).append(v)
        summary = {k: float(np.nanmean(v)) for k, v in all_metrics.items()}
        accelerator.print(f"[summary steps={num_steps}] " + "  ".join(
            f"{k}={v:.4f}" for k, v in summary.items()))


if __name__ == "__main__":
    main()
