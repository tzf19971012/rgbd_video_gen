"""Standalone test-set evaluation for a trained RGB-D NVS checkpoint.

Replicates the model construction of
examples/wanvideo/model_training/train_with_accelerate_rgbd_nvs.py, loads a
checkpoint saved by accelerate.save_state (model.safetensors), then runs
WanRGBDNVSTrainingModule.evaluate over every sample of a TEST manifest.

Single process, single NPU.  Launch one process per NPU with --shard /
--num-shards to evaluate different slices of the test set in parallel; each
shard writes its own output dir, merge with --summarize-only or by hand.

Usage (blendedmvs test, full 50-scene manifest, on NPU 3):
  ASCEND_RT_VISIBLE_DEVICES=3 .venv/bin/python eval_script/eval_testset.py \
      --run-config logs/multi4_nvs_lora20261020_094334/args.yaml \
      --ckpt logs/multi4_nvs_lora20261020_094334/checkpoint-step-800 \
      --manifest /mnt/sfs_turbo/t00845730/data/blendedmvs/manifest_test.jsonl \
      --kubric4d-root /mnt/sfs_turbo/t00845730/data/blendedmvs/extracted \
      --output logs/test_eval/bmvs_test_s800
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

try:
    import torch_npu  # noqa: F401

    torch.npu.set_compile_mode(jit_compile=False)
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from accelerate import Accelerator  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from examples.dataset import Kubric4DRGBDNVSDataset  # noqa: E402
from examples.wanvideo.model_training.train_with_accelerate_rgbd_nvs import (  # noqa: E402
    custom_collate_fn,
)
from examples.wanvideo.model_training.WanRGBDNVSTrainingModule import (  # noqa: E402
    WanRGBDNVSTrainingModule,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-config", required=True,
                        help="args.yaml of the training run (resolved config)")
    parser.add_argument("--ckpt", required=True,
                        help="checkpoint dir containing model.safetensors")
    parser.add_argument("--manifest", required=True, help="test manifest jsonl")
    parser.add_argument("--kubric4d-root", default=None,
                        help="data root for the test manifest (comma separated); "
                             "default: cfg.kubric4d_root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-samples", type=int, default=0,
                        help="cap on evaluated samples; 0 = all scenes")
    parser.add_argument("--start-index", type=int, default=0,
                        help="resume: skip the first N samples of the manifest "
                             "(already covered by a previous run)")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--samples-per-scene", type=int, default=1)
    parser.add_argument("--view-seed", type=int, default=1234,
                        help="deterministic_view_seed for frozen source/target splits")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--save-vis", type=int, default=4,
                        help="render comparison grids for the first N samples")
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--eval-step-tag", type=int, default=0,
                        help="global_step tag used in eval_step_* dir names")
    return parser.parse_args()


def build_model(cfg, accelerator):
    """Same construction as the training entry point."""
    model = WanRGBDNVSTrainingModule(
        args=cfg,
        accelerator=accelerator,
        model_id_with_origin_paths=cfg.model_id_with_origin_paths,
        trainable_models=cfg.trainable_models,
        lora_rank=cfg.lora_rank,
        lora_base_model=cfg.lora_base_model,
        lora_target_modules=cfg.lora_target_modules,
        local_model_path=cfg.get("wan_model_root", "./models"),
        skip_download=cfg.get("skip_download", False),
        strip_i2v_image_input=cfg.get("strip_i2v_image_input", False),
        use_gradient_checkpointing=cfg.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=cfg.use_gradient_checkpointing_offload,
        num_source_views=cfg.num_source_views,
        num_target_views=cfg.num_target_views,
        vae_encoding_mode=cfg.vae_encoding_mode,
        init_dvd_ckpt=cfg.init_dvd_ckpt,
        rgb_loss_weight=cfg.rgb_loss_weight,
        depth_loss_weight=cfg.depth_loss_weight,
        view_condition_plucker_resolution=cfg.get("view_condition_plucker_resolution", "latent"),
        view_condition_pixel_scale=cfg.get("view_condition_pixel_scale", 8),
        view_condition_mode=cfg.get("view_condition_mode", "add"),
        view_condition_use_role=cfg.get("view_condition_use_role", True),
        view_condition_normalize_moment=cfg.get("view_condition_normalize_moment", False),
        latent_scale=cfg.get("latent_scale", 1.0),
        depth_latent_scale=cfg.get("depth_latent_scale", None),
        token_layout=cfg.get("token_layout", "channel"),
        warp_mode=cfg.get("warp_mode", "none"),
        warp_splat_footprint=cfg.get("warp_splat_footprint", "nearest"),
        warp_dropout_prob=cfg.get("warp_dropout_prob", 0.5),
        warp_two_pass=cfg.get("warp_two_pass", True),
        warp_perturb_prob=cfg.get("warp_perturb_prob", 0.0),
        warp_perturb_gain_log_std=cfg.get("warp_perturb_gain_log_std", 0.34),
        warp_perturb_shift_std=cfg.get("warp_perturb_shift_std", 0.059),
        warp_perturb_blur_sigma=cfg.get("warp_perturb_blur_sigma", 2.0),
        warp_perturb_lowfreq_std=cfg.get("warp_perturb_lowfreq_std", 0.05),
        warp_perturb_lowfreq_cells=cfg.get("warp_perturb_lowfreq_cells", 6),
        warp_perturb_fill_invalid=cfg.get("warp_perturb_fill_invalid", True),
        warp_perturb_fill_scale=cfg.get("warp_perturb_fill_scale", 1.0),
        depth_warp_block=cfg.get("depth_warp_block", False),
        depth_consist_scale_weight=cfg.get("depth_consist_scale_weight", 0.0),
        depth_consist_repro_weight=cfg.get("depth_consist_repro_weight", 0.0),
        depth_consist_sigma_max=cfg.get("depth_consist_sigma_max", 0.6),
        depth_consist_num_views=cfg.get("depth_consist_num_views", 3),
        depth_consist_max_samples=cfg.get("depth_consist_max_samples", 2),
        depth_consist_min_support=cfg.get("depth_consist_min_support", 0.02),
        depth_consist_ratio_clip=cfg.get("depth_consist_ratio_clip", 4.0),
        depth_consist_trim_quantile=cfg.get("depth_consist_trim_quantile", 0.9),
        depth_consist_vae_checkpoint=cfg.get("depth_consist_vae_checkpoint", True),
        per_frame_timestep=cfg.get("per_frame_timestep", False),
        clean_source_depth=cfg.get("clean_source_depth", False),
        clean_source_depth_prob=cfg.get("clean_source_depth_prob", 0.5),
        shared_rope_frame_ids=cfg.get("shared_rope_frame_ids", False),
        rope_frame_axis=cfg.get("rope_frame_axis", None),
        rope_pose_translation_scale=cfg.get("rope_pose_translation_scale", None),
        roce=cfg.get("roce", "off"),
        camera_geometry_fp32=cfg.get("camera_geometry_fp32", False),
        rgb_flow_matching=cfg.get("rgb_flow_matching", True),
        rgb_flow_shift=cfg.get("rgb_flow_shift", 5.0),
        rgb_flow_timestep_sampling=cfg.get("rgb_flow_timestep_sampling", "logit_normal"),
        rgb_flow_logit_mean=cfg.get("rgb_flow_logit_mean", 0.0),
        rgb_flow_logit_std=cfg.get("rgb_flow_logit_std", 1.0),
        depth_flow_matching=cfg.get("depth_flow_matching", False),
        depth_head_init=cfg.get("depth_head_init", "pretrained"),
        eval_num_inference_steps=cfg.get("eval_num_inference_steps", 4),
        eval_flow_shift=cfg.get("eval_flow_shift", None),
        eval_sample_seed=cfg.get("eval_sample_seed", 1234),
        eval_lpips_net=cfg.get("eval_lpips_net", "alex"),
        eval_vis_downscale=cfg.get("eval_vis_downscale", 2),
        eval_vis_format=cfg.get("eval_vis_format", "jpg"),
        eval_vis_depth_norm=cfg.get("eval_vis_depth_norm", "shared"),
    )
    model.set_training_param()
    return model


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.run_config)

    accelerator = Accelerator(mixed_precision=cfg.mixed_precision)
    os.makedirs(args.output, exist_ok=True)
    accelerator.print(f"[eval_testset] args: {vars(args)}")

    model = build_model(cfg, accelerator)

    from safetensors.torch import load_file

    ckpt_weights = os.path.join(args.ckpt, "model.safetensors")
    if os.path.isdir(args.ckpt) and os.path.exists(ckpt_weights):
        pass
    elif os.path.isfile(args.ckpt):
        ckpt_weights = args.ckpt
    else:
        raise FileNotFoundError(f"no model.safetensors under {args.ckpt}")
    state = load_file(ckpt_weights, device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    accelerator.print(
        f"[eval_testset] loaded {ckpt_weights}: {len(state)} tensors, "
        f"missing={len(missing)}, unexpected={len(unexpected)}"
    )
    if missing or unexpected:
        accelerator.print(
            f"[eval_testset] WARNING mismatch: missing={list(missing)[:5]} "
            f"unexpected={list(unexpected)[:5]}"
        )
    del state

    # Replicate accelerator.prepare's placement (the training entry point never
    # runs evaluate on a CPU-resident pipeline): move every child (incl. the
    # WanVideoPipeline, whose custom .to() refreshes its .device attribute) to
    # the NPU.  Without this, evaluate's autocast(device_type="npu") is inert on
    # CPU tensors and fp32 inputs collide with bf16 conv weights.
    model.to(accelerator.device)
    if str(getattr(model.pipe, "device", "")) != str(accelerator.device):
        model.pipe.to(accelerator.device)
    model.pipe.torch_dtype = torch.bfloat16
    model.eval()
    _vae_w = next(model.pipe.vae.model.encoder.parameters())
    accelerator.print(
        f"[eval_testset] pipe.device={model.pipe.device} "
        f"torch_dtype={model.pipe.torch_dtype} vae_dtype={_vae_w.dtype} "
        f"vae_device={_vae_w.device}"
    )

    roots = args.kubric4d_root
    if roots is None:
        roots = cfg.kubric4d_root
    roots = [r.strip() for r in str(roots).split(",") if r.strip()]

    dataset = Kubric4DRGBDNVSDataset(
        kubric4d_root=roots,
        manifest_path=args.manifest,
        resolution=tuple(cfg.resolution),
        num_source_views=cfg.num_source_views,
        num_target_views=cfg.num_target_views,
        random_permute_source_views=cfg.random_permute_source_views,
        view_order=cfg.get("view_order", "random"),
        video_tgt_interval_prob=cfg.get("video_tgt_interval_prob", 0.2),
        video_max_src_span=cfg.get("video_max_src_span", None),
        disparity_norm_mode=cfg.get("disparity_norm_mode", "shared_scale"),
        disparity_norm_quantile=cfg.get("disparity_norm_quantile", 0.98),
        disparity_norm_headroom=cfg.get("disparity_norm_headroom", 1.5),
        min_view_valid_fraction=cfg.get("min_view_valid_fraction", 0.0),
        view_validity_path=cfg.get("view_validity_path", None),
        samples_per_scene=args.samples_per_scene,
        split_path=None,
        split=None,
        scene_filter=None,
    )
    dataset.deterministic_view_seed = args.view_seed

    total = len(dataset)
    if args.num_samples > 0:
        total = min(total, args.num_samples)
    indices = list(range(args.start_index, total))[args.shard::args.num_shards]
    accelerator.print(
        f"[eval_testset] dataset scenes={len(dataset.records)} total_samples={total} "
        f"start={args.start_index} shard={args.shard}/{args.num_shards} -> {len(indices)} samples: {indices}"
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=indices,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        collate_fn=custom_collate_fn,
    )

    all_metrics = []
    done = 0
    for batch in loader:
        metrics = model.evaluate(
            batch,
            output_dir=args.output,
            global_step=args.eval_step_tag,
            max_samples=args.batch_size,
            save_vis=(done < args.save_vis),
            vis_offset=done,
        )
        all_metrics.append(metrics)
        done += args.batch_size
        accelerator.print(f"[eval_testset] progress {done}/{len(indices)}: {metrics}")

    merged = {}
    for key in {k for m in all_metrics for k in m}:
        vals = [m[key] for m in all_metrics if key in m and not math.isnan(float(m[key]))]
        merged[key] = float(np.mean(vals)) if vals else float("nan")
    for side in ("src", "tgt"):
        d1 = merged.get(f"{side}_depth_delta1")
        ramp = merged.get(f"{side}_depth_delta1_ramp")
        if d1 is not None and ramp is not None and 1.0 - ramp > 1e-3:
            merged[f"{side}_depth_delta1_skill_agg"] = (d1 - ramp) / (1.0 - ramp)

    accelerator.print(
        f"[eval_testset] FINAL ({done} samples, shard {args.shard}/{args.num_shards})\n"
        + "\n".join(f"  {k}={v:.4f}" for k, v in sorted(merged.items()))
    )
    with open(os.path.join(args.output, "summary_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "ckpt": args.ckpt,
                "manifest": args.manifest,
                "num_samples": done,
                "shard": args.shard,
                "num_shards": args.num_shards,
                "metrics": merged,
                "per_sample": all_metrics,
            },
            fh,
            indent=2,
        )


if __name__ == "__main__":
    main()
