"""Standalone evaluation for the RGB-D NVS model.

Loads an `accelerate.save_state` checkpoint directory (model.safetensors) into
WanRGBDNVSTrainingModule and runs `evaluate()` over a fixed number of batches,
reporting the aggregated metrics.

Run it as a MODULE from the project root -- it imports `utils.*` and
`examples.*` as top-level packages, which a bare `python <path>` invocation puts
out of reach because sys.path[0] becomes the script's own directory.

Example:
  python -m examples.wanvideo.model_training.eval_rgbd_nvs \
      --config train_config/normal_config/blendedmvs_nvs_independent_8gpu_s4_t3_bs8_fullfinetune_dvdinit.yaml \
      --checkpoint logs/.../checkpoint-step-3000 \
      --disparity_norm_mode per_view \
      --num_batches 4 --batch_size 4
"""
import argparse
import os

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Subset

from examples.dataset import Kubric4DRGBDNVSDataset
from examples.wanvideo.model_training.train_with_accelerate_rgbd_nvs import custom_collate_fn
from examples.wanvideo.model_training.WanRGBDNVSTrainingModule import (
    WanRGBDNVSTrainingModule,
    resolve_rope_frame_axis,
)


def _num_list(text, cast):
    return [cast(v) for v in str(text).split(",") if v.strip() != ""]


def frozen_eval_indices(dataset, eval_max_samples, samples_per_scene):
    """Reproduce the frozen eval set built in train_with_accelerate_rgbd_nvs.

    Without a split the training loop strides over SCENES and takes the first
    sample slot of each, so with 102 scenes and eval_max_samples=4 the raw
    indices are [0, 250, 500, 750].  Iterating the first N batches of the full
    loader instead would score a completely different set of scenes and none of
    the numbers would be comparable to the in-training log.

    The stride is also the bug that made the eval set a function of len(records):
    growing the manifest 113 -> 502 scenes moved it from stride 3 to stride 15 and
    onto a disjoint scene set (report_20260905_t0 §2).  When `dataset.split` is
    set the records are already exactly the held-out scenes in split-file order,
    so there is nothing to stride over -- take them all, in order.
    """
    num_scenes = len(dataset.records)
    if getattr(dataset, "split", None) is not None:
        return [i * samples_per_scene for i in range(num_scenes)][:eval_max_samples]
    scene_stride = max(1, num_scenes // eval_max_samples)
    return [
        scene_idx * samples_per_scene
        for scene_idx in range(0, num_scenes, scene_stride)
    ][:eval_max_samples]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="accelerate save_state directory containing model.safetensors")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_batches", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--disparity_norm_mode", type=str, default=None,
                        choices=["shared_scale", "per_view"],
                        help="Override the config value. Use the mode the checkpoint was TRAINED with.")
    parser.add_argument("--disparity_norm_headroom", type=float, default=None)
    parser.add_argument("--num_inference_steps", type=str, default=None,
                        help="Euler steps for target-view RGB sampling (flow matching only). "
                             "Comma-separated values are swept in one process.")
    parser.add_argument("--eval_flow_shift", type=str, default=None,
                        help="Override the sigma-schedule shift used at INFERENCE only. "
                             "Training keeps rgb_flow_shift; shift=5 back-loads the schedule "
                             "so hard that the last Euler step spans >50%% of the trajectory. "
                             "Comma-separated values are swept in one process.")
    parser.add_argument("--eval_max_samples", type=int, default=None,
                        help="Reproduce the training-time FROZEN eval set (strided over scenes) "
                             "instead of taking the first --num_batches batches. Set this to the "
                             "config's eval_max_samples to get numbers comparable to train.log.")
    parser.add_argument("--manifest_path", type=str, default=None,
                        help="Override cfg.manifest_path. REQUIRED to reproduce an old "
                             "checkpoint's numbers after the manifest has grown: the frozen eval "
                             "set strides over scenes with stride len(records)//eval_max_samples, "
                             "so adding scenes silently re-picks which scenes are scored and every "
                             "number becomes incomparable to the run's own train.log. The BlendedMVS "
                             "manifest went 113 -> 502 scenes on 2026-09-05 (BlendedMVS+/++), and "
                             "the original 113 are interleaved at positions 42..501 of the new file, "
                             "not its first 113 lines. Point this at "
                             "manifest.jsonl.bak_113scenes_20260905_112542 for b-sweep comparisons.")
    parser.add_argument("--split_path", type=str, default=None,
                        help="Frozen scene-id split (configs/bmvs_split_v1.json). Overrides "
                             "cfg.split_path.")
    parser.add_argument("--split", type=str, default=None,
                        help="Which subset of --split_path to score, e.g. 'val' or 'test'. "
                             "Overrides cfg.eval_split. With a split the eval set is the "
                             "FULL named scene list in file order, so it no longer depends "
                             "on len(records) the way the stride-derived set did.")
    parser.add_argument("--min_view_valid_fraction", type=float, default=None,
                        help="Overrides cfg. Held-out splits are frozen against a stricter "
                             "threshold than training uses (see tools/make_bmvs_split.py); "
                             "pass the split file's holdout_threshold when scoring them.")
    parser.add_argument("--view_validity_path", type=str, default=None,
                        help="Override cfg.view_validity_path. Must be set whenever "
                             "--manifest_path is a backup: the default sidecar location is "
                             "splitext(manifest)[0] + '.view_validity.json', which for a "
                             "'.bak_*' suffix resolves to a file that does not exist.")
    parser.add_argument("--no_flow_matching", action="store_true",
                        help="Evaluate a legacy deterministic single-step checkpoint.")
    parser.add_argument("--depth_vae_ceiling", action="store_true",
                        help="Also score the VAE round-trip of the GT disparity, giving the depth "
                             "metrics the floor that tgt_rgb_psnr_vae already gives RGB. Costs one "
                             "extra VAE decode per view. The result depends only on the VAE and the "
                             "dataset, not on the checkpoint, so one pass is enough -- it is off by "
                             "default so training-time evals do not pay for it repeatedly.")
    parser.add_argument("--save_vis", action="store_true",
                        help="Also write the side-by-side comparison images. Off by default: "
                             "each target view costs ~9 extra VAE decodes, and the per-view "
                             "metrics JSONL carries everything a diagnostic needs. Metrics and "
                             "RNG are unaffected either way.")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Split the frozen eval set across this many INDEPENDENT processes "
                             "(one GPU each, no cross-process communication -- eval is "
                             "embarrassingly parallel per sample). Batches are assigned "
                             "round-robin by GLOBAL batch index, and `global_step` stays that "
                             "global index, so every shard writes its own eval_step_%06d "
                             "directories and shards can share one --output_dir without "
                             "colliding. Each shard's printed sweep summary then covers ONLY "
                             "its own batches -- merge the per_view_metrics.jsonl files to get "
                             "the eval-set mean (see sbatch/eval_full_val_vis_shard.sbatch).")
    parser.add_argument("--shard_idx", type=int, default=0,
                        help="Which shard this process owns, in [0, --num_shards).")
    parser.add_argument("--plucker_resolution", type=str, default=None,
                        choices=["latent", "pixel"],
                        help="Override the config value. MUST match what the checkpoint was "
                             "trained with, otherwise the first view-condition conv is dropped.")
    parser.add_argument("--depth_flow_matching", type=str, default=None,
                        choices=["true", "false"],
                        help="Override the config value. MUST match what the checkpoint was "
                             "trained with: joint mode widens patch_embedding to 32 input "
                             "channels, so the shapes are incompatible either way.")
    args = parser.parse_args()
    if args.num_shards < 1 or not (0 <= args.shard_idx < args.num_shards):
        parser.error(f"--shard_idx {args.shard_idx} out of range for "
                     f"--num_shards {args.num_shards}")

    cfg = OmegaConf.load(args.config)
    set_seed(args.seed)
    accelerator = Accelerator(mixed_precision=cfg.mixed_precision)

    norm_mode = args.disparity_norm_mode or cfg.get("disparity_norm_mode", "shared_scale")
    headroom = args.disparity_norm_headroom or cfg.get("disparity_norm_headroom", 1.5)
    flow_matching = (not args.no_flow_matching) and cfg.get("rgb_flow_matching", True)
    step_grid = _num_list(
        args.num_inference_steps
        if args.num_inference_steps is not None
        else cfg.get("eval_num_inference_steps", 4),
        int,
    )
    train_shift = float(cfg.get("rgb_flow_shift", 5.0))
    shift_grid = _num_list(
        args.eval_flow_shift if args.eval_flow_shift is not None else train_shift,
        float,
    )
    plucker_resolution = args.plucker_resolution or cfg.get(
        "view_condition_plucker_resolution", "latent"
    )
    if args.depth_flow_matching is None:
        depth_flow_matching = bool(cfg.get("depth_flow_matching", False))
    else:
        depth_flow_matching = args.depth_flow_matching == "true"
    output_dir = args.output_dir or os.path.join(
        args.checkpoint or cfg.output_path, f"eval_{norm_mode}"
    )
    os.makedirs(output_dir, exist_ok=True)
    accelerator.print(
        f"disparity_norm_mode={norm_mode}  headroom={headroom}  "
        f"flow_matching={flow_matching}  steps={step_grid}  "
        f"train_shift={train_shift}  eval_shift={shift_grid}  "
        f"plucker_resolution={plucker_resolution}  "
        f"depth_flow_matching={depth_flow_matching}  out={output_dir}"
    )
    accelerator.print(
        f"token_layout={cfg.get('token_layout', 'channel')}  "
        f"warp_mode={cfg.get('warp_mode', 'none')}  "
        f"warp_two_pass={cfg.get('warp_two_pass', True)}  "
        f"per_frame_timestep={cfg.get('per_frame_timestep', False)}  "
        f"depth_warp_block={cfg.get('depth_warp_block', False)}  "
        f"clean_source_depth={cfg.get('clean_source_depth', False)}  "
        f"shared_rope_frame_ids={cfg.get('shared_rope_frame_ids', False)}  "
        # Resolved, for the reason spelled out in train_with_accelerate_rgbd_nvs.py.
        f"rope_frame_axis={resolve_rope_frame_axis(cfg.get('rope_frame_axis', None), cfg.get('shared_rope_frame_ids', False))}  "
        f"rope_pose_translation_scale={cfg.get('rope_pose_translation_scale', None)}  "
        f"roce={cfg.get('roce', 'off')}  "
        f"view_order={cfg.get('view_order', 'random')}  "
        f"latent_scale={cfg.get('latent_scale', 1.0)}/"
        f"{cfg.get('depth_latent_scale', None)}"
    )

    model = WanRGBDNVSTrainingModule(
        args=cfg,
        accelerator=accelerator,
        model_id_with_origin_paths=cfg.model_id_with_origin_paths,
        trainable_models=cfg.trainable_models,
        lora_rank=cfg.lora_rank,
        lora_base_model=cfg.lora_base_model,
        lora_target_modules=cfg.lora_target_modules,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        local_model_path=cfg.get("wan_model_root", "./models"),
        skip_download=cfg.get("skip_download", False),
        strip_i2v_image_input=cfg.get("strip_i2v_image_input", False),
        num_source_views=cfg.num_source_views,
        num_target_views=cfg.num_target_views,
        vae_encoding_mode=cfg.vae_encoding_mode,
        init_dvd_ckpt=None if args.checkpoint else cfg.init_dvd_ckpt,
        rgb_loss_weight=cfg.rgb_loss_weight,
        depth_loss_weight=cfg.depth_loss_weight,
        view_condition_plucker_resolution=plucker_resolution,
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
        # These four leave the parameter shapes untouched, so a checkpoint
        # trained with them ON still loads cleanly with them OFF -- and then
        # silently reports different numbers than train.log, because they change
        # the DiT input block, the AdaLN modulation or the positional encoding.
        # Read them from the config, never default them for a checkpoint whose
        # config says otherwise.  shared_rope_frame_ids is the worst of the four:
        # getting it wrong is a silent RoPE mismatch, not an error.
        depth_warp_block=cfg.get("depth_warp_block", False),
        per_frame_timestep=cfg.get("per_frame_timestep", False),
        clean_source_depth=cfg.get("clean_source_depth", False),
        clean_source_depth_prob=cfg.get("clean_source_depth_prob", 0.5),
        shared_rope_frame_ids=cfg.get("shared_rope_frame_ids", False),
        rope_frame_axis=cfg.get("rope_frame_axis", None),
        rope_pose_translation_scale=cfg.get("rope_pose_translation_scale", None),
        roce=cfg.get("roce", "off"),
        camera_geometry_fp32=cfg.get("camera_geometry_fp32", False),
        rgb_flow_matching=flow_matching,
        rgb_flow_shift=cfg.get("rgb_flow_shift", 5.0),
        depth_flow_matching=depth_flow_matching,
        depth_head_init=cfg.get("depth_head_init", "pretrained"),
        eval_num_inference_steps=step_grid[0],
        eval_flow_shift=cfg.get("eval_flow_shift", None),
        eval_sample_seed=cfg.get("eval_sample_seed", 1234),
        eval_lpips_net=cfg.get("eval_lpips_net", "alex"),
        eval_depth_vae_ceiling=args.depth_vae_ceiling,
        eval_vis_downscale=cfg.get("eval_vis_downscale", 2),
        eval_vis_format=cfg.get("eval_vis_format", "jpg"),
        eval_vis_depth_norm=cfg.get("eval_vis_depth_norm", "shared"),
    )

    if args.checkpoint:
        weights = os.path.join(args.checkpoint, "model.safetensors")
        state = load_file(weights, device="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        accelerator.print(
            f"Loaded {weights}: {len(state)} tensors, missing={len(missing)}, unexpected={len(unexpected)}"
        )
        if unexpected:
            accelerator.print(f"  first unexpected: {unexpected[:5]}")

    model.to(accelerator.device)
    model.eval()

    manifest_path = args.manifest_path or cfg.manifest_path
    view_validity_path = args.view_validity_path or cfg.get("view_validity_path", None)
    dataset = Kubric4DRGBDNVSDataset(
        kubric4d_root=cfg.kubric4d_root,
        manifest_path=manifest_path,
        resolution=tuple(cfg.resolution),
        num_source_views=cfg.num_source_views,
        num_target_views=cfg.num_target_views,
        random_permute_source_views=cfg.random_permute_source_views,
        view_order=cfg.get("view_order", "random"),
        video_tgt_interval_prob=cfg.get("video_tgt_interval_prob", 0.2),
        video_max_src_span=cfg.get("video_max_src_span", None),
        disparity_norm_mode=norm_mode,
        disparity_norm_headroom=headroom,
        deterministic_view_seed=int(cfg.get("eval_view_seed", 1234)),
        min_view_valid_fraction=(
            args.min_view_valid_fraction
            if args.min_view_valid_fraction is not None
            else cfg.get("min_view_valid_fraction", 0.0)
        ),
        view_validity_path=view_validity_path,
        samples_per_scene=cfg.get("samples_per_scene", 1),
        split_path=args.split_path or cfg.get("split_path", None),
        split=args.split or cfg.get("eval_split", None),
    )
    # The eval set's identity is (manifest, validity sidecar, min_view_valid_fraction,
    # samples_per_scene, eval_view_seed, eval_max_samples) -- print all of it.  A
    # grown manifest loads cleanly and reports a full metric table that simply
    # describes different scenes, which is the same class of silent failure as the
    # per_frame_timestep gap in report_20260904_p0a §1.
    accelerator.print(
        f"eval set identity: manifest={os.path.basename(manifest_path)} "
        f"({len(dataset.records)} scenes after min_view_valid_fraction="
        f"{cfg.get('min_view_valid_fraction', 0.0)})  "
        f"validity={os.path.basename(view_validity_path) if view_validity_path else '<derived>'}  "
        f"samples_per_scene={cfg.get('samples_per_scene', 1)}  "
        f"eval_view_seed={cfg.get('eval_view_seed', 1234)}"
    )
    # shuffle=False + deterministic_view_seed => every checkpoint and every A/B
    # arm is scored on byte-identical inputs.
    if args.eval_max_samples is not None:
        indices = frozen_eval_indices(
            dataset, args.eval_max_samples, cfg.get("samples_per_scene", 1)
        )
        accelerator.print(f"Frozen eval set (matches train.log): indices={indices}")
        # Chunk the frozen set so a large --eval_max_samples does not have to fit
        # in a single forward pass.  When --batch_size >= --eval_max_samples this
        # is one batch, i.e. byte-identical to the in-training eval.
        per_batch = min(args.eval_max_samples, args.batch_size)
        loader = DataLoader(
            Subset(dataset, indices),
            batch_size=per_batch,
            shuffle=False,
            num_workers=cfg.dataloader_num_workers,
            collate_fn=custom_collate_fn,
            drop_last=False,
        )
        batch_limit = (len(indices) + per_batch - 1) // per_batch
    else:
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=cfg.dataloader_num_workers,
            collate_fn=custom_collate_fn,
            drop_last=False,
        )
        batch_limit, per_batch = args.num_batches, args.batch_size

    # Materialise the batches once so every (steps, shift) cell is scored on
    # byte-identical inputs -- the whole point of the sweep is to isolate the
    # sampler, so re-drawing the data per cell would reintroduce exactly the
    # confound we are trying to remove.
    # Carry each batch's GLOBAL index alongside it: with --num_shards > 1 the
    # local position in `batches` is no longer the eval-set position, and
    # `global_step` below must stay the global one or two shards would write into
    # the same eval_step_%06d directory and overwrite each other's b00..b03 jpgs.
    batches = []
    for i, batch in enumerate(loader):
        if i >= batch_limit:
            break
        if args.num_shards > 1 and i % args.num_shards != args.shard_idx:
            continue
        batches.append((i, batch))
    if args.num_shards > 1:
        accelerator.print(
            f"shard {args.shard_idx}/{args.num_shards}: owns {len(batches)} of "
            f"{batch_limit} batches, global indices "
            f"{[i for i, _ in batches]} -- the sweep summary below is over THESE "
            f"batches only, not the whole eval set"
        )

    summary = {}
    for shift in shift_grid:
        for num_steps in step_grid:
            model.eval_flow_shift = shift         # inference_sigmas only
            model.eval_num_inference_steps = num_steps
            sigmas = model.inference_sigmas(num_steps)
            dsig = (sigmas[:-1] - sigmas[1:])
            cell_dir = os.path.join(output_dir, f"shift{shift:g}_steps{num_steps}")
            accelerator.print(
                f"\n--- eval_flow_shift={shift:g}  steps={num_steps}  "
                f"max_step_dsigma={dsig.max():.3f} (fraction of trajectory in the "
                f"single largest Euler step) ---"
            )

            all_metrics = {}
            for i, batch in batches:
                m = model.evaluate(batch, output_dir=cell_dir, global_step=i,
                                   max_samples=per_batch, save_vis=args.save_vis)
                accelerator.print(
                    f"[batch {i}] " + "  ".join(f"{k}={v:.4f}" for k, v in m.items())
                )
                for k, v in m.items():
                    all_metrics.setdefault(k, []).append(v)
            summary[(shift, num_steps)] = {
                k: float(np.nanmean(v)) for k, v in all_metrics.items()
            }
            # Recompute the aggregate skill from the batch-averaged delta1 and
            # ramp instead of averaging the per-batch ratios, so the denominator
            # is the headroom of the whole eval set.
            cell = summary[(shift, num_steps)]
            for side in ("src", "tgt"):
                d1 = cell.get(f"{side}_depth_delta1")
                ramp = cell.get(f"{side}_depth_delta1_ramp")
                if d1 is None or ramp is None or 1.0 - ramp <= 1e-3:
                    continue
                cell[f"{side}_depth_delta1_skill_agg"] = (d1 - ramp) / (1.0 - ramp)

    keys = ["src_depth_delta1_skill", "src_depth_delta1_skill_agg",
            "src_depth_struct_corr", "src_depth_disp_l1",
            "tgt_depth_delta1_skill", "tgt_depth_delta1_skill_agg",
            "tgt_depth_struct_corr", "tgt_depth_disp_l1",
            "tgt_rgb_psnr", "tgt_rgb_lpips"]
    if args.depth_vae_ceiling:
        # Interleave each ceiling next to the model number it bounds, so the row
        # reads as (model, floor) couples instead of two disconnected halves.
        keys = ["src_depth_abs_rel", "src_depth_vae_abs_rel",
                "src_depth_struct_corr", "src_depth_vae_struct_corr",
                "src_depth_disp_l1", "src_depth_vae_disp_l1",
                "tgt_depth_abs_rel", "tgt_depth_vae_abs_rel",
                "tgt_depth_struct_corr", "tgt_depth_vae_struct_corr",
                "tgt_depth_disp_l1", "tgt_depth_vae_disp_l1",
                "tgt_rgb_psnr", "tgt_rgb_psnr_vae"]
    accelerator.print(f"\n=== sweep summary ({norm_mode}, train shift {train_shift:g}) ===")
    header = f"{'shift':>6} {'steps':>6} " + " ".join(f"{k[-22:]:>22}" for k in keys)
    accelerator.print(header)
    for (shift, num_steps), m in summary.items():
        row = f"{shift:>6g} {num_steps:>6} " + " ".join(
            f"{m.get(k, float('nan')):>22.4f}" for k in keys
        )
        accelerator.print(row)


if __name__ == "__main__":
    main()
