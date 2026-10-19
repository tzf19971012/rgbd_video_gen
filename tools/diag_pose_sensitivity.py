"""Diagnostic: does the model actually USE the abstract camera-pose field?

The companion of tools/diag_warp_sensitivity.py, for the other geometric
input.  Two paths carry camera geometry into this model:

  * the forward splat (`warp_block`) -- explicit pixel correspondence.  §6 of
    report_20260904_p0a_coverage_stratification.md measured it: shuffling it
    costs 3.37 dB, so the model demonstrably reads it.
  * the Plucker ray field (`plucker_block`) -- the abstract "where is this
    camera" signal, which is the ONLY thing available in the 47% of target
    pixels no source pixel reaches.  Never measured.

report_20260910_pose_conditioning_magnitude.md tried to settle it in weight
space and got the wrong answer twice in opposite directions: per-channel
weight RMS says the Plucker path is 8-17x WEAKER than the warp path, while
the exact end-to-end contribution (which accounts for the 384 channels being
pooled by a near-uniform average) says it is 2.4x STRONGER.  Neither number
predicts behaviour, because input magnitude is not causal influence -- the
warp path enters 2.4x smaller and yet shuffling it costs 3.37 dB.  Only an
ablation can answer it.

Six variants on the SAME batches, seed and GT-splat partition:
  baseline     -- normal eval-time inputs
  swap_tgt     -- target poses cyclically shifted among themselves: valid
                  poses, wrong assignment.  THE DECISIVE ONE.
  src0         -- every target pose replaced by source view 0's ("do not move").
                  CONFOUNDED, keep as a sanity check only: on top of corrupting
                  the pose it moves the request out of the target-pose marginal
                  (source 0 is a pose the model saw in the SOURCE role) and
                  collapses all targets onto one pose.  Note it is NOT "free
                  copying": the GT is still the true target view, and copying the
                  nearest source scores 2.0 dB BELOW the model
                  (`tgt_rgb_psnr_copy` = 12.17 vs 14.17 on this val split), so a
                  gain here cannot be read as the model taking an easy way out.
  pose_shuffle -- Plucker block's latent positions permuted
  warp_shuffle -- warp block shuffled, pose intact (the p0a cross-check)
  both_shuffle -- both

How to read it.  `all_` (full-frame PSNR) is the headline, because the pose
field conditions the whole frame; `covered_`/`uncovered_` show WHERE the damage
lands; `lpips_` is the tie-breaker and OUTRANKS PSNR whenever the two disagree.
PSNR's optimum is the conditional MEAN, so a corruption that makes the model
hedge harder can RAISE it while the image gets visibly worse -- and LPIPS has
~1.4x PSNR's discrimination power on this sweep (report_20260907_c_sweep
§9-1b).  A variant that is +PSNR and +LPIPS at once is the signature of
increased blur, not of an improvement.  LPIPS is full-frame only: it is a CNN
over patches, so there is no honest masked version of it.
  * swap_tgt barely hurts  -> the abstract path is decorative and the leverage
    is warp coverage, not pose representation capacity.
  * swap_tgt clearly hurts -> the signal is read, and the ceiling is how well
    it is represented (view_condition_mode=add, a real encoder, ...).
  * neither corruption hurts much -> the bottleneck is the objective, not the
    conditioning (report_20260905_t0_measurement.md's hedging result).

Run as a MODULE from the project root, same as eval_rgbd_nvs.py:
  python -m tools.diag_pose_sensitivity \
      --config logs/bmvs_c_sharedrope_s42/args.yaml \
      --checkpoint logs/bmvs_c_sharedrope_s42/checkpoint-step-3500 \
      --eval_max_samples 32 --batch_size 4
"""
import argparse
import math
import os

from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Subset

from examples.dataset import Kubric4DRGBDNVSDataset
from examples.wanvideo.model_training.eval_rgbd_nvs import frozen_eval_indices
from examples.wanvideo.model_training.train_with_accelerate_rgbd_nvs import custom_collate_fn
from examples.wanvideo.model_training.WanRGBDNVSTrainingModule import WanRGBDNVSTrainingModule


MODES = ("baseline", "swap_tgt", "src0", "pose_shuffle", "warp_shuffle", "both_shuffle")


def mean(values):
    vals = [v for v in values if v is not None and not math.isnan(v)]
    return float("nan") if not vals else sum(vals) / len(vals)


def bin_by(rows, key, edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        out.append(((lo, hi), [r for r in rows if lo <= r[key] < hi]))
    return out


def report_line(label, rows, prefix, digits=2):
    if not rows:
        return f"{label:<22}{0:>5}"
    cells = "".join(
        f"{mean([r[f'{prefix}_{m}'] for r in rows]):>13.{digits}f}" for m in MODES
    )
    return f"{label:<22}{len(rows):>5}{cells}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--eval_max_samples", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--manifest_path", type=str, default=None,
                        help="Override cfg.manifest_path -- see eval_rgbd_nvs.py.")
    parser.add_argument("--view_validity_path", type=str, default=None,
                        help="Override cfg.view_validity_path. Required alongside a '.bak_*' "
                             "manifest, whose derived sidecar path does not exist.")
    parser.add_argument("--split_path", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    set_seed(args.seed)
    accelerator = Accelerator(mixed_precision=cfg.mixed_precision)

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
        num_source_views=cfg.num_source_views,
        num_target_views=cfg.num_target_views,
        vae_encoding_mode=cfg.vae_encoding_mode,
        init_dvd_ckpt=None,
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
        # These nine leave the parameter shapes untouched (droce only half so),
        # which means a checkpoint trained with them ON loads cleanly with them
        # OFF and then reports numbers for a DIFFERENT model.  That failure has
        # already cost this project 1.45 dB of pure harness artefact
        # (report_20260907_c_sweep §9-1a).  Read them from the config, never
        # default them for a checkpoint whose config says otherwise.
        depth_warp_block=cfg.get("depth_warp_block", False),
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
        depth_flow_matching=bool(cfg.get("depth_flow_matching", False)),
        depth_head_init=cfg.get("depth_head_init", "pretrained"),
        eval_num_inference_steps=cfg.get("eval_num_inference_steps", 25),
        eval_flow_shift=cfg.get("eval_flow_shift", None),
        eval_sample_seed=cfg.get("eval_sample_seed", 1234),
        eval_lpips_net=cfg.get("eval_lpips_net", "alex"),
    )

    weights = os.path.join(args.checkpoint, "model.safetensors")
    state = load_file(weights, device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    accelerator.print(f"Loaded {weights}: missing={len(missing)}, unexpected={len(unexpected)}")
    if unexpected:
        accelerator.print(f"  first unexpected: {unexpected[:5]}")

    model.to(accelerator.device)
    model.eval()

    manifest_path = args.manifest_path or cfg.manifest_path
    dataset = Kubric4DRGBDNVSDataset(
        kubric4d_root=cfg.kubric4d_root,
        manifest_path=manifest_path,
        resolution=tuple(cfg.resolution),
        num_source_views=cfg.num_source_views,
        num_target_views=cfg.num_target_views,
        random_permute_source_views=cfg.random_permute_source_views,
        view_order=cfg.get("view_order", "random"),
        disparity_norm_mode=cfg.get("disparity_norm_mode", "shared_scale"),
        disparity_norm_headroom=cfg.get("disparity_norm_headroom", 1.5),
        deterministic_view_seed=int(cfg.get("eval_view_seed", 1234)),
        min_view_valid_fraction=cfg.get("min_view_valid_fraction", 0.0),
        view_validity_path=args.view_validity_path or cfg.get("view_validity_path", None),
        samples_per_scene=cfg.get("samples_per_scene", 1),
        split_path=args.split_path or cfg.get("split_path", None),
        split=args.split or cfg.get("eval_split", None),
    )
    indices = frozen_eval_indices(dataset, args.eval_max_samples, cfg.get("samples_per_scene", 1))
    accelerator.print(
        f"eval set identity: manifest={os.path.basename(manifest_path)} "
        f"({len(dataset.records)} scenes)  split={args.split or cfg.get('eval_split', None)}  "
        f"indices={indices}"
    )
    accelerator.print(
        "model identity: "
        f"plucker={cfg.get('view_condition_plucker_resolution', 'latent')} "
        f"view_cond={cfg.get('view_condition_mode', 'add')} "
        f"shared_rope={cfg.get('shared_rope_frame_ids', False)} "
        f"rope_frame_axis={cfg.get('rope_frame_axis', None)} "
        f"roce={cfg.get('roce', 'off')} "
        f"geom_fp32={cfg.get('camera_geometry_fp32', False)} "
        f"view_order={cfg.get('view_order', 'random')}"
    )

    per_batch = min(args.eval_max_samples, args.batch_size)
    loader = DataLoader(
        Subset(dataset, indices), batch_size=per_batch, shuffle=False,
        num_workers=cfg.dataloader_num_workers, collate_fn=custom_collate_fn, drop_last=False,
    )

    rows = []
    for i, batch in enumerate(loader):
        batch_rows = model.evaluate_pose_sensitivity(batch, max_samples=per_batch)
        rows.extend(batch_rows)
        accelerator.print(f"[batch {i}] " + report_line("", batch_rows, "all").strip())

    header = f"{'stratum':<22}{'n':>5}" + "".join(f"{m:>13}" for m in MODES)
    # LPIPS is full-frame only (see `_lpips` in the module) and LOWER is better,
    # so its delta column is signed the other way.  Both metrics are printed
    # because PSNR on its own cannot adjudicate this: its optimum is the
    # conditional mean, so a corruption that makes the model hedge harder can
    # RAISE PSNR while making the image visibly worse.
    for prefix, unit, digits, better in (
        ("all", "PSNR dB", 2, "higher"),
        ("covered", "PSNR dB", 2, "higher"),
        ("uncovered", "PSNR dB", 2, "higher"),
        ("lpips", "LPIPS", 4, "lower"),
    ):
        accelerator.print(
            f"\n=== pose sensitivity, {prefix} ({unit}, {better} is better), "
            f"{len(rows)} target views ==="
        )
        accelerator.print(header)
        accelerator.print("-" * len(header))
        accelerator.print(report_line("ALL", rows, prefix, digits))
        accelerator.print("-" * len(header))
        for (lo, hi), sel in bin_by(
            rows, "tgt_rgb_splat_coverage", [0.0, 0.2, 0.35, 0.5, 0.65, 1.01]
        ):
            accelerator.print(
                report_line(f"coverage [{lo:.2f},{hi:.2f})", sel, prefix, digits)
            )
        base = mean([r[f"{prefix}_baseline"] for r in rows])
        accelerator.print("-" * len(header))
        for m in MODES[1:]:
            d = mean([r[f"{prefix}_{m}"] for r in rows]) - base
            tag = "" if prefix == "lpips" else " dB"
            hurt = (d < 0) if better == "higher" else (d > 0)
            accelerator.print(
                f"  {prefix}_{m:<14} - {prefix}_baseline = {d:+.{digits}f}{tag}"
                f"   ({'worse' if hurt else 'better'})"
            )


if __name__ == "__main__":
    main()
