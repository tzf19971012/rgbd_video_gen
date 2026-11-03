"""Diagnostic: does the model's covered-pixel accuracy depend on the warp
block carrying correct, pixel-precise correspondence -- or on source-depth
accuracy at all?

Motivated by report_20260904_p0a_coverage_stratification.md: the model beats
a fair (predicted-depth) splat baseline everywhere, but the margin it gets
from having more correspondence available (covered - uncovered) does not
grow with coverage. That is consistent with either (a) the model uses the
warp channel but is capped by its own source-depth accuracy, or (b) the
model barely uses the warp channel's pixel-level content at all, regardless
of how accurate it is. This script tells the two apart by comparing, on the
SAME batches/seed/GT-splat partition:
  baseline -- normal eval-time warp block (two-pass, model's own depth)
  shuffle  -- baseline block with spatial positions permuted (same values,
              zero correspondence)
  lowfreq  -- baseline block low-pass filtered (coarse layout kept, detail
              destroyed)
  oracle   -- warp block rebuilt from GT source depth (upper bound on what a
              perfect source-depth predictor could buy)
If shuffle/lowfreq barely hurt and oracle barely helps, the model is not
using pixel-level correspondence from the warp channel.

Run as a MODULE from the project root, same as eval_rgbd_nvs.py:
  python -m tools.diag_warp_sensitivity \
      --config logs/bmvs_b_b3/args.yaml \
      --checkpoint logs/bmvs_b_b3/checkpoint-step-4500 \
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


def mean(values):
    vals = [v for v in values if v is not None and not math.isnan(v)]
    return float("nan") if not vals else sum(vals) / len(vals)


def bin_by(rows, key, edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r in rows if lo <= r[key] < hi]
        out.append(((lo, hi), sel))
    return out


MODES = ("baseline", "shuffle", "lowfreq", "oracle")


def report_line(label, rows):
    if not rows:
        return f"{label:<22}{0:>5}"
    cells = "".join(f"{mean([r[f'covered_{m}'] for r in rows]):>10.2f}" for m in MODES)
    return f"{label:<22}{len(rows):>5}{cells}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--eval_max_samples", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--manifest_path", type=str, default=None,
                        help="Override cfg.manifest_path -- see eval_rgbd_nvs.py. Pin this to "
                             "manifest.jsonl.bak_113scenes_20260905_112542 to keep the frozen "
                             "eval set identical to the b-sweep's.")
    parser.add_argument("--view_validity_path", type=str, default=None,
                        help="Override cfg.view_validity_path. Required alongside a '.bak_*' "
                             "manifest, whose derived sidecar path does not exist.")
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
        # Shape-preserving flags that change the DiT input block, the AdaLN
        # modulation or the positional encoding: a checkpoint trained with them
        # ON loads cleanly with them OFF and then reports numbers for a
        # different model (report_20260907_c_sweep §9-1a, 1.45 dB of pure
        # harness artefact).  This script was written before the c/d rounds
        # existed and defaulted all of them, so every delta it produced on a
        # c-sweep or D-sweep checkpoint was measured on the wrong model.
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

    model.to(accelerator.device)
    model.eval()

    dataset = Kubric4DRGBDNVSDataset(
        kubric4d_root=cfg.kubric4d_root,
        manifest_path=args.manifest_path or cfg.manifest_path,
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
        split_path=cfg.get("split_path", None),
        split=cfg.get("eval_split", None),
    )
    indices = frozen_eval_indices(dataset, args.eval_max_samples, cfg.get("samples_per_scene", 1))
    accelerator.print(
        f"eval set identity: manifest={os.path.basename(args.manifest_path or cfg.manifest_path)} "
        f"({len(dataset.records)} scenes)  indices={indices}"
    )
    per_batch = min(args.eval_max_samples, args.batch_size)
    loader = DataLoader(
        Subset(dataset, indices), batch_size=per_batch, shuffle=False,
        num_workers=cfg.dataloader_num_workers, collate_fn=custom_collate_fn, drop_last=False,
    )

    rows = []
    for i, batch in enumerate(loader):
        batch_rows = model.evaluate_warp_sensitivity(batch, max_samples=per_batch)
        rows.extend(batch_rows)
        accelerator.print(f"[batch {i}] " + report_line("", batch_rows).strip())

    accelerator.print(f"\n=== warp sensitivity, {len(rows)} target views ===")
    header = f"{'stratum':<22}{'n':>5}" + "".join(f"{m:>10}" for m in MODES)
    accelerator.print(header)
    accelerator.print("-" * len(header))
    accelerator.print(report_line("ALL", rows))
    accelerator.print("-" * len(header))
    for (lo, hi), sel in bin_by(rows, "tgt_rgb_splat_coverage", [0.0, 0.2, 0.35, 0.5, 0.65, 1.01]):
        accelerator.print(report_line(f"coverage [{lo:.2f},{hi:.2f})", sel))

    b = mean([r["covered_baseline"] for r in rows])
    accelerator.print("-" * len(header))
    for m in ("shuffle", "lowfreq", "oracle"):
        d = mean([r[f"covered_{m}"] for r in rows]) - b
        accelerator.print(f"  covered_{m:<10} - covered_baseline = {d:+.2f} dB")


if __name__ == "__main__":
    main()
