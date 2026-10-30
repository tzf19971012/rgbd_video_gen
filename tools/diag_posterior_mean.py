"""Diagnostic: how much of the target-RGB gap is the SAMPLING penalty?

`evaluate()` reports PSNR of one stochastic draw.  PSNR is minimised by the
posterior MEAN, so for a well-calibrated model whose GT is an independent draw
from the same posterior,
    E||sample - gt||^2 = 2 * E||mean - gt||^2
exactly -- a fixed 10*log10(2) = 3.01 dB that no architecture change can
recover.  b3@4500 reports tgt_rgb_psnr 13.6 against a VAE ceiling of 30.0, and
before spending training compute on the 16 dB in between we should know how much
of it is definitional.

Draws N independent samples per batch (same conditioning, seeds
eval_sample_seed+i so sample 0 reproduces evaluate() exactly) and reports:

  psnr_mean_k for k=1..N   the Monte-Carlo posterior mean of the first k draws.
                           The k=1 -> k=N rise is the sampling penalty actually
                           being paid.  If it saturates near +3 dB the model is
                           roughly calibrated; much less means the samples are
                           nearly identical (the model is already a regressor
                           and the gap is real error); much more means the
                           posterior is far wider than the data warrants.
  lpips_single vs _mean    averaging should HURT LPIPS.  That is the trade being
                           measured: it separates "sharp but misaligned sampler"
                           from "blurry regressor".
  std_covered/_uncovered   per-pixel posterior std across the N draws, split by
                           the GT-splat partition.  THE key row: a model that
                           knows which pixels its source views determine should
                           be visibly less uncertain on covered pixels.  If the
                           two are equal, its uncertainty is not conditioned on
                           evidence availability -- which would explain the flat
                           covered-vs-uncovered margin in
                           report_20260904_p0a §3.2 and its DiD table, and put
                           the defect in evidence FUSION rather than in warp
                           accuracy (§6 already showed the model reads the warp).

Run as a MODULE from the project root, same as eval_rgbd_nvs.py:
  python -m tools.diag_posterior_mean \
      --config logs/bmvs_b_b3/args.yaml \
      --checkpoint logs/bmvs_b_b3/checkpoint-step-4500 \
      --num_samples 4 --eval_max_samples 32 --batch_size 4 \
      --manifest_path .../manifest.jsonl.bak_113scenes_20260905_112542 \
      --view_validity_path .../manifest.view_validity.json.bak_113scenes_20260905_112542

Cost is N full sampling passes plus N+1 VAE decodes per target view, so it is
~N times an eval.  Pin the manifest: the frozen eval set strides over scenes and
the BlendedMVS manifest grew 113 -> 502 on 2026-09-05.
"""
import argparse
import json
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


def fmt(x, width=8, prec=2):
    return f"{'--':>{width}}" if math.isnan(x) else f"{x:>{width}.{prec}f}"


def bin_by(rows, key, edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r in rows
               if r.get(key) is not None and not math.isnan(r[key]) and lo <= r[key] < hi]
        out.append(((lo, hi), sel))
    return out


def head(n):
    return (
        f"{'stratum':<22}{'n':>5}{'cover':>7}"
        + "".join(f"{f'mean{k}':>8}" for k in range(1, n + 1))
        + f"{'gain':>8}{'std_cov':>9}{'std_unc':>9}{'ratio':>7}"
    )


def line(label, rows, n):
    if not rows:
        return f"{label:<22}{0:>5}"
    m1 = mean([r["psnr_mean_1"] for r in rows])
    mn = mean([r[f"psnr_mean_{n}"] for r in rows])
    sc = mean([r["std_covered"] for r in rows])
    su = mean([r["std_uncovered"] for r in rows])
    return (
        f"{label:<22}{len(rows):>5}"
        + fmt(mean([r["tgt_rgb_splat_coverage"] for r in rows]), 7, 3)
        + "".join(fmt(mean([r[f"psnr_mean_{k}"] for r in rows])) for k in range(1, n + 1))
        + fmt(mn - m1)
        + fmt(sc, 9, 4) + fmt(su, 9, 4)
        + fmt(su / sc if sc and not math.isnan(sc) and sc > 0 else float("nan"), 7)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=4,
                        help="Independent draws per batch. The k-sample Monte-Carlo mean has "
                             "residual variance 1/k of a single draw, so the measured gain "
                             "underestimates the true posterior-mean gain by ~10*log10((1+1/k)/1); "
                             "at k=4 that is 0.97 dB of the 3.01 dB still unrecovered.")
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--eval_flow_shift", type=float, default=None)
    parser.add_argument("--eval_max_samples", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to write per_view_posterior.jsonl. Defaults to the checkpoint.")
    parser.add_argument("--manifest_path", type=str, default=None,
                        help="Override cfg.manifest_path -- see eval_rgbd_nvs.py. Pin this to "
                             "manifest.jsonl.bak_113scenes_20260905_112542 to keep the frozen "
                             "eval set identical to the b-sweep's.")
    parser.add_argument("--view_validity_path", type=str, default=None,
                        help="Override cfg.view_validity_path. Required alongside a '.bak_*' "
                             "manifest, whose derived sidecar path does not exist.")
    parser.add_argument("--split_path", type=str, default=None,
                        help="Frozen scene-id split (configs/bmvs_split_v1.json). Overrides "
                             "cfg.split_path.")
    parser.add_argument("--split", type=str, default=None,
                        help="Which subset of --split_path to score, e.g. 'val'. Overrides "
                             "cfg.eval_split. Without this the dataset has no split, "
                             "frozen_eval_indices falls back to STRIDING over the whole "
                             "manifest, and the numbers describe a different scene set than "
                             "the in-training eval -- see eval_rgbd_nvs.py:41-51.")
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
        # These five leave the parameter shapes untouched, so a checkpoint trained
        # with them ON still loads cleanly with them OFF -- and then silently
        # reports different numbers than train.log, because they change the DiT
        # input block, the AdaLN modulation, the inference path or the positional
        # encoding.  Read them from the config, never default them for a
        # checkpoint whose config says otherwise.  shared_rope_frame_ids is the
        # worst: getting it wrong is a silent RoPE mismatch, not an error, and it
        # cost the c-sweep `sharedrope` cells 1.45 dB of pure harness artefact
        # before the crosscheck against per_view_metrics.jsonl caught it.
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
        eval_num_inference_steps=args.num_inference_steps
        or cfg.get("eval_num_inference_steps", 25),
        eval_flow_shift=args.eval_flow_shift
        if args.eval_flow_shift is not None
        else cfg.get("eval_flow_shift", None),
        eval_sample_seed=cfg.get("eval_sample_seed", 1234),
        eval_lpips_net=cfg.get("eval_lpips_net", "alex"),
    )

    weights = os.path.join(args.checkpoint, "model.safetensors")
    state = load_file(weights, device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    accelerator.print(f"Loaded {weights}: missing={len(missing)}, unexpected={len(unexpected)}")

    # Fail loudly on a CPU fallback.  Without a visible GPU, accelerator.device is
    # cpu, torch.autocast(device_type="cuda") becomes inert, and the bf16 weights
    # meet fp32 inputs 40 stack frames deep inside the VAE as
    # "Input type (float) and bias type (c10::BFloat16) should be the same" --
    # which reads like a dtype bug in this file rather than a missing --gres.
    if accelerator.device.type != "cuda":
        raise SystemExit(
            f"accelerator.device={accelerator.device}: no GPU visible. This script "
            f"needs one (srun/sbatch must allocate --gres=gpu, and --overlap steps "
            f"must not pass --gres=gpu:0)."
        )
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
        # Mirror the in-training eval dataset exactly
        # (train_with_accelerate_rgbd_nvs.py:458-475), or the posterior mean is
        # not comparable to the tgt_rgb_psnr it is meant to replace.
        min_view_valid_fraction=cfg.get(
            "eval_min_view_valid_fraction",
            cfg.get("min_view_valid_fraction", 0.0)),
        view_validity_path=args.view_validity_path or cfg.get("view_validity_path", None),
        samples_per_scene=cfg.get("samples_per_scene", 1),
        split_path=args.split_path or cfg.get("split_path", None),
        split=args.split or cfg.get("eval_split", None),
    )
    indices = frozen_eval_indices(dataset, args.eval_max_samples, cfg.get("samples_per_scene", 1))
    accelerator.print(
        f"eval set identity: manifest={os.path.basename(manifest_path)} "
        f"split={args.split or cfg.get('eval_split', None)!r} "
        f"({len(dataset.records)} scenes)  indices={indices}"
    )
    accelerator.print(
        f"num_samples={args.num_samples}  steps={model.eval_num_inference_steps}  "
        f"eval_flow_shift={model.eval_flow_shift}"
    )
    # Echoed from the constructed MODEL, not from cfg: the point is to catch a
    # flag that the config sets and this file forgets to forward, which is
    # exactly the failure that invalidated the first posterior run.
    accelerator.print(
        # shared_rope_frame_ids stays on this line even though rope_frame_axis
        # supersedes it: the identity gate in sbatch/c_diag_posterior.sbatch greps
        # for `<key>=` per key and would read a missing key as a mismatch.
        f"model identity: shared_rope_frame_ids={model.shared_rope_frame_ids}  "
        f"rope_frame_axis={model.rope_frame_axis}  "
        f"rope_pose_translation_scale={model.rope_pose_translation_scale}  "
        f"roce={model.roce_mode}  "
        f"view_order={dataset.view_order}  "
        f"clean_source_depth={model.clean_source_depth}  "
        f"depth_warp_block={model.depth_warp_block}  "
        f"per_frame_timestep={model.per_frame_timestep}  "
        f"rgb_flow_shift={model.rgb_flow_shift}"
    )
    per_batch = min(args.eval_max_samples, args.batch_size)
    loader = DataLoader(
        Subset(dataset, indices), batch_size=per_batch, shuffle=False,
        num_workers=cfg.dataloader_num_workers, collate_fn=custom_collate_fn, drop_last=False,
    )

    n = args.num_samples
    output_dir = args.output_dir or args.checkpoint
    os.makedirs(output_dir, exist_ok=True)
    jsonl = os.path.join(output_dir, "per_view_posterior.jsonl")

    rows = []
    with open(jsonl, "w", encoding="utf-8") as handle:
        for i, batch in enumerate(loader):
            batch_rows = model.evaluate_posterior_mean(
                batch, num_samples=n, max_samples=per_batch
            )
            for row in batch_rows:
                handle.write(json.dumps(row) + "\n")
            handle.flush()
            rows.extend(batch_rows)
            accelerator.print(
                f"[batch {i}] mean1={mean([r['psnr_mean_1'] for r in batch_rows]):.2f} "
                f"mean{n}={mean([r[f'psnr_mean_{n}'] for r in batch_rows]):.2f} "
                f"std_cov={mean([r['std_covered'] for r in batch_rows]):.4f} "
                f"std_unc={mean([r['std_uncovered'] for r in batch_rows]):.4f}"
            )

    accelerator.print(f"\n=== posterior mean, {len(rows)} target views, N={n} ===")
    accelerator.print(f"wrote {jsonl}")
    hdr = head(n)
    accelerator.print(hdr)
    accelerator.print("-" * len(hdr))
    accelerator.print(line("ALL", rows, n))
    accelerator.print("-" * len(hdr))
    for (lo, hi), sel in bin_by(rows, "tgt_rgb_splat_coverage", [0.0, 0.2, 0.35, 0.5, 0.65, 1.01]):
        accelerator.print(line(f"coverage [{lo:.2f},{hi:.2f})", sel, n))

    accelerator.print("-" * len(hdr))
    m1 = mean([r["psnr_mean_1"] for r in rows])
    mn = mean([r[f"psnr_mean_{n}"] for r in rows])
    # Extrapolate to the true posterior mean.  A k-draw average still carries
    # 1/k of a single draw's variance, so if the single-draw error is
    # e^2 = b^2 + v (bias^2 + posterior variance) then the k-average has
    # b^2 + v/k, and the k-draw gain recovers only (1 - 1/k) of the v term.
    # Solving the two measurements for b^2 and v gives the k->inf limit without
    # assuming calibration -- which is the point, since the 3.01 dB figure is
    # exactly the calibrated special case (v = b^2 -> gain_inf = 3.01 dB).
    e1 = 10 ** (-m1 / 10)
    en = 10 ** (-mn / 10)
    v = (e1 - en) / (1.0 - 1.0 / n) if n > 1 else float("nan")
    b2 = e1 - v
    accelerator.print(
        f"  single draw          {m1:6.2f} dB\n"
        f"  {n}-draw MC mean       {mn:6.2f} dB   (gain {mn - m1:+.2f} dB)"
    )
    if n > 1 and b2 > 0:
        minf = -10 * math.log10(b2)
        accelerator.print(
            f"  extrapolated k->inf  {minf:6.2f} dB   "
            f"(gain {minf - m1:+.2f} dB; posterior variance is "
            f"{100 * v / e1:.0f}% of single-draw MSE, calibrated would be 50%)"
        )
    elif n > 1:
        accelerator.print(
            f"  extrapolated k->inf  diverges: fitted bias^2 = {b2:.2e} <= 0, i.e. the "
            f"{n}-draw gain already exceeds what pure posterior variance can explain "
            f"(draws are anti-correlated, or n is too small to fit two parameters)"
        )
    accelerator.print(
        f"  LPIPS  single {mean([r['lpips_single'] for r in rows]):.4f} -> "
        f"mean-of-{n} {mean([r['lpips_mean'] for r in rows]):.4f}  "
        f"(worse is EXPECTED; averaging trades sharpness for MSE)"
    )
    sc = mean([r["std_covered"] for r in rows])
    su = mean([r["std_uncovered"] for r in rows])
    accelerator.print(
        f"  posterior std  covered {sc:.4f}  uncovered {su:.4f}  "
        f"ratio {su / sc if sc > 0 else float('nan'):.3f}  "
        f"(>1 means uncertainty IS conditioned on evidence availability)"
    )


if __name__ == "__main__":
    main()
