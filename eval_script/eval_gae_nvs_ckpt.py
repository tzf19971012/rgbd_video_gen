"""Standalone GAE-NVS checkpoint eval with pred-vs-GT comparison grids.

Based on the in-training validation path (train_with_accelerate_gae_nvs.py +
WanGAENVSTrainingModule.evaluate): rebuilds the frozen eval set byte-identically
(same cache view-pool clipping, num_train_samples truncation, eval_view_seed,
strided scene indices and per-batch sampling seed), loads a checkpoint, samples
the target latents with the same ODE schedule, decodes through the frozen GAE
codec, and reports PSNR / SSIM / LPIPS / metric-depth numbers.

Unlike the trainer's b*_pred_rgb.jpg strip (predicted views only), every sample
gets a column-aligned comparison grid:

    row 0  SRC GT        -- the conditioning views (rescaled to grid width)
    row 1  PRED tgt      -- sampled novel views
    row 2  GT-DECODE tgt -- cached GT latents decoded straight through the codec
                            (the codec ceiling: blur here is the codec's fault,
                            not the model's)
    row 3  NOVEL RGB GT  -- the ground truth of the novel views

Usage (reproduce the trainer's step-6000 val with GT rows added, NPU 4):
  ASCEND_RT_VISIBLE_DEVICES=4 .venv/bin/python eval_script/eval_gae_nvs_ckpt.py \
      --run-config logs/gae_re10k_rgb_aux20261030_022403/args.yaml \
      --ckpt logs/gae_re10k_rgb_aux20261030_022403/checkpoint-step-6000

The default data path is the trainer's frozen TRAIN-scene eval set, so the
summary numbers are directly comparable with the "[eval step N]" lines in
train.log / tensorboard.  Pass --manifest to point at a different jsonl (the
scenes must exist in cfg.gae_dataset_name's latent cache).
"""

import argparse
import copy
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from omegaconf import OmegaConf
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Subset

try:
    import torch_npu  # noqa: F401

    torch.npu.set_compile_mode(jit_compile=False)
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from accelerate import Accelerator  # noqa: E402
from accelerate.utils import set_seed  # noqa: E402

from examples.wanvideo.model_training.WanGAENVSTrainingModule import (  # noqa: E402
    WanGAENVSTrainingModule,
)
from examples.wanvideo.model_training.train_with_accelerate_gae_nvs import (  # noqa: E402
    build_dataset,
    restrict_views_to_cache,
)
from examples.wanvideo.model_training.train_with_accelerate_rgbd_nvs import (  # noqa: E402
    custom_collate_fn,
    load_dotenv,
)

LABEL_H = 18


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-config", required=True,
                        help="args.yaml of the training run (resolved config)")
    parser.add_argument("--ckpt", required=True,
                        help="checkpoint dir containing model.safetensors")
    parser.add_argument("--output", default=None,
                        help="output dir; default <run_dir>/eval_ckpt_<tag>")
    parser.add_argument("--eval-step-tag", type=int, default=None,
                        help="step tag in filenames/labels; parsed from --ckpt dir name by default")
    parser.add_argument("--manifest", default=None,
                        help="override the manifest jsonl; default = cfg (trainer's frozen eval set)")
    parser.add_argument("--num-samples", type=int, default=0,
                        help="cap on evaluated samples; 0 = cfg.eval_max_samples")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="0 = cfg.eval_batch_size")
    parser.add_argument("--num-inference-steps", type=int, default=0,
                        help="0 = cfg.eval_num_inference_steps")
    parser.add_argument("--no-gt-decode", action="store_true",
                        help="skip the GT-latent codec-ceiling row and its metrics")
    parser.add_argument("--save-vis", type=int, default=10 ** 9,
                        help="render comparison grids for the first N samples")
    parser.add_argument("--seed", type=int, default=0,
                        help="process-level seed (dataset/view/sampling rngs are independently seeded)")
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    return parser.parse_args()


def build_model(cfg, accelerator):
    """Same construction as the GAE training entry point / step-sweep eval."""
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
        gae_latent_channels=cfg.get("gae_latent_channels", 64),
        view_condition_out_channels=cfg.get("view_condition_out_channels", 16),
        keep_pretrained_projections=cfg.get("keep_pretrained_projections", False),
        view_condition_plucker_resolution=cfg.get(
            "view_condition_plucker_resolution", "latent"),
        view_condition_pixel_scale=cfg.get("view_condition_pixel_scale", 8),
        view_condition_mode=cfg.get("view_condition_mode", "add"),
        view_condition_use_role=cfg.get("view_condition_use_role", True),
        view_condition_normalize_moment=cfg.get(
            "view_condition_normalize_moment", False),
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
        gae_prediction=cfg.get("gae_prediction", "v"),
        gae_rope_frame_axis=cfg.get("gae_rope_frame_axis", "view"),
        gae_auto_flow_shift=cfg.get("gae_auto_flow_shift", True),
        gae_codec_device=cfg.get("gae_codec_device", None),
    )
    model.set_training_param()
    return model


def _load_checkpoint(model, ckpt, accelerator):
    weights = os.path.join(ckpt, "model.safetensors")
    if os.path.isdir(ckpt) and os.path.exists(weights):
        pass
    elif os.path.isfile(ckpt):
        weights = ckpt
    else:
        raise FileNotFoundError(f"no model.safetensors under {ckpt}")
    state = load_file(weights, device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    accelerator.print(
        f"[eval] loaded {weights}: {len(state)} tensors, "
        f"missing={len(missing)}, unexpected={len(unexpected)}")
    if missing or unexpected:
        accelerator.print(
            f"[eval] WARNING mismatch: missing={list(missing)[:5]} "
            f"unexpected={list(unexpected)[:5]}")
    del state


def _label_row(text, width):
    margin = Image.new("RGB", (width, LABEL_H), (0, 0, 0))
    ImageDraw.Draw(margin).text((4, 3), text, fill=(255, 255, 255))
    return np.asarray(margin)


def _tiles_to_strip(tiles, out_w):
    """[N, 3, H, W] float in [0,1] -> uint8 HMI of width out_w (aspect kept)."""
    n, _, h, w = tiles.shape
    strip = tiles.permute(1, 0, 2, 3).reshape(1, 3, h, n * w).float()
    out_h = max(1, round(h * out_w / (n * w)))
    strip = F.interpolate(
        strip, size=(out_h, out_w), mode="bilinear", align_corners=False,
    )[0]
    return (strip.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def save_comparison_grid(path, src_gt, pred_tgt, gt_tgt, gt_dec_tgt, caption):
    """Rows: src GT | pred novel | GT-latent decode novel | novel RGB GT.

    All novel rows are full-res and column-aligned (column v = novel view v);
    the source row is rescaled to the same total width for context.
    """
    n_tgt, _, height, width = pred_tgt.shape
    grid_w = n_tgt * width
    rows = [_label_row(caption, grid_w)]

    rows.append(_label_row(
        f"SRC GT ({src_gt.shape[0]} conditioning views)", grid_w))
    rows.append(_tiles_to_strip(src_gt, grid_w))

    rows.append(_label_row("PRED novel views", grid_w))
    rows.append(_tiles_to_strip(pred_tgt, grid_w))

    if gt_dec_tgt is not None:
        rows.append(_label_row("GT-LATENT DECODE novel views (codec ceiling)", grid_w))
        rows.append(_tiles_to_strip(gt_dec_tgt, grid_w))

    rows.append(_label_row("NOVEL-VIEW RGB GT", grid_w))
    rows.append(_tiles_to_strip(gt_tgt, grid_w))

    sep = np.full((2, grid_w, 3), 255, dtype=np.uint8)
    interleaved = []
    for j, row in enumerate(rows):
        if j:
            interleaved.append(sep)
        interleaved.append(row)
    Image.fromarray(np.concatenate(interleaved, axis=0)).save(path)


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.run_config)
    if args.manifest is not None:
        cfg.manifest_path = [args.manifest]

    accelerator = Accelerator(mixed_precision=cfg.mixed_precision)
    set_seed(args.seed)
    load_dotenv()

    tag = args.eval_step_tag
    if tag is None:
        base = os.path.basename(os.path.normpath(args.ckpt))
        if base.startswith("checkpoint-step-"):
            tag = int(base.rsplit("-", 1)[-1])
        else:
            tag = 0
    num_steps = args.num_inference_steps or int(cfg.get("eval_num_inference_steps", 4))
    num_samples = args.num_samples or int(cfg.get("eval_max_samples", 2))
    batch_size = args.batch_size or max(
        1, min(num_samples, int(cfg.get("eval_batch_size", 4))))

    output = args.output or os.path.join(
        os.path.dirname(os.path.normpath(args.ckpt)), f"eval_ckpt_{tag}")
    os.makedirs(output, exist_ok=True)
    accelerator.print(f"[eval] args: {vars(args)}")
    accelerator.print(
        f"[eval] step_tag={tag} steps={num_steps} samples={num_samples} "
        f"batch={batch_size} output={output}")

    model = build_model(cfg, accelerator)
    _load_checkpoint(model, args.ckpt, accelerator)
    model.to(accelerator.device)
    if str(getattr(model.pipe, "device", "")) != str(accelerator.device):
        model.pipe.to(accelerator.device)
    model.pipe.torch_dtype = torch.bfloat16
    model.eval()
    accelerator.print(
        f"[eval] pipe.device={model.pipe.device} torch_dtype={model.pipe.torch_dtype}")

    # -- frozen eval set, byte-identical to the trainer ----------------------
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

    eval_dataset = copy.copy(train_dataset)
    eval_dataset.deterministic_view_seed = int(cfg.get("eval_view_seed", 1234))
    samples_per_scene = getattr(eval_dataset, "samples_per_scene", 1)
    num_scenes = len(eval_dataset.records)
    scene_stride = max(1, num_scenes // num_samples)
    eval_indices = [
        scene_idx * samples_per_scene
        for scene_idx in range(0, num_scenes, scene_stride)
    ][:num_samples]
    loader = DataLoader(
        Subset(eval_dataset, eval_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        collate_fn=custom_collate_fn,
        drop_last=False,
    )
    accelerator.print(f"[eval] frozen eval set: {len(eval_indices)} samples")

    s = model.num_source_views
    lpips_net = model._get_lpips(accelerator.device)
    from skimage.metrics import structural_similarity

    per_view_rows = []
    per_sample = []
    done = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            bs = batch["images"].shape[0]
            height = batch["images"].shape[-2]
            width = batch["images"].shape[-1]

            model._eval_inputs = True
            try:
                inputs = model.forward_preprocess(batch)
            finally:
                model._eval_inputs = False

            # Same per-batch generator seeding as the trainer's evaluate().
            gen = torch.Generator(device=accelerator.device)
            gen.manual_seed(int(cfg.get("eval_sample_seed", 1234)))
            pred_z = model.sample_latents(inputs, num_steps=num_steps, generator=gen)
            pred_rgb, pred_depth = model._decode_views(pred_z, height, width)

            gt_dec_rgb = None
            if not args.no_gt_decode:
                h0, w0 = model._gae_true_grid
                z_tgt = inputs["gae_latents"][:, :, s:][:, :, :, :h0, :w0]
                gt_dec_rgb, _ = model._decode_views(z_tgt, height, width)

            gt_rgb = batch["images"].to(accelerator.device).float().clamp(0, 1)
            gt_depth = batch["depth_metric"][:, :, 0].to(accelerator.device).float()
            valid = (gt_depth > 1e-3).float()

            sample_vals = {"tgt_rgb_psnr": [], "tgt_rgb_ssim": [],
                           "tgt_rgb_lpips": [], "tgt_psnr_vs_gt_decode": [],
                           "gt_decode_rgb_psnr": [], "gt_decode_rgb_ssim": [],
                           "gt_decode_rgb_lpips": [],
                           "tgt_depth_abs_rel": [], "tgt_depth_abs_rel_aligned": [],
                           "tgt_depth_delta1": []}
            src_vals = []
            for b in range(bs):
                gb = done + b
                scene_id = batch["scene_id"][b]
                sample_id = batch.get("sample_id", [""] * bs)[b]
                for v in range(s, model.num_views):
                    pr = pred_rgb[b, v].to(accelerator.device)
                    gr = gt_rgb[b, v]
                    mse = float(((pr - gr) ** 2).mean())
                    psnr = 100.0 if mse < 1e-10 else -10 * math.log10(mse)
                    pn = pred_rgb[b, v].clamp(0, 1).permute(1, 2, 0).numpy()
                    gn = gt_rgb[b, v].cpu().permute(1, 2, 0).numpy()
                    ssim = float(structural_similarity(
                        pn, gn, channel_axis=2, data_range=1.0))
                    row = {
                        "global_index": gb, "batch": i, "b": b,
                        "scene_id": scene_id, "sample_id": sample_id,
                        "tgt_view": v,
                        "tgt_rgb_psnr": psnr, "tgt_rgb_ssim": ssim,
                    }
                    if lpips_net is not None:
                        with torch.no_grad():
                            row["tgt_rgb_lpips"] = float(lpips_net(
                                pr[None].float() * 2 - 1, gr[None].float() * 2 - 1))
                        sample_vals["tgt_rgb_lpips"].append(row["tgt_rgb_lpips"])
                    if gt_dec_rgb is not None:
                        gmse = float(((pred_rgb[b, v] - gt_dec_rgb[b, v - s]) ** 2).mean())
                        vpsnr = 100.0 if gmse < 1e-10 else -10 * math.log10(gmse)
                        row["tgt_psnr_vs_gt_decode"] = vpsnr
                        sample_vals["tgt_psnr_vs_gt_decode"].append(vpsnr)
                        # Codec ceiling: cached GT latents straight through the
                        # codec vs the pixels. Quantifies how much of the error
                        # is the codec vs the generated latent.
                        gdr = gt_dec_rgb[b, v - s].to(accelerator.device)
                        dmse = float(((gdr - gr) ** 2).mean())
                        row["gt_decode_rgb_psnr"] = (
                            100.0 if dmse < 1e-10 else -10 * math.log10(dmse))
                        gdn = gt_dec_rgb[b, v - s].clamp(0, 1).permute(1, 2, 0).numpy()
                        row["gt_decode_rgb_ssim"] = float(structural_similarity(
                            gdn, gn, channel_axis=2, data_range=1.0))
                        sample_vals["gt_decode_rgb_psnr"].append(row["gt_decode_rgb_psnr"])
                        sample_vals["gt_decode_rgb_ssim"].append(row["gt_decode_rgb_ssim"])
                        if lpips_net is not None:
                            with torch.no_grad():
                                row["gt_decode_rgb_lpips"] = float(lpips_net(
                                    gdr[None].float() * 2 - 1, gr[None].float() * 2 - 1))
                            sample_vals["gt_decode_rgb_lpips"].append(
                                row["gt_decode_rgb_lpips"])

                    pd, gd = pred_depth[b, v].to(accelerator.device), gt_depth[b, v]
                    m = valid[b, v].bool() & torch.isfinite(gd)
                    if m.sum() >= 16:
                        p, g = pd[m].float(), gd[m].float()
                        row["tgt_depth_abs_rel"] = float(((p - g).abs() / g).mean())
                        scale = float(g.median() / p.median().clamp_min(1e-6))
                        pa = p * scale
                        row["tgt_depth_abs_rel_aligned"] = float(
                            ((pa - g).abs() / g).mean())
                        row["tgt_depth_delta1"] = float(
                            (torch.maximum(pa / g, g / pa) < 1.25).float().mean())
                    for k in ("tgt_depth_abs_rel", "tgt_depth_abs_rel_aligned",
                              "tgt_depth_delta1"):
                        if k in row:
                            sample_vals[k].append(row[k])
                    for k in ("tgt_rgb_psnr", "tgt_rgb_ssim"):
                        sample_vals[k].append(row[k])
                    per_view_rows.append(row)

                for v in range(s):
                    pd = pred_depth[b, v].to(accelerator.device)
                    gd = gt_depth[b, v]
                    m = valid[b, v].bool() & torch.isfinite(gd)
                    if m.sum() >= 16:
                        p, g = pd[m].float(), gd[m].float()
                        src_vals.append(float(((p - g).abs() / g).mean()))

                if gb < args.save_vis:
                    save_comparison_grid(
                        os.path.join(output, f"b{gb:02d}_pred_vs_gt.jpg"),
                        src_gt=batch["images"][b, :s].float().clamp(0, 1).cpu(),
                        pred_tgt=pred_rgb[b, s:].clamp(0, 1).cpu(),
                        gt_tgt=gt_rgb[b, s:].cpu(),
                        gt_dec_tgt=(gt_dec_rgb[b].clamp(0, 1).cpu()
                                    if gt_dec_rgb is not None else None),
                        caption=(
                            f"{scene_id} | {sample_id} | ckpt-{tag} | "
                            f"{num_steps}-step ODE | sample {gb}"
                        ),
                    )
                sample_vals["src_depth_abs_rel"] = src_vals
                per_sample.append({
                    "global_index": gb, "scene_id": scene_id, "sample_id": sample_id,
                    **{k: float(np.mean(v_)) for k, v_ in sample_vals.items() if v_},
                })

            summary = {k: float(np.nanmean(v_))
                       for k, v_ in sample_vals.items() if v_}
            accelerator.print(
                f"[eval] progress {done + bs}/{len(eval_indices)}: {summary}")
            done += bs
            del inputs, pred_z, pred_rgb, pred_depth, gt_dec_rgb, gt_rgb, gt_depth
            accelerator.free_memory()

    merged = {}
    for key in {k for r in per_view_rows for k in r}:
        if key in ("global_index", "batch", "b", "scene_id", "sample_id", "tgt_view"):
            continue
        vals = [r[key] for r in per_view_rows
                if key in r and not math.isnan(float(r[key]))]
        merged[key] = float(np.mean(vals)) if vals else float("nan")
    if src_vals:
        merged["src_depth_abs_rel"] = float(np.mean(src_vals))

    accelerator.print(
        f"[eval] FINAL ({done} samples, ckpt {args.ckpt}, {num_steps} steps)\n"
        + "\n".join(f"  {k}={v:.4f}" for k, v in sorted(merged.items())))
    with open(os.path.join(output, "summary_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "ckpt": args.ckpt,
            "manifest": str(cfg.manifest_path),
            "num_samples": done,
            "num_inference_steps": num_steps,
            "eval_view_seed": int(cfg.get("eval_view_seed", 1234)),
            "eval_sample_seed": int(cfg.get("eval_sample_seed", 1234)),
            "metrics": merged,
            "per_sample": per_sample,
        }, fh, indent=2)
    with open(os.path.join(output, "per_view_metrics.jsonl"), "w", encoding="utf-8") as fh:
        for row in per_view_rows:
            fh.write(json.dumps(row) + "\n")
    accelerator.print(f"[eval] outputs under {output}")


if __name__ == "__main__":
    main()
