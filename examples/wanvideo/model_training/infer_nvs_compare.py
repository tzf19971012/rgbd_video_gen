"""Qualitative inference for the GAE / GeoColor-Token NVS runs.

Loads the LATEST (or --checkpoint) checkpoint of a run directory, samples
--num_samples targets on a deterministic scene-strided subset of the
cache-restricted dataset pool, and writes one labelled grid per sample in the
reference eval-grid layout (WanRGBDNVSTrainingModule._compose_eval_grid):
rows rgb / depth gt / depth pred, columns src1..S then (tgt gt, tgt pred)
per target.  Source depth tiles are the codec round-trip of the conditioning
latent (what the model actually sees as evidence); GT rgb / GT depth come
straight from the dataset batch.

The dataset pool is restricted to BOTH caches (GAE + Wan) even for the
pure-GAE runs, so all three arms score byte-identical samples.

Run from the repo root:
  ASCEND_RT_VISIBLE_DEVICES=3 python -m examples.wanvideo.model_training.infer_nvs_compare \
      --run_dir logs/ovf_gae_re10k_aux0_patch1 --num_samples 20
"""
import argparse
import json
import math
import os

try:
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
except ImportError:
    pass

import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from examples.dataset import Kubric4DRGBDNVSDataset
from examples.wanvideo.model_training.train_with_accelerate_rgbd_nvs import custom_collate_fn
from examples.wanvideo.model_training.train_with_accelerate_gae_nvs import (
    build_dataset,
    restrict_views_to_cache,
)
from examples.wanvideo.model_training.WanRGBDNVSTrainingModule import (
    _compose_eval_grid,
)


def latest_checkpoint(run_dir):
    steps = []
    for name in os.listdir(run_dir):
        if name.startswith("checkpoint-step-"):
            try:
                steps.append(int(name.rsplit("-", 1)[1]))
            except ValueError:
                continue
    if not steps:
        raise FileNotFoundError(f"no checkpoint-step-* under {run_dir}")
    return os.path.join(run_dir, f"checkpoint-step-{max(steps)}"), max(steps)


def build_model(kind, cfg, accelerator):
    common = dict(
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
        view_condition_normalize_moment=cfg.get(
            "view_condition_normalize_moment", False),
        camera_geometry_fp32=cfg.get("camera_geometry_fp32", False),
        rgb_flow_shift=cfg.get("rgb_flow_shift", 5.0),
        rgb_flow_timestep_sampling=cfg.get(
            "rgb_flow_timestep_sampling", "logit_normal"),
        rgb_flow_logit_mean=cfg.get("rgb_flow_logit_mean", 0.0),
        rgb_flow_logit_std=cfg.get("rgb_flow_logit_std", 1.0),
        eval_num_inference_steps=cfg.get("eval_num_inference_steps", 25),
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
        gae_rgb_loss_weight=cfg.get("gae_rgb_loss_weight", 0.0),
        gae_rgb_t_max=cfg.get("gae_rgb_t_max", 0.4),
        gae_rgb_max_views=cfg.get("gae_rgb_max_views", 1),
    )
    if kind == "geocolor_token":
        from examples.wanvideo.model_training.WanGeoColorTokenNVSTrainingModule import (
            WanGeoColorTokenNVSTrainingModule,
        )

        return WanGeoColorTokenNVSTrainingModule(
            gae_patch_mode=cfg.get("gae_patch_mode", "native"),
            wan_latent_cache_root=cfg.get(
                "wan_latent_cache_root", "/mnt/sfs_turbo/t00845730/wan_latent_cache"),
            wan_latent_dataset_name=cfg.get(
                "wan_latent_dataset_name", "re10k_ovf8_wan"),
            wan_latent_stats_path=cfg.get("wan_latent_stats_path", None),
            color_loss_weight=cfg.get("color_loss_weight", 1.0),
            **common,
        )
    from examples.wanvideo.model_training.WanGAENVSTrainingModule import (
        WanGAENVSTrainingModule,
    )

    return WanGAENVSTrainingModule(
        gae_patch_mode=cfg.get("gae_patch_mode", "native"),
        v_loss_t_eps=cfg.get("v_loss_t_eps", 0.05),
        camera_drop_prob=cfg.get("camera_drop_prob", 0.0),
        ref_drop_prob=cfg.get("ref_drop_prob", 0.0),
        eval_cfg_scale=cfg.get("eval_cfg_scale", 1.0),
        **common,
    )


def _rgb_np(t):
    """[3,H,W] float in [0,1] -> HxWx3 uint8."""
    return (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def _depth_np(d, lo, hi):
    """metric depth HxW -> plasma HxWx3 uint8 on the SHARED lo/hi scale."""
    import matplotlib.cm as cm

    t = np.clip((d - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return (cm.plasma(t)[:, :, :3] * 255).astype(np.uint8)


def _align_median(pred, gt, valid):
    """Scale pred so its median matches the GT median (valid pixels only)."""
    p = float(pred[valid].median()) if valid.any() else 0.0
    g = float(gt[valid].median()) if valid.any() else 0.0
    return pred * (g / max(p, 1e-6))


@torch.no_grad()
def run_one(run_dir, checkpoint, output_dir, num_samples, batch_size,
            num_inference_steps=None, downscale=2):
    cfg = OmegaConf.load(os.path.join(run_dir, "args.yaml"))
    kind = ("geocolor_token"
            if cfg.get("wan_latent_cache_root", None) is not None else "gae")
    accelerator = Accelerator(mixed_precision=cfg.mixed_precision)
    accelerator.print(f"kind={kind}  ckpt={checkpoint}")
    accelerator.print(OmegaConf.to_yaml(cfg))

    model = build_model(kind, cfg, accelerator)
    model.set_training_param()
    from safetensors.torch import load_file

    weights = os.path.join(checkpoint, "model.safetensors")
    state = load_file(weights, device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    accelerator.print(
        f"Loaded {weights}: {len(state)} tensors, missing={len(missing)}, "
        f"unexpected={len(unexpected)}")
    if missing:
        accelerator.print(f"  first missing: {missing[:5]}")
    model.to(accelerator.device)
    model.eval()
    if num_inference_steps is not None:
        model.eval_num_inference_steps = int(num_inference_steps)

    os.makedirs(output_dir, exist_ok=True)

    # Same pool recipe as the trainers, restricted to BOTH caches so every arm
    # scores identical scenes (the Wan color cache holds only 8 re10k scenes).
    num_views = int(cfg.num_source_views) + int(cfg.num_target_views)
    dataset = build_dataset(cfg, split=cfg.get("eval_split", None))
    restrict_views_to_cache(
        dataset, cfg.get("gae_cache_root"), cfg.get("gae_dataset_name", "bmvs"),
        num_views, accelerator)
    restrict_views_to_cache(
        dataset, cfg.get(
            "wan_latent_cache_root", "/mnt/sfs_turbo/t00845730/wan_latent_cache"),
        cfg.get("wan_latent_dataset_name", "re10k_ovf8_wan"),
        num_views, accelerator)
    dataset.deterministic_view_seed = int(cfg.get("eval_view_seed", 1234))

    pool = len(dataset)
    if num_samples > pool:
        accelerator.print(f"pool only has {pool} samples; clamping")
        num_samples = pool
    stride = max(1, pool // num_samples)
    indices = list(range(0, pool, stride))[:num_samples]
    accelerator.print(
        f"pool={pool} scenes={len(dataset.records)} -> {num_samples} samples, "
        f"indices={indices}")

    loader = DataLoader(
        Subset(dataset, indices), batch_size=batch_size, shuffle=False,
        num_workers=cfg.dataloader_num_workers, collate_fn=custom_collate_fn,
        drop_last=False)

    s = int(cfg.num_source_views)
    V = s + int(cfg.num_target_views)
    device = accelerator.device
    lpips_net = model._get_lpips(device)
    per_sample = []

    for bi, batch in enumerate(loader):
        save_dir = output_dir
        b0 = bi * batch_size
        def _slice(x, n):
            if isinstance(x, torch.Tensor):
                return x[:n]
            if isinstance(x, list):
                return x[:n]
            return x
        batch_small = {k: _slice(v, batch_size) for k, v in batch.items()}
        model._eval_inputs = True
        try:
            inputs = model.forward_preprocess(batch_small)
        finally:
            model._eval_inputs = False

        gen = torch.Generator(device=device)
        gen.manual_seed(int(cfg.get("eval_sample_seed", 1234)))
        if kind == "geocolor_token":
            pred_geo, pred_color = model.sample_latents(
                inputs=inputs, generator=gen)
            height = batch_small["images"].shape[-2]
            width = batch_small["images"].shape[-1]
            pred_rgb = model._decode_color_views(pred_color, height, width)
            pred_depth = model._decode_geo_depth(pred_geo, height, width)
        else:
            pred_z = model.sample_latents(
                inputs, num_steps=model.eval_num_inference_steps, generator=gen)
            height = batch_small["images"].shape[-2]
            width = batch_small["images"].shape[-1]
            pred_rgb, pred_depth = model._decode_views(pred_z, height, width)

        gt_rgb = batch_small["images"].float().clamp(0, 1)
        gt_depth = batch_small["depth_metric"][:, :, 0].float()
        B = pred_rgb.shape[0]

        for b in range(B):
            sid = batch_small["scene_id"][b] if "scene_id" in batch_small else f"b{b}"
            sample_idx = b0 + b
            # Per-sample shared depth window from the VALID GT of all views.
            vals = torch.cat([
                gt_depth[b, v][gt_depth[b, v] > 1e-3] for v in range(V)])
            lo = float(np.percentile(vals.numpy(), 2))
            hi = float(np.percentile(vals.numpy(), 98))

            src_rgb_nps, src_gt_nps, src_pred_nps, src_caps = [], [], [], []
            tgt_tiles, tgt_caps = [], []
            sample_metrics = {"sample_id": sample_idx, "scene_id": sid}
            rels, d1s, psnrs, lpipss, rels_raw = [], [], [], [], []

            for v in range(s):
                gd = gt_depth[b, v]
                valid = (gd > 1e-3) & torch.isfinite(gd)
                pd = pred_depth[b, v]
                pa = _align_median(pd, gd, valid)
                rel = float((((pa - gd) / gd).abs()[valid]).mean()) if valid.sum() >= 16 else float("nan")
                src_rgb_nps.append(_rgb_np(gt_rgb[b, v]))
                src_gt_nps.append(_depth_np(gd.numpy(), lo, hi))
                src_pred_nps.append(_depth_np(pa.numpy(), lo, hi))
                src_caps.append([f"src{v + 1} absrel={rel:.3f}"])

            for t in range(s, V):
                gr = gt_rgb[b, t]
                pr = pred_rgb[b, t].clamp(0, 1)
                gd = gt_depth[b, t]
                pd = pred_depth[b, t]
                valid = (gd > 1e-3) & torch.isfinite(gd)
                mse = float(((pr - gr) ** 2).mean())
                psnr = 100.0 if mse < 1e-10 else -10 * math.log10(mse)
                lp = float("nan")
                if lpips_net is not None:
                    with torch.no_grad():
                        lp = float(lpips_net(
                            pr[None].to(device).float() * 2 - 1,
                            gr[None].to(device).float() * 2 - 1))
                pa = _align_median(pd, gd, valid)
                if valid.sum() >= 16:
                    rel = float((((pa - gd) / gd).abs()[valid]).mean())
                    rel_raw = float((((pd - gd) / gd).abs()[valid]).mean())
                    d1 = float((torch.maximum(
                        pa / gd, gd / pa)[valid] < 1.25).float().mean())
                else:
                    rel = rel_raw = d1 = float("nan")
                tgt_tiles.append((
                    _rgb_np(gr), _rgb_np(pr),
                    _depth_np(gd.numpy(), lo, hi),
                    _depth_np(pa.numpy(), lo, hi),
                ))
                tgt_caps.append([
                    f"psnr={psnr:.2f} lpips={lp:.3f}",
                    f"d1={d1:.3f} rel={rel:.3f}",
                ])
                psnrs.append(psnr); lpipss.append(lp)
                d1s.append(d1); rels.append(rel); rels_raw.append(rel_raw)

            grid, warning = _compose_eval_grid(
                src_rgb_nps, src_gt_nps, src_pred_nps, tgt_tiles,
                downscale=downscale, src_captions=src_caps, tgt_captions=tgt_caps)
            if warning:
                accelerator.print(f"[sample {sample_idx}] {warning}")
            out_fp = os.path.join(save_dir, f"sample_{sample_idx:02d}_{sid}.jpg")
            grid.save(out_fp, quality=92)

            def _mean(xs):
                xs = [x for x in xs if x == x]
                return float(np.mean(xs)) if xs else float("nan")
            sample_metrics.update(
                tgt_rgb_psnr=_mean(psnrs), tgt_rgb_lpips=_mean(lpipss),
                tgt_depth_delta1=_mean(d1s),
                tgt_depth_abs_rel_aligned=_mean(rels),
                tgt_depth_abs_rel=_mean(rels_raw),
            )
            per_sample.append(sample_metrics)
            accelerator.print(
                f"[sample {sample_idx:02d}] {sid}  "
                + "  ".join(f"{k}={v:.4f}" for k, v in sample_metrics.items()
                            if k not in ("scene_id",)))

    with open(os.path.join(output_dir, "per_sample_metrics.jsonl"), "w",
              encoding="utf-8") as fh:
        for row in per_sample:
            fh.write(json.dumps(row) + "\n")

    accelerator.print(f"\n=== {run_dir} @{os.path.basename(checkpoint)} "
                      f"({len(per_sample)} samples, "
                      f"{model.eval_num_inference_steps} steps) ===")
    for k in ("tgt_rgb_psnr", "tgt_rgb_lpips", "tgt_depth_delta1",
              "tgt_depth_abs_rel_aligned", "tgt_depth_abs_rel"):
        vals = [r[k] for r in per_sample if r[k] == r[k]]
        accelerator.print(f"  {k}: {float(np.mean(vals)):.4f}" if vals
                          else f"  {k}: nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="defaults to the LATEST checkpoint-step-* in --run_dir")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--downscale", type=int, default=2)
    args = parser.parse_args()

    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint, step = latest_checkpoint(args.run_dir)
    else:
        step = int(checkpoint.rsplit("-", 1)[1])
    output_dir = args.output_dir or os.path.join(
        args.run_dir, f"infer_compare_step{step}")
    run_one(args.run_dir, checkpoint, output_dir, args.num_samples,
            args.batch_size, args.num_inference_steps, args.downscale)


if __name__ == "__main__":
    main()
