"""Measure training loss as a function of a FIXED sigma on a frozen checkpoint.

Training draws sigma from logit-normal(0, 1) pushed through the shift-5 map, which
puts 99.3% of the mass above sigma=0.3 and 50% above sigma=0.83.  The question this
script answers is: at each noise level, does the model actually use the image
content, or is it just reproducing the content-free solution?

For every sigma we report three numbers on the same masked elements:

  mse_model   the model's velocity MSE
  mse_lin     the best possible CONTENT-FREE prediction, i.e. the least-squares
              scalar a* minimising ||v - a*x_t||^2.  Any predictor that ignores
              the conditioning views cannot beat this.
  mse_zero    predicting v = 0.

skill = 1 - mse_model / mse_lin  is the fraction of the loss that is explained by
actually looking at the images.  skill ~ 0 means the model has learned nothing
usable at that noise level.

Usage:
  python tools/sigma_sweep_loss.py \
      --config train_config/normal_config/bmvs_jf_pix_cam_concat.yaml \
      --checkpoint logs/bmvs_cam_concat/checkpoint-step-5000 \
      --eval_max_samples 32 --batch_size 4
"""
import argparse
import json
import os

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Subset

from examples.dataset import Kubric4DRGBDNVSDataset
from examples.wanvideo.model_training.eval_rgbd_nvs import frozen_eval_indices
from examples.wanvideo.model_training.train_with_accelerate_rgbd_nvs import (
    _masked_mse,
    _resize_mask,
    custom_collate_fn,
)
from examples.wanvideo.model_training.WanRGBDNVSTrainingModule import (
    WanRGBDNVSTrainingModule,
)


def _stats(pred, target, x_t, mask):
    """Model MSE plus the two content-free references, on masked elements only."""
    mask = mask.expand_as(target)
    v = target.float()
    x = x_t.float()
    m = mask.float()
    n = m.sum().clamp_min(1.0)
    svv = (m * v * v).sum()
    sxx = (m * x * x).sum().clamp_min(1e-12)
    sxv = (m * x * v).sum()
    return {
        "mse_model": float(_masked_mse(pred, target, mask)),
        "mse_lin": float((svv - sxv * sxv / sxx) / n),
        "mse_zero": float(svv / n),
        "a_star": float(sxv / sxx),
        "n": float(n),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--sigmas", type=str,
                   default="0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.833,0.9,0.95")
    p.add_argument("--eval_max_samples", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--repeats", type=int, default=2,
                   help="Independent noise draws per (batch, sigma); averaged.")
    p.add_argument("--ablate_source", action="store_true",
                   help="Replace the clean source-view latents with those of a "
                        "DIFFERENT sample in the batch, leaving poses, target "
                        "slots and noise untouched.  If the metrics barely move, "
                        "the model is not using the conditioning views at all and "
                        "is simply refining x_t.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = OmegaConf.load(args.config)
    set_seed(args.seed)
    accelerator = Accelerator(mixed_precision=cfg.mixed_precision)
    sigmas = [float(s) for s in args.sigmas.split(",") if s.strip()]
    norm_mode = cfg.get("disparity_norm_mode", "shared_scale")
    depth_flow_matching = bool(cfg.get("depth_flow_matching", False))

    output_dir = args.output_dir or os.path.join(
        args.checkpoint or cfg.output_path, "sigma_sweep"
    )
    os.makedirs(output_dir, exist_ok=True)

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
        init_dvd_ckpt=None if args.checkpoint else cfg.init_dvd_ckpt,
        rgb_loss_weight=cfg.rgb_loss_weight,
        depth_loss_weight=cfg.depth_loss_weight,
        view_condition_plucker_resolution=cfg.get(
            "view_condition_plucker_resolution", "latent"),
        view_condition_pixel_scale=cfg.get("view_condition_pixel_scale", 8),
        view_condition_mode=cfg.get("view_condition_mode", "add"),
        view_condition_use_role=cfg.get("view_condition_use_role", True),
        view_condition_normalize_moment=cfg.get("view_condition_normalize_moment", False),
        rgb_flow_matching=cfg.get("rgb_flow_matching", True),
        rgb_flow_shift=cfg.get("rgb_flow_shift", 5.0),
        depth_flow_matching=depth_flow_matching,
        depth_head_init=cfg.get("depth_head_init", "pretrained"),
        eval_num_inference_steps=cfg.get("eval_num_inference_steps", 5),
        eval_sample_seed=cfg.get("eval_sample_seed", 1234),
        eval_lpips_net=cfg.get("eval_lpips_net", "alex"),
    )

    if args.checkpoint:
        weights = os.path.join(args.checkpoint, "model.safetensors")
        state = load_file(weights, device="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        accelerator.print(
            f"Loaded {weights}: {len(state)} tensors, "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
    model.to(accelerator.device)
    model.eval()

    dataset = Kubric4DRGBDNVSDataset(
        kubric4d_root=cfg.kubric4d_root,
        manifest_path=cfg.manifest_path,
        resolution=tuple(cfg.resolution),
        num_source_views=cfg.num_source_views,
        num_target_views=cfg.num_target_views,
        random_permute_source_views=cfg.random_permute_source_views,
        disparity_norm_mode=norm_mode,
        disparity_norm_headroom=cfg.get("disparity_norm_headroom", 1.5),
        deterministic_view_seed=int(cfg.get("eval_view_seed", 1234)),
        min_view_valid_fraction=cfg.get("min_view_valid_fraction", 0.0),
        view_validity_path=cfg.get("view_validity_path", None),
        samples_per_scene=cfg.get("samples_per_scene", 1),
    )
    indices = frozen_eval_indices(
        dataset, args.eval_max_samples, cfg.get("samples_per_scene", 1)
    )
    per_batch = min(args.eval_max_samples, args.batch_size)
    loader = DataLoader(
        Subset(dataset, indices), batch_size=per_batch, shuffle=False,
        num_workers=cfg.dataloader_num_workers, collate_fn=custom_collate_fn,
        drop_last=False,
    )
    accelerator.print(
        f"checkpoint={args.checkpoint}\nsigmas={sigmas}  repeats={args.repeats}  "
        f"samples={len(indices)}  depth_flow_matching={depth_flow_matching}\n"
    )

    device = accelerator.device
    num_train_timesteps = model.pipe.scheduler.num_train_timesteps
    acc = {s: {} for s in sigmas}

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            # Match the training mixed-precision context; Accelerate's autocast
            # is not active here, so it has to be entered explicitly.
            with torch.autocast(device_type="cuda", dtype=model.pipe.torch_dtype,
                                enabled=True):
                # VAE-encode once per batch; every sigma reuses the same latents so
                # the comparison across sigma is on byte-identical conditioning.
                inputs = model.forward_preprocess(batch)
                if args.ablate_source:
                    # _build_flow_input reads the clean conditioning views from
                    # inputs["rgb_latents"]; rolling only the source slots swaps
                    # in another scene's appearance while keeping the Plucker
                    # conditioning, the target slots and the noise identical.
                    s = model.num_source_views
                    lat = inputs["rgb_latents"]
                    if lat.shape[0] < 2:
                        raise ValueError("--ablate_source needs batch_size >= 2")
                    lat[:, :, :s] = lat[:, :, :s].roll(1, dims=0).clone()
                rgb_gt = inputs["rgb_gt"]
                depth_gt = inputs["depth_gt"]
                bsz = rgb_gt.shape[0]
                masks = inputs["loss_masks"]

                for s in sigmas:
                    sigma = torch.full((bsz, 1, 1, 1, 1), s,
                                       device=rgb_gt.device, dtype=rgb_gt.dtype)
                    timestep = (sigma.reshape(bsz) * num_train_timesteps).to(
                        dtype=model.pipe.torch_dtype)
                    for r in range(args.repeats):
                        g = torch.Generator(device=device).manual_seed(
                            1000 * bi + 17 * r + int(s * 1e4))
                        noise = torch.randn(rgb_gt.shape, generator=g, device=device,
                                            dtype=rgb_gt.dtype)
                        x_t = (1.0 - sigma) * rgb_gt + sigma * noise
                        v_rgb = noise - rgb_gt

                        if depth_flow_matching:
                            dn = torch.randn(depth_gt.shape, generator=g, device=device,
                                             dtype=depth_gt.dtype)
                            depth_x_t = (1.0 - sigma) * depth_gt + sigma * dn
                            v_depth = dn - depth_gt
                        else:
                            depth_x_t, v_depth = None, depth_gt

                        pred_depth, pred_rgb = model._forward_model(
                            inputs, model._build_flow_input(inputs, x_t, depth_x_t), timestep
                        )
                        dmask = _resize_mask(masks[:, 0], pred_depth.shape[-2:]).to(
                            pred_depth.dtype).permute(0, 2, 1, 3, 4)
                        rmask = _resize_mask(masks[:, 1], pred_rgb.shape[-2:]).to(
                            pred_rgb.dtype).permute(0, 2, 1, 3, 4)

                        cell = acc[s]
                        for tag, st in (
                            ("rgb", _stats(pred_rgb, v_rgb, x_t, rmask)),
                            ("depth", _stats(pred_depth, v_depth,
                                             depth_x_t if depth_flow_matching else depth_gt,
                                             dmask)),
                        ):
                            for k, val in st.items():
                                cell.setdefault(f"{tag}_{k}", []).append(val)
            accelerator.print(f"  batch {bi} done ({bsz} samples)")

    def mean(xs):
        return sum(xs) / len(xs)

    rows = []
    for s in sigmas:
        row = {"sigma": s}
        for tag in ("rgb", "depth"):
            mm, ml, mz = (mean(acc[s][f"{tag}_{k}"])
                          for k in ("mse_model", "mse_lin", "mse_zero"))
            row[f"{tag}_mse"] = mm
            row[f"{tag}_lin"] = ml
            row[f"{tag}_zero"] = mz
            row[f"{tag}_skill"] = 1.0 - mm / ml if ml > 1e-12 else float("nan")
        rows.append(row)

    hdr = (f"{'sigma':>7} | {'rgb_mse':>9} {'rgb_lin':>9} {'rgb_skill':>10} | "
           f"{'dep_mse':>9} {'dep_lin':>9} {'dep_skill':>10}")
    accelerator.print("\n" + hdr)
    accelerator.print("-" * len(hdr))
    for r in rows:
        accelerator.print(
            f"{r['sigma']:>7.3f} | {r['rgb_mse']:>9.4f} {r['rgb_lin']:>9.4f} "
            f"{r['rgb_skill']:>10.4f} | {r['depth_mse']:>9.4f} "
            f"{r['depth_lin']:>9.4f} {r['depth_skill']:>10.4f}"
        )
    accelerator.print(
        "\nskill = 1 - mse_model/mse_lin.  <=0 means the model is no better than "
        "the best predictor that ignores the images entirely."
    )
    out = os.path.join(output_dir, "sigma_sweep.json")
    with open(out, "w") as f:
        json.dump({"checkpoint": args.checkpoint, "rows": rows}, f, indent=2)
    accelerator.print(f"wrote {out}")


if __name__ == "__main__":
    main()
