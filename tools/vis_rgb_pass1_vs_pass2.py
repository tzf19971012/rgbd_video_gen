"""Visualize the target-view RGB the model produces on PASS 1 (blind, warp
zeroed) vs PASS 2 (warp rebuilt from pass-1's predicted depth) side by side
with GT -- pass 1's target RGB is a full clean latent that sample_rgb()
computes and then discards (WanRGBDNVSTrainingModule.py:1930-1934); only
pass 2's RGB ever gets decoded/saved by evaluate(). This script keeps both.

Reproduces the frozen-eval indexing of train_with_accelerate_rgbd_nvs.py so
b{NN} in a saved eval_step_*/b{NN}_step*.jpg maps to the same dataset item:
chunk_idx, local_b = divmod(NN, eval_batch_size).

Run as a MODULE from the project root, one GPU:
  python -m tools.vis_rgb_pass1_vs_pass2 \
      --run logs/bmvs_d_drocefp32_s42 --step 3750 --b 5 --target 2
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Subset

from examples.wanvideo.model_training.train_with_accelerate_rgbd_nvs import custom_collate_fn
from tools.vis_rgb_warp_pred_depth import build_model, build_dataset, psnr_masked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--b", type=int, required=True, help="b-index of the saved grid, e.g. 5")
    ap.add_argument("--target", type=int, required=True, help="0-indexed target view")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = OmegaConf.load(os.path.join(args.run, "args.yaml"))
    OmegaConf.resolve(cfg)
    set_seed(0)
    accelerator = Accelerator(mixed_precision=cfg.mixed_precision)

    model = build_model(cfg, accelerator)
    ckpt = os.path.join(args.run, f"checkpoint-step-{args.step}")
    nfiles = len(os.listdir(ckpt))
    if nfiles != 11:
        raise SystemExit(f"{ckpt} has {nfiles} files, expected 11 -- write was interrupted")
    state = load_file(os.path.join(ckpt, "model.safetensors"), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"loaded {ckpt}: missing={len(missing)} unexpected={len(unexpected)}")
    model.to(accelerator.device)
    model.eval()
    device = model.pipe.device

    dataset = build_dataset(cfg)
    eval_max_samples = int(cfg.get("eval_max_samples", 2))
    eval_batch_size = max(1, min(eval_max_samples, int(cfg.get("eval_batch_size", 4))))
    samples_per_scene = cfg.get("samples_per_scene", 1)
    eval_indices = [i * samples_per_scene for i in range(len(dataset.records))][:eval_max_samples]
    chunk_idx, local_b = divmod(args.b, eval_batch_size)
    loader = DataLoader(
        Subset(dataset, eval_indices), batch_size=eval_batch_size, shuffle=False,
        num_workers=0, collate_fn=custom_collate_fn, drop_last=False,
    )
    batch = None
    for i, candidate in enumerate(loader):
        if i == chunk_idx:
            batch = candidate
            break
    if batch is None:
        raise SystemExit(f"chunk {chunk_idx} not in loader")
    print(f"b{args.b:02d} -> chunk {chunk_idx}, local {local_b}; "
          f"sample_id={batch['sample_id'][local_b]}")

    num_src = model.num_source_views
    t = args.target
    vi = num_src + t
    b = local_b

    autocast_ctx = torch.autocast(device_type="cuda", dtype=model.pipe.torch_dtype, enabled=True)
    with torch.no_grad(), autocast_ctx:
        model._eval_inputs = True
        try:
            inputs = model.forward_preprocess(batch)
        finally:
            model._eval_inputs = False

        # ---- pass 1: blind (warp zeroed) -- keep its target RGB, normally discarded
        gen = torch.Generator(device=device)
        gen.manual_seed(model.eval_sample_seed)
        blind = dict(inputs)
        blind["warp_block"] = torch.zeros_like(inputs["warp_block"])
        if inputs.get("depth_warp_block") is not None:
            blind["depth_warp_block"] = torch.zeros_like(inputs["depth_warp_block"])
        pred_depth_p1, pred_rgb_p1 = model._sample_rgb_once(
            blind, model.eval_num_inference_steps, gen)

        # ---- pass 2: warp rebuilt from pass-1's predicted depth
        warp_block, depth_warp_block = model._warp_block_from_predicted_depth(
            inputs, pred_depth_p1)
        second = dict(inputs)
        second["warp_block"] = warp_block
        second["depth_warp_block"] = depth_warp_block
        clean_src_depth = (
            pred_depth_p1[:, :, :num_src] if model.clean_source_depth else None)
        pred_depth_p2, pred_rgb_p2 = model._sample_rgb_once(
            second, model.eval_num_inference_steps, gen, clean_src_depth=clean_src_depth)

        if model.latent_scale != 1.0:
            pred_rgb_p1 = pred_rgb_p1 / model.latent_scale
            pred_rgb_p2 = pred_rgb_p2 / model.latent_scale

        def decode_view(latents):
            v_lat = latents[b:b + 1, :, vi:vi + 1].to(device).to(model.pipe.torch_dtype)
            return model.pipe.vae.decode(v_lat, device=device)[:, :, 0].float().clamp(0, 1)

        pred_rgb_px_p1 = decode_view(pred_rgb_p1)
        pred_rgb_px_p2 = decode_view(pred_rgb_p2)

        rgb_px = batch["images"].to(device).float().clamp(0.0, 1.0)  # [B,V,3,H,W]

    gt_px = rgb_px[b:b + 1, vi]  # [1,3,H,W]

    ones = torch.ones_like(gt_px[:, :1])
    print()
    print(f"=== target {t} of {batch['sample_id'][b]} ===")
    print(f"  pass 1 (blind, no warp)  psnr_full={psnr_masked(pred_rgb_px_p1, gt_px, ones):6.2f}")
    print(f"  pass 2 (warp @ pred depth) psnr_full={psnr_masked(pred_rgb_px_p2, gt_px, ones):6.2f}")

    def img(x):
        return x[0].permute(1, 2, 0).cpu().numpy().clip(0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(3 * 3.2, 3.2))
    for ax in axes:
        ax.axis("off")
    axes[0].imshow(img(gt_px))
    axes[0].set_title(f"tgt{t+1} GT", fontsize=10)
    axes[1].imshow(img(pred_rgb_px_p1))
    axes[1].set_title(
        f"tgt{t+1} pass 1 (blind)\npsnr={psnr_masked(pred_rgb_px_p1, gt_px, ones):.2f}",
        fontsize=10)
    axes[2].imshow(img(pred_rgb_px_p2))
    axes[2].set_title(
        f"tgt{t+1} pass 2 (final)\npsnr={psnr_masked(pred_rgb_px_p2, gt_px, ones):.2f}",
        fontsize=10)

    fig.suptitle(
        f"{os.path.basename(args.run)} step{args.step}  {batch['sample_id'][b]}  "
        f"b{args.b:02d} target#{t}", fontsize=11)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.82, bottom=0.02, wspace=0.05)
    out = args.out or (f"tools/rgb_pass1_vs_pass2_{os.path.basename(args.run)}"
                        f"_step{args.step}_b{args.b:02d}_t{t}.png")
    plt.savefig(out, dpi=140)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
