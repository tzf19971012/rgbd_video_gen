"""Visualize the RGB warp the model ACTUALLY receives at inference -- the one
rebuilt from its own predicted source depth -- side by side with the GT-depth
warp (the oracle upper bound) for the same sample/target.

WHY.  tools/vis_rgb_warp_one_sample.py splats with GT source disparity, so it
shows a warp the model never sees: forward_preprocess zeroes warp_block under
`_eval_inputs` (WanRGBDNVSTrainingModule.py:1314-1328) and sample_rgb's two-pass
rebuilds it from the pass-1 source-depth estimate (:1693-1732).  On
57f8d9bbe73f6760f10e916a/tt0/s50 target 2 that is the difference between
22.56 dB (GT depth) and 10.55 dB (predicted depth) on covered pixels, which is
the whole reason the prediction looks wrong.  This script renders both so the
gap is visible rather than inferred, plus the per-source-view disparity
comparison that causes it.

Reproduces the frozen-eval indexing of train_with_accelerate_rgbd_nvs.py so
b{NN} in a saved eval_step_*/b{NN}_step*.jpg maps to the same dataset item:
chunk_idx, local_b = divmod(NN, eval_batch_size).

Run as a MODULE from the project root, one GPU:
  python -m tools.vis_rgb_warp_pred_depth \
      --run logs/bmvs_d_drocefp32_s42 --step 3750 --b 5 --target 2
"""
import argparse
import inspect
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Subset

from examples.dataset import Kubric4DRGBDNVSDataset
from examples.wanvideo.model_training.train_with_accelerate_rgbd_nvs import custom_collate_fn
from examples.wanvideo.model_training.WanRGBDNVSTrainingModule import WanRGBDNVSTrainingModule


def build_model(cfg, accelerator):
    """Construct the module with EVERY __init__ param the run's config sets.

    Introspection rather than a hand-written kwargs list: the shape-preserving
    flags (shared_rope_frame_ids, roce, camera_geometry_fp32, per_frame_timestep,
    ...) load cleanly when omitted and then silently describe a different model
    -- 1.45 dB of pure harness artefact in report_20260907_c_sweep §9-1a.  A
    hand-written list is exactly what went stale there.
    """
    params = inspect.signature(WanRGBDNVSTrainingModule.__init__).parameters
    skip = {"self", "args", "accelerator", "model_paths"}
    kwargs = {}
    for name in params:
        if name in skip:
            continue
        if name in cfg:
            kwargs[name] = cfg[name]
    # Not training: checkpointing only costs time here.
    kwargs["use_gradient_checkpointing"] = False
    kwargs["use_gradient_checkpointing_offload"] = False
    kwargs["init_dvd_ckpt"] = None
    model = WanRGBDNVSTrainingModule(args=cfg, accelerator=accelerator, **kwargs)
    print(
        "model identity: "
        f"warp_mode={model.warp_mode} warp_two_pass={model.warp_two_pass} "
        f"shared_rope_frame_ids={model.shared_rope_frame_ids} "
        f"roce={model.roce_mode} camera_geometry_fp32={model.camera_geometry_fp32} "
        f"per_frame_timestep={model.per_frame_timestep} "
        f"clean_source_depth={model.clean_source_depth} "
        f"token_layout={model.token_layout} "
        f"eval_num_inference_steps={model.eval_num_inference_steps} "
        f"eval_flow_shift={model.eval_flow_shift} "
        f"eval_sample_seed={model.eval_sample_seed}"
    )
    return model


def build_dataset(cfg):
    dataset = Kubric4DRGBDNVSDataset(
        kubric4d_root=cfg.kubric4d_root,
        manifest_path=cfg.manifest_path,
        resolution=tuple(cfg.resolution),
        num_source_views=cfg.num_source_views,
        num_target_views=cfg.num_target_views,
        random_permute_source_views=cfg.random_permute_source_views,
        view_order=cfg.get("view_order", "random"),
        disparity_norm_mode=cfg.get("disparity_norm_mode", "shared_scale"),
        disparity_norm_quantile=cfg.get("disparity_norm_quantile", 0.98),
        disparity_norm_headroom=cfg.get("disparity_norm_headroom", 1.5),
        min_view_valid_fraction=cfg.get(
            "eval_min_view_valid_fraction", cfg.get("min_view_valid_fraction", 0.0)),
        view_validity_path=cfg.get("view_validity_path", None),
        samples_per_scene=cfg.get("samples_per_scene", 1),
        split_path=cfg.get("split_path", None),
        split=cfg.get("eval_split", None),
    )
    dataset.deterministic_view_seed = int(cfg.get("eval_view_seed", 1234))
    return dataset


def psnr_masked(pred, gt, mask):
    m = mask.expand_as(pred)
    n = m.sum()
    if n < 1:
        return float("nan")
    mse = ((((pred - gt) ** 2) * m).sum() / n).item()
    return 100.0 if mse < 1e-10 else -10 * math.log10(mse)


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
    vi = num_src + args.target

    # ---- replicate evaluate(): preprocess under _eval_inputs (warp zeroed) ----
    autocast_ctx = torch.autocast(device_type="cuda", dtype=model.pipe.torch_dtype, enabled=True)
    with torch.no_grad(), autocast_ctx:
        model._eval_inputs = True
        try:
            inputs = model.forward_preprocess(batch)
        finally:
            model._eval_inputs = False

        # ---- replicate sample_rgb's two pass, but keep the pass-1 depth -------
        # The warp block the model is fed on pass 2 is built from PASS 1's
        # source-depth estimate.  evaluate() logs tgt_rgb_psnr_splat_pred_* from
        # the pass-2 depth instead, so both are computed here and printed
        # separately; the pass-1 one is what the prediction actually saw.
        gen = torch.Generator(device=device)
        gen.manual_seed(model.eval_sample_seed)
        blind = dict(inputs)
        blind["warp_block"] = torch.zeros_like(inputs["warp_block"])
        if inputs.get("depth_warp_block") is not None:
            blind["depth_warp_block"] = torch.zeros_like(inputs["depth_warp_block"])
        pred_depth_p1, _ = model._sample_rgb_once(
            blind, model.eval_num_inference_steps, gen)

        warp_block, depth_warp_block = model._warp_block_from_predicted_depth(
            inputs, pred_depth_p1)
        second = dict(inputs)
        second["warp_block"] = warp_block
        second["depth_warp_block"] = depth_warp_block
        clean_src_depth = (
            pred_depth_p1[:, :, :num_src] if model.clean_source_depth else None)
        pred_depth_p2, pred_rgb = model._sample_rgb_once(
            second, model.eval_num_inference_steps, gen, clean_src_depth=clean_src_depth)

        if model.latent_scale != 1.0:
            pred_rgb = pred_rgb / model.latent_scale

        def decode_src_disp(depth_latents):
            """Same read-out _warp_block_from_predicted_depth uses: [B,S,H,W] in [0,1]."""
            lat = depth_latents[:, :, :num_src] / model.depth_latent_scale
            return torch.stack([
                model.pipe.vae.decode(
                    lat[:, :, v:v + 1].to(model.pipe.torch_dtype), device=device
                )[:, :, 0].float().mean(dim=1)
                for v in range(num_src)
            ], dim=1).clamp(0.0, 1.0)

        pred_disp_p1 = decode_src_disp(pred_depth_p1)
        pred_disp_p2 = decode_src_disp(pred_depth_p2)

        rgb_px = batch["images"].to(device).float().clamp(0.0, 1.0)     # [B,V,3,H,W]
        disp_gt = batch["disparity"].to(device)[:, :num_src, 0].float()  # [B,S,H,W]

        splat_gt, mask_gt = model._splat_sources_pixel(
            rgb_px, disp_gt, batch["K"].to(device).float(), inputs["c2w"])
        splat_p1, mask_p1 = model._splat_sources_pixel(
            rgb_px, pred_disp_p1, batch["K"].to(device).float(), inputs["c2w"])
        splat_p2, mask_p2 = model._splat_sources_pixel(
            rgb_px, pred_disp_p2, batch["K"].to(device).float(), inputs["c2w"])

        # Slice the BATCH here, not just the view: b{NN} lives at local_b inside
        # the chunk, and decoding the whole batch then taking [0] silently shows
        # (and scores, via broadcasting against a [1,3,H,W] GT) a different
        # sample -- which is exactly why an earlier run of this script rendered
        # an unrelated scene as "tgt3 model pred".
        b = local_b
        t = args.target
        v_lat = pred_rgb[b:b + 1, :, vi:vi + 1].to(device).to(model.pipe.torch_dtype)
        pred_rgb_px = model.pipe.vae.decode(v_lat, device=device)[:, :, 0].float().clamp(0, 1)
    gt_px = rgb_px[b:b + 1, vi]                                    # [1,3,H,W]

    print()
    print(f"=== target {t} of {batch['sample_id'][b]} ===")
    for label, splat, mask in (("warp @ GT depth      ", splat_gt, mask_gt),
                               ("warp @ pred depth p1 ", splat_p1, mask_p1),
                               ("warp @ pred depth p2 ", splat_p2, mask_p2)):
        w = splat[b:b + 1, t]
        m = mask[b:b + 1, t]
        print(f"  {label} coverage={float(m.mean()):.4f}  "
              f"psnr_valid={psnr_masked(w, gt_px, m):6.2f}")
    print(f"  model prediction      "
          f"psnr_covered={psnr_masked(pred_rgb_px, gt_px, mask_gt[b:b+1, t]):6.2f}  "
          f"psnr_full={psnr_masked(pred_rgb_px, gt_px, torch.ones_like(mask_gt[b:b+1, t])):6.2f}")
    print("  (compare against per_view_metrics.jsonl: splat_pred_* is logged from p2)")

    # -------------------------------- figure ---------------------------------
    def img(x):
        return x[0].permute(1, 2, 0).cpu().numpy().clip(0, 1)

    ncols = max(num_src, 6)
    fig, axes = plt.subplots(4, ncols, figsize=(2.9 * ncols, 4 * 2.9 * 0.62))
    for ax in axes.ravel():
        ax.axis("off")

    for s in range(num_src):
        axes[0, s].imshow(img(rgb_px[b:b + 1, s]))
        axes[0, s].set_title(f"src{s+1} RGB", fontsize=9)
        axes[1, s].imshow(disp_gt[b, s].cpu().numpy(), cmap="plasma", vmin=0, vmax=1)
        axes[1, s].set_title(f"src{s+1} disp GT", fontsize=9)
        axes[2, s].imshow(pred_disp_p1[b, s].cpu().numpy(), cmap="plasma", vmin=0, vmax=1)
        l1 = (pred_disp_p1[b, s] - disp_gt[b, s]).abs().mean().item()
        axes[2, s].set_title(f"src{s+1} disp PRED\nraw L1={l1:.3f}", fontsize=9)

    panels = [
        (img(gt_px), None, f"tgt{t+1} GT"),
        (img(splat_gt[b:b + 1, t]), None,
         f"warp @ GT depth\ncov={float(mask_gt[b,t].mean()):.3f} "
         f"psnr={psnr_masked(splat_gt[b:b+1,t], gt_px, mask_gt[b:b+1,t]):.2f}"),
        (img(splat_p1[b:b + 1, t]), None,
         f"warp @ PRED depth  <-- model input\ncov={float(mask_p1[b,t].mean()):.3f} "
         f"psnr={psnr_masked(splat_p1[b:b+1,t], gt_px, mask_p1[b:b+1,t]):.2f}"),
        (img(pred_rgb_px), None,
         f"tgt{t+1} model pred\npsnr={psnr_masked(pred_rgb_px, gt_px, mask_gt[b:b+1,t]):.2f}"),
        (mask_gt[b, t, 0].cpu().numpy(), "gray", "mask @ GT depth"),
        (mask_p1[b, t, 0].cpu().numpy(), "gray", "mask @ PRED depth"),
    ]
    for c, (im, cmap, title) in enumerate(panels):
        ax = axes[3, c]
        ax.imshow(im, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
        ax.set_title(title, fontsize=9)

    fig.suptitle(
        f"{os.path.basename(args.run)} step{args.step}  {batch['sample_id'][b]}  "
        f"b{args.b:02d} target#{t}", fontsize=12)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.01, wspace=0.05, hspace=0.28)
    out = args.out or (f"tools/rgb_warp_pred_{os.path.basename(args.run)}"
                       f"_step{args.step}_b{args.b:02d}_t{t}.png")
    plt.savefig(out, dpi=130)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
