"""Batch version of vis_rgb_pass1_vs_pass2.py: render GT / pass-1 (blind) /
pass-2 (final) target-view RGB for N samples of a run's frozen eval set, one
PNG per sample, annotated with PSNR, LPIPS, AbsRel and delta_acc (delta1,
i.e. fraction of pixels with depth ratio < 1.25) for both passes -- so a
"pass 1 looks better than pass 2" impression can be checked across many
samples instead of one.

Pass 1's target RGB/depth are full clean latents that sample_rgb() computes
and then discards (WanRGBDNVSTrainingModule.py:1930-1934); only pass 2's
outputs ever get decoded/saved by evaluate(). This keeps both.

Reproduces the frozen-eval indexing of train_with_accelerate_rgbd_nvs.py so
b{NN} in a saved eval_step_*/b{NN}_step*.jpg maps to the same dataset item:
chunk_idx, local_b = divmod(NN, eval_batch_size). Samples sharing a chunk are
preprocessed and sampled together (one forward_preprocess + two-pass sample
per chunk, not per sample), mirroring evaluate()'s own batching.

Run as a MODULE from the project root, one GPU:
  python -m tools.vis_rgb_pass1_vs_pass2_batch \
      --run logs/bmvs_d_drocefp32_s42 --step 3750 \
      --b-start 0 --num-samples 20 --target 2 \
      --out-dir tools/pass1_vs_pass2_bmvs_d_drocefp32_s42_step3750
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Subset

from examples.wanvideo.model_training.train_with_accelerate_rgbd_nvs import custom_collate_fn
from tools.vis_rgb_warp_pred_depth import build_model, build_dataset, psnr_masked


def fit_affine(p, g):
    """Least squares  min_{a,b} ||a*p + b - g||^2  (float64, closed form)."""
    n = p.numel()
    sp, spp = p.sum(), (p * p).sum()
    sg, spg = g.sum(), (p * g).sum()
    det = n * spp - sp * sp
    if torch.abs(det) < 1e-12:
        a = g.median() / p.median().clamp_min(1e-8)
        b = torch.zeros((), dtype=torch.float64, device=p.device)
    else:
        a = (n * spg - sp * sg) / det
        b = (spp * sg - sp * spg) / det
    return a, b


def decode_src_disp(pred_depth_latents, num_src, model, device):
    """Decode source-view disparity from a RAW (un-divided) predicted depth
    latent -- same convention _warp_block_from_predicted_depth uses
    (WanRGBDNVSTrainingModule.py:2002): [B,S,H,W] in [0,1]."""
    lat = pred_depth_latents[:, :, :num_src] / model.depth_latent_scale
    return torch.stack([
        model.pipe.vae.decode(
            lat[:, :, v:v + 1].to(model.pipe.torch_dtype), device=device
        )[:, :, 0].float().mean(dim=1)
        for v in range(num_src)
    ], dim=1).clamp(0.0, 1.0)


def depth_metrics_light(pred_lv, gt_metric_hw, mask_hw, model, device):
    """AbsRel / delta_acc (delta1, ratio<1.25) after affine alignment in
    disparity space -- same protocol as evaluate()'s _depth_metrics/
    _align_and_score, minus the clamp/ramp/struct_corr side numbers.

    pred_lv must already be divided by model.depth_latent_scale, mirroring
    evaluate()'s pre-division at WanRGBDNVSTrainingModule.py:2195-2197 --
    _warp_block_from_predicted_depth expects the RAW (un-divided) latent
    instead, so this must not share a pre-divided tensor with that call."""
    if pred_lv.shape[2] == 0:
        return float("nan"), float("nan")
    v_lat = pred_lv.to(device).to(model.pipe.torch_dtype)  # [1,16,1,h,w]
    pred_disp = model.pipe.vae.decode(v_lat, device=device)[0, :, 0].float().mean(0)  # [H,W]
    gt = gt_metric_hw.to(device).float()
    if pred_disp.shape != gt.shape:
        pred_disp = F.interpolate(
            pred_disp[None, None], size=gt.shape, mode="bilinear", align_corners=False)[0, 0]
    valid = mask_hw.bool().to(device) & torch.isfinite(gt) & (gt > 0)
    if valid.sum() < 16:
        return float("nan"), float("nan")
    p = pred_disp[valid].double()
    gt_v = gt[valid].double()
    g = 1.0 / gt_v.clamp_min(1e-8)
    a, b = fit_affine(p, g)
    disp_floor = (g.min() * 0.5).clamp_min(1e-8)
    pred_depth_m = 1.0 / (a * p + b).clamp_min(disp_floor)
    abs_rel = ((pred_depth_m - gt_v).abs() / gt_v).mean().item()
    ratio = torch.max(pred_depth_m / gt_v, gt_v / pred_depth_m)
    delta_acc = (ratio < 1.25).double().mean().item()
    return abs_rel, delta_acc


def psnr_full(pred_img, gt_img):
    mse = ((pred_img - gt_img) ** 2).mean().item()
    if mse < 1e-10:
        return 100.0
    import math
    return -10 * math.log10(mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--b-start", type=int, default=0)
    ap.add_argument("--num-samples", type=int, default=20)
    ap.add_argument("--target", type=int, default=2, help="0-indexed target view")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or (
        f"tools/pass1_vs_pass2_{os.path.basename(args.run)}_step{args.step}")
    os.makedirs(out_dir, exist_ok=True)

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
    lpips_net = model._get_lpips(device)
    if lpips_net is None:
        print("WARNING: lpips not installed, lpips column will be NaN")

    def lpips(pred_img, gt_img):
        if lpips_net is None:
            return float("nan")
        with torch.no_grad():
            d = lpips_net(pred_img.float() * 2 - 1, gt_img.float() * 2 - 1)
        return float(d.mean())

    dataset = build_dataset(cfg)
    eval_max_samples = int(cfg.get("eval_max_samples", 2))
    eval_batch_size = max(1, min(eval_max_samples, int(cfg.get("eval_batch_size", 4))))
    samples_per_scene = cfg.get("samples_per_scene", 1)
    eval_indices = [i * samples_per_scene for i in range(len(dataset.records))][:eval_max_samples]
    loader = DataLoader(
        Subset(dataset, eval_indices), batch_size=eval_batch_size, shuffle=False,
        num_workers=0, collate_fn=custom_collate_fn, drop_last=False,
    )

    sample_bs = list(range(args.b_start, args.b_start + args.num_samples))
    needed_chunks = {}
    for b in sample_bs:
        chunk_idx, local_b = divmod(b, eval_batch_size)
        needed_chunks.setdefault(chunk_idx, []).append((b, local_b))
    max_chunk = max(needed_chunks)

    num_src = model.num_source_views
    t = args.target
    vi = num_src + t
    autocast_ctx = torch.autocast(device_type="cuda", dtype=model.pipe.torch_dtype, enabled=True)

    rows = []
    for chunk_idx, batch in enumerate(loader):
        if chunk_idx > max_chunk:
            break
        if chunk_idx not in needed_chunks:
            continue

        with torch.no_grad(), autocast_ctx:
            model._eval_inputs = True
            try:
                inputs = model.forward_preprocess(batch)
            finally:
                model._eval_inputs = False

            gen = torch.Generator(device=device)
            gen.manual_seed(model.eval_sample_seed)
            blind = dict(inputs)
            blind["warp_block"] = torch.zeros_like(inputs["warp_block"])
            if inputs.get("depth_warp_block") is not None:
                blind["depth_warp_block"] = torch.zeros_like(inputs["depth_warp_block"])
            pred_depth_p1, pred_rgb_p1 = model._sample_rgb_once(
                blind, model.eval_num_inference_steps, gen)

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

            # depth_metrics_light expects the pre-divided convention evaluate()
            # uses (WanRGBDNVSTrainingModule.py:2195-2197); keep the RAW
            # pred_depth_p1/p2 for _warp_block_from_predicted_depth /
            # decode_src_disp above, which divide internally themselves.
            depth_scale = model.depth_latent_scale
            pred_depth_p1_m = pred_depth_p1 / depth_scale if depth_scale != 1.0 else pred_depth_p1
            pred_depth_p2_m = pred_depth_p2 / depth_scale if depth_scale != 1.0 else pred_depth_p2

            rgb_px = batch["images"].to(device).float().clamp(0.0, 1.0)      # [B,V,3,H,W]
            disp_gt_batch = batch["disparity"].to(device)                    # [B,V,3,H,W]
            depth_metric_batch = batch["depth_metric"].to(device)            # [B,V,3,H,W]
            loss_masks = inputs["loss_masks"]                                 # [B,2,V,1,H,W]

            # Coverage of the warp actually fed into pass 2: splat the GT
            # source RGB using pass-1's PREDICTED source depth (the same
            # depth _warp_block_from_predicted_depth used to build pass 2's
            # conditioning), then read out what fraction of the target view
            # got a valid splatted pixel -- low coverage here is the usual
            # explanation for a pass-2-only artefact.
            pred_disp_p1 = decode_src_disp(pred_depth_p1, num_src, model, device)
            _, mask_warp_p2 = model._splat_sources_pixel(
                rgb_px, pred_disp_p1, batch["K"].to(device).float(), inputs["c2w"])

            def decode_view(latents, b_local):
                v_lat = latents[b_local:b_local + 1, :, vi:vi + 1].to(device).to(model.pipe.torch_dtype)
                return model.pipe.vae.decode(v_lat, device=device)[:, :, 0].float().clamp(0, 1)

            for b, local_b in needed_chunks[chunk_idx]:
                gt_px = rgb_px[local_b:local_b + 1, vi]  # [1,3,H,W]
                p1_px = decode_view(pred_rgb_p1, local_b)
                p2_px = decode_view(pred_rgb_p2, local_b)

                psnr_p1 = psnr_full(p1_px, gt_px)
                psnr_p2 = psnr_full(p2_px, gt_px)
                lpips_p1 = lpips(p1_px, gt_px)
                lpips_p2 = lpips(p2_px, gt_px)

                mask_hw = loss_masks[local_b, 0, vi, 0]
                gt_metric_hw = depth_metric_batch[local_b, vi, 0]
                abs_rel_p1, delta_acc_p1 = depth_metrics_light(
                    pred_depth_p1_m[local_b:local_b + 1, :, vi:vi + 1], gt_metric_hw, mask_hw, model, device)
                abs_rel_p2, delta_acc_p2 = depth_metrics_light(
                    pred_depth_p2_m[local_b:local_b + 1, :, vi:vi + 1], gt_metric_hw, mask_hw, model, device)
                cov_p2 = float(mask_warp_p2[local_b:local_b + 1, t].mean())

                sample_id = batch["sample_id"][local_b]
                rows.append({
                    "b": b, "sample_id": sample_id,
                    "psnr_p1": psnr_p1, "psnr_p2": psnr_p2,
                    "lpips_p1": lpips_p1, "lpips_p2": lpips_p2,
                    "abs_rel_p1": abs_rel_p1, "abs_rel_p2": abs_rel_p2,
                    "delta_acc_p1": delta_acc_p1, "delta_acc_p2": delta_acc_p2,
                    "cov_p2": cov_p2,
                })

                def img(x):
                    return x[0].permute(1, 2, 0).cpu().numpy().clip(0, 1)

                fig, axes = plt.subplots(1, 3, figsize=(3 * 3.4, 3.9))
                for ax in axes:
                    ax.axis("off")
                axes[0].imshow(img(gt_px))
                axes[0].set_title(f"tgt{t+1} GT", fontsize=10)
                axes[1].imshow(img(p1_px))
                axes[1].set_title(
                    f"tgt{t+1} pass 1 (blind)\n"
                    f"psnr={psnr_p1:.2f}  lpips={lpips_p1:.3f}\n"
                    f"abs_rel={abs_rel_p1:.3f}  delta_acc={delta_acc_p1:.3f}",
                    fontsize=9)
                axes[2].imshow(img(p2_px))
                axes[2].set_title(
                    f"tgt{t+1} pass 2 (final)\n"
                    f"psnr={psnr_p2:.2f}  lpips={lpips_p2:.3f}\n"
                    f"abs_rel={abs_rel_p2:.3f}  delta_acc={delta_acc_p2:.3f}  "
                    f"warp_cov={cov_p2:.3f}",
                    fontsize=9)

                fig.suptitle(
                    f"{os.path.basename(args.run)} step{args.step}  {sample_id}  "
                    f"b{b:02d} target#{t}", fontsize=11)
                fig.subplots_adjust(left=0.01, right=0.99, top=0.78, bottom=0.02, wspace=0.05)
                safe_id = sample_id.replace("/", "_")
                out_path = os.path.join(out_dir, f"b{b:02d}_t{t}_{safe_id}.png")
                plt.savefig(out_path, dpi=130)
                plt.close(fig)
                print(f"b{b:02d} {sample_id}: "
                      f"psnr p1={psnr_p1:6.2f} p2={psnr_p2:6.2f}  "
                      f"lpips p1={lpips_p1:.3f} p2={lpips_p2:.3f}  "
                      f"abs_rel p1={abs_rel_p1:.3f} p2={abs_rel_p2:.3f}  "
                      f"delta_acc p1={delta_acc_p1:.3f} p2={delta_acc_p2:.3f}  "
                      f"warp_cov(p2)={cov_p2:.3f}  -> {out_path}")

    n = len(rows)
    if n:
        import math
        def mean(key):
            vals = [r[key] for r in rows if not math.isnan(r[key])]
            return sum(vals) / len(vals) if vals else float("nan")
        print(f"\n=== mean over {n} samples ===")
        print(f"  psnr      p1={mean('psnr_p1'):.2f}  p2={mean('psnr_p2'):.2f}")
        print(f"  lpips     p1={mean('lpips_p1'):.3f}  p2={mean('lpips_p2'):.3f}")
        print(f"  abs_rel   p1={mean('abs_rel_p1'):.3f}  p2={mean('abs_rel_p2'):.3f}")
        print(f"  delta_acc p1={mean('delta_acc_p1'):.3f}  p2={mean('delta_acc_p2'):.3f}")
        print(f"  warp_cov(p2)={mean('cov_p2'):.3f}")
    print(f"\nsaved {n} images to {out_dir}")


if __name__ == "__main__":
    main()
