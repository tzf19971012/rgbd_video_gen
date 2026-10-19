"""Same rendering as tools/vis_rgb_mvs_vs_p2.py (GT vs p2 vs warp_mvs, every
target view) but for EVERY sample in the frozen eval set, one PNG per sample.

Multi-GPU: chunks are assigned round-robin across independent processes, same
convention as diag_cross_stream_rails.py's --num-shards/--shard-idx (see
sbatch/vis_rgb_mvs_vs_p2_batch_shard.sbatch, a SLURM job array).  No merge step
needed afterwards: each image's filename is keyed by its own sample_id, so the
shards' output directories can simply be pooled (they never collide).

Run as a MODULE from the project root, one GPU:
  python -m tools.vis_rgb_mvs_vs_p2_batch --run logs/bmvs_d_droce_s1337 --step 4000
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
from tools.diag_cross_stream_rails import MVS_GRID, MVS_LAMBDA, decode_src_disparity, ordered_pairs, sample_rails
from tools.diag_mvs_scale_fit import scan_global
from tools.diag_pass1_prior import decode_disp, decode_rgb, depth_metrics, psnr_full
from tools.vis_rgb_warp_pred_depth import build_dataset, build_model


def safe_name(sample_id):
    return sample_id.replace("/", "_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--max-chunks", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-idx", type=int, default=0)
    args = ap.parse_args()
    if not (0 <= args.shard_idx < max(args.num_shards, 1)):
        raise SystemExit(f"--shard-idx {args.shard_idx} out of range for "
                         f"--num-shards {args.num_shards}")

    out_dir = args.out_dir or (
        f"tools/vis_rgb_mvs_vs_p2_{os.path.basename(args.run)}_step{args.step}")
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
    missing, unexpected = model.load_state_dict(
        load_file(os.path.join(ckpt, "model.safetensors"), device="cpu"), strict=False)
    print(f"loaded {ckpt}: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model.to(accelerator.device)
    model.eval()
    device = model.pipe.device
    if not model.depth_flow_matching:
        raise SystemExit("this diagnostic is about the depth stream; "
                         "depth_flow_matching is off in this run")
    lpips_net = model._get_lpips(device)

    def lpips(a, b):
        if lpips_net is None:
            return float("nan")
        with torch.no_grad():
            return float(lpips_net(a.float() * 2 - 1, b.float() * 2 - 1).mean())

    dataset = build_dataset(cfg)
    eval_max_samples = int(cfg.get("eval_max_samples", 2))
    eval_batch_size = max(1, min(eval_max_samples, int(cfg.get("eval_batch_size", 4))))
    samples_per_scene = cfg.get("samples_per_scene", 1)
    eval_indices = [i * samples_per_scene for i in range(len(dataset.records))][:eval_max_samples]
    loader = DataLoader(Subset(dataset, eval_indices), batch_size=eval_batch_size,
                        shuffle=False, num_workers=0, collate_fn=custom_collate_fn,
                        drop_last=False)

    num_src = model.num_source_views
    num_tgt = model.num_target_views
    tgt_views = list(range(num_src, num_src + num_tgt))
    num_steps = model.eval_num_inference_steps
    seed = model.eval_sample_seed
    autocast_ctx = torch.autocast(device_type="cuda", dtype=model.pipe.torch_dtype, enabled=True)
    pairs = ordered_pairs(num_src)

    def img(x):
        return x[0].permute(1, 2, 0).cpu().numpy().clip(0, 1)

    n_saved = 0
    for chunk_idx, batch in enumerate(loader):
        if args.max_chunks is not None and chunk_idx >= args.max_chunks:
            break
        if args.num_shards > 1 and chunk_idx % args.num_shards != args.shard_idx:
            continue
        with torch.no_grad(), autocast_ctx:
            model._eval_inputs = True
            try:
                inputs = model.forward_preprocess(batch)
            finally:
                model._eval_inputs = False

            rgb_gt_lat = inputs["rgb_gt"]
            dep_gt_lat = inputs["depth_gt"]
            g = torch.Generator(device=device)
            g.manual_seed(seed)
            eps_rgb = torch.randn(rgb_gt_lat.shape, device=device, dtype=rgb_gt_lat.dtype, generator=g)
            eps_dep = torch.randn(dep_gt_lat.shape, device=device, dtype=dep_gt_lat.dtype, generator=g)

            blind = dict(inputs)
            blind["warp_block"] = torch.zeros_like(inputs["warp_block"])
            if inputs.get("depth_warp_block") is not None:
                blind["depth_warp_block"] = torch.zeros_like(inputs["depth_warp_block"])
            dep1, rgb1 = sample_rails(model, blind, num_steps, eps_rgb, eps_dep)

            wb_p2, dwb_p2 = model._warp_block_from_predicted_depth(inputs, dep1)
            base_p2 = dict(inputs)
            base_p2["warp_block"] = wb_p2
            base_p2["depth_warp_block"] = dwb_p2
            dep_p2, rgb_p2 = sample_rails(model, base_p2, num_steps, eps_rgb, eps_dep)

            # GT-free per-sample scale fit, identical to the `warp_mvs` arm.
            sdisp_p1 = decode_src_disparity(model, dep1)
            src_valid = inputs["loss_masks"][:, 0, :num_src, 0] > 0.5
            src_K = inputs["pixel_K"][:, :num_src]
            src_c2w = inputs["c2w"][:, :num_src]
            src_rgb = inputs["source_images"].float()
            k_hats, k_shrunks, mvs_disps = [], [], []
            for bi in range(sdisp_p1.shape[0]):
                errs, _ = scan_global(src_rgb[bi], sdisp_p1[bi], src_valid[bi],
                                      src_K[bi], src_c2w[bi], pairs, MVS_GRID)
                fin = [(e, k) for e, k in zip(errs, MVS_GRID) if e == e]
                k_hat = min(fin, key=lambda t: t[0])[1] if fin else 1.0
                k_shrunk = k_hat ** MVS_LAMBDA
                k_hats.append(k_hat)
                k_shrunks.append(k_shrunk)
                mvs_disps.append((sdisp_p1[bi] * k_shrunk).clamp(0.0, 1.0))
            sdisp_mvs = torch.stack(mvs_disps, dim=0)
            wb_mvs, dwb_mvs = model._build_warp_block(
                inputs["source_images"], sdisp_mvs, inputs["pixel_K"], inputs["c2w"],
                tuple(dep1.shape[-2:]))
            base_mvs = dict(inputs)
            base_mvs["warp_block"] = wb_mvs.to(dep_gt_lat.dtype)
            base_mvs["depth_warp_block"] = None if dwb_mvs is None else dwb_mvs.to(dep_gt_lat.dtype)
            dep_mvs, rgb_mvs = sample_rails(model, base_mvs, num_steps, eps_rgb, eps_dep)

            rgb_px = batch["images"].to(device).float().clamp(0.0, 1.0)
            disp_gt = batch["disparity"].to(device).float()
            depth_metric = batch["depth_metric"].to(device).float()
            loss_masks = inputs["loss_masks"]

            dec_rgb = {"p2": decode_rgb(model, rgb_p2, tgt_views, device),
                      "mvs": decode_rgb(model, rgb_mvs, tgt_views, device)}
            dec_disp = {"p2": decode_disp(model, dep_p2, tgt_views, device),
                       "mvs": decode_disp(model, dep_mvs, tgt_views, device)}

        for b in range(sdisp_p1.shape[0]):
            sample_id = batch["sample_id"][b]
            num_t = len(tgt_views)
            fig, axes = plt.subplots(num_t, 3, figsize=(3 * 3.4, num_t * 3.4))
            if num_t == 1:
                axes = axes[None, :]
            for ax in axes.flat:
                ax.axis("off")
            for t, vi in enumerate(tgt_views):
                gt = rgb_px[b:b + 1, vi]
                p2_img = dec_rgb["p2"][vi][b:b + 1]
                mvs_img = dec_rgb["mvs"][vi][b:b + 1]
                p2_psnr, mvs_psnr = psnr_full(p2_img, gt), psnr_full(mvs_img, gt)
                p2_lp, mvs_lp = lpips(p2_img, gt), lpips(mvs_img, gt)
                ar_p2, _, _ = depth_metrics(dec_disp["p2"][vi][b], depth_metric[b, vi, 0],
                                            disp_gt[b, vi, 0], loss_masks[b, 0, vi, 0])
                ar_mvs, _, _ = depth_metrics(dec_disp["mvs"][vi][b], depth_metric[b, vi, 0],
                                             disp_gt[b, vi, 0], loss_masks[b, 0, vi, 0])
                axes[t, 0].imshow(img(gt))
                axes[t, 0].set_title(f"tgt{t+1} GT", fontsize=10)
                axes[t, 1].imshow(img(p2_img))
                axes[t, 1].set_title(f"tgt{t+1} p2\npsnr={p2_psnr:.2f} lpips={p2_lp:.4f} "
                                     f"abs_rel={ar_p2:.4f}", fontsize=9)
                axes[t, 2].imshow(img(mvs_img))
                better = "better" if mvs_lp < p2_lp else "WORSE"
                axes[t, 2].set_title(f"tgt{t+1} warp_mvs [{better}]\npsnr={mvs_psnr:.2f} "
                                     f"lpips={mvs_lp:.4f} abs_rel={ar_mvs:.4f}", fontsize=9)
            fig.suptitle(f"{os.path.basename(args.run)} step{args.step}  {sample_id}  "
                        f"k_hat={k_hats[b]:.4f} k_shrunk={k_shrunks[b]:.4f}", fontsize=11)
            fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.02,
                                wspace=0.05, hspace=0.35)
            out = os.path.join(out_dir, f"{safe_name(sample_id)}.png")
            plt.savefig(out, dpi=110)
            plt.close(fig)
            n_saved += 1
            print(f"  [{n_saved}] chunk {chunk_idx} b={b} {sample_id} "
                 f"k_hat={k_hats[b]:.4f} -> {out}", flush=True)

    print(f"\nsaved {n_saved} images to {out_dir}")


if __name__ == "__main__":
    main()
