"""Visualize ONE frozen-eval sample: p2 (production baseline) vs `warp_mvs`
(report §16i's GT-free inference-time source-disparity scale fit) side by
side with GT, for every target view.

Written for `b05_t2_57f8d9bbe73f6760f10e916a_tt0_s50` -- the church-door
sample the whole report opens with (report_20260916_pass1_prior_diag.md §0's
origin story, revisited at §8b/§8c/§9): a near-baseline target where the
source depth's per-view SCALE disagreement makes the same door arch splat to
two different target pixels ("double arch" ghosting).  §16i's aggregate
table found `warp_mvs` improves 143/192 target pairs on average but this
specific scene has 3 target views with 3 different outcomes (b=5 in
records.jsonl): t0 improved (psnr +0.43, abs_rel -0.018), t1 mixed
(psnr +0.83 but lpips +0.019), t2 -- the one with the visible double-arch --
got WORSE (psnr -1.05, lpips +0.027).  This renders all three so the
improvement (or lack of it) can be seen directly rather than inferred from
the aggregate numbers.

Reuses the EXACT same code path as the `warp_mvs` arm in
tools/diag_cross_stream_rails.py (sample_rails, decode_src_disparity) and the
GT-free scale fit in tools/diag_mvs_scale_fit.py (ordered_pairs, scan_global),
with the same shared noise draw convention, so the images match the numbers
in tools/cross_stream_rails_mvs_bmvs_d_drocefp32_s42_step3750/records.jsonl.

Run as a MODULE from the project root, one GPU:
  python -m tools.vis_rgb_mvs_vs_p2 --run logs/bmvs_d_drocefp32_s42 --step 3750 --b 5
"""
import argparse
import math
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
from tools.diag_cross_stream_rails import MVS_GRID, MVS_LAMBDA, decode_src_disparity, sample_rails
from tools.diag_mvs_scale_fit import ordered_pairs, scan_global
from tools.diag_pass1_prior import decode_disp, decode_rgb, depth_metrics, psnr_full
from tools.vis_rgb_warp_pred_depth import build_dataset, build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--b", type=int, default=5, help="b-index from the saved eval grid, e.g. 5 for b05")
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
    missing, unexpected = model.load_state_dict(
        load_file(os.path.join(ckpt, "model.safetensors"), device="cpu"), strict=False)
    print(f"loaded {ckpt}: missing={len(missing)} unexpected={len(unexpected)}")
    model.to(accelerator.device)
    model.eval()
    device = model.pipe.device
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
    chunk_idx, local_b = divmod(args.b, eval_batch_size)
    loader = DataLoader(Subset(dataset, eval_indices), batch_size=eval_batch_size,
                        shuffle=False, num_workers=0, collate_fn=custom_collate_fn,
                        drop_last=False)
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
    num_tgt = model.num_target_views
    tgt_views = list(range(num_src, num_src + num_tgt))
    num_steps = model.eval_num_inference_steps
    seed = model.eval_sample_seed
    b = local_b
    autocast_ctx = torch.autocast(device_type="cuda", dtype=model.pipe.torch_dtype, enabled=True)

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

        # ---- pass 1: blind (same for every arm)
        blind = dict(inputs)
        blind["warp_block"] = torch.zeros_like(inputs["warp_block"])
        if inputs.get("depth_warp_block") is not None:
            blind["depth_warp_block"] = torch.zeros_like(inputs["depth_warp_block"])
        dep1, rgb1 = sample_rails(model, blind, num_steps, eps_rgb, eps_dep)

        # ---- p2: warp rebuilt from pass 1's own predicted source depth
        wb_p2, dwb_p2 = model._warp_block_from_predicted_depth(inputs, dep1)
        base_p2 = dict(inputs)
        base_p2["warp_block"] = wb_p2
        base_p2["depth_warp_block"] = dwb_p2
        dep_p2, rgb_p2 = sample_rails(model, base_p2, num_steps, eps_rgb, eps_dep)

        # ---- warp_mvs: EXACT same GT-free fit as the arm in
        # diag_cross_stream_rails.py -- 1-DOF photometric scan per sample,
        # then MVS_LAMBDA shrinkage, applied to pass 1's own source disparity.
        sdisp_p1 = decode_src_disparity(model, dep1)
        src_valid = inputs["loss_masks"][:, 0, :num_src, 0] > 0.5
        src_K = inputs["pixel_K"][:, :num_src]
        src_c2w = inputs["c2w"][:, :num_src]
        src_rgb = inputs["source_images"].float()
        pairs = ordered_pairs(num_src)
        # index `b` (local_b), NOT 0: sdisp_p1/src_valid/src_K/src_c2w/src_rgb
        # are the FULL batch, and the fit must match the sample we render.
        errs, _ = scan_global(src_rgb[b], sdisp_p1[b], src_valid[b], src_K[b], src_c2w[b],
                              pairs, MVS_GRID)
        fin = [(e, k) for e, k in zip(errs, MVS_GRID) if e == e]
        k_hat = min(fin, key=lambda t: t[0])[1] if fin else 1.0
        k_shrunk = k_hat ** MVS_LAMBDA
        print(f"  k_hat={k_hat:.4f}  k_shrunk(^{MVS_LAMBDA})={k_shrunk:.4f}")
        sdisp_mvs = (sdisp_p1 * k_shrunk).clamp(0.0, 1.0)
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

    def img(x):
        # x is already sliced to the single sample (x[b:b+1] at the call site,
        # so index 0 here) -- indexing by `b` again would silently grab the
        # wrong (out-of-range, for b>0) element.
        return x[0].permute(1, 2, 0).cpu().numpy().clip(0, 1)

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
        ar_p2, d1_p2, _ = depth_metrics(dec_disp["p2"][vi][b], depth_metric[b, vi, 0],
                                        disp_gt[b, vi, 0], loss_masks[b, 0, vi, 0])
        ar_mvs, d1_mvs, _ = depth_metrics(dec_disp["mvs"][vi][b], depth_metric[b, vi, 0],
                                          disp_gt[b, vi, 0], loss_masks[b, 0, vi, 0])
        axes[t, 0].imshow(img(gt))
        axes[t, 0].set_title(f"tgt{t+1} GT", fontsize=10)
        axes[t, 1].imshow(img(p2_img))
        axes[t, 1].set_title(f"tgt{t+1} p2 (baseline)\npsnr={p2_psnr:.2f} lpips={p2_lp:.4f} "
                             f"abs_rel={ar_p2:.4f}", fontsize=9)
        axes[t, 2].imshow(img(mvs_img))
        better = "better" if mvs_lp < p2_lp else "WORSE"
        axes[t, 2].set_title(f"tgt{t+1} warp_mvs [{better}]\npsnr={mvs_psnr:.2f} "
                             f"lpips={mvs_lp:.4f} abs_rel={ar_mvs:.4f}", fontsize=9)
        print(f"  t{t}: p2  psnr={p2_psnr:.3f} lpips={p2_lp:.4f} abs_rel={ar_p2:.4f} delta1={d1_p2:.4f}")
        print(f"       mvs psnr={mvs_psnr:.3f} lpips={mvs_lp:.4f} abs_rel={ar_mvs:.4f} delta1={d1_mvs:.4f}")

    fig.suptitle(f"{os.path.basename(args.run)} step{args.step}  {batch['sample_id'][b]}  "
                f"b{args.b:02d}  k_hat={k_hat:.4f} k_shrunk={k_shrunk:.4f}", fontsize=11)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.02, wspace=0.05, hspace=0.35)
    out = args.out or (f"tools/vis_rgb_mvs_vs_p2_{os.path.basename(args.run)}"
                       f"_step{args.step}_b{args.b:02d}.png")
    plt.savefig(out, dpi=140)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
