"""Pose-swap diagnostic: does the model READ the target camera pose?

For one scene, render the target frames twice with the SAME sampling noise:
  pass A  true poses                -> pred_true   (scored vs true GT)
  pass B  target poses cyclically
          shifted among themselves  -> pred_swap   (scored vs true GT AND vs
                                                      the GT of the pose it got)

Headline metric: swap_gain = mean(psnr(pred_swap, gt_swap) - psnr(pred_swap,
gt_true)).  gain >> 0 means the output follows the assigned pose (camera is
read); gain ~ 0 means the output ignores the pose (camera is decorative).
Secondary: pixel L1(pred_swap, pred_true) relative to L1(pred_true, gt_true) --
how much of the model's output variability is attributable to the pose.

Unlike tools/diag_pose_sensitivity.py this works in BOTH add and concat modes
(it perturbs the batch poses before forward_preprocess, no module hooks), so
all arms are comparable with one instrument.

Run from the repo root:
  ASCEND_RT_VISIBLE_DEVICES=6 python -m tools.diag_pose_swap_nvs \
      --run_dir logs/ovf_gae_re10k_aux0_patch1 --checkpoint .../checkpoint-step-5600
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

from examples.wanvideo.model_training.infer_nvs_compare import (
    _align_median,
    build_model,
    latest_checkpoint,
)
from examples.wanvideo.model_training.infer_trajectory import (
    _FILL_CTX,
    _install_zero_fill_shim,
)
from examples.wanvideo.model_training.train_with_accelerate_gae_nvs import build_dataset


def _psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    return 100.0 if mse < 1e-10 else -10 * math.log10(mse)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--scene_id", default="00028da87cc5a4c4")
    ap.add_argument("--max_targets", type=int, default=21)
    ap.add_argument("--cfg_scale", type=float, default=None,
                    help="override eval_cfg_scale (default: config value)")
    ap.add_argument("--num_inference_steps", type=int, default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    ckpt = args.checkpoint or latest_checkpoint(args.run_dir)[0]
    step = int(ckpt.rsplit("-", 1)[1])
    cfg = OmegaConf.load(os.path.join(args.run_dir, "args.yaml"))
    kind = ("geocolor_token"
            if cfg.get("wan_latent_cache_root", None) is not None else "gae")
    accelerator = Accelerator(mixed_precision=cfg.mixed_precision)
    accelerator.print(f"kind={kind} ckpt={ckpt} scene={args.scene_id}")

    model = build_model(kind, cfg, accelerator)
    model.set_training_param()
    if args.cfg_scale is not None:
        model.eval_cfg_scale = float(args.cfg_scale)
    if args.num_inference_steps is not None:
        model.eval_num_inference_steps = int(args.num_inference_steps)
    from safetensors.torch import load_file

    state = load_file(os.path.join(ckpt, "model.safetensors"), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    accelerator.print(f"loaded {len(state)} tensors, missing={len(missing)}")
    model.to(accelerator.device).eval()

    dataset = build_dataset(cfg, split=cfg.get("eval_split", None))
    record = next(r for r in dataset.records if r["scene_id"] == args.scene_id)
    views = sorted(record["views"], key=lambda v: str(v.view_id))
    cache_dir = os.path.join(
        cfg.get("gae_cache_root"), cfg.get("gae_dataset_name", "bmvs"))
    with open(os.path.join(cache_dir, f"{args.scene_id}.meta.json")) as fh:
        cached = {str(v) for v in json.load(fh)["views"]}
    cached_views = [v for v in views if str(v.view_id) in cached]
    step_s = max(1, len(cached_views) // model.num_source_views)
    src_views = cached_views[::step_s][:model.num_source_views]
    src_ids = {str(v.view_id) for v in src_views}
    tgt_views = [v for v in views if str(v.view_id) not in src_ids]
    n_full = (len(tgt_views) // model.num_target_views) * model.num_target_views
    if len(tgt_views) > args.max_targets:
        pick = np.linspace(0, len(tgt_views) - 1, args.max_targets).round().astype(int)
        tgt_views = [tgt_views[i] for i in dict.fromkeys(pick)]
        n_full = (len(tgt_views) // model.num_target_views) * model.num_target_views
    tgt_views = tgt_views[:n_full]
    accelerator.print(
        f"{len(views)} frames -> {len(src_views)} src + {len(tgt_views)} tgt")

    _FILL_CTX["scene"] = args.scene_id
    _FILL_CTX["fill_views"] = [str(v.view_id) for v in tgt_views]
    _install_zero_fill_shim()

    s = model.num_source_views
    device = accelerator.device
    lpips_net = model._get_lpips(device)

    def render(chunk_views, swap_perm=None):
        """Sample targets for a chunk; optionally cyclic-shift TARGET c2w rows."""
        items = [dataset._load_view(v)
                 for v in list(src_views) + list(chunk_views)]
        V = len(items)
        K = torch.stack([it["K"] for it in items]).unsqueeze(0)
        c2w = torch.stack([it["c2w"] for it in items]).unsqueeze(0).clone()
        if swap_perm is not None:
            c2w[:, s:] = c2w[:, s:][:, swap_perm]
        depth_masks = torch.cat([it["valid_mask"] for it in items], dim=0)
        rgb_masks = torch.cat(
            [torch.zeros_like(it["valid_mask"]) for it in items[:s]]
            + [torch.ones_like(it["valid_mask"]) for it in items[s:]], dim=0)
        batch = {
            "scene_id": [args.scene_id],
            "source_view_ids": [[str(v.view_id) for v in src_views]],
            "target_view_ids": [[str(v.view_id) for v in chunk_views]],
            "images": torch.stack([it["image"] for it in items]).unsqueeze(0),
            "depth_metric": torch.stack(
                [it["depth_metric"] for it in items]).unsqueeze(0),
            "K": K, "c2w": c2w,
            "view_roles": torch.tensor(
                [0] * s + [1] * len(chunk_views)).unsqueeze(0),
            "loss_masks": torch.stack(
                [depth_masks, rgb_masks], dim=0).unsqueeze(0),
            "sample_id": [f"{args.scene_id}"],
        }
        model._eval_inputs = True
        try:
            inputs = model.forward_preprocess(batch)
        finally:
            model._eval_inputs = False
        gen = torch.Generator(device=device)
        gen.manual_seed(int(cfg.get("eval_sample_seed", 1234)))
        if kind == "geocolor_token":
            pg, pc = model.sample_latents(inputs=inputs, generator=gen)
            pr = model._decode_color_views(
                pc[:, :, s:], batch["images"].shape[-2], batch["images"].shape[-1])
            pd = model._decode_geo_depth(
                pg[:, :, s:], batch["images"].shape[-2], batch["images"].shape[-1])
        else:
            pz = model.sample_latents(
                inputs, num_steps=model.eval_num_inference_steps, generator=gen)
            pr, pd = model._decode_views(
                pz[:, :, s:], batch["images"].shape[-2], batch["images"].shape[-1])
        return (pr[0].clamp(0, 1).cpu(), pd[0].cpu(),
                batch["images"][0, s:].float().clamp(0, 1),
                batch["depth_metric"][0, s:, 0].float())

    rows = []
    for ti in range(0, len(tgt_views), model.num_target_views):
        chunk = tgt_views[ti:ti + model.num_target_views]
        T = len(chunk)
        # cyclic shift of the TARGET pose block: [t0..t2] -> [t2,t0,t1]
        pred_t, depth_t, gt_rgb, gt_d = render(chunk)                      # true
        shift = torch.roll(torch.arange(T), 1)
        pred_s, depth_s, _, _ = render(chunk, swap_perm=shift)
        # gt_swap[j] = GT of the view whose pose slot j received = chunk[shift[j]]
        gt_rgb_swap = gt_rgb[shift]
        gt_d_swap = gt_d[shift]

        for j, view in enumerate(chunk):
            vid = str(view.view_id)
            m = {
                "view_id": vid,
                "psnr_true": _psnr(pred_t[j], gt_rgb[j]),
                "psnr_swap_vs_true": _psnr(pred_s[j], gt_rgb[j]),
                "psnr_swap_vs_swapGT": _psnr(pred_s[j], gt_rgb_swap[j]),
                "l1_swap_shift": float((pred_s[j] - pred_t[j]).abs().mean()),
                "l1_true_err": float((pred_t[j] - gt_rgb[j]).abs().mean()),
            }
            gd, pd = gt_d[j], depth_t[j]
            valid = (gd > 1e-3) & torch.isfinite(gd)
            if valid.sum() >= 16:
                pa = _align_median(pd, gd, valid)
                m["depth_delta1_true"] = float((torch.maximum(
                    pa / gd, gd / pa)[valid] < 1.25).float().mean())
            if lpips_net is not None:
                with torch.no_grad():
                    m["lpips_swap_vs_swapGT"] = float(lpips_net(
                        pred_s[j][None].to(device).float() * 2 - 1,
                        gt_rgb_swap[j][None].to(device).float() * 2 - 1))
                    m["lpips_swap_vs_true"] = float(lpips_net(
                        pred_s[j][None].to(device).float() * 2 - 1,
                        gt_rgb[j][None].to(device).float() * 2 - 1))
            rows.append(m)
            accelerator.print(
                f"  [{vid}] true={m['psnr_true']:.2f}  "
                f"swap|trueGT={m['psnr_swap_vs_true']:.2f}  "
                f"swap|swapGT={m['psnr_swap_vs_swapGT']:.2f}  "
                f"gain={m['psnr_swap_vs_swapGT'] - m['psnr_swap_vs_true']:+.2f}")

    def _mean(k):
        v = [r[k] for r in rows if r[k] == r[k]]
        return float(np.mean(v)) if v else float("nan")

    summary = {
        "run_dir": args.run_dir, "ckpt": os.path.basename(ckpt),
        "scene": args.scene_id, "frames": len(rows),
        "psnr_true": _mean("psnr_true"),
        "psnr_swap_vs_true": _mean("psnr_swap_vs_true"),
        "psnr_swap_vs_swapGT": _mean("psnr_swap_vs_swapGT"),
        "swap_gain": _mean("psnr_swap_vs_swapGT") - _mean("psnr_swap_vs_true"),
        "lpips_swap_vs_swapGT": _mean("lpips_swap_vs_swapGT"),
        "lpips_swap_vs_true": _mean("lpips_swap_vs_true"),
        "pose_var_ratio": _mean("l1_swap_shift") / max(_mean("l1_true_err"), 1e-6),
        "depth_delta1_true": _mean("depth_delta1_true"),
        "eval_cfg_scale": float(model.eval_cfg_scale),
        "steps": int(model.eval_num_inference_steps),
    }
    accelerator.print(
        f"\n=== pose-swap {args.run_dir} @{os.path.basename(ckpt)} ===\n"
        f"  psnr(true)={summary['psnr_true']:.2f}  "
        f"swap|trueGT={summary['psnr_swap_vs_true']:.2f}  "
        f"swap|swapGT={summary['psnr_swap_vs_swapGT']:.2f}  "
        f"SWAP_GAIN={summary['swap_gain']:+.3f} dB\n"
        f"  pose_var_ratio={summary['pose_var_ratio']:.3f}  "
        f"(swap-shift L1 / true-err L1; ~0 = pose ignored)\n"
        f"  lpips swap|swapGT={summary['lpips_swap_vs_swapGT']:.3f} vs "
        f"swap|true={summary['lpips_swap_vs_true']:.3f}  "
        f"depth_delta1(true)={summary['depth_delta1_true']:.3f}")

    out_fp = args.output or os.path.join(
        os.path.dirname(args.checkpoint or args.run_dir),
        f"pose_swap_step{step}.json")
    with open(out_fp, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "per_frame": rows}, fh, indent=2)
    accelerator.print(f"wrote {out_fp}")


if __name__ == "__main__":
    main()
