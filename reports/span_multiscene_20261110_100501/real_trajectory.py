"""Trajectory extrapolation test for the GAE / GeoColor-Token NVS runs.

Pick one video scene, pin S source frames (from the latent cache), then render
ALL remaining frames of the scene with their GT camera poses and compare each
render against GT.  This is the direct camera-control probe: a model that
follows the conditioning must track the whole trajectory; a model that copies
the nearest source stays glued to it.

Target views do NOT need cached latents (their slots start from noise), so the
`load_file` shim below zero-fills uncached target views -- only the SOURCE
views must be cached.  A nearest-source PSNR baseline is reported per frame so
"pred == copy of closest source" is visible numerically.

Outputs under <run_dir>/traj_step<k>/<scene>/:
  montage_XX.jpg          rows gt rgb / pred rgb / gt depth / pred depth
  frames/<vid>_side.jpg   per-frame [gt | pred | gt depth | pred depth]
  per_frame_metrics.jsonl view_id, psnr, lpips, depth metrics, copy baseline
  video_<run>_step<k>.mp4 rendered trajectory video (rows [GT|PRED] rgb over
                          [GT|PRED] depth, captioned; --no_video to skip)

Run from the repo root:
  ASCEND_RT_VISIBLE_DEVICES=4 python -m examples.wanvideo.model_training.infer_trajectory \
      --run_dir logs/ovf_gae_re10k_aux0_patch1 [--scene_id 00028da87cc5a4c4]
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
    _depth_np,
    _rgb_np,
    build_model,
    latest_checkpoint,
)
from examples.wanvideo.model_training.train_with_accelerate_gae_nvs import build_dataset
from examples.wanvideo.model_training.WanRGBDNVSTrainingModule import _annotate_grid

_FILL_CTX = {}


def _install_zero_fill_shim():
    """Zero-fill uncached views of the target scene inside safetensors.load_file.

    Both cache loaders do `from safetensors.torch import load_file` INSIDE the
    function body, so patching the module attribute intercepts them without
    touching loader-specific grid/pad logic.  Only shape correctness matters:
    sample_latents reads SOURCE slots from the cache and takes target slots
    from pure noise.
    """
    import safetensors.torch as st

    orig = st.load_file

    def patched(fp, device="cpu", **kwargs):
        bucket = orig(fp, device=device, **kwargs)
        scene = _FILL_CTX.get("scene")
        if scene and os.path.basename(str(fp)) == f"{scene}.safetensor":
            fill = _FILL_CTX.get("fill_views") or ()
            ref = bucket[next(iter(bucket))]
            C, h, w = ref.shape
            for vid in fill:
                key = f"v{vid}"
                if key not in bucket:
                    bucket[key] = torch.zeros(C, h, w, dtype=ref.dtype)
        return bucket

    st.load_file = patched


@torch.no_grad()
def run_scene(run_dir, checkpoint, scene_id, num_source_views, max_targets,
              output_dir, downscale=2, cfg_scales=(1.0,), shared_target_noise=False,
              cfg_scale_override=None, manifest=None, cache_name=None):
    cfg = OmegaConf.load(os.path.join(run_dir, "args.yaml"))
    if cfg.get("gae_stats_path") is None:
        cfg.gae_stats_path = os.path.join(cfg.gae_cache_root, cfg.gae_dataset_name, "latent_stats.pt")
    if manifest is not None:
        cfg.manifest_path = manifest
    if cache_name is not None:
        cfg.gae_dataset_name = cache_name
    kind = ("geocolor_token"
            if cfg.get("wan_latent_cache_root", None) is not None else "gae")
    accelerator = Accelerator(mixed_precision=cfg.mixed_precision)
    accelerator.print(f"kind={kind}  ckpt={checkpoint}  scene={scene_id}")

    model = build_model(kind, cfg, accelerator)
    model.set_training_param()
    if cfg_scale_override is not None:
        # Eval-time CFG sweep: override the checkpoint's trained-in
        # eval_cfg_scale so the same weights can be probed at guidance 1.0
        # (pure conditional) vs >1 (classifier-free guidance).
        model.eval_cfg_scale = float(cfg_scale_override)
    from safetensors.torch import load_file

    weights = os.path.join(checkpoint, "model.safetensors")
    state = load_file(weights, device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    accelerator.print(
        f"Loaded {weights}: {len(state)} tensors, missing={len(missing)}, "
        f"unexpected={len(unexpected)}")
    model.to(accelerator.device)
    model.eval()

    # Full record (no cache restriction): the trajectory must span the WHOLE
    # video, sources just have to be cached.
    dataset = build_dataset(cfg, split=cfg.get("eval_split", None))
    record = next(r for r in dataset.records if r["scene_id"] == scene_id)
    views = sorted(record["views"], key=lambda v: str(v.view_id))

    cache_dir = os.path.join(
        cfg.get("gae_cache_root"), cfg.get("gae_dataset_name", "bmvs"))
    meta_fp = os.path.join(cache_dir, f"{scene_id}.meta.json")
    with open(meta_fp, "r", encoding="utf-8") as fh:
        cached = {str(v) for v in json.load(fh)["views"]}
    cached_views = [v for v in views if str(v.view_id) in cached]
    if len(cached_views) < num_source_views:
        raise ValueError(f"only {len(cached_views)} cached views in {scene_id}")
    step = max(1, len(cached_views) // num_source_views)
    src_views = cached_views[::step][:num_source_views]
    src_ids = {str(v.view_id) for v in src_views}
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir,"sources.json"),"w") as fh:
        json.dump({"source_view_ids":[str(v.view_id) for v in src_views]},fh)

    tgt_views = [v for v in views if str(v.view_id) not in src_ids]
    if max_targets and len(tgt_views) > max_targets:
        pick = np.linspace(0, len(tgt_views) - 1, max_targets).round().astype(int)
        tgt_views = [tgt_views[i] for i in dict.fromkeys(pick)]
    accelerator.print(
        f"{len(views)} frames, {len(cached_views)} cached -> "
        f"{len(src_views)} sources {[str(v.view_id) for v in src_views]}, "
        f"{len(tgt_views)} targets")

    _FILL_CTX["scene"] = scene_id
    _FILL_CTX["fill_views"] = [str(v.view_id) for v in tgt_views]
    _install_zero_fill_shim()

    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    s = model.num_source_views
    device = accelerator.device
    lpips_net = model._get_lpips(device)

    def _load_batch(tgt_chunk):
        items = [dataset._load_view(v) for v in list(src_views) + list(tgt_chunk)]
        V = len(items)
        depth_masks = torch.cat([it["valid_mask"] for it in items], dim=0)
        rgb_masks = torch.cat(
            [torch.zeros_like(it["valid_mask"]) for it in items[:s]]
            + [torch.ones_like(it["valid_mask"]) for it in items[s:]], dim=0)
        return {
            "scene_id": [scene_id],
            "source_view_ids": [[str(v.view_id) for v in src_views]],
            "target_view_ids": [[str(v.view_id) for v in tgt_chunk]],
            "images": torch.stack([it["image"] for it in items]).unsqueeze(0),
            "depth_metric": torch.stack(
                [it["depth_metric"] for it in items]).unsqueeze(0),
            "K": torch.stack([it["K"] for it in items]).unsqueeze(0),
            "c2w": torch.stack([it["c2w"] for it in items]).unsqueeze(0),
            "view_roles": torch.tensor(
                [0] * s + [1] * len(tgt_chunk)).unsqueeze(0),
            "loss_masks": torch.stack(
                [depth_masks, rgb_masks], dim=0).unsqueeze(0),
            "sample_id": [f"{scene_id}"],
        }

    # Nearest-source-copy baseline per target: best PSNR against any source GT.
    src_gt = torch.stack([
        dataset._load_view(v)["image"] for v in src_views])          # [S,3,H,W]

    metrics = []
    n_full = (len(tgt_views) // model.num_target_views) * model.num_target_views
    if n_full < len(tgt_views):
        accelerator.print(
            f"dropping {len(tgt_views) - n_full} trailing frame(s): a ragged "
            "final chunk changes the DiT input shape and NPU graph compile "
            "rejects it mid-run")
    for ti in range(0, n_full, model.num_target_views):
        chunk = tgt_views[ti:ti + model.num_target_views]
        batch = _load_batch(chunk)
        model._eval_inputs = True
        try:
            inputs = model.forward_preprocess(batch)
        finally:
            model._eval_inputs = False

        gen = torch.Generator(device=device)
        gen.manual_seed(int(cfg.get("eval_sample_seed", 1234)))
        if kind == "geocolor_token":
            pred_geo, pred_color = model.sample_latents(
                inputs=inputs, generator=gen, shared_target_noise=shared_target_noise)
            pred_rgb = model._decode_color_views(
                pred_color[:, :, s:], batch["images"].shape[-2],
                batch["images"].shape[-1])
            pred_depth = model._decode_geo_depth(
                pred_geo[:, :, s:], batch["images"].shape[-2],
                batch["images"].shape[-1])
        else:
            pred_z = model.sample_latents(
                inputs, num_steps=model.eval_num_inference_steps, generator=gen,
                shared_target_noise=shared_target_noise)
            pred_rgb, pred_depth = model._decode_views(
                pred_z[:, :, s:], batch["images"].shape[-2],
                batch["images"].shape[-1])

        gt_rgb = batch["images"][0].float().clamp(0, 1)
        gt_depth = batch["depth_metric"][0, :, 0].float()

        chunk_rows = {k: [] for k in ("gt_rgb", "pred_rgb", "gt_d", "pred_d")}
        chunk_caps = []
        for j, view in enumerate(chunk):
            vid = str(view.view_id)
            gr, pr = gt_rgb[s + j], pred_rgb[0, j].clamp(0, 1)
            gd, pd = gt_depth[s + j], pred_depth[0, j]
            valid = (gd > 1e-3) & torch.isfinite(gd)
            mse = float(((pr - gr) ** 2).mean())
            psnr = 100.0 if mse < 1e-10 else -10 * math.log10(mse)
            copy_psnr = min(float(((gr - sg) ** 2).mean()) for sg in src_gt)
            copy_psnr = 100.0 if copy_psnr < 1e-10 else -10 * math.log10(copy_psnr)
            lp = float("nan")
            if lpips_net is not None:
                with torch.no_grad():
                    lp = float(lpips_net(
                        pr[None].to(device).float() * 2 - 1,
                        gr[None].to(device).float() * 2 - 1))
            pa = _align_median(pd, gd, valid)
            if valid.sum() >= 16:
                rel = float((((pa - gd) / gd).abs()[valid]).mean())
                d1 = float((torch.maximum(
                    pa / gd, gd / pa)[valid] < 1.25).float().mean())
            else:
                rel = d1 = float("nan")
            vals = gd[valid]
            lo = float(np.percentile(vals.numpy(), 2))
            hi = float(np.percentile(vals.numpy(), 98))
            gt_d_np = _depth_np(gd.numpy(), lo, hi)
            pd_np = _depth_np(pa.numpy(), lo, hi)

            row = {
                "view_id": vid, "tgt_rgb_psnr": psnr, "tgt_rgb_lpips": lp,
                "tgt_depth_delta1": d1, "tgt_depth_abs_rel_aligned": rel,
                "nearest_src_psnr": copy_psnr,
            }
            metrics.append(row)
            accelerator.print(
                f"  [{vid}] psnr={psnr:.2f} lpips={lp:.3f} d1={d1:.3f} "
                f"rel={rel:.3f} | copy_src={copy_psnr:.2f}")

            side = np.concatenate([
                _rgb_np(gr), _rgb_np(pr),
                gt_d_np, pd_np], axis=1)
            from PIL import Image as PILImage

            PILImage.fromarray(side).save(
                os.path.join(frames_dir, f"{vid}_side.jpg"), quality=92)
            chunk_rows["gt_rgb"].append(_rgb_np(gr))
            chunk_rows["pred_rgb"].append(_rgb_np(pr))
            chunk_rows["gt_d"].append(gt_d_np)
            chunk_rows["pred_d"].append(pd_np)
            chunk_caps.append([
                f"v{vid} psnr={psnr:.2f}",
                f"lpips={lp:.3f} d1={d1:.3f}",
                f"copy={copy_psnr:.2f}",
            ])

        H, W = chunk_rows["gt_rgb"][0].shape[:2]
        sep = np.ones((H, 4, 3), np.uint8) * 200
        row_sep = np.ones((4, W * len(chunk) + 4 * (len(chunk) - 1), 3),
                          np.uint8) * 200

        def hstack(imgs):
            out_ = imgs[0]
            for im in imgs[1:]:
                out_ = np.concatenate([out_, sep, im], axis=1)
            return out_

        grid = np.concatenate([
            hstack(chunk_rows["gt_rgb"]), row_sep,
            hstack(chunk_rows["pred_rgb"]), row_sep,
            hstack(chunk_rows["gt_d"]), row_sep,
            hstack(chunk_rows["pred_d"])], axis=0)
        from PIL import Image as PILImage

        img = PILImage.fromarray(grid)
        img = img.resize((grid.shape[1] // downscale, grid.shape[0] // downscale),
                         PILImage.LANCZOS)
        tile_h = H / downscale
        tile_w = W / downscale
        col_labels = [f"v{str(v.view_id)}" for v in chunk]
        captions = [(i, 1, lines) for i, lines in enumerate(chunk_caps)]
        try:
            img = _annotate_grid(
                img,
                ["gt rgb", "pred rgb", "gt depth", "pred depth"],
                col_labels, (tile_h, tile_w), sep=4.0 / downscale,
                captions=captions)
        except Exception as exc:
            accelerator.print(f"  labels unavailable ({exc})")
        img.save(os.path.join(
            output_dir, f"montage_{ti // model.num_target_views:02d}.jpg"),
            quality=92)

    with open(os.path.join(output_dir, "per_frame_metrics.jsonl"), "w",
              encoding="utf-8") as fh:
        for row in metrics:
            fh.write(json.dumps(row) + "\n")

    def _mean(key):
        vals = [r[key] for r in metrics if r[key] == r[key]]
        return float(np.mean(vals)) if vals else float("nan")

    accelerator.print(
        f"\n=== trajectory {scene_id} @{os.path.basename(checkpoint)} "
        f"({len(metrics)} frames, {model.eval_num_inference_steps} steps) ===\n"
        f"  psnr={_mean('tgt_rgb_psnr'):.4f}  lpips={_mean('tgt_rgb_lpips'):.4f}  "
        f"delta1={_mean('tgt_depth_delta1'):.4f}  "
        f"rel_aligned={_mean('tgt_depth_abs_rel_aligned'):.4f}  "
        f"nearest_src_psnr={_mean('nearest_src_psnr'):.4f}")


def _synthesize_video(output_dir, run_name, step, scene_id, fps=3.0):
    """Encode the rendered trajectory into an mp4 next to the frame dumps.

    Layout matches tools/make_traj_videos.py's per-run video: rows
    [GT | PRED] rgb over [GT | PRED] depth, per-frame metric captions.
    Encoding failures (missing imageio-ffmpeg, odd dims, ...) must never
    discard the frame dumps, so every failure path only warns.
    """
    metrics_fp = os.path.join(output_dir, "per_frame_metrics.jsonl")
    with open(metrics_fp) as fh:
        rows = [json.loads(l) for l in fh]
    try:
        from tools.make_traj_videos import (
            _label, _sep_h, _sep_v, _split_side, encode)
    except ImportError as exc:
        print(f"video skipped (cannot import encoder helpers: {exc})")
        return

    frames = []
    for r in rows:
        fp = os.path.join(output_dir, "frames", f"{r['view_id']}_side.jpg")
        if not os.path.isfile(fp):
            print(f"video skipped (missing frame dump {fp})")
            return
        gt_rgb, pred_rgb, gt_d, pred_d = _split_side(fp)
        h = gt_rgb.shape[0]
        rgb = np.concatenate([gt_rgb, _sep_v(h), pred_rgb], axis=1)
        dep = np.concatenate([gt_d, _sep_v(h), pred_d], axis=1)
        img = np.concatenate([rgb, _sep_h(rgb.shape[1], 4), dep], axis=0)
        frames.append(_label(img, [
            f"v{r['view_id']}  psnr={r['tgt_rgb_psnr']:.2f}  "
            f"lpips={r['tgt_rgb_lpips']:.3f}  d1={r['tgt_depth_delta1']:.3f}  "
            f"rel={r['tgt_depth_abs_rel_aligned']:.3f}  "
            f"(copy-src {r['nearest_src_psnr']:.2f})"],
            title=f"{run_name} @step{step}  {scene_id}"))

    out_fp = os.path.join(output_dir, f"video_{run_name}_step{step}.mp4")
    try:
        encode(frames, fps, out_fp)
    except Exception as exc:
        print(f"video encode failed: {exc}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--scene_id", type=str, default="00028da87cc5a4c4")
    parser.add_argument("--max_targets", type=int, default=30)
    parser.add_argument("--downscale", type=int, default=2)
    parser.add_argument("--cfg_scale", type=float, default=None,
                        help="override the model's eval_cfg_scale (e.g. 1.0 "
                             "to disable CFG); output goes to "
                             "traj_step<N>_cfg<X> to keep runs apart")
    parser.add_argument("--shared_target_noise", action="store_true",
                        help="all target slots start from ONE noise field so "
                             "adjacent frames pick the same conditional mode "
                             "(temporal-consistency aid, inference-only)")
    parser.add_argument("--fps", type=float, default=3.0,
                        help="frame rate of the synthesized trajectory mp4")
    parser.add_argument("--no_video", action="store_true",
                        help="skip mp4 synthesis after rendering")
    parser.add_argument("--manifest", type=str, default=None,
                        help="override cfg.manifest_path, e.g. the re10k "
                             "manifest_test.jsonl to probe a held-out scene")
    parser.add_argument("--cache_name", type=str, default=None,
                        help="override gae_dataset_name for the latent cache "
                             "dir, e.g. re10k_test to match --manifest")
    args = parser.parse_args()

    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint, step = latest_checkpoint(args.run_dir)
    else:
        step = int(checkpoint.rsplit("-", 1)[1])
    cfg = OmegaConf.load(os.path.join(args.run_dir, "args.yaml"))
    suffix = ("" if args.cfg_scale is None
              else f"_cfg{args.cfg_scale:g}")
    if args.manifest is not None or args.cache_name is not None:
        suffix += "_test"
    output_dir = os.path.join(
        args.run_dir, f"traj_step{step}{suffix}", args.scene_id)
    run_scene(args.run_dir, checkpoint, args.scene_id,
              int(cfg.num_source_views), args.max_targets, output_dir,
              args.downscale, shared_target_noise=args.shared_target_noise,
              cfg_scale_override=args.cfg_scale,
              manifest=args.manifest, cache_name=args.cache_name)
    if not args.no_video:
        _synthesize_video(output_dir,
                          os.path.basename(args.run_dir.rstrip("/")),
                          step, args.scene_id, fps=args.fps)


if __name__ == "__main__":
    main()
