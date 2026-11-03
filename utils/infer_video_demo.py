"""Video-interpolation inference on the DL3DV overfit set, rendered as an mp4.

Loads an accelerate checkpoint, samples the target RGBD for interpolated views,
then rebuilds the camera trajectory as a timeline: source slots show GT only,
target slots show GT next to the prediction.  Encodes the timeline into an mp4
(GT | PRED, RGB over depth) plus a static contact grid.

Two modes:

* default   -- one video-mode sample (frozen eval selection, 6 src + 3 tgt)
  per --sample_idx, mirroring eval_rgbd_nvs.py inputs.
* --dense   -- sliding-window reconstruction of a long trajectory: sources
  every --src_gap indices, each forward interpolates the midpoints of 3 gaps,
  window slides by 3 gaps.  All windows are stitched into one video ordered
  by view_id, so an --end_idx--start_idx span of ~120 frames yields ~40
  timeline frames instead of 9.

Run as a MODULE from the project root, mirroring eval_rgbd_nvs.py:
  python -m tools.infer_dl3dv_interp_video \
      --config train_config/normal_config/dl3dv2_overfit_seq6s3t_video_interp.yaml \
      --checkpoint logs/dl3dv2_overfit_seq6s3t_video_interp/best_tgt_rgbd \
      --output_dir logs/dl3dv2_overfit_seq6s3t_video_interp/infer_video
  # dense:
  python -m tools.infer_dl3dv_interp_video ... --dense --end_idx 180
"""
import argparse
import math
import os
import shutil
import subprocess

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from safetensors.torch import load_file

from examples.dataset import Kubric4DRGBDNVSDataset
from examples.wanvideo.model_training.train_with_accelerate_rgbd_nvs import custom_collate_fn
from examples.wanvideo.model_training.WanRGBDNVSTrainingModule import (
    WanRGBDNVSTrainingModule,
)


def _psnr(pred, gt):
    # Inputs are uint8 0..255, so normalise by 255**2 (evaluate()'s copy of
    # this formula runs on [0,1] floats and must NOT be "fixed" there).
    mse = ((pred.astype(np.float64) - gt.astype(np.float64)) ** 2).mean()
    if mse < 1e-10:
        return 100.0
    return -10 * math.log10(mse / (255.0 ** 2))


def _plasma(gray):
    import matplotlib.cm as cm
    return (cm.plasma(np.clip(gray, 0.0, 1.0))[:, :, :3] * 255).astype(np.uint8)


def _label_strip(width, lines):
    """Caption strip with text."""
    h = 18 * len(lines) + 8
    strip = Image.new("RGB", (width, h), (16, 16, 16))
    draw = ImageDraw.Draw(strip)
    for i, line in enumerate(lines):
        draw.text((6, 4 + 18 * i), line, fill=(240, 240, 240))
    return strip


def _tile(arr_u8, header, captions):
    """[H,W,3] uint8 + column header + caption lines -> labelled PIL image."""
    img = Image.fromarray(arr_u8)
    head = Image.new("RGB", (img.width, 26), (30, 60, 120))
    ImageDraw.Draw(head).text((8, 6), header, fill=(255, 255, 255))
    col = Image.new("RGB", (img.width, head.height + img.height), (0, 0, 0))
    col.paste(head, (0, 0))
    col.paste(img, (0, head.height))
    strip = _label_strip(img.width, captions)
    out = Image.new("RGB", (img.width, col.height + strip.height), (0, 0, 0))
    out.paste(col, (0, 0))
    out.paste(strip, (0, col.height))
    return out


def _dense_windows(record, num_src, num_tgt, start_idx, end_idx, src_gap):
    """Sliding-window plan over a scene's views (index space, view_id order).

    Window k places num_src sources at start+k*3*src_gap + j*src_gap and asks
    for the midpoints of gaps 0/2/4 as targets.  The next window starts 3 gaps
    later, so gaps 3/4 of window k become gaps 0/1 of window k+1 and every gap
    except the final window's last two gets exactly one interpolated frame.
    Returns a list of (source_positions, target_positions).
    """
    slide = num_tgt * src_gap
    span = (num_src - 1) * src_gap
    windows = []
    base = start_idx
    while base + span <= end_idx:
        src_pos = [base + j * src_gap for j in range(num_src)]
        tgt_pos = [base + src_gap // 2 + 2 * i * src_gap for i in range(num_tgt)]
        windows.append((src_pos, tgt_pos))
        base += slide
    return windows


def _run_window(model, dataset, record, source_views, target_views, device,
                latent_scale, depth_latent_scale, num_steps, accelerator):
    """One 6-src + 3-tgt forward; returns per-view render results.

    Selection is forced by monkeypatching _select_views so dataset[0] loads
    exactly the requested views (the normal path freezes them behind the
    eval rng, which cannot express a custom sliding-window plan).
    """
    orig_select = dataset._select_views
    dataset._select_views = lambda record_, idx=0: (source_views, target_views)
    try:
        batch = custom_collate_fn([dataset[0]])
    finally:
        dataset._select_views = orig_select

    with torch.autocast(device_type="npu", dtype=model.pipe.torch_dtype, enabled=True):
        model._eval_inputs = True
        try:
            inputs = model.forward_preprocess(batch)
        finally:
            model._eval_inputs = False
        gen = torch.Generator(device=device)
        gen.manual_seed(model.eval_sample_seed)
        pred_depth, pred_rgb = model.sample_rgb(
            inputs, num_steps=num_steps, generator=gen
        )

    if latent_scale != 1.0:
        pred_rgb = pred_rgb / latent_scale
    if depth_latent_scale != 1.0:
        pred_depth = pred_depth / depth_latent_scale

    gt_images = batch["images"][0].numpy()          # [V,3,H,W] pixel GT
    gt_disp = batch["disparity"][0].numpy()          # [V,3,H,W] shared-scale
    num_src = len(source_views)
    out = {}
    for vi, view in enumerate(list(source_views) + list(target_views)):
        slot = vi
        gt_rgb = (gt_images[slot].transpose(1, 2, 0) * 255).astype(np.uint8)
        gt_d = gt_disp[slot][0]
        role = "src" if vi < num_src else "tgt"
        entry = {"role": role, "gt_rgb": gt_rgb, "gt_disp": gt_d}
        if role == "tgt":
            # pred_* latents cover ALL views: [sources..., targets...], so the
            # slice index is the global slot vi, not the target-local t.
            rgb = model.pipe.vae.decode(
                pred_rgb[:, :, vi:vi + 1].to(device).to(model.pipe.torch_dtype), device=device
            )[0, :, 0].float().clamp(0, 1).cpu().numpy()
            raw = model.pipe.vae.decode(
                pred_depth[:, :, vi:vi + 1].to(device).to(model.pipe.torch_dtype), device=device
            )[0, :, 0].float().mean(0).cpu().numpy()
            pred_rgb_u8 = (rgb.transpose(1, 2, 0) * 255).astype(np.uint8)
            entry["pred_rgb"] = pred_rgb_u8
            entry["pred_disp"] = raw
            entry["psnr"] = _psnr(pred_rgb_u8, gt_rgb)
        out[view.view_id] = entry
    del inputs, pred_rgb, pred_depth, batch
    torch.npu.empty_cache()
    return out


def _ffmpeg_bin():
    """System ffmpeg, or the binary bundled with imageio_ffmpeg (NPU boxes
    often have no system ffmpeg)."""
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _compose_video(sample_dir, timeline, framerate):
    """timeline: ordered list of (view_id, entry dict) -> source pngs + grid png + mp4.

    Sources are exported as plain GT pngs under sources/; the grid and mp4
    hold only the interpolated targets as a two-column GT | PRED comparison.
    """
    src_dir = os.path.join(sample_dir, "sources")
    os.makedirs(src_dir, exist_ok=True)
    grid_frames = []
    for vid, e in timeline:
        gt_rgb = e["gt_rgb"]
        if e["role"] == "src":
            Image.fromarray(gt_rgb).save(os.path.join(src_dir, f"{vid}.png"))
            continue
        cap = [f"INTERP TARGET  view_id {vid}", f"pred psnr {e['psnr']:.2f} dB"]
        cols = (
            _tile(gt_rgb, "GT", cap),
            _tile(e["pred_rgb"], "PRED", cap),
            _tile(_plasma(e["gt_disp"]), "GT depth", cap),
            _tile(_plasma(e["pred_disp"]), "PRED depth", cap),
        )
        grid_frames.append(np.concatenate([np.array(c) for c in cols], axis=1))

    grid = np.concatenate(grid_frames, axis=0)
    Image.fromarray(grid).save(os.path.join(sample_dir, "timeline_grid.png"))

    tmp = os.path.join(sample_dir, "frames")
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)  # stale frames from a previous run would be re-encoded
    os.makedirs(tmp)
    for i, fr in enumerate(grid_frames):
        im = Image.fromarray(fr)
        # libx264 wants even dimensions.
        canvas = Image.new("RGB", (im.width + im.width % 2, im.height + im.height % 2), (0, 0, 0))
        canvas.paste(im, (0, 0))
        canvas.save(os.path.join(tmp, f"frame_{i:03d}.png"))
    mp4 = os.path.join(sample_dir, "interp_compare.mp4")
    subprocess.run(
        [_ffmpeg_bin(), "-y", "-framerate", str(framerate),
         "-i", os.path.join(tmp, "frame_%03d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", mp4],
        check=True, capture_output=True)
    return mp4, len(grid_frames)


def build_model(cfg, accelerator, num_steps):
    """Construct the WanRGBDNVSTrainingModule exactly as this script's main
    always did.  Shared with tools/infer_dl3dv_extrap_video.py."""
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
        local_model_path=cfg.get("wan_model_root", "./models"),
        skip_download=cfg.get("skip_download", False),
        strip_i2v_image_input=cfg.get("strip_i2v_image_input", False),
        num_source_views=cfg.num_source_views,
        num_target_views=cfg.num_target_views,
        vae_encoding_mode=cfg.vae_encoding_mode,
        init_dvd_ckpt=None,
        rgb_loss_weight=cfg.rgb_loss_weight,
        depth_loss_weight=cfg.depth_loss_weight,
        view_condition_plucker_resolution=cfg.get("view_condition_plucker_resolution", "latent"),
        view_condition_pixel_scale=cfg.get("view_condition_pixel_scale", 8),
        view_condition_mode=cfg.get("view_condition_mode", "add"),
        view_condition_use_role=cfg.get("view_condition_use_role", True),
        view_condition_normalize_moment=cfg.get("view_condition_normalize_moment", False),
        latent_scale=cfg.get("latent_scale", 1.0),
        depth_latent_scale=cfg.get("depth_latent_scale", None),
        token_layout=cfg.get("token_layout", "channel"),
        warp_mode=cfg.get("warp_mode", "none"),
        warp_splat_footprint=cfg.get("warp_splat_footprint", "nearest"),
        warp_dropout_prob=cfg.get("warp_dropout_prob", 0.5),
        warp_two_pass=cfg.get("warp_two_pass", True),
        warp_perturb_prob=cfg.get("warp_perturb_prob", 0.0),
        warp_perturb_gain_log_std=cfg.get("warp_perturb_gain_log_std", 0.34),
        warp_perturb_shift_std=cfg.get("warp_perturb_shift_std", 0.059),
        warp_perturb_blur_sigma=cfg.get("warp_perturb_blur_sigma", 2.0),
        warp_perturb_lowfreq_std=cfg.get("warp_perturb_lowfreq_std", 0.05),
        warp_perturb_lowfreq_cells=cfg.get("warp_perturb_lowfreq_cells", 6),
        warp_perturb_fill_invalid=cfg.get("warp_perturb_fill_invalid", True),
        warp_perturb_fill_scale=cfg.get("warp_perturb_fill_scale", 1.0),
        depth_warp_block=cfg.get("depth_warp_block", False),
        per_frame_timestep=cfg.get("per_frame_timestep", False),
        clean_source_depth=cfg.get("clean_source_depth", False),
        clean_source_depth_prob=cfg.get("clean_source_depth_prob", 0.5),
        shared_rope_frame_ids=cfg.get("shared_rope_frame_ids", False),
        rope_frame_axis=cfg.get("rope_frame_axis", None),
        rope_pose_translation_scale=cfg.get("rope_pose_translation_scale", None),
        roce=cfg.get("roce", "off"),
        camera_geometry_fp32=cfg.get("camera_geometry_fp32", False),
        rgb_flow_matching=cfg.get("rgb_flow_matching", True),
        rgb_flow_shift=cfg.get("rgb_flow_shift", 5.0),
        depth_flow_matching=cfg.get("depth_flow_matching", False),
        depth_head_init=cfg.get("depth_head_init", "pretrained"),
        eval_num_inference_steps=num_steps,
        eval_flow_shift=cfg.get("eval_flow_shift", None),
        eval_sample_seed=cfg.get("eval_sample_seed", 1234),
        eval_lpips_net=cfg.get("eval_lpips_net", "alex"),
        eval_depth_vae_ceiling=False,
        eval_vis_downscale=1,
        eval_vis_format="png",
        eval_vis_depth_norm="shared",
    )
    return model


def build_dataset(cfg):
    """The eval dataset used for inference (shared with the extrap tool)."""
    return Kubric4DRGBDNVSDataset(
        kubric4d_root=cfg.kubric4d_root,
        manifest_path=cfg.manifest_path,
        resolution=tuple(cfg.resolution),
        num_source_views=cfg.num_source_views,
        num_target_views=cfg.num_target_views,
        random_permute_source_views=cfg.random_permute_source_views,
        view_order=cfg.get("view_order", "random"),
        video_tgt_interval_prob=cfg.get("video_tgt_interval_prob", 0.2),
        video_max_src_span=cfg.get("video_max_src_span", None),
        disparity_norm_mode=cfg.get("disparity_norm_mode", "shared_scale"),
        disparity_norm_headroom=cfg.get("disparity_norm_headroom", 1.5),
        deterministic_view_seed=int(cfg.get("eval_view_seed", 1234)),
        min_view_valid_fraction=cfg.get("min_view_valid_fraction", 0.0),
        view_validity_path=cfg.get("view_validity_path", None),
        samples_per_scene=cfg.get("samples_per_scene", 1),
        split_path=cfg.get("split_path", None),
        split=cfg.get("eval_split", None),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sample_idx", type=str, default="0",
                        help="Comma-separated dataset indices (default single-sample mode).")
    parser.add_argument("--dense", action="store_true",
                        help="Sliding-window dense trajectory reconstruction.")
    parser.add_argument("--record_idx", type=int, default=0,
                        help="Scene index into the manifest (dense mode).")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="First view index (view_id order) of the dense span.")
    parser.add_argument("--end_idx", type=int, default=120,
                        help="Last view index (inclusive) of the dense span.")
    parser.add_argument("--src_gap", type=int, default=6,
                        help="Index gap between consecutive source views.")
    parser.add_argument("--num_inference_steps", type=int, default=None,
                        help="Euler steps for target RGB sampling. Defaults to "
                             "cfg.eval_num_inference_steps (what train.log used).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--framerate", type=int, default=4)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    set_seed(args.seed)
    accelerator = Accelerator(mixed_precision=cfg.mixed_precision)
    device = accelerator.device

    num_steps = args.num_inference_steps or int(cfg.get("eval_num_inference_steps", 4))

    model = build_model(cfg, accelerator, num_steps)

    weights = os.path.join(args.checkpoint, "model.safetensors")
    state = load_file(weights, device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    accelerator.print(f"Loaded {weights}: {len(state)} tensors, "
                      f"missing={len(missing)}, unexpected={len(unexpected)}")
    if unexpected:
        accelerator.print(f"  first unexpected: {unexpected[:5]}")
    model.to(device)
    model.eval()

    dataset = build_dataset(cfg)

    os.makedirs(args.output_dir, exist_ok=True)
    latent_scale = float(cfg.get("latent_scale", 1.0))
    depth_latent_scale = model.depth_latent_scale
    num_src = cfg.num_source_views

    if args.dense:
        record = dataset.records[args.record_idx % len(dataset.records)]
        views_sorted = sorted(record["views"], key=lambda v: v.view_id)
        end_idx = min(args.end_idx, len(views_sorted) - 1)
        windows = _dense_windows(record, num_src, cfg.num_target_views,
                                 args.start_idx, end_idx, args.src_gap)
        scene_tag = str(record["scene_id"]).split("/")[-1][:24]
        sample_dir = os.path.join(
            args.output_dir,
            f"dense_r{args.record_idx}_{args.start_idx}-{end_idx}_g{args.src_gap}_{scene_tag}")
        os.makedirs(sample_dir, exist_ok=True)
        accelerator.print(f"[dense] scene={record['scene_id']} "
                          f"views={len(views_sorted)} span={args.start_idx}..{end_idx} "
                          f"src_gap={args.src_gap} -> {len(windows)} forward windows")

        by_id = {}
        src_ids_all = []
        for w, (src_pos, tgt_pos) in enumerate(windows):
            source_views = [views_sorted[min(p, len(views_sorted) - 1)] for p in src_pos]
            target_views = [views_sorted[min(p, len(views_sorted) - 1)] for p in tgt_pos]
            accelerator.print(f"[dense] window {w + 1}/{len(windows)}: "
                              f"src={[v.view_id for v in source_views]} "
                              f"tgt={[v.view_id for v in target_views]}")
            res = _run_window(model, dataset, record, source_views, target_views,
                              device, latent_scale, depth_latent_scale, num_steps,
                              accelerator)
            for vid, e in res.items():
                if e["role"] == "src":
                    src_ids_all.append(vid)
                # Targets: keep the first render of each view_id (windows do not
                # overlap on targets by construction, so this never triggers).
                by_id.setdefault(vid, e)
        timeline = sorted(by_id.items(), key=lambda kv: kv[0])
        mp4, n = _compose_video(sample_dir, timeline, args.framerate)
        tgt_psnrs = [e["psnr"] for _, e in timeline if e["role"] == "tgt"]
        accelerator.print(
            f"[dense] wrote {mp4}: {n} timeline frames "
            f"({len(src_ids_all)} unique source slots, {len(tgt_psnrs)} interpolated "
            f"targets, mean psnr {np.mean(tgt_psnrs):.2f} dB)")
    else:
        for idx in [int(v) for v in args.sample_idx.split(",") if v.strip()]:
            record = dataset.records[idx % len(dataset.records)]
            # Same deterministic rng path as __getitem__, so these are exactly
            # the views inside dataset[idx] -- but with the view_ids needed to
            # order the timeline.
            source_views, target_views = dataset._select_views(record, idx=idx)
            accelerator.print(f"[sample {idx}] scene={record['scene_id']} "
                              f"src_ids={[v.view_id for v in source_views]} "
                              f"tgt_ids={[v.view_id for v in target_views]}")
            res = _run_window(model, dataset, record, source_views, target_views,
                              device, latent_scale, depth_latent_scale, num_steps,
                              accelerator)
            timeline = sorted(res.items(), key=lambda kv: kv[0])
            scene_tag = str(record["scene_id"]).split("/")[-1][:24]
            sample_dir = os.path.join(args.output_dir, f"idx{idx:04d}_{scene_tag}")
            os.makedirs(sample_dir, exist_ok=True)
            mp4, n = _compose_video(sample_dir, timeline, args.framerate)
            accelerator.print(f"[sample {idx}] wrote {mp4} ({n} timeline frames)")


if __name__ == "__main__":
    main()
