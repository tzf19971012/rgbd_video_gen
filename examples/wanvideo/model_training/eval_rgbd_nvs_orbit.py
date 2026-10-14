"""Directional NVS inference: render front/back/left/right/up/down target views.

Like eval_rgbd_nvs, but the 3 target views per batch are not drawn randomly --
they are the captured views that best match each of six canonical directions
(front/back/left/right/up/down) around the scene's trajectory centroid, so one
run produces an interpretable surround set instead of a random draw.  Sources
are six views evenly spaced in azimuth.  Target views are REAL captures, so GT
RGB/depth metrics and the standard eval grid images still work.

Additionally renders VIRTUAL forward-tilted targets (--tilt_degrees, default
30/60/90, where 90 = straight-down BEV).  Each virtual camera sits above the
captured front view with image-up tracking the front camera's heading; these
poses have no captured GT, so they go through the virtual-view path (no
metrics) and are saved as visualizations only.

Run from the repo root, same env as eval_rgbd_nvs:
  python -m examples.wanvideo.model_training.eval_rgbd_nvs_orbit \
      --config <args.yaml> --checkpoint <dir> --manifest_path <jsonl> \
      --split_path <json> --output_dir <dir>
"""
import argparse
import math
import os

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from PIL import Image
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from examples.dataset import Kubric4DRGBDNVSDataset
from examples.wanvideo.model_training.eval_rgbd_nvs import _num_list
from examples.wanvideo.model_training.infer_virtual_views import (
    VirtualView,
    _patch_load_view,
)
from examples.wanvideo.model_training.train_with_accelerate_rgbd_nvs import custom_collate_fn
from examples.wanvideo.model_training.WanRGBDNVSTrainingModule import (
    WanRGBDNVSTrainingModule,
    resolve_rope_frame_axis,
)

AXES = ["front", "back", "left", "right", "up", "down"]


def _directional_targets(record):
    """Pick one captured view per canonical direction around the trajectory."""
    centers, forwards = [], []
    for v in record["views"]:
        c2w = v.c2w.double().numpy()
        centers.append(c2w[:3, 3])
        forwards.append(c2w[:3, 2])  # OpenCV camera z-axis
    centers = np.stack(centers)
    forwards = np.stack(forwards)
    centroid = centers.mean(0)

    mean_f = forwards.mean(0)
    mean_f /= np.linalg.norm(mean_f)
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(mean_f, world_up)) > 0.95:      # near-vertical trajectory
        world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(mean_f, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, mean_f)

    basis = {
        "front": mean_f, "back": -mean_f,
        "right": right, "left": -right,
        "up": up, "down": -up,
    }
    pos = centers - centroid
    pos_dir = pos / np.maximum(np.linalg.norm(pos, axis=1, keepdims=True), 1e-9)
    fwd_dir = forwards / np.maximum(np.linalg.norm(forwards, axis=1, keepdims=True), 1e-9)

    picked = {}
    for name, axis in basis.items():
        if name in ("up", "down"):
            # Indoor captures barely vary in height; direction is the signal.
            score = fwd_dir @ axis
        else:
            score = 0.5 * (pos_dir @ axis) + 0.5 * (fwd_dir @ axis)
        picked[name] = int(np.argmax(score))
    return picked


def _azimuth_sources(record, exclude, n_src):
    """Six views evenly spaced in azimuth around the trajectory centroid."""
    views = record["views"]
    c2ws = [v.c2w.double().numpy() for v in views]
    centers = np.stack([c2w[:3, 3] for c2w in c2ws])
    centroid = centers.mean(0)
    mean_f = np.stack([c2w[:3, 2] for c2w in c2ws]).mean(0)
    mean_f /= np.linalg.norm(mean_f)
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(mean_f, world_up)) > 0.95:
        world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(mean_f, world_up)
    right /= np.linalg.norm(right)
    rel = centers - centroid
    az = np.arctan2(rel @ right, rel @ mean_f)
    order = np.argsort(az)
    picked, step = [], 0
    while len(picked) < n_src:
        idx = int(order[(step * len(order)) // n_src % len(order)])
        if idx not in picked and idx not in exclude:
            picked.append(idx)
        step += 1
        if step > 10 * n_src:                     # degenerate fallback
            picked = list(range(n_src))
            break
    return picked


def _tilt_views(record, tilt_degrees, tilt_height=None):
    """Virtual forward-tilted poses anchored at the record's front capture.

    The heading is taken from the captured view that best matches the canonical
    front direction (see ``_directional_targets``); image-up tracks that
    heading at every pitch, so pitch = 90 deg is a heading-aligned BEV.
    """
    front = record["views"][_directional_targets(record)["front"]]
    c2w = front.c2w.double().numpy()
    eye0, fwd = c2w[:3, 3], c2w[:3, 2]
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(fwd, world_up)) > 0.95:
        world_up = np.array([0.0, 1.0, 0.0])
    heading = fwd - np.dot(fwd, world_up) * world_up      # horizontalise
    norm = np.linalg.norm(heading)
    heading = c2w[:3, 0] if norm < 1e-6 else heading / norm
    if tilt_height is None:
        # scale-aware default: median camera distance to the capture centroid
        centers = np.stack(
            [v.c2w.double().numpy()[:3, 3] for v in record["views"]])
        tilt_height = float(
            np.median(np.linalg.norm(centers - centers.mean(0), axis=1)))
    eye = eye0 + world_up * tilt_height
    views = {}
    for deg in tilt_degrees:
        pitch = math.radians(deg)
        z = math.cos(pitch) * heading - math.sin(pitch) * world_up
        y = -(math.sin(pitch) * heading + math.cos(pitch) * world_up)
        x = np.cross(y, z)
        tilt_c2w = np.eye(4)
        tilt_c2w[:3, 0], tilt_c2w[:3, 1], tilt_c2w[:3, 2], tilt_c2w[:3, 3] = x, y, z, eye
        name = "bev" if deg == 90 else f"tilt{deg:g}"
        views[name] = VirtualView(name, None, torch.from_numpy(tilt_c2w).float())
    return views


def _decode_rgb(model, lat, vi, device):
    v = lat[:, :, vi:vi + 1].to(device).to(model.pipe.torch_dtype)
    rgb = model.pipe.vae.decode(v, device=device)[0, :, 0].float()
    return (rgb.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)


def _decode_disp(model, lat, vi, device):
    import matplotlib.cm as cm
    v = lat[:, :, vi:vi + 1].to(device).to(model.pipe.torch_dtype)
    raw = model.pipe.vae.decode(v, device=device)[0, :, 0].float().mean(0).cpu().numpy()
    return raw, (cm.plasma(np.clip(raw, 0, 1))[:, :, :3] * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--manifest_path", type=str, required=True)
    parser.add_argument("--split_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_inference_steps", type=str, default="25")
    parser.add_argument("--eval_flow_shift", type=str, default="0.5")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tilt_degrees", type=str, default="30,60,90",
                        help="comma-separated downward pitch angles rendered "
                             "beyond the six captured directions; 90 = BEV")
    parser.add_argument("--tilt_height", type=float, default=None,
                        help="virtual eye height above the front capture "
                             "(world units); default = median source radius")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    set_seed(args.seed)
    accelerator = Accelerator(mixed_precision=cfg.mixed_precision)
    flow_matching = bool(cfg.get("rgb_flow_matching", True))
    step_grid = _num_list(args.num_inference_steps, int)
    shift = float(args.eval_flow_shift)
    norm_mode = cfg.get("disparity_norm_mode", "shared_scale")
    headroom = cfg.get("disparity_norm_headroom", 1.5)
    plucker_resolution = cfg.get("view_condition_plucker_resolution", "latent")
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- model: identical construction to eval_rgbd_nvs.main() -------------
    model = WanRGBDNVSTrainingModule(
        args=cfg, accelerator=accelerator,
        model_id_with_origin_paths=cfg.model_id_with_origin_paths,
        trainable_models=cfg.trainable_models,
        lora_rank=cfg.lora_rank, lora_base_model=cfg.lora_base_model,
        lora_target_modules=cfg.lora_target_modules,
        use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False,
        num_source_views=cfg.num_source_views, num_target_views=cfg.num_target_views,
        vae_encoding_mode=cfg.vae_encoding_mode, init_dvd_ckpt=cfg.init_dvd_ckpt,
        rgb_loss_weight=cfg.rgb_loss_weight, depth_loss_weight=cfg.depth_loss_weight,
        view_condition_plucker_resolution=plucker_resolution,
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
        warp_perturb_prob=0.0,
        warp_perturb_gain_log_std=0.34, warp_perturb_shift_std=0.059,
        warp_perturb_blur_sigma=2.0, warp_perturb_lowfreq_std=0.05,
        warp_perturb_lowfreq_cells=6, warp_perturb_fill_invalid=True,
        warp_perturb_fill_scale=1.0,
        depth_warp_block=cfg.get("depth_warp_block", False),
        per_frame_timestep=cfg.get("per_frame_timestep", False),
        clean_source_depth=cfg.get("clean_source_depth", False),
        clean_source_depth_prob=cfg.get("clean_source_depth_prob", 0.5),
        shared_rope_frame_ids=cfg.get("shared_rope_frame_ids", False),
        rope_frame_axis=cfg.get("rope_frame_axis", None),
        rope_pose_translation_scale=cfg.get("rope_pose_translation_scale", None),
        roce=cfg.get("roce", "off"),
        camera_geometry_fp32=cfg.get("camera_geometry_fp32", False),
        rgb_flow_matching=flow_matching,
        rgb_flow_shift=cfg.get("rgb_flow_shift", 5.0),
        depth_flow_matching=cfg.get("depth_flow_matching", False),
        depth_head_init=cfg.get("depth_head_init", "pretrained"),
        eval_num_inference_steps=step_grid[0],
        eval_flow_shift=shift,
        eval_sample_seed=cfg.get("eval_sample_seed", 1234),
        eval_lpips_net=cfg.get("eval_lpips_net", "alex"),
        eval_depth_vae_ceiling=False,
        eval_vis_downscale=cfg.get("eval_vis_downscale", 2),
        eval_vis_format=cfg.get("eval_vis_format", "jpg"),
        eval_vis_depth_norm=cfg.get("eval_vis_depth_norm", "shared"),
    )
    weights = os.path.join(args.checkpoint, "model.safetensors")
    state = load_file(weights, device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    accelerator.print(f"Loaded {weights}: {len(state)} tensors, "
                      f"missing={len(missing)}, unexpected={len(unexpected)}")
    assert not missing and not unexpected, "checkpoint/model structure mismatch"
    model.to(accelerator.device)
    model.eval()

    # ---- dataset: one scene, deterministic -------------------------------
    dataset = Kubric4DRGBDNVSDataset(
        kubric4d_root=cfg.kubric4d_root,
        manifest_path=args.manifest_path,
        resolution=tuple(cfg.resolution),
        num_source_views=cfg.num_source_views,
        num_target_views=cfg.num_target_views,
        random_permute_source_views=cfg.random_permute_source_views,
        view_order=cfg.get("view_order", "random"),
        disparity_norm_mode=norm_mode,
        disparity_norm_headroom=headroom,
        deterministic_view_seed=int(cfg.get("eval_view_seed", 1234)),
        min_view_valid_fraction=cfg.get("min_view_valid_fraction", 0.0),
        view_validity_path=cfg.get("view_validity_path", None),
        samples_per_scene=cfg.get("samples_per_scene", 1),
        split_path=args.split_path,
        split="val",
    )
    assert len(dataset.records) >= 1, "empty split"
    record = dataset.records[0]
    picked = _directional_targets(record)
    accelerator.print("directional targets: " + "  ".join(
        f"{a}={record['views'][i].view_id}" for a, i in picked.items()))

    batches = []
    for chunk in (["front", "left", "right"], ["back", "up", "down"]):
        tgt_idx = [picked[a] for a in chunk]
        src_idx = _azimuth_sources(record, set(tgt_idx), cfg.num_source_views)
        sources = [record["views"][i] for i in src_idx]
        targets = [record["views"][i] for i in tgt_idx]
        dataset._select_views = lambda rec, idx=0, s=sources, t=targets: (list(s), list(t))
        loader = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=0, collate_fn=custom_collate_fn,
                            drop_last=False)
        batches.append((chunk, next(iter(loader))))

    # ---- sample -----------------------------------------------------------
    model.eval_flow_shift = shift
    model.eval_num_inference_steps = step_grid[0]
    cell_dir = os.path.join(args.output_dir, f"shift{shift:g}_steps{step_grid[0]}")
    os.makedirs(cell_dir, exist_ok=True)
    for gi, (chunk, batch) in enumerate(batches):
        m = model.evaluate(batch, output_dir=cell_dir, global_step=gi,
                           max_samples=1, save_vis=True)
        accelerator.print(f"[{'+'.join(chunk)}] " +
                          "  ".join(f"{k}={v:.4f}" for k, v in m.items()))

    # ---- virtual forward-tilted targets (pitch 30/60/90; 90 = BEV) ---------
    # Anchored at the captured front view, image-up tracking its heading.
    # These poses have no captured GT -> virtual-view sampling, no metrics.
    tilt_degrees = [float(d) for d in str(args.tilt_degrees).split(",") if d.strip()]
    if tilt_degrees:
        device = accelerator.device
        _patch_load_view(dataset, int(cfg.resolution[0]), int(cfg.resolution[1]))
        tilt_views = _tilt_views(record, tilt_degrees, args.tilt_height)
        src_views = [record["views"][i]
                     for i in _azimuth_sources(record, set(), cfg.num_source_views)]
        tilt_k = torch.stack(
            [dataset._load_view(v)["K"] for v in src_views]).median(0).values
        for vv in tilt_views.values():
            vv.K = tilt_k.clone()
        accelerator.print("tilt views: " + "  ".join(
            f"{n}(eye={np.round(v.c2w.numpy()[:3, 3], 2).tolist()})"
            for n, v in tilt_views.items()))

        tilt_dir = os.path.join(cell_dir, "tilt")
        os.makedirs(tilt_dir, exist_ok=True)
        names = list(tilt_views)
        for start in range(0, len(names), cfg.num_target_views):
            chunk = names[start:start + cfg.num_target_views]
            targets = [tilt_views[n] for n in chunk]
            dataset._select_views = (
                lambda rec, idx=0, s=src_views, t=targets: (list(s), list(t)))
            batch = custom_collate_fn([dataset[0]])
            with torch.autocast(device_type="npu", dtype=model.pipe.torch_dtype,
                                enabled=True):
                model._eval_inputs = True
                try:
                    inputs = model.forward_preprocess(batch)
                finally:
                    model._eval_inputs = False
                gen = torch.Generator(device=model.pipe.device)
                gen.manual_seed(int(cfg.get("eval_sample_seed", 1234)))
                pred_depth, pred_rgb = model.sample_rgb(
                    inputs, num_steps=step_grid[0], generator=gen)

            if model.latent_scale != 1.0:
                pred_rgb = pred_rgb / model.latent_scale
            if model.depth_latent_scale != 1.0:
                pred_depth = pred_depth / model.depth_latent_scale
            rgb_gt = inputs["rgb_gt"]
            if model.latent_scale != 1.0:
                rgb_gt = rgb_gt / model.latent_scale

            for i, name in enumerate(chunk):
                vi = model.num_source_views + i
                Image.fromarray(_decode_rgb(model, pred_rgb, vi, device)).save(
                    os.path.join(tilt_dir, f"pred_rgb_{name}.jpg"), quality=95)
                raw, vis = _decode_disp(model, pred_depth, vi, device)
                Image.fromarray(vis).save(
                    os.path.join(tilt_dir, f"pred_disp_{name}.jpg"), quality=95)
                np.save(os.path.join(tilt_dir, f"pred_disp_raw_{name}.npy"), raw)
            if start == 0:
                for si in range(model.num_source_views):
                    rgb = _decode_rgb(model, rgb_gt, si, device)
                    Image.fromarray(rgb).save(
                        os.path.join(tilt_dir, f"src_rgb_{si + 1}.jpg"), quality=95)


if __name__ == "__main__":
    main()
