"""Directional NVS inference: render front/back/left/right/up/down target views.

Like eval_rgbd_nvs, but the 3 target views per batch are not drawn randomly --
they are the captured views that best match each of six canonical directions
(front/back/left/right/up/down) around the scene's trajectory centroid, so one
run produces an interpretable surround set instead of a random draw.  Sources
are six views evenly spaced in azimuth.  Target views are REAL captures, so GT
RGB/depth metrics and the standard eval grid images still work.

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
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from examples.dataset import Kubric4DRGBDNVSDataset
from examples.wanvideo.model_training.eval_rgbd_nvs import _num_list
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


if __name__ == "__main__":
    main()
