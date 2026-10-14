"""Inference at VIRTUAL canonical views (front/back/left/right/up/down).

Unlike eval_rgbd_nvs(_orbit) the six target poses here have NO captured GT:
each camera is placed at centroid + axis * R (R = median camera radius) and
look-at the trajectory centroid, with K = median source K at the eval
resolution.  The batch is assembled through the normal dataset path (so the
shared-scale disparity normalisation applies to the virtual c2w translations
exactly as to real ones) with zero-filled GT for the virtual targets, then the
script drives forward_preprocess + sample_rgb directly and decodes the
predictions.  No metrics are computed -- there is nothing to score against.

Output layout (one image per scene, plus per-view dumps):
  row 1: the 6 source views (RGB, as fed to the model)
  row 2: predicted RGB at the six virtual views
  row 3: predicted depth (shared-scale disparity, plasma) at the same views

Run from the repo root, same env as eval_rgbd_nvs:
  python -m examples.wanvideo.model_training.infer_virtual_views \
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
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file

from examples.dataset import Kubric4DRGBDNVSDataset
from examples.wanvideo.model_training.eval_rgbd_nvs import _num_list
from examples.wanvideo.model_training.train_with_accelerate_rgbd_nvs import custom_collate_fn
from examples.wanvideo.model_training.WanRGBDNVSTrainingModule import (
    WanRGBDNVSTrainingModule,
)

AXES = ["front", "back", "left", "right", "up", "down"]


class VirtualView:
    """Stand-in for dataset._ViewRecord at a synthesised pose (no GT)."""

    virtual = True

    def __init__(self, view_id, K, c2w):
        self.view_id = view_id
        self.K = K
        self.c2w = c2w
        self.rgb_path = None
        self.depth_path = None
        self.depth_scale = 1.0


def _trajectory_basis(views):
    centers = np.stack([v.c2w.double().numpy()[:3, 3] for v in views])
    forwards = np.stack([v.c2w.double().numpy()[:3, 2] for v in views])
    centroid = centers.mean(0)
    mean_f = forwards.mean(0)
    mean_f /= np.linalg.norm(mean_f)
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(mean_f, world_up)) > 0.95:
        world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(mean_f, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, mean_f)
    radius = float(np.median(np.linalg.norm(centers - centroid, axis=1)))
    return centroid, mean_f, right, up, radius


def _look_at(eye, target, down_ref):
    """OpenCV c2w (x right, y down, z forward), looking eye -> target."""
    z = target - eye
    z = z / np.linalg.norm(z)
    x = np.cross(down_ref, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, eye
    return c2w


def _virtual_poses(record_views, num_source_views, virtual_radius=None):
    """Six canonical poses + the source views used as conditioning."""
    real = [v for v in record_views if not getattr(v, "virtual", False)]
    centroid, mean_f, right, up, radius = _trajectory_basis(real)
    if virtual_radius is not None:
        # Waymo-style rigs cluster all cameras within centimetres, so the
        # median source radius would put the virtual eyes inside the car roof.
        # Override with a scene-scale distance (normalised by the same shared
        # disparity scale as everything else).
        radius = float(virtual_radius)
    down = -up
    axes = {
        "front": mean_f, "back": -mean_f,
        "right": right, "left": -right,
        "up": up, "down": down,
    }
    poses = {}
    for name, axis in axes.items():
        eye = centroid + axis * radius
        if name in ("up", "down"):
            # look-at reference direction is degenerate for vertical cameras;
            # roll the image so the trajectory's mean forward is "screen up".
            down_ref = -mean_f
        else:
            down_ref = np.array([0.0, 0.0, -1.0])   # OpenCV y points down
        poses[name] = _look_at(eye, centroid, down_ref)
    return poses, centroid


def _azimuth_sources(record_views, exclude, n_src):
    import numpy as _np
    real = [v for v in record_views if not getattr(v, "virtual", False)]
    centroid, mean_f, right, up, radius = _trajectory_basis(real)
    centers = _np.stack([v.c2w.double().numpy()[:3, 3] for v in real])
    rel = centers - centroid
    az = _np.arctan2(rel @ right, rel @ mean_f)
    order = _np.argsort(az)
    picked, step = [], 0
    while len(picked) < n_src and step <= 10 * n_src:
        idx = int(order[(step * len(order)) // n_src % len(order)])
        if idx not in picked and idx not in exclude:
            picked.append(idx)
        step += 1
    return [real[i] for i in picked[:n_src]]


def _patch_load_view(dataset, H, W):
    """Handle VirtualView items: zero GT, K already at eval resolution."""
    original = dataset._load_view

    def _load_view(view):
        if not getattr(view, "virtual", False):
            return original(view)
        zero = torch.zeros(3, H, W)
        return {
            "image": zero,
            "inv_depth": torch.zeros(H, W),
            "positive": torch.zeros(H, W, dtype=torch.bool),
            "depth_metric": zero.clone(),
            "K": view.K.clone().float(),
            "c2w": view.c2w.clone().float(),
            "valid_mask": torch.zeros(1, 1, H, W),
            "view_id": view.view_id,
        }

    dataset._load_view = _load_view


def _compose_grid(src_rgbs, tgt_rgbs, tgt_depths, labels):
    """3 rows x len(labels) cols with a top label strip; no GT anywhere."""
    H, W = src_rgbs[0].shape[:2]
    sep, pad = 4, 34
    rows = [
        np.concatenate([np.concatenate([c, np.ones((H, sep, 3), np.uint8) * 200], 1)
                        for c in row], 1)[:, :-sep]
        for row in (src_rgbs, tgt_rgbs, tgt_depths)
    ]
    col_sep = np.ones((sep, rows[0].shape[1], 3), np.uint8) * 200
    grid = np.concatenate([rows[0], col_sep, rows[1], col_sep, rows[2]], 0)
    img = Image.fromarray(grid)
    img = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)
    scale = img.width / grid.shape[1]

    out = Image.new("RGB", (img.width + 70, img.height + pad + 10), (255, 255, 255))
    out.paste(img, (70, pad))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    tile_w = (W + sep) * scale
    for c, lab in enumerate(labels):
        x = 70 + c * tile_w + tile_w / 2
        draw.text((x - 30, 8), lab[:9], fill=(0, 0, 0), font=font)
    for r, lab in enumerate(["source", "pred RGB", "pred depth"]):
        draw.text((6, pad + r * ((img.height + sep * scale) / 3) + 40), lab, fill=(0, 0, 0), font=font)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--manifest_path", type=str, required=True)
    parser.add_argument("--split_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_inference_steps", type=str, default="25")
    parser.add_argument("--eval_flow_shift", type=str, default="0.5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--virtual_radius", type=float, default=None,
                        help="override the virtual-view distance from the "
                             "trajectory centroid (world units); default is "
                             "the median source-camera radius")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    set_seed(args.seed)
    accelerator = Accelerator(mixed_precision=cfg.mixed_precision)
    num_steps = _num_list(args.num_inference_steps, int)[0]
    shift = float(args.eval_flow_shift)
    H, W = int(cfg.resolution[0]), int(cfg.resolution[1])
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- model, identical construction to eval_rgbd_nvs --------------------
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
        rgb_flow_matching=bool(cfg.get("rgb_flow_matching", True)),
        rgb_flow_shift=cfg.get("rgb_flow_shift", 5.0),
        depth_flow_matching=cfg.get("depth_flow_matching", False),
        depth_head_init=cfg.get("depth_head_init", "pretrained"),
        eval_num_inference_steps=num_steps,
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

    # ---- dataset (single scene) with virtual-view support ------------------
    dataset = Kubric4DRGBDNVSDataset(
        kubric4d_root=cfg.kubric4d_root,
        manifest_path=args.manifest_path,
        resolution=tuple(cfg.resolution),
        num_source_views=cfg.num_source_views,
        num_target_views=cfg.num_target_views,
        random_permute_source_views=cfg.random_permute_source_views,
        view_order=cfg.get("view_order", "random"),
        disparity_norm_mode=cfg.get("disparity_norm_mode", "shared_scale"),
        disparity_norm_headroom=cfg.get("disparity_norm_headroom", 1.5),
        deterministic_view_seed=int(cfg.get("eval_view_seed", 1234)),
        min_view_valid_fraction=cfg.get("min_view_valid_fraction", 0.0),
        view_validity_path=cfg.get("view_validity_path", None),
        samples_per_scene=cfg.get("samples_per_scene", 1),
        split_path=args.split_path,
        split="val",
    )
    assert len(dataset.records) >= 1, "empty split"
    record = dataset.records[0]
    _patch_load_view(dataset, H, W)

    # Virtual poses in ORIGINAL world scale; __getitem__ rescales c2w
    # translations by the shared disparity scale for every view alike.
    poses, _ = _virtual_poses(record["views"], cfg.num_source_views,
                              virtual_radius=args.virtual_radius)
    src_views = _azimuth_sources(record["views"], set(), cfg.num_source_views)

    # Virtual K = median of the (post-resize) source K.
    src_k = torch.stack([dataset._load_view(v)["K"] for v in src_views])
    virtual_k = src_k.median(0).values

    accelerator.print("virtual views: " + "  ".join(
        f"{a}: eye={np.round(p[:3, 3], 2)}" for a, p in poses.items()))

    device = accelerator.device
    model.eval_flow_shift = shift
    model.eval_num_inference_steps = num_steps

    pred_rgbs, pred_disps, src_rgbs = {}, {}, None
    for chunk in (["front", "back", "up"], ["left", "right", "down"]):
        targets = [VirtualView(a, virtual_k, torch.from_numpy(poses[a]).float())
                   for a in chunk]
        dataset._select_views = lambda rec, idx=0, s=src_views, t=targets: (list(s), list(t))
        batch = custom_collate_fn([dataset[0]])

        with torch.autocast(device_type="npu", dtype=model.pipe.torch_dtype, enabled=True):
            model._eval_inputs = True
            try:
                inputs = model.forward_preprocess(batch)
            finally:
                model._eval_inputs = False
            gen = torch.Generator(device=model.pipe.device)
            gen.manual_seed(int(cfg.get("eval_sample_seed", 1234)))
            pred_depth, pred_rgb = model.sample_rgb(
                inputs, num_steps=num_steps, generator=gen
            )

        rgb_gt = inputs["rgb_gt"]
        if model.latent_scale != 1.0:
            rgb_gt = rgb_gt / model.latent_scale
            pred_rgb = pred_rgb / model.latent_scale
        if model.depth_latent_scale != 1.0:
            pred_depth = pred_depth / model.depth_latent_scale

        def _decode_rgb(lat, vi):
            v = lat[:, :, vi:vi + 1].to(device).to(model.pipe.torch_dtype)
            return (model.pipe.vae.decode(v, device=device)[0, :, 0].float()
                    .clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

        def _decode_disp(lat, vi):
            import matplotlib.cm as cm
            v = lat[:, :, vi:vi + 1].to(device).to(model.pipe.torch_dtype)
            raw = model.pipe.vae.decode(v, device=device)[0, :, 0].float().mean(0).cpu().numpy()
            return raw, (cm.plasma(np.clip(raw, 0, 1))[:, :, :3] * 255).astype(np.uint8)

        if src_rgbs is None:
            src_rgbs = [_decode_rgb(rgb_gt, sv) for sv in range(model.num_source_views)]
        for i, name in enumerate(chunk):
            vi = model.num_source_views + i
            pred_rgbs[name] = _decode_rgb(pred_rgb, vi)
            pred_disps[name] = _decode_disp(pred_depth, vi)

    # ---- save --------------------------------------------------------------
    labels = AXES
    grid = _compose_grid(
        src_rgbs,
        [pred_rgbs[a] for a in labels],
        [pred_disps[a][1] for a in labels],
        labels,
    )
    grid.save(os.path.join(args.output_dir, "virtual_views_grid.jpg"), quality=92)
    for i in range(len(src_rgbs)):
        Image.fromarray(src_rgbs[i]).save(
            os.path.join(args.output_dir, f"src_rgb_{i + 1}.jpg"), quality=95)
    for a in labels:
        Image.fromarray(pred_rgbs[a]).save(
            os.path.join(args.output_dir, f"pred_rgb_{a}.jpg"), quality=95)
        np.save(os.path.join(args.output_dir, f"pred_disp_raw_{a}.npy"), pred_disps[a][0])
    # Disparity -> metric depth needs the shared scale; recompute it from the
    # source disparity the same way the dataset does and store it alongside.
    accelerator.print(f"saved grid + per-view dumps -> {args.output_dir}")


if __name__ == "__main__":
    main()
