"""Single-image NVS demo over re10k: 1 source view -> 3 virtual views/scene.

For each of --num_scenes test scenes pick ONE capture (the anchor), clone it
into the 6 source slots with centimetre jitter (keeps the 6-source layout the
model was trained with while all information still comes from that one image),
then render three virtual targets:

  forward  translate along the anchor heading (0.35 x median anchor depth)
  bev      straight DOWN from above the anchor (0.9 x median anchor depth),
           image-top tracking the heading
  yaw+90   pure look-rotate at the anchor eye

World-up is derived FROM THE CAPTURES (up = -mean camera y-column): re10k c2w
is y-down, so the earlier z-up tilt views placed the "bev" camera under the
floor looking up -- this script fixes that.

Outputs per scene: src/pred rgb + disparity.  Merged deliverables: one grid
sheet per view type (all scenes), a source grid, and a combined
[src|forward|bev|yaw] preview sheet.

Run from the repo root:
  python -m tools.infer_re10k_1src_3views \
      --config <args.yaml> --checkpoint <dir> --manifest_path <jsonl> \
      --output_dir <dir>
"""
import argparse
import dataclasses
import math
import os

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from examples.dataset import Kubric4DRGBDNVSDataset
from examples.dataset.kubric4d_rgbd_nvs_dataset import _ViewRecord
from examples.wanvideo.model_training.infer_virtual_views import (
    VirtualView,
    _patch_load_view,
)
from examples.wanvideo.model_training.train_with_accelerate_rgbd_nvs import custom_collate_fn
from examples.wanvideo.model_training.WanRGBDNVSTrainingModule import (
    WanRGBDNVSTrainingModule,
)

VIEW_NAMES = ("forward", "bev", "yaw+90")


def _unit(v):
    return v / max(np.linalg.norm(v), 1e-9)


def _scene_up(views):
    """Gravity up from the captures themselves (re10k y-col points DOWN)."""
    downs = np.stack([v.c2w.double().numpy()[:3, 1] for v in views])
    return _unit(-downs.mean(0))


def _rot_from_zy(z, up):
    """OpenCV c2w rotation with optical axis z and image-up as close to `up`."""
    z = _unit(z)
    y = -_unit(up)                      # camera y is the image-DOWN direction
    x = np.cross(y, z)
    x = _unit(x)
    y = np.cross(z, x)                  # re-orthogonalise
    r = np.eye(4)
    r[:3, 0], r[:3, 1], r[:3, 2] = x, y, z
    return r


def _pose_at(eye, z, up):
    p = _rot_from_zy(z, up)
    p[:3, 3] = eye
    return p


def _virtual_targets(anchor_c2w, up, med_depth):
    eye0 = anchor_c2w[:3, 3]
    fwd = anchor_c2w[:3, 2]
    heading = fwd - np.dot(fwd, up) * up
    if np.linalg.norm(heading) < 1e-6:
        heading = np.cross(up, anchor_c2w[:3, 0])
    heading = _unit(heading)

    d_fwd = float(np.clip(0.35 * med_depth, 0.15, 1.2))
    h_bev = float(np.clip(0.90 * med_depth, 1.0, 3.0))

    poses = {
        "forward": _pose_at(eye0 + heading * d_fwd, heading, up),
        # straight down (z = -up), image top along the heading
        "bev": _pose_at(eye0 + up * h_bev, -up, heading),
        # pure look-rotate: +90 deg about up (left turn), eye unchanged
        "yaw+90": _pose_at(eye0, np.cross(up, heading), up),
    }
    return poses


def _jitter_sources(anchor, up, right, rng):
    """6 copies of the single anchor view with centimetre pose jitter."""
    offsets = [(0.0, 0.0), (0.04, 0.0), (-0.04, 0.0),
               (0.0, 0.03), (0.0, -0.03), (0.03, 0.03)]
    srcs = []
    for dr, du in offsets:
        c2w = anchor.c2w.clone()
        c2w[:3, 3] = c2w[:3, 3] + right * dr + up * du
        srcs.append(dataclasses.replace(anchor, c2w=c2w))
    return srcs


def _pick_anchor(dataset, views):
    """Middle capture with usable depth; walk outward if the depth is bad."""
    n = len(views)
    order = sorted(range(n), key=lambda i: abs(i - n // 2))
    for i in order:
        item = dataset._load_view(views[i])
        valid = item["valid_mask"][0, 0] > 0.5
        if float(valid.float().mean()) < 0.85:
            continue
        depth = item["depth_metric"][0]
        med = float(depth[valid].median()) if bool(valid.any()) else 0.0
        if not math.isfinite(med) or med <= 0:
            continue
        return views[i], item, med
    raise RuntimeError("no capture with usable depth")


def _decode_rgb(model, lat, vi, device):
    v = lat[:, :, vi:vi + 1].to(device).to(model.pipe.torch_dtype)
    rgb = model.pipe.vae.decode(v, device=device)[0, :, 0].float()
    return (rgb.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)


def _decode_disp(model, lat, vi, device):
    import matplotlib.cm as cm
    v = lat[:, :, vi:vi + 1].to(device).to(model.pipe.torch_dtype)
    raw = model.pipe.vae.decode(v, device=device)[0, :, 0].float().mean(0).cpu().numpy()
    return raw, (cm.plasma(np.clip(raw, 0, 1))[:, :, :3] * 255).astype(np.uint8)


def _label(tile, text):
    img = Image.fromarray(tile)
    draw = ImageDraw.Draw(img)
    draw.text((6, 4), text, fill=(255, 255, 0))
    return np.asarray(img)


def _grid(tiles, cols, tile_h, tile_w, path, title):
    rows = math.ceil(len(tiles) / cols)
    sep = 4
    pad = 24 if title else 0
    sheet = np.ones(((tile_h + sep) * rows + pad, (tile_w + sep) * cols, 3),
                    np.uint8) * 255
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[pad + r * (tile_h + sep):pad + r * (tile_h + sep) + tile_h,
              c * (tile_w + sep):c * (tile_w + sep) + tile_w] = t
    img = Image.fromarray(sheet)
    if title:
        draw = ImageDraw.Draw(img)
        draw.text((8, 6), title, fill=(0, 0, 0))
    img.save(path, quality=90)
    print(f"[grid] {path} ({len(tiles)} tiles)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--manifest_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_scenes", type=int, default=50)
    parser.add_argument("--min_views", type=int, default=24)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--eval_flow_shift", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    set_seed(args.seed)
    accelerator = Accelerator(mixed_precision=cfg.mixed_precision)
    shift = float(args.eval_flow_shift)
    num_steps = int(args.num_inference_steps)
    H, W = int(cfg.resolution[0]), int(cfg.resolution[1])
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- model: identical construction to eval_rgbd_nvs --------------------
    model = WanRGBDNVSTrainingModule(
        args=cfg, accelerator=accelerator,
        model_id_with_origin_paths=cfg.model_id_with_origin_paths,
        trainable_models=cfg.trainable_models,
        lora_rank=cfg.lora_rank, lora_base_model=cfg.lora_base_model,
        lora_target_modules=cfg.lora_target_modules,
        use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False,
        local_model_path=cfg.get("wan_model_root", "./models"),
        skip_download=cfg.get("skip_download", False),
        strip_i2v_image_input=cfg.get("strip_i2v_image_input", False),
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

    # ---- dataset: full manifest, we pick scenes/anchors ourselves ----------
    dataset = Kubric4DRGBDNVSDataset(
        kubric4d_root=cfg.kubric4d_root,
        manifest_path=args.manifest_path,
        resolution=tuple(cfg.resolution),
        num_source_views=cfg.num_source_views,
        num_target_views=cfg.num_target_views,
        random_permute_source_views=False,
        view_order="video",
        disparity_norm_mode=cfg.get("disparity_norm_mode", "shared_scale"),
        disparity_norm_headroom=cfg.get("disparity_norm_headroom", 1.5),
        deterministic_view_seed=int(cfg.get("eval_view_seed", 1234)),
        min_view_valid_fraction=cfg.get("min_view_valid_fraction", 0.0),
        view_validity_path=cfg.get("view_validity_path", None),
        samples_per_scene=1,
        split_path=None,
        split=None,
    )
    _patch_load_view(dataset, H, W)

    rng = np.random.default_rng(args.seed)
    pool = [r for r in dataset.records if len(r["views"]) >= args.min_views]
    picked = rng.choice(len(pool), size=min(args.num_scenes, len(pool)), replace=False)
    accelerator.print(f"scenes available: {len(pool)}; running on {len(picked)}")

    device = accelerator.device
    model.eval_flow_shift = shift
    model.eval_num_inference_steps = num_steps
    cell_dir = os.path.join(args.output_dir, f"shift{shift:g}_steps{num_steps}")
    os.makedirs(cell_dir, exist_ok=True)

    ns = model.num_source_views
    src_tiles, view_tiles = [], {n: [] for n in VIEW_NAMES}
    disp_tiles = {n: [] for n in VIEW_NAMES}

    for si, pi in enumerate(picked):
        record = pool[int(pi)]
        views = record["views"]
        anchor, anchor_item, med_depth = _pick_anchor(dataset, views)

        up = _scene_up(views)
        a_np = anchor.c2w.double().numpy()
        fwd = a_np[:3, 2]
        heading = _unit(fwd - np.dot(fwd, up) * up)
        right = np.cross(heading, up)

        poses = _virtual_targets(a_np, up, med_depth)
        tgt_k = anchor_item["K"]
        targets = [VirtualView(n, tgt_k.clone(), torch.from_numpy(p).float())
                   for n, p in poses.items()]
        src_views = _jitter_sources(anchor, up, right, rng)

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
                inputs, num_steps=num_steps, generator=gen)

        if model.latent_scale != 1.0:
            pred_rgb = pred_rgb / model.latent_scale
        if model.depth_latent_scale != 1.0:
            pred_depth = pred_depth / model.depth_latent_scale
        rgb_gt = inputs["rgb_gt"]
        if model.latent_scale != 1.0:
            rgb_gt = rgb_gt / model.latent_scale

        scene_dir = os.path.join(cell_dir, f"{si:03d}_{record['scene_id'][:12]}")
        os.makedirs(scene_dir, exist_ok=True)
        tag = f"{si:03d} {record['scene_id'][:8]} a={anchor.view_id}"

        src_rgb = _decode_rgb(model, rgb_gt, 0, device)
        Image.fromarray(src_rgb).save(os.path.join(scene_dir, "src_rgb.jpg"), quality=95)
        src_tiles.append(_label(src_rgb, tag))

        row = [src_rgb]
        for vi, name in enumerate(VIEW_NAMES):
            rgb = _decode_rgb(model, pred_rgb, ns + vi, device)
            raw, vis = _decode_disp(model, pred_depth, ns + vi, device)
            Image.fromarray(rgb).save(
                os.path.join(scene_dir, f"pred_rgb_{name}.jpg"), quality=95)
            Image.fromarray(vis).save(
                os.path.join(scene_dir, f"pred_disp_{name}.jpg"), quality=95)
            np.save(os.path.join(scene_dir, f"pred_disp_raw_{name}.npy"), raw)
            view_tiles[name].append(_label(rgb, tag))
            disp_tiles[name].append(vis)
            row.append(rgb)
        _grid(row, len(row), H, W,
              os.path.join(scene_dir, "row_src_fwd_bev_yaw.jpg"), "")

        accelerator.print(
            f"[{si + 1}/{len(picked)}] {record['scene_id']} "
            f"anchor={anchor.view_id} med_d={med_depth:.2f} "
            f"up={np.round(up, 2).tolist()} done", flush=True)
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    # ---- merged deliverables: one sheet per view ---------------------------
    _grid(src_tiles, 10, H, W, os.path.join(cell_dir, "grid_source.jpg"),
          "source (1 capture / scene)")
    for name in VIEW_NAMES:
        _grid(view_tiles[name], 10, H, W,
              os.path.join(cell_dir, f"grid_{name.replace('+', 'p')}.jpg"),
              f"pred RGB - {name}")
        _grid(disp_tiles[name], 10, H, W,
              os.path.join(cell_dir, f"grid_{name.replace('+', 'p')}_disp.jpg"),
              f"pred disparity - {name}")

    # combined [src|forward|bev|yaw] preview, first 10 scenes
    preview = [np.concatenate([src_tiles[i], view_tiles["forward"][i],
                               view_tiles["bev"][i], view_tiles["yaw+90"][i]],
                              axis=1)
               for i in range(min(10, len(src_tiles)))]
    if preview:
        _grid(preview, 1, H, 4 * W,
              os.path.join(cell_dir, "preview_rows_src_fwd_bev_yaw.jpg"),
              "rows: src | forward | bev | yaw+90 (first 10 scenes)")
    accelerator.print("all done")


if __name__ == "__main__":
    main()
