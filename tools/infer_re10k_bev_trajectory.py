"""re10k BEV-trajectory inference demo.

Renders a sequence of BEV (straight-down) predictions along the camera
trajectory of one scene: 9 virtual top-down cameras -- one above each of the 6
consecutive source frames plus 3 constant-velocity extrapolated waypoints
beyond them -- all pitched 90 deg down with image-up tracking the local
heading (the `_tilt_views` deg=90 recipe, per waypoint).  Sources are the
same 6 consecutive frames as tools/infer_re10k_consecutive.

Outputs (under --output_dir):
  bev_traj/pred_rgb_trajKK.jpg + pred_disp_trajKK.jpg + raw npy
  bev_trajectory.mp4   (PRED RGB | PRED depth, ordered along the path)
  cameras_bev_traj.html  plotly 3D: src frustums + BEV frustums + path
  cameras_bev_traj.png   matplotlib top-down trajectory plot

Run from the project root:
  .venv/bin/python -u -m tools.infer_re10k_bev_trajectory \
      --config logs/<exp>/args_re10k_infer.yaml \
      --checkpoint logs/<exp>/checkpoint-step-2700 \
      --output_dir logs/<exp>/infer_re10k_bev_traj --record_idx 1
"""
import argparse
import math
import os
import shutil
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from PIL import Image
from safetensors.torch import load_file

from examples.dataset import Kubric4DRGBDNVSDataset
from examples.wanvideo.model_training.infer_virtual_views import (
    VirtualView,
    _patch_load_view,
)
from examples.wanvideo.model_training.train_with_accelerate_rgbd_nvs import custom_collate_fn
from utils.infer_video_demo import build_model


def _ffmpeg_bin():
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _bev_waypoints(src_views, num_extrap, height):
    """BEV poses: straight-down above each source centre + extrapolated ones.

    Returns (c2ws numpy [N,4,4], src_centers [6,3], extrap_centers [M,3]).
    """
    centers = np.stack([v.c2w.double().numpy()[:3, 3] for v in src_views])
    world_up = np.array([0.0, 0.0, 1.0])

    def heading(c2w):
        f = c2w[:3, 2]
        h = f - np.dot(f, world_up) * world_up
        n = np.linalg.norm(h)
        return c2w[:3, 0] if n < 1e-6 else h / n

    heads = [heading(v.c2w.double().numpy()) for v in src_views]

    vel = (centers[-1] - centers[0]) / (len(centers) - 1)
    extrap = [centers[-1] + vel * (k + 1) for k in range(num_extrap)]
    eyes = [c + world_up * height for c in centers] + \
           [c + world_up * height for c in extrap]

    c2ws = []
    for i, eye in enumerate(eyes):
        head = heads[i] if i < len(heads) else heads[-1]
        z = -world_up                                  # optical axis: straight down
        y = -head                                      # image-up tracks heading
        x = np.cross(y, z)
        m = np.eye(4)
        m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, eye
        c2ws.append(m)
    return np.stack(c2ws), centers, np.stack(extrap)


def _decode_rgb(model, lat, vi, device):
    v = lat[:, :, vi:vi + 1].to(device).to(model.pipe.torch_dtype)
    rgb = model.pipe.vae.decode(v, device=device)[0, :, 0].float()
    return (rgb.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)


def _decode_disp(model, lat, vi, device):
    import matplotlib.cm as cm
    v = lat[:, :, vi:vi + 1].to(device).to(model.pipe.torch_dtype)
    raw = model.pipe.vae.decode(v, device=device)[0, :, 0].float().mean(0).cpu().numpy()
    return raw, (cm.plasma(np.clip(raw, 0, 1))[:, :, :3] * 255).astype(np.uint8)


def _plotly_html(path, src_views, src_centers, extrap_centers, bev_c2ws, height):
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=src_centers[:, 0], y=src_centers[:, 1], z=src_centers[:, 2],
        mode="lines+markers", name="src trajectory",
        line=dict(color="#1f77b4", width=5)))

    last = np.vstack([src_centers[-1:], extrap_centers])
    fig.add_trace(go.Scatter3d(
        x=last[:, 0], y=last[:, 1], z=last[:, 2],
        mode="lines+markers", name="extrapolated",
        line=dict(color="#1f77b4", width=3, dash="dash")))

    bev_centers = bev_c2ws[:, :3, 3]
    fig.add_trace(go.Scatter3d(
        x=bev_centers[:, 0], y=bev_centers[:, 1], z=bev_centers[:, 2],
        mode="lines+markers", name="BEV cams (predicted)",
        line=dict(color="#ff7f0e", width=5)))

    def add_frustum(c2w, K, color, name, scale):
        o = c2w[:3, 3]
        R = c2w[:3, :3]
        corners = np.array([
            [-0.5, -0.5, 1.0], [0.5, -0.5, 1.0],
            [0.5, 0.5, 1.0], [-0.5, 0.5, 1.0],
        ])
        dirs = R @ (corners * scale).T
        pts = o[:, None] + dirs
        seg_x, seg_y, seg_z = [], [], []
        for i in range(4):
            j = (i + 1) % 4
            for a, b in ((o, pts[:, i]), (o, pts[:, j]), (pts[:, i], pts[:, j])):
                seg_x += [a[0], b[0], None]
                seg_y += [a[1], b[1], None]
                seg_z += [a[2], b[2], None]
        fig.add_trace(go.Scatter3d(
            x=seg_x, y=seg_y, z=seg_z, mode="lines", name=name,
            line=dict(color=color, width=3), showlegend=True))

    k_unit = np.array([[0.8, 0, 0], [0, 0.8, 0], [0, 0, 1.0]])
    for i, v in enumerate(src_views):
        add_frustum(v.c2w.double().numpy(), k_unit, "#1f77b4",
                    "src" if i == 0 else f"src{i}", scale=float(height))
    for i in range(len(bev_c2ws)):
        add_frustum(bev_c2ws[i], k_unit, "#ff7f0e",
                    "BEV" if i == 0 else f"BEV{i}", scale=float(height) * 0.6)

    all_pts = np.vstack([src_centers, extrap_centers, bev_centers])
    rng = all_pts.max(0) - all_pts.min(0)
    c = (all_pts.max(0) + all_pts.min(0)) / 2
    r = float(rng.max()) / 2 * 1.2
    fig.update_layout(scene=dict(
        xaxis=dict(range=[c[0] - r, c[0] + r], title="X"),
        yaxis=dict(range=[c[1] - r, c[1] + r], title="Y"),
        zaxis=dict(range=[c[2] - r, c[2] + r], title="Z"),
        aspectmode="cube"),
        legend=dict(x=0.85, y=0.05))
    fig.write_html(path, include_plotlyjs=True)


def _plot_png(path, src_centers, extrap_centers, bev_centers, scene_id):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(src_centers[:, 0], src_centers[:, 1], "-o", color="#1f77b4",
            label="src trajectory", lw=2, ms=4)
    last = np.vstack([src_centers[-1:], extrap_centers])
    ax.plot(last[:, 0], last[:, 1], "--o", color="#1f77b4", alpha=0.6,
            label="extrapolated", lw=1.5, ms=4)
    ax.plot(bev_centers[:, 0], bev_centers[:, 1], "s", color="#ff7f0e",
            label="BEV cams (predicted)", ms=8, mfc="none", mew=2)
    for i in range(len(bev_centers)):
        ax.annotate(f"{i}", bev_centers[i, :2], fontsize=8, color="#ff7f0e")
    ax.set_aspect("equal")
    ax.set_title(f"BEV trajectory  scene={scene_id} (top-down X-Y)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--manifest_path", type=str, default=None,
                        help="override cfg manifest (default: cfg value)")
    parser.add_argument("--record_idx", type=int, default=1)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--num_extrap", type=int, default=3,
                        help="BEV waypoints extrapolated beyond the sources")
    parser.add_argument("--height", type=float, default=None,
                        help="BEV eye height above the path (world units); "
                             "default = median source-camera radius")
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--framerate", type=int, default=2)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.manifest_path:
        cfg.manifest_path = [args.manifest_path]
    set_seed(args.seed)
    accelerator = Accelerator(mixed_precision=cfg.mixed_precision)
    device = accelerator.device
    H, W = int(cfg.resolution[0]), int(cfg.resolution[1])
    num_steps = args.num_inference_steps or int(cfg.get("eval_num_inference_steps", 4))

    model = build_model(cfg, accelerator, num_steps)
    weights = os.path.join(args.checkpoint, "model.safetensors")
    state = load_file(weights, device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    accelerator.print(f"Loaded {weights}: {len(state)} tensors, "
                      f"missing={len(missing)}, unexpected={len(unexpected)}")
    model.to(device)
    model.eval()

    dataset = Kubric4DRGBDNVSDataset(
        kubric4d_root=cfg.kubric4d_root,
        manifest_path=cfg.manifest_path,
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
        split_path=cfg.get("split_path", None),
        split=cfg.get("eval_split", None),
    )
    record = dataset.records[args.record_idx % len(dataset.records)]
    _patch_load_view(dataset, H, W)

    views_sorted = sorted(record["views"], key=lambda v: v.view_id)
    src_views = views_sorted[args.start_idx:args.start_idx + cfg.num_source_views]
    assert len(src_views) == cfg.num_source_views

    centers = np.stack([v.c2w.double().numpy()[:3, 3] for v in views_sorted])
    height = args.height if args.height is not None else float(
        np.median(np.linalg.norm(centers - centers.mean(0), axis=1)))
    bev_c2ws, src_centers, extrap_centers = _bev_waypoints(
        src_views, args.num_extrap, height)

    src_k = torch.stack([dataset._load_view(v)["K"] for v in src_views])
    virtual_k = src_k.median(0).values
    bev_views = [VirtualView(f"traj{i:02d}", virtual_k.clone(),
                             torch.from_numpy(m).float())
                 for i, m in enumerate(bev_c2ws)]
    accelerator.print(
        f"[bev-traj] scene={record['scene_id']} views={len(views_sorted)} "
        f"src_ids={[v.view_id for v in src_views]} height={height:.3f}")
    accelerator.print("[bev-traj] BEV eyes: " + "  ".join(
        f"traj{i:02d}({np.round(m[:3, 3], 2).tolist()})"
        for i, m in enumerate(bev_c2ws)))

    os.makedirs(args.output_dir, exist_ok=True)
    bev_dir = os.path.join(args.output_dir, "bev_traj")
    os.makedirs(bev_dir, exist_ok=True)

    names = [v.view_id for v in bev_views]
    frames = []
    for start in range(0, len(names), cfg.num_target_views):
        chunk = names[start:start + cfg.num_target_views]
        targets = [bev_views[names.index(n)] for n in chunk]
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

        for i, name in enumerate(chunk):
            vi = model.num_source_views + i
            rgb = _decode_rgb(model, pred_rgb, vi, device)
            raw, vis = _decode_disp(model, pred_depth, vi, device)
            Image.fromarray(rgb).save(
                os.path.join(bev_dir, f"pred_rgb_{name}.jpg"), quality=95)
            Image.fromarray(vis).save(
                os.path.join(bev_dir, f"pred_disp_{name}.jpg"), quality=95)
            np.save(os.path.join(bev_dir, f"pred_disp_raw_{name}.npy"), raw)
            cap = f"BEV {name}"
            rgb_img = Image.fromarray(rgb)
            dep_img = Image.fromarray(vis)
            strip = Image.new("RGB", (rgb_img.width, 22), (16, 16, 16))
            from PIL import ImageDraw
            ImageDraw.Draw(strip).text((6, 4), cap, fill=(240, 240, 240))
            col = Image.new("RGB", (rgb_img.width * 2 + 4, rgb_img.height + 22), (0, 0, 0))
            col.paste(strip, (0, 0))
            col.paste(rgb_img, (0, 22))
            col.paste(dep_img, (rgb_img.width + 4, 22))
            frames.append(col)
        del inputs, pred_depth, pred_rgb, batch
        torch.npu.empty_cache()
        accelerator.print(f"[bev-traj] rendered {chunk}")

    # mp4: PRED RGB | PRED depth along the trajectory
    tmp = os.path.join(args.output_dir, "frames")
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    for i, fr in enumerate(frames):
        canvas = Image.new("RGB", (fr.width + fr.width % 2, fr.height + fr.height % 2), (0, 0, 0))
        canvas.paste(fr, (0, 0))
        canvas.save(os.path.join(tmp, f"frame_{i:03d}.png"))
    mp4 = os.path.join(args.output_dir, "bev_trajectory.mp4")
    subprocess.run(
        [_ffmpeg_bin(), "-y", "-framerate", str(args.framerate),
         "-i", os.path.join(tmp, "frame_%03d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", mp4],
        check=True, capture_output=True)
    shutil.rmtree(tmp)

    bev_centers = bev_c2ws[:, :3, 3]
    _plotly_html(os.path.join(args.output_dir, "cameras_bev_traj.html"),
                 src_views, src_centers, extrap_centers, bev_c2ws, height)
    _plot_png(os.path.join(args.output_dir, "cameras_bev_traj.png"),
              src_centers, extrap_centers, bev_centers, record["scene_id"])
    accelerator.print(f"[bev-traj] wrote {mp4} + cameras_bev_traj.html/.png "
                      f"({len(frames)} BEV frames)")


if __name__ == "__main__":
    main()
