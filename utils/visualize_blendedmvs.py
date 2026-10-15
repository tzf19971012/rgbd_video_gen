"""
Visualize BlendedMVS scenes for a quick initial understanding.

Produces per-scene PNG grids saved to --out_dir:
  <out_dir>/
    scene_overview.png        — RGB thumbnails of all scenes (one row per scene)
    <scene_id>/
      rgb_depth_grid.png      — RGB + pseudo-color depth side-by-side for each view
      camera_poses.png        — 3-D scatter of camera positions + view directions
      nvs_triplet.png         — Example (source1, source2, target) training triplet

Usage:
  cd /path/to/DVD
  source .venv/bin/activate
  python utils/visualize_blendedmvs.py \
      --data_root /mnt/lustre/.../datasets/BlendedMVS/BlendedMVS \
      --manifest  /mnt/lustre/.../datasets/BlendedMVS/manifest.jsonl \
      --out_dir   /mnt/lustre/.../datasets/BlendedMVS/viz \
      --max_scenes 6
"""

import argparse
import json
import os
import random
import struct
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from PIL import Image, ImageOps


# ── helpers ──────────────────────────────────────────────────────────────────

def _read_pfm(path):
    with open(path, "rb") as f:
        header = f.readline().decode("utf-8").rstrip()
        color = header == "PF"
        dim_match = re.match(r"^(\d+)\s(\d+)\s$", f.readline().decode("utf-8"))
        w, h = map(int, dim_match.groups())
        scale = float(f.readline().decode("utf-8").rstrip())
        endian = "<" if scale < 0 else ">"
        data = np.frombuffer(f.read(), dtype=np.dtype(endian + "f4"))
        data = data.reshape((h, w, 3) if color else (h, w))
        return np.flipud(data).astype(np.float32)


def _load_depth(path):
    if path.endswith(".npy"):
        return np.load(path).astype(np.float32)
    return _read_pfm(path)


def _depth_to_color(depth):
    """Return H×W×3 uint8 pseudo-color depth (invalid → black)."""
    valid = np.isfinite(depth) & (depth > 0)
    out = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if valid.any():
        d = depth.copy()
        lo, hi = np.percentile(d[valid], [2, 98])
        d = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
        rgba = cm.plasma(d)
        out = (rgba[..., :3] * 255).astype(np.uint8)
        out[~valid] = 0
    return out


def _parse_cam_txt(path):
    """Return (w2c 4×4, K 3×3)."""
    words = open(path).read().split()
    w2c = np.array([float(words[4 * i + j + 1])
                    for i in range(4) for j in range(4)], np.float32).reshape(4, 4)
    K   = np.array([float(words[3 * i + j + 18])
                    for i in range(3) for j in range(3)], np.float32).reshape(3, 3)
    return w2c, K


def _w2c_to_c2w(w2c):
    return np.linalg.inv(w2c).astype(np.float32)


def _load_image(img_path):
    """Open image with EXIF orientation applied."""
    img = Image.open(img_path).convert("RGB")
    return ImageOps.exif_transpose(img)


def _thumb(img_path, size=128):
    img = _load_image(img_path)
    img.thumbnail((size, size))
    return np.array(img)


# ── per-scene plots ───────────────────────────────────────────────────────────

def plot_rgb_depth_grid(scene_dir, view_ids, out_path, max_views=12):
    """Side-by-side RGB / depth strips for up to max_views."""
    view_ids = view_ids[:max_views]
    n = len(view_ids)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    fig.suptitle(os.path.basename(scene_dir), fontsize=10)

    for col, vid in enumerate(view_ids):
        rgb_path   = os.path.join(scene_dir, "blended_images",      f"{vid}.jpg")
        depth_path = os.path.join(scene_dir, "rendered_depth_maps",  f"{vid}.pfm")
        # prefer pre-converted npy
        npy_path   = depth_path.replace(".pfm", ".npy")
        if os.path.isfile(npy_path):
            depth_path = npy_path

        rgb = np.array(_load_image(rgb_path))
        depth = _load_depth(depth_path)
        depth_color = _depth_to_color(depth)

        ax_rgb   = axes[0, col] if n > 1 else axes[0]
        ax_depth = axes[1, col] if n > 1 else axes[1]

        ax_rgb.imshow(rgb)
        ax_rgb.set_title(f"view {vid}", fontsize=7)
        ax_rgb.axis("off")

        ax_depth.imshow(depth_color)
        valid = np.isfinite(depth) & (depth > 0)
        med = np.median(depth[valid]) if valid.any() else 0
        ax_depth.set_title(f"depth  med={med:.2f}m", fontsize=7)
        ax_depth.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()


def plot_camera_poses(scene_dir, view_ids, out_path):
    """3-D scatter of camera centers + forward vectors."""
    centers = []
    forwards = []
    for vid in view_ids:
        cam_path = os.path.join(scene_dir, "cams", f"{vid}_cam.txt")
        if not os.path.isfile(cam_path):
            continue
        w2c, _ = _parse_cam_txt(cam_path)
        c2w = _w2c_to_c2w(w2c)
        centers.append(c2w[:3, 3])
        forwards.append(c2w[:3, 2])   # camera looks along -Z in OpenCV, +Z col here

    if not centers:
        return
    centers  = np.array(centers)
    forwards = np.array(forwards)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
               c=np.arange(len(centers)), cmap="rainbow", s=40, zorder=5)
    scale = np.ptp(centers, axis=0).max() * 0.1 + 1e-6
    for c, f in zip(centers, forwards):
        ax.quiver(c[0], c[1], c[2], f[0], f[1], f[2],
                  length=scale, normalize=True, color="gray", alpha=0.6)
    ax.set_title(f"{os.path.basename(scene_dir)} — {len(centers)} cameras", fontsize=9)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()


def plot_nvs_triplet(scene_dir, view_ids, out_path):
    """Show a random (src1, src2, target) triplet as the training code sees it."""
    if len(view_ids) < 3:
        return
    chosen = random.sample(view_ids, 3)
    labels = ["source 1", "source 2", "target (query)"]

    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    fig.suptitle(f"Example NVS triplet — {os.path.basename(scene_dir)}", fontsize=10)

    for col, (vid, label) in enumerate(zip(chosen, labels)):
        rgb_path   = os.path.join(scene_dir, "blended_images",     f"{vid}.jpg")
        depth_path = os.path.join(scene_dir, "rendered_depth_maps", f"{vid}.pfm")
        npy_path   = depth_path.replace(".pfm", ".npy")
        if os.path.isfile(npy_path):
            depth_path = npy_path

        rgb   = np.array(_load_image(rgb_path))
        depth = _load_depth(depth_path)
        depth_color = _depth_to_color(depth)

        axes[0, col].imshow(rgb)
        axes[0, col].set_title(f"{label}\nview {vid}", fontsize=8)
        axes[0, col].axis("off")

        axes[1, col].imshow(depth_color)
        axes[1, col].axis("off")
        if col == 2:
            axes[1, col].set_xlabel("← model predicts this", fontsize=7)

    axes[1, 0].set_ylabel("depth (pseudo-color)", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()


# ── scene overview ────────────────────────────────────────────────────────────

def plot_scene_overview(scene_dirs, out_path, thumbs_per_scene=6):
    """One row per scene, each cell = RGB thumbnail."""
    n_scenes = len(scene_dirs)
    n_cols   = thumbs_per_scene
    fig, axes = plt.subplots(n_scenes, n_cols,
                             figsize=(2.5 * n_cols, 2.5 * n_scenes))
    if n_scenes == 1:
        axes = axes[np.newaxis, :]

    for row, sd in enumerate(scene_dirs):
        imgs_dir = os.path.join(sd, "blended_images")
        jpgs = sorted(f for f in os.listdir(imgs_dir)
                      if f.endswith(".jpg") and "_masked" not in f)
        chosen = jpgs[::max(1, len(jpgs) // n_cols)][:n_cols]
        for col in range(n_cols):
            ax = axes[row, col]
            if col < len(chosen):
                ax.imshow(_thumb(os.path.join(imgs_dir, chosen[col]), size=120))
                if col == 0:
                    ax.set_ylabel(os.path.basename(sd)[:16], fontsize=6,
                                  rotation=0, labelpad=60, va="center")
            ax.axis("off")

    fig.suptitle("BlendedMVS — scene overview (evenly-sampled views)", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root",  required=True,
                    help="BlendedMVS/BlendedMVS/ directory (contains PID sub-dirs)")
    ap.add_argument("--manifest",   default=None,
                    help="manifest.jsonl (optional; falls back to all sub-dirs)")
    ap.add_argument("--out_dir",    required=True)
    ap.add_argument("--max_scenes", type=int, default=6)
    ap.add_argument("--seed",       type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # Collect scene directories
    if args.manifest and os.path.isfile(args.manifest):
        with open(args.manifest) as f:
            records = [json.loads(l) for l in f if l.strip()]
        scene_ids = [r["scene_id"] for r in records[:args.max_scenes]]
    else:
        scene_ids = sorted(
            e for e in os.listdir(args.data_root)
            if os.path.isdir(os.path.join(args.data_root, e))
        )[:args.max_scenes]

    scene_dirs = [os.path.join(args.data_root, sid) for sid in scene_ids]
    scene_dirs = [sd for sd in scene_dirs if os.path.isdir(sd)]

    print(f"Visualizing {len(scene_dirs)} scenes → {args.out_dir}")

    # 1. Scene overview grid
    plot_scene_overview(scene_dirs,
                        os.path.join(args.out_dir, "scene_overview.png"))

    # 2. Per-scene plots
    for sd in scene_dirs:
        sid = os.path.basename(sd)
        imgs_dir = os.path.join(sd, "blended_images")
        view_ids = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(imgs_dir)
            if f.endswith(".jpg") and "_masked" not in f
        )

        scene_out = os.path.join(args.out_dir, sid)
        os.makedirs(scene_out, exist_ok=True)

        print(f"  {sid}: {len(view_ids)} views")

        plot_rgb_depth_grid(sd, view_ids,
                            os.path.join(scene_out, "rgb_depth_grid.png"))
        plot_camera_poses(sd, view_ids,
                          os.path.join(scene_out, "camera_poses.png"))
        plot_nvs_triplet(sd, view_ids,
                         os.path.join(scene_out, "nvs_triplet.png"))

    print(f"\nDone. Open {args.out_dir}/ to browse the PNGs.")


if __name__ == "__main__":
    main()
