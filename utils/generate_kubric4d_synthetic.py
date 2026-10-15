"""
Generate a small synthetic Kubric-4D style dataset for smoke testing.

Creates N_SCENES scenes, each with N_VIEWS cameras arranged around a simple 3D
scene containing colored quads and a ground plane.  Outputs:
  <out_dir>/scene_XXXX/view_YY/rgb.png
  <out_dir>/scene_XXXX/view_YY/depth.npy
  <out_dir>/manifest.jsonl   (one JSON record per scene-time pair)

Usage:
  python utils/generate_kubric4d_synthetic.py \
      --out_dir /mnt/lustre/.../datasets/kubric4d \
      --n_scenes 8 \
      --n_views 5 \
      --width 256 --height 256
"""

import argparse
import json
import math
import os
import random

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def make_lookat(eye: np.ndarray, target: np.ndarray, up: np.ndarray = None) -> np.ndarray:
    """Return 4x4 camera-to-world matrix looking from eye toward target."""
    if up is None:
        up = np.array([0.0, 0.0, 1.0])
    z = eye - target
    z = z / (np.linalg.norm(z) + 1e-8)
    x = np.cross(up, z)
    x = x / (np.linalg.norm(x) + 1e-8)
    y = np.cross(z, x)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = x
    c2w[:3, 1] = y
    c2w[:3, 2] = z
    c2w[:3, 3] = eye
    return c2w


def make_intrinsics(width: int, height: int, fov_deg: float = 60.0) -> np.ndarray:
    """Return 3x3 camera intrinsics matrix."""
    fov_rad = math.radians(fov_deg)
    fx = (width / 2.0) / math.tan(fov_rad / 2.0)
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    K = np.array([[fx, 0., cx],
                  [0., fy, cy],
                  [0.,  0., 1.]], dtype=np.float32)
    return K


# ---------------------------------------------------------------------------
# Simple ray-marched scene renderer
# ---------------------------------------------------------------------------

def _sphere_intersect(ro: np.ndarray, rd: np.ndarray,
                      center: np.ndarray, radius: float):
    """Analytic ray-sphere intersection. Returns t or np.inf."""
    oc = ro - center  # [..., 3]
    b = (oc * rd).sum(axis=-1)
    c = (oc * oc).sum(axis=-1) - radius ** 2
    disc = b * b - c
    mask = disc >= 0
    t = np.full_like(b, np.inf)
    sq = np.sqrt(np.maximum(disc, 0.0))
    t1 = -b - sq
    t2 = -b + sq
    t[mask] = np.where(t1[mask] > 1e-4, t1[mask],
                       np.where(t2[mask] > 1e-4, t2[mask], np.inf))
    return t


def _plane_intersect(ro: np.ndarray, rd: np.ndarray,
                     normal: np.ndarray, d: float):
    """Ray-plane intersection.  plane: dot(n, p) = d.  Returns t or np.inf."""
    denom = (rd * normal).sum(axis=-1)
    t = np.full(ro.shape[:-1], np.inf)
    valid = np.abs(denom) > 1e-8
    t[valid] = (d - (ro[valid] * normal).sum(axis=-1)) / denom[valid]
    t[t < 1e-4] = np.inf
    return t


def render_scene(c2w: np.ndarray, K: np.ndarray,
                 width: int, height: int,
                 scene_objects: list) -> tuple:
    """
    Software-render a scene from the given camera.

    scene_objects: list of dicts with keys
        type: "sphere" | "plane"
        color: (R, G, B) in [0,1]
        For sphere: center [3], radius float
        For plane:  normal [3], d float, checkerboard: bool

    Returns (rgb [H,W,3] float32 [0,1], depth [H,W] float32 metres)
    """
    # Build ray directions in world space
    iy, ix = np.meshgrid(np.arange(height, dtype=np.float32),
                         np.arange(width, dtype=np.float32), indexing="ij")
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    rd_cam = np.stack([(ix - cx) / fx,
                       (iy - cy) / fy,
                       np.ones_like(ix)], axis=-1)  # [H,W,3]
    rd_cam = rd_cam / (np.linalg.norm(rd_cam, axis=-1, keepdims=True) + 1e-8)
    R = c2w[:3, :3].astype(np.float32)  # [3,3]
    rd = (R @ rd_cam.reshape(-1, 3).T).T.reshape(height, width, 3)  # [H,W,3]
    ro = np.broadcast_to(c2w[:3, 3].astype(np.float32), (height, width, 3))

    # Initialize buffers
    depth = np.full((height, width), np.inf, dtype=np.float32)
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    bg_color = np.array([0.6, 0.7, 0.9], dtype=np.float32)

    for obj in scene_objects:
        if obj["type"] == "sphere":
            t = _sphere_intersect(ro, rd,
                                  np.array(obj["center"], dtype=np.float32),
                                  obj["radius"])
        elif obj["type"] == "plane":
            t = _plane_intersect(ro, rd,
                                 np.array(obj["normal"], dtype=np.float32),
                                 obj["d"])
        else:
            continue

        hit = (t < depth) & np.isfinite(t)
        if not hit.any():
            continue

        depth[hit] = t[hit]
        color = np.array(obj["color"], dtype=np.float32)

        hit_flat = hit.ravel()
        t_flat = t.ravel()
        ro_flat = ro.reshape(-1, 3)
        rd_flat = rd.reshape(-1, 3)

        if obj.get("checkerboard", False):
            # Checkerboard pattern on the plane
            p = ro_flat[hit_flat] + t_flat[hit_flat, None] * rd_flat[hit_flat]
            checker = ((np.floor(p[:, 0] * 0.5) + np.floor(p[:, 1] * 0.5)).astype(int)) % 2
            c = np.where(checker[:, None], color, color * 0.5)
            rgb_flat = rgb.reshape(-1, 3)
            rgb_flat[hit_flat] = c
            rgb[:] = rgb_flat.reshape(height, width, 3)
        else:
            # Simple Lambertian shading from a fixed directional light
            if obj["type"] == "sphere":
                center = np.array(obj["center"], dtype=np.float32)
                pts = ro_flat[hit_flat] + t_flat[hit_flat, None] * rd_flat[hit_flat]
                normals = (pts - center) / (obj["radius"] + 1e-8)
            else:
                normals = np.broadcast_to(
                    np.array(obj["normal"], dtype=np.float32), (hit_flat.sum(), 3))
            light = np.array([0.5, 0.7, 1.0], dtype=np.float32)
            light = light / (np.linalg.norm(light) + 1e-8)
            diff = np.clip((normals * light).sum(axis=-1), 0.1, 1.0)
            rgb_flat = rgb.reshape(-1, 3)
            rgb_flat[hit_flat] = color * diff[:, None]
            rgb[:] = rgb_flat.reshape(height, width, 3)

    # Background for rays that hit nothing
    bg = ~np.isfinite(depth)
    rgb[bg] = bg_color
    depth[bg] = 0.0  # invalid depth

    return rgb.clip(0, 1), depth


def build_scene_objects(rng: np.random.Generator) -> list:
    """Random scene: ground plane + 2-5 coloured spheres."""
    objects = []

    # Ground plane
    objects.append({
        "type": "plane",
        "normal": [0.0, 0.0, 1.0],
        "d": -0.5,
        "color": [0.5, 0.45, 0.4],
        "checkerboard": True,
    })

    n_spheres = rng.integers(2, 6)
    for _ in range(n_spheres):
        cx = rng.uniform(-2.0, 2.0)
        cy = rng.uniform(0.5, 3.5)
        cz = rng.uniform(0.0, 1.5)
        r = rng.uniform(0.2, 0.7)
        color = rng.uniform(0.2, 0.95, size=3).tolist()
        objects.append({
            "type": "sphere",
            "center": [cx, cy, cz],
            "radius": r,
            "color": color,
        })

    return objects


def cameras_for_scene(n_views: int, rng: np.random.Generator,
                      radius_min=4.0, radius_max=7.0,
                      elev_min=10.0, elev_max=50.0) -> list:
    """Return list of (eye [3], c2w [4,4]) tuples looking toward origin."""
    cams = []
    target = np.array([0.0, 0.0, 0.5], dtype=np.float32)
    for i in range(n_views):
        azimuth = rng.uniform(0, 360.0)
        elevation = rng.uniform(elev_min, elev_max)
        radius = rng.uniform(radius_min, radius_max)
        az_rad = math.radians(azimuth)
        el_rad = math.radians(elevation)
        x = radius * math.cos(el_rad) * math.cos(az_rad)
        y = radius * math.cos(el_rad) * math.sin(az_rad)
        z = radius * math.sin(el_rad)
        eye = np.array([x, y, z], dtype=np.float32)
        c2w = make_lookat(eye, target)
        cams.append(c2w)
    return cams


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate(out_dir: str, n_scenes: int, n_views: int,
             width: int, height: int, seed: int):
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "manifest.jsonl")
    rng = np.random.default_rng(seed)
    K = make_intrinsics(width, height, fov_deg=60.0)

    records = []

    for scene_idx in range(n_scenes):
        scene_id = f"scene_{scene_idx:04d}"
        scene_dir = os.path.join(out_dir, scene_id)
        objects = build_scene_objects(rng)
        cams = cameras_for_scene(n_views, rng)

        views = []
        for view_idx, c2w in enumerate(cams):
            view_id = f"view_{view_idx:02d}"
            view_dir = os.path.join(scene_dir, view_id)
            os.makedirs(view_dir, exist_ok=True)

            rgb_np, depth_np = render_scene(c2w, K, width, height, objects)

            rgb_path = os.path.join(view_dir, "rgb.png")
            depth_path = os.path.join(view_dir, "depth.npy")

            Image.fromarray((rgb_np * 255).astype(np.uint8)).save(rgb_path)
            np.save(depth_path, depth_np)

            views.append({
                "view_id": view_id,
                "rgb_path": os.path.relpath(rgb_path, out_dir),
                "depth_path": os.path.relpath(depth_path, out_dir),
                "K": K.tolist(),
                "c2w": c2w.tolist(),
                "depth_scale": 1.0,
            })

        record = {
            "scene_id": scene_id,
            "time_id": "t0",
            "views": views,
        }
        records.append(record)
        print(f"  rendered {scene_id} ({n_views} views)")

    with open(manifest_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"\nManifest written: {manifest_path} ({len(records)} records)")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Kubric4D dataset")
    parser.add_argument(
        "--out_dir",
        default=os.path.join(os.environ.get("DVD_DATA_ROOT", "."), "datasets", "kubric4d"),
    )
    parser.add_argument("--n_scenes", type=int, default=8)
    parser.add_argument("--n_views", type=int, default=5)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Generating {args.n_scenes} scenes x {args.n_views} views "
          f"({args.width}x{args.height}) -> {args.out_dir}")
    generate(args.out_dir, args.n_scenes, args.n_views,
             args.width, args.height, args.seed)


if __name__ == "__main__":
    main()
