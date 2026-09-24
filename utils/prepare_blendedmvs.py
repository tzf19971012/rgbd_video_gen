"""
Prepare BlendedMVS data for NVS training.

For each scene in the dataset, this script:
  1. Reads per-view cam.txt files → intrinsics K and camera-to-world c2w
  2. Converts rendered_depth_maps/*.pfm → *.npy (metric, in same dir)
  3. Writes a manifest.jsonl where every record = one scene with N views

Usage:
  python utils/prepare_blendedmvs.py \
      --data_root /path/to/BlendedMVS \
      --out_manifest /path/to/BlendedMVS/manifest.jsonl \
      [--scene_list BlendedMVS_list.txt] \
      [--max_scenes 20] \
      [--min_views 5] \
      [--skip_depth_convert]
"""

import argparse
import json
import os
import re
import struct
import sys

import numpy as np


# ---------------------------------------------------------------------------
# PFM reader (ported from MVSNet)
# ---------------------------------------------------------------------------

def _read_pfm(path):
    """Read a PFM file. Returns (data: float32 H×W, scale: float)."""
    with open(path, "rb") as f:
        header = f.readline().decode("utf-8").rstrip()
        if header == "PF":
            color = True
        elif header == "Pf":
            color = False
        else:
            raise ValueError(f"Not a PFM file: {path!r}")

        dim_match = re.match(r"^(\d+)\s(\d+)\s$", f.readline().decode("utf-8"))
        if not dim_match:
            raise ValueError(f"Malformed PFM header in {path!r}")
        width, height = map(int, dim_match.groups())

        scale = float(f.readline().decode("utf-8").rstrip())
        if scale < 0:
            endian = "<"
            scale = -scale
        else:
            endian = ">"

        dtype = np.dtype(endian + "f4")
        channels = 3 if color else 1
        data = np.frombuffer(f.read(), dtype=dtype)
        data = data.reshape((height, width, channels) if color else (height, width))
        data = np.flipud(data)  # PFM stores bottom-up
        return data.astype(np.float32), scale


# ---------------------------------------------------------------------------
# cam.txt reader
# ---------------------------------------------------------------------------

def _read_cam_txt(path):
    """
    Parse MVSNet-format cam.txt.
    Returns:
        w2c  : (4, 4) float32 – world-to-camera extrinsic
        K    : (3, 3) float32 – intrinsic matrix
    File layout (words split by whitespace):
        word 0  : 'extrinsic'
        words 1-16 : 4×4 w2c (row-major)
        word 17 : 'intrinsic'
        words 18-26 : 3×3 K (row-major)
        optional words 27-28: depth_min, depth_interval
    """
    with open(path, "r") as f:
        words = f.read().split()

    w2c = np.array([float(words[4 * i + j + 1]) for i in range(4) for j in range(4)],
                   dtype=np.float32).reshape(4, 4)
    K = np.array([float(words[3 * i + j + 18]) for i in range(3) for j in range(3)],
                 dtype=np.float32).reshape(3, 3)
    return w2c, K


def _w2c_to_c2w(w2c):
    """Invert a 4×4 rigid-body transform."""
    c2w = np.linalg.inv(w2c)
    return c2w.astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _iter_scenes(data_root, scene_list_file, max_scenes):
    """Yield scene directory paths."""
    if scene_list_file and os.path.isfile(scene_list_file):
        scene_ids = open(scene_list_file).read().splitlines()
    else:
        # fall back: every sub-directory that has both cams/ and blended_images/
        scene_ids = sorted(
            e for e in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, e))
            and os.path.isdir(os.path.join(data_root, e, "cams"))
            and os.path.isdir(os.path.join(data_root, e, "blended_images"))
        )

    if max_scenes is not None and max_scenes > 0:
        scene_ids = scene_ids[:max_scenes]

    for sid in scene_ids:
        yield sid, os.path.join(data_root, sid)


def _convert_pfm_to_npy(pfm_path, npy_path, force=False):
    """Convert pfm → npy if npy does not exist yet (or force=True)."""
    if not force and os.path.isfile(npy_path):
        return  # already done
    depth, _scale = _read_pfm(pfm_path)
    np.save(npy_path, depth)


def prepare(data_root, out_manifest, scene_list_file, max_scenes, min_views, skip_depth_convert, force_convert):
    records = []
    total_views = 0

    for scene_id, scene_dir in _iter_scenes(data_root, scene_list_file, max_scenes):
        cams_dir = os.path.join(scene_dir, "cams")
        images_dir = os.path.join(scene_dir, "blended_images")
        depth_dir = os.path.join(scene_dir, "rendered_depth_maps")

        if not (os.path.isdir(cams_dir) and os.path.isdir(images_dir)):
            print(f"  [skip] {scene_id}: missing cams/ or blended_images/")
            continue

        # collect cam files
        cam_files = sorted(f for f in os.listdir(cams_dir) if f.endswith("_cam.txt"))
        if len(cam_files) < min_views:
            print(f"  [skip] {scene_id}: only {len(cam_files)} views < {min_views}")
            continue

        views = []
        for cam_file in cam_files:
            view_id = cam_file.replace("_cam.txt", "")   # e.g. '00000000'
            cam_path = os.path.join(cams_dir, cam_file)
            rgb_path = os.path.join(images_dir, f"{view_id}.jpg")
            pfm_path = os.path.join(depth_dir, f"{view_id}.pfm")
            npy_path = os.path.join(depth_dir, f"{view_id}.npy")

            if not os.path.isfile(rgb_path):
                continue  # image missing – skip this view
            if not os.path.isfile(pfm_path):
                continue  # depth missing – skip this view

            # parse camera
            try:
                w2c, K = _read_cam_txt(cam_path)
            except Exception as e:
                print(f"  [warn] cannot parse {cam_path}: {e}")
                continue

            c2w = _w2c_to_c2w(w2c)

            # convert depth if needed
            if not skip_depth_convert:
                try:
                    _convert_pfm_to_npy(pfm_path, npy_path, force=force_convert)
                    depth_path_for_manifest = os.path.relpath(npy_path, data_root)
                except Exception as e:
                    print(f"  [warn] cannot convert {pfm_path}: {e}")
                    continue
            else:
                # keep pfm reference (loader does not support pfm natively,
                # but you can add it later or set skip_depth_convert=False)
                depth_path_for_manifest = os.path.relpath(pfm_path, data_root)

            views.append({
                "view_id": view_id,
                "rgb_path": os.path.relpath(rgb_path, data_root),
                "depth_path": depth_path_for_manifest,
                "K": K.tolist(),
                "c2w": c2w.tolist(),
                "depth_scale": 1.0,
            })

        if len(views) < min_views:
            print(f"  [skip] {scene_id}: only {len(views)} valid views after checks")
            continue

        records.append({
            "scene_id": scene_id,
            "time_id": "t0",
            "views": views,
        })
        total_views += len(views)
        print(f"  [ok]   {scene_id}: {len(views)} views")

    os.makedirs(os.path.dirname(os.path.abspath(out_manifest)), exist_ok=True)
    with open(out_manifest, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"\nDone: {len(records)} scenes, {total_views} total views → {out_manifest}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True,
                        help="Root directory of extracted BlendedMVS dataset")
    parser.add_argument("--out_manifest", required=True,
                        help="Output manifest.jsonl path")
    parser.add_argument("--scene_list", default=None,
                        help="Path to scene-list .txt (e.g. BlendedMVS_list.txt). "
                             "If omitted, all valid sub-dirs are used.")
    parser.add_argument("--max_scenes", type=int, default=None,
                        help="Limit to first N scenes (useful for smoke tests)")
    parser.add_argument("--min_views", type=int, default=5,
                        help="Skip scenes with fewer than this many views (default: 5)")
    parser.add_argument("--skip_depth_convert", action="store_true",
                        help="Do NOT convert pfm → npy (leave pfm references in manifest)")
    parser.add_argument("--force_convert", action="store_true",
                        help="Re-convert pfm → npy even if .npy already exists")
    args = parser.parse_args()

    print(f"Processing BlendedMVS at: {args.data_root}")
    prepare(
        data_root=args.data_root,
        out_manifest=args.out_manifest,
        scene_list_file=args.scene_list,
        max_scenes=args.max_scenes,
        min_views=args.min_views,
        skip_depth_convert=args.skip_depth_convert,
        force_convert=args.force_convert,
    )


if __name__ == "__main__":
    main()
