"""
Prepare GTA-SfM / MatrixCity / TartanAir manifests for NVS training.

For datasets that ship without a manifest.jsonl, this scans the raw layout
and writes one record per scene:
    {"scene_id", "time_id", "views": [{"view_id", "rgb_path", "depth_path",
                                       "K", "c2w", "depth_scale"}]}
All paths are written RELATIVE to --data_root so the manifest is portable;
rgb/depth files that live inside archives (GTA-SfM) are extracted to disk
first, because the training dataset loads images/depths from files.

Pose conventions were reverse-engineered against the reference manifests and
verified numerically view-by-view:

  tartanair
    Raw layout: <root>/<env>/<Easy|Hard>/<P###>/ with image_left/*.png,
    depth_left/ or depth_clean/*.npy, pose_left.txt.
    pose_left.txt rows: "x y z qx qy qz qw" (position + quaternion).
    c2w_R = quat2mat(qx,qy,qz,qw) @ PERM, PERM = [[0,0,1],[1,0,0],[0,1,0]]
    (TartanAir camera axes -> OpenCV c2w, no sign flips); t = (x, y, z) as-is.
    K follows the TartanAir convention f = W/2, cx = W/2, cy = H/2.
    Depth prefers depth_clean/ (less noise), falls back to depth_left/.
  matrixcity
    Raw layout: <root>/scene_*/ with rgb/*.png, depth_npy/*.npy (or depth/*.exr)
    and transforms.json holding {camera_angle_x, frames[].rot_mat}.
    rot_mat is a 4x4 [0.01*R | t] (the fbx cm-scale lands on the rotation
    block), so c2w_R = rot_mat[:3,:3] / 0.01, t = rot_mat[:3,3] -- NO transpose.
    K: f = W / (2 * tan(camera_angle_x / 2)), principal point at the center.
  gta_sfm
    Raw layout: <root>/<split>/<name>.hdf5, one archive per scene, holding
    per-frame image_<i> (BMP byte blobs), depth_<i> (H×W float32),
    K_<i> (3×3) and pose_<i> (4x4 c2w, used verbatim). Decoded images and
    depths are written to <data_root>/extracted/<scene_id>/{rgb,depth}/ and
    the manifest references those files.

Usage:
  python utils/prepare_multi4d_manifests.py tartanair \
      --data_root /path/to/TartanAir --out_manifest /path/to/TartanAir/manifest.jsonl
  python utils/prepare_multi4d_manifests.py matrixcity \
      --data_root /path/to/MatrixCity --out_manifest .../manifest.jsonl
  python utils/prepare_multi4d_manifests.py gta_sfm \
      --data_root /path/to/GTA-SfM --out_manifest .../manifest.jsonl [--stride 5]

Common options: --stride N (keep every N-th frame, default 1 = all frames),
--min_views N (drop scenes with fewer usable views, default 5),
--tartanair-k "fx,fy,cx,cy" (override the default W/2,H/2 intrinsics).
"""

import argparse
import io
import json
import math
import os
import re
import sys

import numpy as np
from PIL import Image

# TartanAir camera-axis permutation: manifest c2w_R = R_quat @ PERM
# (columns reordered to [e_y, e_z, e_x]); verified against the reference
# manifest on abandonedfactory/Easy/P000.
_TARTANAIR_PERM = np.array([[0.0, 0.0, 1.0],
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0]])


def _quat_to_mat(qx, qy, qz, qw):
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def _image_size(path):
    with Image.open(path) as img:
        return img.size  # (W, H)


def _write_manifest(records, out_manifest):
    os.makedirs(os.path.dirname(os.path.abspath(out_manifest)), exist_ok=True)
    with open(out_manifest, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    n_views = sum(len(r["views"]) for r in records)
    print(f"Done: {len(records)} scenes, {n_views} views -> {out_manifest}")


def _record(scene_id, views):
    return {"scene_id": scene_id, "time_id": "t0", "views": views}


def _view(view_id, rgb_path, depth_path, K, c2w, depth_scale=1.0):
    return {
        "view_id": view_id,
        "rgb_path": rgb_path,
        "depth_path": depth_path,
        "K": [[float(v) for v in row] for row in K],
        "c2w": [[float(v) for v in row] for row in c2w],
        "depth_scale": float(depth_scale),
    }


# ---------------------------------------------------------------------------
# TartanAir
# ---------------------------------------------------------------------------

def _tartanair_scenes(root):
    """All <env>/<difficulty>/<traversal>/ dirs that hold pose_left.txt."""
    scenes = []
    for env in sorted(os.listdir(root)):
        env_dir = os.path.join(root, env)
        if not os.path.isdir(env_dir):
            continue
        for difficulty in sorted(os.listdir(env_dir)):
            diff_dir = os.path.join(env_dir, difficulty)
            if not os.path.isdir(diff_dir):
                continue
            for traversal in sorted(os.listdir(diff_dir)):
                trav_dir = os.path.join(diff_dir, traversal)
                if os.path.isfile(os.path.join(trav_dir, "pose_left.txt")):
                    scenes.append((os.path.join(env, difficulty, traversal), trav_dir))
    return scenes


def prepare_tartanair(data_root, out_manifest, stride, min_views, k_override):
    records = []
    for scene_id, scene_dir in _tartanair_scenes(data_root):
        rel_dir = scene_id  # already relative, "/" separated
        poses = np.loadtxt(os.path.join(scene_dir, "pose_left.txt"), ndmin=2)
        n_frames = poses.shape[0]
        if n_frames < 1:
            continue

        depth_subdir = None
        for sub in ("depth_clean", "depth_left"):
            if os.path.isdir(os.path.join(scene_dir, sub)):
                depth_subdir = sub
                break
        if depth_subdir is None:
            print(f"[skip] {scene_id}: no depth_clean/ or depth_left/", file=sys.stderr)
            continue

        # intrinsics from the first frame's image size (TartanAir: f = W/2)
        first_img = os.path.join(scene_dir, "image_left", f"{0:06d}_left.png")
        if not os.path.exists(first_img):
            print(f"[skip] {scene_id}: no image_left/{first_idx:06d}_left.png", file=sys.stderr)
            continue
        W, H = _image_size(first_img)
        if k_override is not None:
            fx, fy, cx, cy = k_override
        else:
            fx = fy = W / 2.0
            cx, cy = W / 2.0, H / 2.0
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

        views = []
        for idx in range(0, n_frames, stride):
            rgb_file = f"image_left/{idx:06d}_left.png"
            if not os.path.exists(os.path.join(scene_dir, rgb_file)):
                continue
            depth_file = f"{depth_subdir}/{idx:06d}_left_depth.npy"
            if not os.path.exists(os.path.join(scene_dir, depth_file)):
                continue
            x, y, z, qx, qy, qz, qw = poses[idx]
            c2w = np.eye(4)
            c2w[:3, :3] = _quat_to_mat(qx, qy, qz, qw) @ _TARTANAIR_PERM
            c2w[:3, 3] = (x, y, z)
            views.append(_view(f"{idx:06d}",
                               f"{rel_dir}/{rgb_file}", f"{rel_dir}/{depth_file}", K, c2w))
        if len(views) >= min_views:
            records.append(_record(scene_id, views))
        else:
            print(f"[skip] {scene_id}: {len(views)} usable views < min_views={min_views}",
                  file=sys.stderr)
    _write_manifest(records, out_manifest)


# ---------------------------------------------------------------------------
# MatrixCity
# ---------------------------------------------------------------------------

def _read_exr(path):
    """Read a single-channel .exr depth map, trying cv2 then imageio."""
    try:
        import cv2
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        data = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if data is not None:
            return np.asarray(data, dtype=np.float32)
    except Exception:
        pass
    try:
        import imageio.v2 as imageio
        data = imageio.imread(path)
        return np.asarray(data, dtype=np.float32)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot read EXR {path}: install imageio (pip install imageio) or an "
            "OpenEXR-enabled opencv, or pre-convert depth/*.exr to depth_npy/*.npy"
        ) from exc


def _matrixcity_scenes(root):
    return sorted(d for d in os.listdir(root)
                  if os.path.isfile(os.path.join(root, d, "transforms.json")))


def prepare_matrixcity(data_root, out_manifest, stride, min_views, rot_scale,
                       convert_exr):
    records = []
    for scene_id in _matrixcity_scenes(data_root):
        scene_dir = os.path.join(data_root, scene_id)
        with open(os.path.join(scene_dir, "transforms.json"), encoding="utf-8") as handle:
            transforms = json.load(handle)
        angle_x = float(transforms["camera_angle_x"])
        frames = sorted(transforms["frames"], key=lambda f: int(f["frame_index"]))

        first_rgb = os.path.join(scene_dir, "rgb", f"{int(frames[0]['frame_index']):04d}.png")
        if not os.path.exists(first_rgb):
            print(f"[skip] {scene_id}: rgb/ empty", file=sys.stderr)
            continue
        W, H = _image_size(first_rgb)
        f = W / (2.0 * math.tan(angle_x / 2.0))
        K = np.array([[f, 0.0, W / 2.0], [0.0, f, H / 2.0], [0.0, 0.0, 1.0]])

        views = []
        for frame in frames[::stride]:
            idx = int(frame["frame_index"])
            rgb_file = f"rgb/{idx:04d}.png"
            if not os.path.exists(os.path.join(scene_dir, rgb_file)):
                continue
            npy_file = f"depth_npy/{idx:04d}.npy"
            if os.path.exists(os.path.join(scene_dir, npy_file)):
                depth_file = npy_file
            else:
                exr_file = f"depth/{idx:04d}.exr"
                if convert_exr and os.path.exists(os.path.join(scene_dir, exr_file)):
                    depth = _read_exr(os.path.join(scene_dir, exr_file))
                    if depth.ndim == 3:
                        depth = depth[..., 0]
                    os.makedirs(os.path.join(scene_dir, "depth_npy"), exist_ok=True)
                    np.save(os.path.join(scene_dir, npy_file), depth.astype(np.float32))
                    depth_file = npy_file
                else:
                    continue
            rot = np.asarray(frame["rot_mat"], dtype=np.float64)
            c2w = np.eye(4)
            c2w[:3, :3] = rot[:3, :3] / rot_scale  # 0.01*R -> R, NO transpose
            c2w[:3, 3] = rot[:3, 3]
            views.append(_view(f"{idx:04d}",
                               f"{scene_id}/{rgb_file}", f"{scene_id}/{depth_file}", K, c2w))
        if len(views) >= min_views:
            records.append(_record(scene_id, views))
        else:
            print(f"[skip] {scene_id}: {len(views)} usable views < min_views={min_views}",
                  file=sys.stderr)
    _write_manifest(records, out_manifest)


# ---------------------------------------------------------------------------
# GTA-SfM
# ---------------------------------------------------------------------------

def _gta_sfm_archives(root):
    """<split>/*.hdf5 archives; scene_id = "<split>_<stem>".

    Splits are enumerated with "train" first (then the rest alphabetically) to
    match the reference manifests.
    """
    archives = []
    splits = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    splits.sort(key=lambda d: d != "train")
    for split in splits:
        split_dir = os.path.join(root, split)
        for name in sorted(os.listdir(split_dir)):
            if name.endswith(".hdf5"):
                archives.append((f"{split}_{os.path.splitext(name)[0]}",
                                 os.path.join(split_dir, name)))
    return archives


def prepare_gta_sfm(data_root, out_manifest, stride, min_views, extract_root,
                    skip_extract, invalid_depth):
    import h5py

    if extract_root is None:
        extract_root = os.path.join(data_root, "extracted")
    records = []
    for scene_id, archive in _gta_sfm_archives(data_root):
        with h5py.File(archive, "r") as handle:
            index_re = re.compile(r"^image_(\d+)$")
            n_frames = max((int(m.group(1)) for key in handle.keys()
                            if (m := index_re.match(key))), default=-1) + 1
            if n_frames < 1:
                print(f"[skip] {scene_id}: no image_* in {archive}", file=sys.stderr)
                continue

            scene_out = os.path.join(extract_root, scene_id)
            rgb_out = os.path.join(scene_out, "rgb")
            depth_out = os.path.join(scene_out, "depth")
            if not skip_extract:
                os.makedirs(rgb_out, exist_ok=True)
                os.makedirs(depth_out, exist_ok=True)

            views = []
            for idx in range(0, n_frames, stride):
                rgb_file = f"extracted/{scene_id}/rgb/{idx:05d}.png"
                depth_file = f"extracted/{scene_id}/depth/{idx:05d}.npy"
                if not skip_extract:
                    with Image.open(io.BytesIO(bytes(handle[f"image_{idx}"][:, 0]))) as img:
                        img.convert("RGB").save(os.path.join(rgb_out, f"{idx:05d}.png"))
                    depth = np.asarray(handle[f"depth_{idx}"], dtype=np.float32)
                    if invalid_depth is not None:
                        # hdf5 uses a large sentinel (default 10000) where no
                        # depth was captured; 0 marks it invalid downstream
                        # (_inverse_depth excludes depth <= 0).
                        depth = np.where(depth >= invalid_depth, 0.0, depth)
                    np.save(os.path.join(depth_out, f"{idx:05d}.npy"), depth)
                elif not (os.path.exists(os.path.join(data_root, rgb_file))
                          and os.path.exists(os.path.join(data_root, depth_file))):
                    continue
                K = np.asarray(handle[f"K_{idx}"], dtype=np.float64)
                c2w = np.asarray(handle[f"pose_{idx}"], dtype=np.float64)
                views.append(_view(f"{idx:05d}", rgb_file, depth_file, K, c2w))
        if len(views) >= min_views:
            records.append(_record(scene_id, views))
        else:
            print(f"[skip] {scene_id}: {len(views)} usable views < min_views={min_views}",
                  file=sys.stderr)
    _write_manifest(records, out_manifest)


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=["tartanair", "matrixcity", "gta_sfm"])
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_manifest", required=True)
    parser.add_argument("--stride", type=int, default=1,
                        help="keep every N-th frame (default 1 = all)")
    parser.add_argument("--min_views", type=int, default=5)
    parser.add_argument("--tartanair-k", default=None,
                        help="override intrinsics as 'fx,fy,cx,cy' "
                             "(default: W/2,W/2,W/2,H/2 from the image size)")
    parser.add_argument("--rot-scale", type=float, default=0.01,
                        help="matrixcity: scale baked into transforms.json rot_mat "
                             "rotation block (default 0.01)")
    parser.add_argument("--convert-exr", action="store_true",
                        help="matrixcity: convert depth/*.exr to depth_npy/*.npy on the fly")
    parser.add_argument("--extract-root", default=None,
                        help="gta_sfm: where to write extracted rgb/depth "
                             "(default <data_root>/extracted)")
    parser.add_argument("--skip-extract", action="store_true",
                        help="gta_sfm: only write the manifest, assume extraction exists")
    parser.add_argument("--invalid-depth", type=float, default=10000.0,
                        help="gta_sfm: depth values >= this in the hdf5 are 'no data' "
                             "sentinels and are written as 0 (invalid); "
                             "pass a negative value to disable")
    args = parser.parse_args()

    if args.stride < 1:
        parser.error("--stride must be >= 1")
    k_override = (tuple(float(v) for v in args.tartanair_k.split(","))
                  if args.tartanair_k else None)

    if args.dataset == "tartanair":
        prepare_tartanair(args.data_root, args.out_manifest, args.stride,
                          args.min_views, k_override)
    elif args.dataset == "matrixcity":
        prepare_matrixcity(args.data_root, args.out_manifest, args.stride,
                           args.min_views, args.rot_scale, args.convert_exr)
    elif args.dataset == "gta_sfm":
        prepare_gta_sfm(args.data_root, args.out_manifest, args.stride,
                        args.min_views, args.extract_root, args.skip_extract,
                        args.invalid_depth)


if __name__ == "__main__":
    main()
