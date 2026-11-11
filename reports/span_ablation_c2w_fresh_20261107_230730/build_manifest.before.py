"""Build manifest_train.jsonl for the re10k WAI-style extraction.

Scans <out_root>/<scene>/poses.json (written by re10k_prep_worker.py), pads
3x4 c2w matrices to 4x4, and keeps scenes with >= --min-views views.

Usage:
  python build_manifest.py --out-root /mnt/sfs_turbo/t00845730/re10k/extracted \
      --out /mnt/sfs_turbo/t00845730/re10k/manifest_train.jsonl
"""

import argparse
import json
import os

import numpy as np


def poses_valid(frames):
    """DA3 偶尔对某些场景预测出全零/奇异外参，这类场景必须剔除。"""
    for fr in frames:
        c2w = np.asarray(fr["c2w"], dtype=np.float64)
        if c2w.shape == (3, 4):
            c2w = np.vstack([c2w, [0.0, 0.0, 0.0, 1.0]])
        if c2w.shape != (4, 4) or not np.isfinite(c2w).all():
            return False
        if abs(np.linalg.det(c2w)) < 1e-9:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="/mnt/sfs_turbo/t00845730/re10k/extracted")
    ap.add_argument("--out", default="/mnt/sfs_turbo/t00845730/re10k/manifest_train.jsonl")
    ap.add_argument("--min-views", type=int, default=9)
    args = ap.parse_args()

    scenes = sorted(os.listdir(args.out_root))
    n_ok = n_short = n_bad_pose = 0
    tmp = args.out + ".tmp"
    with open(tmp, "w") as out:
        for scene in scenes:
            pj = os.path.join(args.out_root, scene, "poses.json")
            if not os.path.exists(pj):
                continue
            with open(pj) as fh:
                meta = json.load(fh)
            frames = meta["frames"]
            if len(frames) < args.min_views:
                n_short += 1
                continue
            if not poses_valid(frames):
                n_bad_pose += 1
                continue
            views = []
            for fr in frames:
                c2w = np.array(fr["c2w"], dtype=np.float64)
                if c2w.shape == (3, 4):
                    c2w = np.vstack([c2w, [0.0, 0.0, 0.0, 1.0]])
                views.append({
                    "view_id": fr["view_id"],
                    "rgb_path": f"{scene}/{fr['view_id']}.jpg",
                    "depth_path": f"{scene}/{fr['view_id']}.npz",
                    "K": np.asarray(fr["K"], dtype=np.float64).tolist(),
                    "c2w": c2w.tolist(),
                    "depth_scale": 1.0,
                })
            rec = {"scene_id": scene, "time_id": "t0", "views": views}
            out.write(json.dumps(rec) + "\n")
            n_ok += 1
    os.replace(tmp, args.out)
    print(f"scenes ok={n_ok} too_short={n_short} bad_pose={n_bad_pose} -> {args.out}")


if __name__ == "__main__":
    main()
