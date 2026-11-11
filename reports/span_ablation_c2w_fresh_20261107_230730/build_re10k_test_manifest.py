"""Build manifest_test.jsonl for extracted re10k TEST scenes.

Reads the scene keys dumped by re10k_test_prep_worker.py, scans
<mid-out-root>/<scene>/poses.json for those keys only, applies the same
validity filters as build_manifest.py (>= --min-views views, finite
non-singular poses), and writes the manifest.

Usage:
  python build_re10k_test_manifest.py --keys /mnt/sfs_turbo/t00845730/re10k/test_keys.txt \
      --out /mnt/sfs_turbo/t00845730/re10k/manifest_test.jsonl
"""

import argparse
import json
import os

import numpy as np


def poses_valid(frames):
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
    ap.add_argument("--keys", required=True)
    ap.add_argument("--out-root", default="/mnt/sfs_turbo/t00845730/re10k/extracted")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-views", type=int, default=9)
    args = ap.parse_args()

    with open(args.keys) as fh:
        test_keys = {line.strip() for line in fh if line.strip()}
    print(f"test keys seen: {len(test_keys)}")

    n_ok = n_missing = n_short = n_bad_pose = 0
    tmp = args.out + ".tmp"
    with open(tmp, "w") as out:
        for scene in sorted(test_keys):
            pj = os.path.join(args.out_root, scene, "poses.json")
            if not os.path.exists(pj):
                n_missing += 1
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
                if meta.get("camera_convention") != "c2w":
                    c2w = np.linalg.inv(c2w)  # legacy extraction stored DA3 w2c
                views.append({
                    "view_id": fr["view_id"],
                    "rgb_path": f"{scene}/{fr['view_id']}.jpg",
                    "depth_path": f"{scene}/{fr['view_id']}.npz",
                    "K": np.asarray(fr["K"], dtype=np.float64).tolist(),
                    "c2w": c2w.tolist(),
                    "depth_scale": 1.0,
                })
            rec = {"scene_id": scene, "time_id": "t0", "camera_convention": "c2w", "views": views}
            out.write(json.dumps(rec) + "\n")
            n_ok += 1
    os.replace(tmp, args.out)
    print(f"test scenes ok={n_ok} missing={n_missing} too_short={n_short} "
          f"bad_pose={n_bad_pose} -> {args.out}")


if __name__ == "__main__":
    main()
