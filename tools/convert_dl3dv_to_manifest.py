"""Convert DL3DV_WAI scenes (scene_meta.json) into BlendMVS-compatible manifest.jsonl.

Each scene becomes one JSONL record; every real frame becomes a view with
    rgb_path   -> images_use/frame_xxxxx.png  (undistorted, real file)
    depth_path -> moge/v0/depth/moge2/frame_xxxxx.exr  (MoGe-2 metric depth)
    K          -> from shared fl_x/fl_y/cx/cy (3x3, pixel units, images_use frame)
    c2w        -> transform_matrix verbatim (COLMAP scale, OpenCV convention)

Scenes whose images_use/ is empty (e.g. dead /efs_data symlinks) are skipped.

Usage:
    python tools/convert_dl3dv_to_manifest.py \
        --dl3dv_root /mi/data2T/y00959808/data/sample_data/DL3DV_WAI \
        --out /path/to/dl3dv/manifest.jsonl
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dl3dv_root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = os.path.abspath(args.dl3dv_root)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    scenes = sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and not d.startswith(("_", "copy"))
    )
    n_records, n_skipped = 0, 0
    with open(args.out, "w", encoding="utf-8") as out:
        for scene in scenes:
            sdir = os.path.join(root, scene)
            meta_path = os.path.join(sdir, "scene_meta.json")
            use_dir = os.path.join(sdir, "images_use")
            depth_dir = os.path.join(sdir, "moge", "v0", "depth", "moge2")
            if not (os.path.exists(meta_path) and os.path.isdir(use_dir) and os.path.isdir(depth_dir)):
                print(f"[skip] {scene}: missing scene_meta / images_use / moge2 depth")
                n_skipped += 1
                continue
            available = set(os.listdir(use_dir))
            if not available:
                print(f"[skip] {scene}: images_use is empty (RGB data missing)")
                n_skipped += 1
                continue

            meta = json.load(open(meta_path, encoding="utf-8"))
            K = [[meta["fl_x"], 0.0, meta["cx"]],
                 [0.0, meta["fl_y"], meta["cy"]],
                 [0.0, 0.0, 1.0]]

            views = []
            for fr in meta["frames"]:
                fname = fr["frame_name"] + ".png"  # images_use stores PNG
                if fname not in available:
                    continue
                depth_name = fr["frame_name"] + ".exr"
                if not os.path.exists(os.path.join(depth_dir, depth_name)):
                    continue
                views.append({
                    "view_id": fr["frame_name"],
                    "rgb_path": f"{scene}/images_use/{fname}",
                    "depth_path": f"{scene}/moge/v0/depth/moge2/{depth_name}",
                    "K": K,
                    "c2w": fr["transform_matrix"],
                    "depth_scale": 1.0,
                })
            if len(views) < 8:
                print(f"[skip] {scene}: only {len(views)} usable views")
                n_skipped += 1
                continue

            record = {"scene_id": scene, "time_id": "v0", "views": views}
            out.write(json.dumps(record) + "\n")
            print(f"[ok]   {scene[:16]}: {len(views)} views, {meta['w']}x{meta['h']}")
            n_records += 1

    print(f"\nwrote {n_records} records ({n_skipped} skipped) -> {args.out}")


if __name__ == "__main__":
    main()
