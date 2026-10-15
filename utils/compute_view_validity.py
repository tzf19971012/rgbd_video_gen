"""Precompute a per-view depth-validity fraction sidecar for a BlendedMVS manifest.

Some BlendedMVS captures only reconstruct a foreground object, so large parts
of many frames have no geometry at all (turntable captures on a black backdrop
are the extreme case: >90% of every frame is empty).  Those views give the
depth branch almost nothing to learn from while still contributing a mostly
empty VAE target.

This script measures, for every view, the fraction of pixels that carry a
valid depth value, and writes it next to the manifest so the dataset can filter
views without paying the cost at every DataLoader init.

Two measurement sources:

  masked_jpg (default)
      BlendedMVS ships ``blended_images/<id>_masked.jpg``, which is the blended
      image with all mesh-uncovered pixels blacked out.  Its black region
      matches the depth-invalid region almost exactly (IoU ~0.997 measured on
      this dataset).  Decoding it at 1/8 scale via PIL's DCT-domain ``draft()``
      is ~60x cheaper than loading the depth ``.npy``.
      Accuracy vs. the .npy on 120 random views: mean abs err 0.009,
      max 0.099, Pearson r 0.998.

  depth
      Loads the depth map itself.  Exact, but reads hundreds of GB.  Use it if
      you need views near the decision boundary classified exactly.

Usage:
    python utils/compute_view_validity.py --manifest /path/to/manifest.jsonl \
        --root /path/to/BlendedMVS
"""

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
from PIL import Image


def default_sidecar_path(manifest_path):
    base = os.path.splitext(manifest_path)[0]
    return f"{base}.view_validity.json"


def _valid_fraction_masked_jpg(abs_rgb_path):
    masked = os.path.splitext(abs_rgb_path)[0] + "_masked.jpg"
    if not os.path.isfile(masked):
        return None
    try:
        image = Image.open(masked)
        # DCT-domain 1/8 downscale; the fraction of non-black pixels is
        # preserved to well within the accuracy we need here.
        image.draft("L", (max(1, image.width // 8), max(1, image.height // 8)))
        gray = np.asarray(image.convert("L"))
    except Exception:
        return None
    return float((gray > 8).mean())


def _valid_fraction_depth(abs_depth_path):
    if not os.path.isfile(abs_depth_path):
        return None
    try:
        depth = np.load(abs_depth_path)
    except Exception:
        return None
    return float((np.isfinite(depth) & (depth > 0)).mean())


def _work(job):
    scene_id, view_id, abs_rgb, abs_depth, source = job
    if source == "depth":
        frac = _valid_fraction_depth(abs_depth)
    else:
        frac = _valid_fraction_masked_jpg(abs_rgb)
    return scene_id, view_id, frac


def _read_records(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as handle:
        text = handle.read().strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", required=True, help="Dataset root the manifest paths are relative to.")
    parser.add_argument("--output", default=None, help="Defaults to <manifest>.view_validity.json")
    parser.add_argument("--source", default="masked_jpg", choices=["masked_jpg", "depth"])
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    output = args.output or default_sidecar_path(args.manifest)
    records = _read_records(args.manifest)

    jobs = []
    for record_idx, record in enumerate(records):
        scene_id = str(record.get("scene_id", record.get("scene", record_idx)))
        views = record.get("views") or [record]
        for view_idx, view in enumerate(views):
            rgb_rel = view.get("rgb_path") or view.get("image_path")
            depth_rel = view.get("depth_path") or view.get("metric_depth_path")
            if rgb_rel is None:
                continue
            view_id = str(view.get("view_id", view.get("camera_id", view.get("cam_id", view_idx))))
            jobs.append(
                (
                    scene_id,
                    view_id,
                    os.path.join(args.root, rgb_rel),
                    os.path.join(args.root, depth_rel) if depth_rel else "",
                    args.source,
                )
            )

    print(f"scenes={len(records)} views={len(jobs)} source={args.source}")
    with Pool(args.workers) as pool:
        results = pool.map(_work, jobs, chunksize=64)

    scenes = {}
    missing = 0
    for scene_id, view_id, frac in results:
        if frac is None:
            missing += 1
            continue
        scenes.setdefault(scene_id, {})[view_id] = round(frac, 5)

    payload = {
        "source": args.source,
        "manifest": os.path.abspath(args.manifest),
        "scenes": scenes,
    }
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    fractions = np.array([f for views in scenes.values() for f in views.values()])
    print(f"wrote {output}")
    print(f"  measured {len(fractions)} views ({missing} unmeasurable)")
    for quantile in (0, 10, 25, 50, 75, 90, 100):
        print(f"  p{quantile:<3d} = {np.percentile(fractions, quantile):.4f}")
    for threshold in (0.3, 0.5, 0.7, 0.9):
        keep = int((fractions >= threshold).sum())
        scenes_ok = sum(
            1 for views in scenes.values() if sum(f >= threshold for f in views.values()) >= 7
        )
        print(
            f"  >= {threshold:.0%}: {keep}/{len(fractions)} views  "
            f"({scenes_ok}/{len(scenes)} scenes keep >= 7 views)"
        )


if __name__ == "__main__":
    main()
