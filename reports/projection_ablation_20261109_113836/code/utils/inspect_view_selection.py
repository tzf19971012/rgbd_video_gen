"""Diagnose source/target view selection WITHOUT loading any images.

Usage:
    python utils/inspect_view_selection.py \
        --config train_config/normal_config/multi4_nvs_lora_ordered.yaml \
        [--num-samples 64] [--samples-per-scene 10] [--seed 42]

Builds Kubric4DRGBDNVSDataset exactly like the training script does (same cfg
fields), then replays _select_views on the first `--num-samples` dataset slots
with a deterministic rng (random.Random(seed + idx)) -- the identical seeding
rule _select_views itself uses when deterministic_view_seed is set.  No images
are decoded: view records carry K / c2w, which is all the geometry stats need.

Reported per protocol group (pose / video / random):
  video   -- source span indices, interp vs extrapolation fraction of targets,
             rank distance from each target to the nearest source.
  pose    -- greedy camera-path length vs mean pairwise source distance
             (ratio ~1 means the block is walked as a chain, not scattered),
             per-target min axis angle / min centre distance to the sources,
             and an extrapolation proxy (target farther from every source than
             the widest source-source gap).
  random  -- the same target stats under the old protocol, for contrast.
"""

import argparse
import os
import random
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch

from examples.dataset.kubric4d_rgbd_nvs_dataset import Kubric4DRGBDNVSDataset


def cfg_get(cfg, key, default=None):
    if hasattr(cfg, "get"):
        value = cfg.get(key, default)
    else:
        value = getattr(cfg, key, default)
    if value is None and default is not None and key not in cfg:
        return default
    return value


def as_ints(view_ids):
    out = []
    for vid in view_ids:
        try:
            out.append(int(vid))
        except (TypeError, ValueError):
            return None
    return out


def analyse(protocol, record, source_views, target_views, view_index):
    src_idx = sorted(view_index[v.view_id] for v in source_views)
    tgt_idx = [view_index[v.view_id] for v in target_views]
    stats = {"protocol": protocol, "n_source": len(src_idx), "n_target": len(tgt_idx)}
    stats["source_ids_ordered"] = [v.view_id for v in source_views]
    stats["target_ids"] = [v.view_id for v in target_views]

    centers = {v.view_id: v.c2w[:3, 3].float() for v in record["views"]}
    axes = {
        v.view_id: torch.nn.functional.normalize(v.c2w[:3, 2].float(), dim=-1)
        for v in record["views"]
    }
    src_centers = torch.stack([centers[v.view_id] for v in source_views])

    record_ids = as_ints([v.view_id for v in record["views"]])
    if protocol in ("video", "vggt") and record_ids is not None:
        lo, hi = min(src_idx), max(src_idx)
        interp = [lo <= t <= hi for t in tgt_idx]
        stats["src_span"] = (lo, hi)
        stats["interp_frac"] = sum(interp) / len(interp)
        stats["tgt_rank_dist"] = sum(min(abs(t - s) for s in src_idx) for t in tgt_idx) / len(tgt_idx)

    if len(src_centers) > 1:
        pdiff = torch.cdist(src_centers, src_centers)
        n = len(src_centers)
        path_len = sum(
            float((src_centers[i + 1] - src_centers[i]).norm()) for i in range(n - 1)
        )
        mean_pair = float(pdiff[pdiff > 0].mean())
        stats["path_over_mean_pair"] = path_len / mean_pair if mean_pair > 0 else float("nan")
        stats["max_src_gap"] = float(pdiff[pdiff > 0].max())

    angles, dists, extrap = [], [], 0
    for t in target_views:
        d = torch.stack(
            [(centers[s.view_id] - centers[t.view_id]).norm() for s in source_views]
        )
        cos = torch.stack(
            [(axes[s.view_id] * axes[t.view_id]).sum(-1) for s in source_views]
        )
        angles.append(float(torch.rad2deg(torch.acos(cos.clamp(-1.0, 1.0))).min()))
        dists.append(float(d.min()))
        if stats.get("max_src_gap") is not None and float(d.min()) > stats["max_src_gap"]:
            extrap += 1
    stats["tgt_min_angle_deg"] = sum(angles) / len(angles)
    stats["tgt_min_dist"] = sum(dists) / len(dists)
    if stats.get("max_src_gap") is not None:
        stats["extrap_frac_proxy"] = extrap / len(target_views)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from omegaconf import OmegaConf

    cfg = OmegaConf.load(args.config)

    dataset = Kubric4DRGBDNVSDataset(
        kubric4d_root=cfg_get(cfg, "kubric4d_root"),
        manifest_path=cfg_get(cfg, "manifest_path"),
        resolution=tuple(cfg_get(cfg, "resolution", [384, 672])),
        num_source_views=cfg_get(cfg, "num_source_views", 6),
        num_target_views=cfg_get(cfg, "num_target_views", 3),
        random_permute_source_views=cfg_get(cfg, "random_permute_source_views", True),
        view_order=cfg_get(cfg, "view_order", "random"),
        video_tgt_interval_prob=cfg_get(cfg, "video_tgt_interval_prob", 0.2),
        video_max_src_span=cfg_get(cfg, "video_max_src_span", None),
        disparity_norm_mode=cfg_get(cfg, "disparity_norm_mode", "shared_scale"),
        min_view_valid_fraction=cfg_get(cfg, "min_view_valid_fraction", 0.0),
        view_validity_path=cfg_get(cfg, "view_validity_path", None),
        samples_per_scene=cfg_get(cfg, "samples_per_scene", 1),
        split_path=cfg_get(cfg, "split_path", None),
        split=cfg_get(cfg, "train_split", None),
        scene_filter=cfg_get(cfg, "scene_filter", None),
    )

    sps = dataset.samples_per_scene
    total_slots = min(args.num_samples * sps, len(dataset))
    per_protocol = {}
    examples_shown = 0

    for idx in range(total_slots):
        record = dataset.records[(idx // sps) % len(dataset.records)]
        rng = random.Random(args.seed + idx)
        source_views, target_views = dataset._select_views(record, idx=idx)
        protocol = record.get("view_order", "random")
        view_index = {v.view_id: i for i, v in enumerate(record["views"])}
        stats = analyse(protocol, record, source_views, target_views, view_index)
        bucket = per_protocol.setdefault(
            protocol,
            {
                "n": 0,
                "path": [],
                "interp": [],
                "extrap": [],
                "angle": [],
                "dist": [],
                "rank": [],
            },
        )
        bucket["n"] += 1
        if "path_over_mean_pair" in stats:
            bucket["path"].append(stats["path_over_mean_pair"])
        if "interp_frac" in stats:
            bucket["interp"].append(stats["interp_frac"])
            bucket["rank"].append(stats["tgt_rank_dist"])
        if "extrap_frac_proxy" in stats:
            bucket["extrap"].append(stats["extrap_frac_proxy"])
        bucket["angle"].append(stats["tgt_min_angle_deg"])
        bucket["dist"].append(stats["tgt_min_dist"])

        if examples_shown < 4:
            src_ids = [v.view_id for v in source_views]
            print(
                f"[example] protocol={protocol} scene={record['scene_id']}"
                f"{'/t' + str(record.get('time_id')) if record.get('time_id') else ''}\n"
                f"    source (in order): {src_ids}\n"
                f"    target           : {[v.view_id for v in target_views]}"
            )
            examples_shown += 1

    print(f"\n==== {total_slots} slots over {len(dataset.records)} records ====")
    header = (
        f"{'protocol':<8} {'tuples':>6} {'path/meanpair':>13} "
        f"{'interp%':>8} {'extrap%':>8} {'rankdist':>9} {'angle°':>7} {'dist':>7}"
    )
    print(header)
    for protocol, b in sorted(per_protocol.items()):
        mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
        print(
            f"{protocol:<8} {b['n']:>6} {mean(b['path']):>13.3f} "
            f"{100 * mean(b['interp']):>7.1f}% {100 * mean(b['extrap']):>7.1f}% "
            f"{mean(b['rank']):>9.2f} {mean(b['angle']):>7.2f} {mean(b['dist']):>7.3f}"
        )
    print(
        "\npath/meanpair ~1.0 => source block walked as a contiguous camera chain "
        "(video/pose protocols; >>1 or NaN-free random ~ scattered).  interp%: "
        "video targets inside the source time span.  extrap%: proxy = target "
        "farther from every source than the widest source-source gap."
    )


if __name__ == "__main__":
    main()
