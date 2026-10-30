"""Bucketed test-set evaluation: interpolation / extrapolation / large-viewpoint.

Each scene contributes THREE tuples (samples_per_scene=3, bucket = idx % 3):
  bucket 0 "interp": sources spread over [15%, 85%] of the trajectory; targets
          are the midpoints of the 3 largest gaps between adjacent sources
          (graded-difficulty interpolation).
  bucket 1 "extrap": sources confined to [30%, 60%]; targets = middle of the
          left tail, middle of the right tail, far end of the longer tail.
  bucket 2 "large":  sources clustered in a 6-view local window; targets = the
          3 views with the largest viewpoint gap (rotation + normalized
          translation) to every source.

The bucket is recovered post-hoc from the sample_id suffix "s{idx}" that the
dataset appends when return_metadata is on, so per_view_metrics.jsonl rows are
self-describing.  Run this script for each arm with the SAME filtered manifest
(written on first invocation) and identical seeds; compare bucket_summary.json.
"""

import argparse
import json
import math
import os
import random
import sys

import numpy as np
import torch

try:
    import torch_npu  # noqa: F401

    torch.npu.set_compile_mode(jit_compile=False)
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from accelerate import Accelerator  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from eval_script.eval_testset import build_model  # noqa: E402
from examples.dataset import Kubric4DRGBDNVSDataset  # noqa: E402
from examples.wanvideo.model_training.train_with_accelerate_rgbd_nvs import (  # noqa: E402
    custom_collate_fn,
)

BUCKETS = ("interp", "extrap", "large")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--kubric4d-root", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-scenes", type=int, default=20)
    parser.add_argument("--min-views", type=int, default=15)
    parser.add_argument("--view-seed", type=int, default=1234)
    parser.add_argument("--save-vis", type=int, default=6)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--eval-step-tag", type=int, default=2000)
    return parser.parse_args()


def _cam_angle_deg(a, b):
    ra, rb = a.c2w[:3, :3].double(), b.c2w[:3, :3].double()
    cos = (torch.trace(ra @ rb.T) - 1) / 2
    return math.degrees(math.acos(max(-1.0, min(1.0, float(cos)))))


def _spread_sources(n, lo, hi, num, rng):
    a = int(round(lo * n))
    b = min(n - 1, int(round(hi * n)))
    if b - a + 1 < num:
        a = max(0, b - num + 1)
    picks = set()
    for i in range(num):
        f = i / (num - 1)
        j = a + int(round(f * (b - a))) + rng.randint(-1, 1)
        picks.add(max(a, min(b, j)))
    pool = [j for j in range(a, b + 1) if j not in picks]
    while len(picks) < num and pool:
        picks.add(pool.pop(len(pool) // 2))
    return sorted(picks)[:num]


def _interp_targets(n, src):
    gaps = []
    for s0, s1 in zip(src[:-1], src[1:]):
        if s1 - s0 >= 2:
            gaps.append((s1 - s0 - 2, (s0 + s1) // 2, s0, s1))
    gaps.sort(reverse=True)
    picked, used = [], set()
    for _, mid, s0, s1 in gaps:
        if mid in used or mid in src:
            continue
        picked.append(mid)
        used.add(mid)
        if len(picked) == 3:
            return picked
    interior = [
        i for i in range(src[0] + 1, src[-1]) if i not in src and i not in used
    ]
    interior.sort(key=lambda i: -min(abs(i - s) for s in src))
    for i in interior:
        if len(picked) == 3:
            break
        picked.append(i)
    return picked


def _extrap_targets(n, src, rng):
    lt = list(range(0, src[0]))
    rt = list(range(src[-1] + 1, n))
    picked = []
    if lt:
        picked.append(lt[len(lt) // 2])
    if rt:
        picked.append(rt[len(rt) // 2])
    end = 0 if len(lt) >= len(rt) else n - 1
    if end not in picked:
        picked.append(end)
    else:
        other = n - 1 if end == 0 else 0
        if other not in picked:
            picked.append(other)
    return picked[:3]


def _cam_dist_norm(a, b, diam):
    ta, tb = a.c2w[:3, 3].double(), b.c2w[:3, 3].double()
    return float(torch.norm(ta - tb)) / max(diam, 1e-8)


def _scene_diam(views):
    ts = [v.c2w[:3, 3].double() for v in views]
    return max(float(torch.norm(p - q)) for p in ts for q in ts)


def _large_targets(src_idx, views):
    # Large viewpoint = far in BOTH rotation and (normalized) translation:
    # re10k aerial/lateral shots rotate little while translating a lot, so a
    # rotation-only score would miss them.
    diam = _scene_diam(views)
    def gap(i):
        return min(
            _cam_angle_deg(views[i], views[s]) + 60.0 * _cam_dist_norm(views[i], views[s], diam)
            for s in src_idx
        )
    src_set = set(src_idx)
    min_sep = max(3, int(0.08 * len(views)))
    cands = [i for i in range(len(views)) if i not in src_set]
    cands.sort(key=lambda i: -gap(i))
    picked = []
    for i in cands:
        if all(abs(i - p) >= min_sep for p in picked):
            picked.append(i)
        if len(picked) == 3:
            break
    for i in cands:
        if len(picked) == 3:
            break
        if i not in picked:
            picked.append(i)
    return picked


class BucketedKubric4D(Kubric4DRGBDNVSDataset):
    """Deterministic bucketed source/target split; bucket = idx % samples_per_scene."""

    bucket_log_path = None

    def _select_views(self, record, idx=0):
        views = sorted(record["views"], key=lambda v: int(v.view_id))
        n = len(views)
        k = int(idx) % max(1, self.samples_per_scene)
        seed = self.deterministic_view_seed if self.deterministic_view_seed is not None else 0
        rng = random.Random(seed + int(idx))

        if k == 0:
            src = _spread_sources(n, 0.15, 0.85, self.num_source_views, rng)
            tgt = _interp_targets(n, src)
        elif k == 1:
            src = _spread_sources(n, 0.30, 0.60, self.num_source_views, rng)
            tgt = _extrap_targets(n, src, rng)
        else:
            w_lo = int(0.05 * n)
            w_hi = max(w_lo, min(int(0.55 * n), n - self.num_source_views))
            w = rng.randint(w_lo, w_hi)
            src = list(range(w, min(n, w + self.num_source_views)))
            if len(src) < self.num_source_views:
                src = list(range(n - self.num_source_views, n))
            tgt = _large_targets(src, views)

        need = self.num_target_views
        if len(tgt) < need:
            raise ValueError(
                f"bucket {BUCKETS[k]} could not draw {need} targets from scene "
                f"{record['scene_id']} (N={n})"
            )
        tgt = tgt[:need]

        diam = _scene_diam(views)
        info_extra = {
            "tgt_min_angle_deg": [
                round(min(_cam_angle_deg(views[t], views[s]) for s in src), 3)
                for t in tgt
            ],
            "tgt_min_dist_frac": [
                round(min(_cam_dist_norm(views[t], views[s], diam) for s in src), 3)
                for t in tgt
            ],
        }

        info = {
            "idx": int(idx),
            "scene_id": record["scene_id"],
            "bucket": BUCKETS[k],
            "n_views": n,
            "src_idx": src,
            "tgt_idx": tgt,
            **info_extra,
        }
        if self.bucket_log_path:
            with open(self.bucket_log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(info) + "\n")

        source_views = [views[i] for i in src]
        target_views = [views[i] for i in tgt]
        if self.random_permute_source_views:
            rng.shuffle(source_views)
        return source_views, target_views


def filter_manifest(manifest, out_path, num_scenes, min_views):
    records = [json.loads(line) for line in open(manifest, encoding="utf-8")]
    kept = [r for r in records if len(r["views"]) >= min_views][:num_scenes]
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in kept:
            fh.write(json.dumps(r) + "\n")
    return out_path, [r["scene_id"] for r in kept]


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.run_config)

    accelerator = Accelerator(mixed_precision=cfg.mixed_precision)
    os.makedirs(args.output, exist_ok=True)
    accelerator.print(f"[eval_bucketed] args: {vars(args)}")

    manifest = args.manifest
    if args.num_scenes > 0:
        manifest, scenes = filter_manifest(
            manifest,
            os.path.abspath(os.path.join(args.output, "manifest_bucketed.jsonl")),
            args.num_scenes,
            args.min_views,
        )
        accelerator.print(f"[eval_bucketed] {len(scenes)} scenes: {scenes}")

    model = build_model(cfg, accelerator)

    from safetensors.torch import load_file

    ckpt_weights = os.path.join(args.ckpt, "model.safetensors")
    state = load_file(ckpt_weights, device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    accelerator.print(
        f"[eval_bucketed] loaded {ckpt_weights}: {len(state)} tensors, "
        f"missing={len(missing)}, unexpected={len(unexpected)}"
    )
    del state

    model.to(accelerator.device)
    if str(getattr(model.pipe, "device", "")) != str(accelerator.device):
        model.pipe.to(accelerator.device)
    model.pipe.torch_dtype = torch.bfloat16
    model.eval()

    roots = args.kubric4d_root or cfg.kubric4d_root
    roots = [r.strip() for r in str(roots).split(",") if r.strip()]

    bucket_log = os.path.join(args.output, "bucket_log.jsonl")
    if os.path.exists(bucket_log):
        os.remove(bucket_log)

    samples_per_scene = len(BUCKETS)
    dataset = BucketedKubric4D(
        kubric4d_root=roots,
        manifest_path=manifest,
        resolution=tuple(cfg.resolution),
        num_source_views=cfg.num_source_views,
        num_target_views=cfg.num_target_views,
        random_permute_source_views=cfg.random_permute_source_views,
        view_order=cfg.get("view_order", "random"),
        video_tgt_interval_prob=cfg.get("video_tgt_interval_prob", 0.2),
        video_max_src_span=cfg.get("video_max_src_span", None),
        disparity_norm_mode=cfg.get("disparity_norm_mode", "shared_scale"),
        disparity_norm_quantile=cfg.get("disparity_norm_quantile", 0.98),
        disparity_norm_headroom=cfg.get("disparity_norm_headroom", 1.5),
        min_view_valid_fraction=cfg.get("min_view_valid_fraction", 0.0),
        view_validity_path=cfg.get("view_validity_path", None),
        samples_per_scene=samples_per_scene,
        split_path=None,
        split=None,
        scene_filter=None,
    )
    dataset.deterministic_view_seed = args.view_seed
    dataset.return_metadata = True
    dataset.bucket_log_path = bucket_log

    indices = list(range(len(dataset)))
    accelerator.print(
        f"[eval_bucketed] scenes={len(dataset.records)} samples={len(indices)}"
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=indices,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        collate_fn=custom_collate_fn,
    )

    all_metrics = []
    done = 0
    for batch in loader:
        metrics = model.evaluate(
            batch,
            output_dir=args.output,
            global_step=args.eval_step_tag,
            max_samples=1,
            save_vis=(done < args.save_vis),
            vis_offset=done,
        )
        all_metrics.append(metrics)
        done += 1
        accelerator.print(f"[eval_bucketed] progress {done}/{len(indices)}: {metrics}")

    # Aggregate per bucket from per_view_metrics.jsonl (rows carry s{idx}).
    step_dir = os.path.join(args.output, f"eval_step_{args.eval_step_tag:06d}")
    rows = []
    pv = os.path.join(step_dir, "per_view_metrics.jsonl")
    if os.path.exists(pv):
        for line in open(pv, encoding="utf-8"):
            row = json.loads(line)
            suffix = row["sample_id"].rsplit("/s", 1)[-1]
            bucket = BUCKETS[int(suffix) % samples_per_scene]
            rows.append((bucket, row))

    summary = {"ckpt": args.ckpt, "manifest": manifest, "num_samples": done, "buckets": {}}
    for bname in BUCKETS:
        brows = [r for b, r in rows if b == bname]
        agg = {}
        for key in {k for r in brows for k in r}:
            if key in ("sample_id", "target_index", "view_index"):
                continue
            vals = [r[key] for r in brows if key in r and not math.isnan(float(r[key]))]
            agg[key] = float(np.mean(vals)) if vals else float("nan")
        agg["_num_views"] = len(brows)
        summary["buckets"][bname] = agg
        accelerator.print(f"\n[eval_bucketed] === {bname} ({len(brows)} views) ===")
        for key in sorted(agg):
            if key.startswith(("tgt_rgb_psnr", "tgt_depth", "src_depth", "tgt_view")):
                accelerator.print(f"  {key}={agg[key]:.4f}")

    with open(os.path.join(args.output, "bucket_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
