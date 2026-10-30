"""Merge sharded eval_testset outputs into one summary_metrics.json.

Usage: python merge_shard_metrics.py logs/test_eval/re10k_test_s1600_shard0 \
           logs/test_eval/re10k_test_s1600_shard1 [...] [--out merged.json]
"""

import argparse
import json
import math
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shard_dirs", nargs="+")
    ap.add_argument("--out", default=None,
                    help="output json path (default: <first_dir_wo_shardN>/summary_metrics.json)")
    args = ap.parse_args()

    per_sample = []
    meta = {}
    for d in args.shard_dirs:
        with open(os.path.join(d, "summary_metrics.json"), encoding="utf-8") as fh:
            s = json.load(fh)
        meta.setdefault("ckpt", s["ckpt"])
        meta.setdefault("manifest", s["manifest"])
        meta["num_samples"] = meta.get("num_samples", 0) + s["num_samples"]
        per_sample.extend(s.get("per_sample", []))

    keys = {k for m in per_sample for k in m}
    merged = {}
    for key in keys:
        vals = [m[key] for m in per_sample if key in m and not math.isnan(float(m[key]))]
        merged[key] = float(np.mean(vals)) if vals else float("nan")
    for side in ("src", "tgt"):
        d1 = merged.get(f"{side}_depth_delta1")
        ramp = merged.get(f"{side}_depth_delta1_ramp")
        if d1 is not None and ramp is not None and 1.0 - ramp > 1e-3:
            merged[f"{side}_depth_delta1_skill_agg"] = (d1 - ramp) / (1.0 - ramp)

    out = args.out
    if out is None:
        base = os.path.dirname(args.shard_dirs[0].rstrip("/")) or "."
        name = os.path.basename(args.shard_dirs[0].rstrip("/"))
        name = name.rsplit("_shard", 1)[0]
        out = os.path.join(base, name, "summary_metrics.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"ckpt": meta.get("ckpt"), "manifest": meta.get("manifest"),
                   "num_samples": meta.get("num_samples"), "num_shards": len(args.shard_dirs),
                   "metrics": merged}, fh, indent=2)
    print(f"merged {len(per_sample)} samples from {len(args.shard_dirs)} shards -> {out}")
    for k in sorted(merged):
        print(f"  {k}={merged[k]:.4f}")


if __name__ == "__main__":
    main()
