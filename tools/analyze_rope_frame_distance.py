"""D0 of reports/plan_20260906_rope_geometry.md: does the RoPE frame axis
carry positional information the model actually uses?

random_permute_source_views shuffles source order on every sample, so which
frame index the geometrically-nearest source view lands on is randomised BY
CONSTRUCTION -- a natural experiment that is already sitting in the frozen
eval set, no training required.  This reads the per_view_metrics.jsonl the
C-sweep arms have already written, reconstructs each row's true
source/target view assignment offline (sample_id -> scene_id/time_id/idx ->
Kubric4DRGBDNVSDataset._select_views, same RNG the training run used because
deterministic_view_seed=eval_view_seed is frozen), and regresses target-view
quality on

    delta_frame = |view_index_tgt - frame_index_of_nearest_source|

controlling for the geometric-difficulty covariates each row already carries
(tgt_view_min_src_dist, tgt_view_min_src_angle_deg, tgt_rgb_splat_coverage) --
without controlling for these, a delta_frame effect could just be "hard
targets happen to fall at certain frame offsets", not a RoPE effect.

See reports/plan_20260906_rope_geometry.md §D0 for the full design and its
power caveat: the eval set is frozen, so delta_frame and the controls are
IDENTICAL across every arm and checkpoint for a given (sample_id,
target_index) -- the independent sample size is the ~192 (scene, target)
pairs, not (arms x checkpoints x 192).  Repeated measurements are averaged
per pair before regressing to reflect that; treat the t-statistics below as
descriptive (views are not independent draws), same caveat as
analyze_c_sweep_window.pearson_ols.

Usage:
  python tools/analyze_rope_frame_distance.py logs/bmvs_c_*
"""
import argparse
import glob
import json
import math
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from examples.dataset.kubric4d_rgbd_nvs_dataset import Kubric4DRGBDNVSDataset  # noqa: E402

_DATA_ROOT = os.environ.get("DVD_DATA_ROOT", "")
DEFAULT_MANIFEST = os.path.join(_DATA_ROOT, "datasets/BlendedMVS/manifest.jsonl")
DEFAULT_KUBRIC_ROOT = os.path.join(_DATA_ROOT, "datasets/BlendedMVS/BlendedMVS")
DEFAULT_SPLIT_PATH = "configs/bmvs_split_v1.json"

METRICS = ("tgt_rgb_lpips", "tgt_rgb_psnr", "tgt_depth_abs_rel", "tgt_depth_delta1")
CONTROLS = ("tgt_view_min_src_dist", "tgt_view_min_src_angle_deg", "tgt_rgb_splat_coverage")


def load_rows(roots):
    """Same dedup discipline as analyze_coverage_strat.load_rows: key on
    (path, sample_id, view_index) so the frozen eval set's repeated keys
    across arms/checkpoints stay distinct cells, while a re-run appending to
    one file collapses to its last write."""
    by_key = {}
    for root in roots:
        files = sorted(glob.glob(os.path.join(root, "**", "per_view_metrics.jsonl"), recursive=True))
        for path in files:
            arm = os.path.basename(os.path.normpath(root))
            m = re.search(r"eval_step_(\d+)", path)
            step = int(m.group(1)) if m else -1
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    row["_arm"] = arm
                    row["_step"] = step
                    by_key[(path, row.get("sample_id"), row.get("view_index"))] = row
    return list(by_key.values())


def parse_sample_id(sample_id):
    """'<scene_id>/t<time_id>/s<idx>' -> (scene_id, time_id, idx).

    The 't' prefix doubles up when time_id itself already starts with 't'
    (BlendedMVS stores time_id="t0"), see
    kubric4d_rgbd_nvs_dataset.py:555-559 -- that is reproduced here, not
    worked around, so the string round-trips exactly.
    """
    parts = sample_id.split("/")
    idx = int(parts[-1][1:])  # strip leading 's'
    if len(parts) == 3:
        scene_id, time_id = parts[0], parts[1][1:]  # strip leading 't'
    else:
        scene_id, time_id = parts[0], None
    return scene_id, time_id, idx


def build_dataset(args):
    return Kubric4DRGBDNVSDataset(
        kubric4d_root=args.kubric4d_root,
        manifest_path=args.manifest_path,
        num_source_views=args.num_source_views,
        num_target_views=args.num_target_views,
        random_permute_source_views=True,
        deterministic_view_seed=args.eval_view_seed,
        min_view_valid_fraction=args.min_view_valid_fraction,
        samples_per_scene=args.samples_per_scene,
        split_path=args.split_path,
        split="val",
    )


def nearest_source_index(source_views, target_view):
    """argmin center distance, matching WanRGBDNVSTrainingModule.py:2130-2131
    exactly (rotation ignored on purpose there too)."""
    centers = torch.stack([v.c2w[:3, 3] for v in source_views])
    d = (centers - target_view.c2w[:3, 3]).norm(dim=-1)
    idx = int(torch.argmin(d).item())
    return idx, float(d[idx].item())


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float("nan") if not xs else sum(xs) / len(xs)


def ols(X, y):
    """Plain multiple OLS with intercept.  beta[0] is the intercept, beta[1:]
    line up with X's columns.  Homoskedastic SE -- there is no clustering by
    scene here because the input is already one row per (scene, target)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = X.shape[0]
    Xd = np.column_stack([np.ones(n), X])
    beta, _, _, _ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ beta
    dof = n - Xd.shape[1]
    if dof <= 0:
        return beta, np.full_like(beta, float("nan")), np.full_like(beta, float("nan")), dof
    sigma2 = float(resid @ resid) / dof
    try:
        cov = sigma2 * np.linalg.inv(Xd.T @ Xd)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full_like(beta, float("nan"))
    with np.errstate(invalid="ignore", divide="ignore"):
        t = beta / se
    return beta, se, t, dof


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("roots", nargs="+", help="one directory per arm, e.g. logs/bmvs_c_*")
    ap.add_argument("--manifest-path", default=DEFAULT_MANIFEST)
    ap.add_argument("--kubric4d-root", default=DEFAULT_KUBRIC_ROOT)
    ap.add_argument("--split-path", default=DEFAULT_SPLIT_PATH)
    ap.add_argument("--num-source-views", type=int, default=6)
    ap.add_argument("--num-target-views", type=int, default=3)
    ap.add_argument("--samples-per-scene", type=int, default=10)
    ap.add_argument("--eval-view-seed", type=int, default=1234)
    ap.add_argument("--min-view-valid-fraction", type=float, default=0.5)
    ap.add_argument("--metrics", nargs="+", default=list(METRICS))
    args = ap.parse_args()

    rows = load_rows(args.roots)
    if not rows:
        print(f"no per_view_metrics.jsonl found under {args.roots}", file=sys.stderr)
        sys.exit(1)

    ds = build_dataset(args)
    by_scene = {(r["scene_id"], r.get("time_id")): r for r in ds.records}

    select_cache = {}
    per_pair = {}
    missing = 0

    for row in rows:
        scene_id, time_id, idx = parse_sample_id(row["sample_id"])
        record = by_scene.get((scene_id, time_id))
        if record is None:
            missing += 1
            continue
        cache_key = (scene_id, time_id, idx)
        if cache_key not in select_cache:
            select_cache[cache_key] = ds._select_views(record, idx=idx)
        source_views, target_views = select_cache[cache_key]
        tv = row["target_index"]
        nearest_idx, computed_min_dist = nearest_source_index(source_views, target_views[tv])
        delta_frame = abs(row["view_index"] - nearest_idx)

        key = (row["sample_id"], tv)
        cell = per_pair.get(key)
        if cell is None:
            logged_min_dist = row.get("tgt_view_min_src_dist")
            dist_ratio = (
                computed_min_dist / logged_min_dist
                if logged_min_dist not in (None, 0) and not math.isnan(logged_min_dist)
                else float("nan")
            )
            cell = {
                "scene_id": scene_id,
                "delta_frame": delta_frame,
                "dist_ratio": dist_ratio,
                "controls": {c: row.get(c) for c in CONTROLS},
                "metrics": {m: [] for m in args.metrics},
            }
            per_pair[key] = cell
        for m in args.metrics:
            cell["metrics"][m].append(row.get(m))

    if missing:
        print(
            f"warning: {missing} rows had no matching (scene_id, time_id) in the "
            "reconstructed dataset -- check split/filtering flags match the run",
            file=sys.stderr,
        )

    # Internal consistency gate.  All targets of one scene share a single
    # disparity_norm_mode="shared_scale" scalar (derived once per __getitem__
    # call from the source depths, and eval draws exactly one idx per scene),
    # so computed_min_dist / logged_min_dist must equal that scalar's inverse
    # for EVERY target in the scene.  A pair whose ratio departs from its
    # scene's median by more than a few percent is not measurement noise (the
    # ~1% spread seen elsewhere is consistent with bf16 rounding in the
    # training-time distance) -- it means this script picked the wrong
    # source/target view for that one pair, so it is dropped rather than fed
    # into the regression as if it were a real delta_frame.  This is the
    # guard against plan §6-2's silent failure mode.
    by_scene_ratios = {}
    for cell in per_pair.values():
        if math.isfinite(cell["dist_ratio"]):
            by_scene_ratios.setdefault(cell["scene_id"], []).append(cell["dist_ratio"])
    scene_median = {s: sorted(rs)[len(rs) // 2] for s, rs in by_scene_ratios.items()}

    dropped = []
    kept = {}
    for key, cell in per_pair.items():
        med = scene_median.get(cell["scene_id"])
        r = cell["dist_ratio"]
        if med and math.isfinite(r) and abs(r - med) / med > 0.03:
            dropped.append((key, r, med))
        else:
            kept[key] = cell
    per_pair = kept
    print(
        f"[sanity] {len(dropped)}/{len(dropped) + len(per_pair)} (scene, target) pairs "
        "dropped: computed/logged min-source-distance ratio departs >3% from "
        "the scene's median (reconstruction mismatch for that one pair, not a "
        "weak effect)"
    )
    for (sample_id, tv), r, med in dropped:
        print(f"    {sample_id} tv={tv}: ratio={r:.4f} vs scene median={med:.4f}", file=sys.stderr)

    n_cells = len({(r["_arm"], r["_step"]) for r in rows})
    print(
        f"[data] {len(rows)} rows across {n_cells} (arm, step) cells -> "
        f"{len(per_pair)} independent (scene, target) pairs after averaging and dropping"
    )

    delta_arr = np.array([cell["delta_frame"] for cell in per_pair.values()], dtype=np.float64)
    X_controls = np.array(
        [[cell["controls"].get(c) for c in CONTROLS] for cell in per_pair.values()], dtype=np.float64
    )
    metric_cols = {
        m: np.array([mean(cell["metrics"][m]) for cell in per_pair.values()], dtype=np.float64)
        for m in args.metrics
    }

    print()
    print(f"{'metric':<20}{'n':>5}   {'raw slope':>10}{'raw t':>8}   "
          f"{'partial slope':>14}{'partial t':>10}   controls: " + ", ".join(CONTROLS))
    for m in args.metrics:
        y = metric_cols[m]
        base_mask = np.isfinite(y) & np.isfinite(delta_arr)
        full_mask = base_mask & np.all(np.isfinite(X_controls), axis=1)

        beta_raw, _, t_raw, _ = ols(delta_arr[base_mask, None], y[base_mask])
        beta_full, _, t_full, _ = ols(
            np.column_stack([delta_arr[full_mask], X_controls[full_mask]]), y[full_mask]
        )
        print(
            f"{m:<20}{full_mask.sum():>5}   {beta_raw[1]:>10.4f}{t_raw[1]:>8.2f}   "
            f"{beta_full[1]:>14.4f}{t_full[1]:>10.2f}"
        )


if __name__ == "__main__":
    main()
