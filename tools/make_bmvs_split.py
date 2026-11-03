"""Freeze a scene-level train/val/test split for BlendedMVS.

WHY THIS EXISTS.  Today there is no hold-out at all: the training loop builds
its eval set as

    eval_dataset = copy.copy(train_dataset)      # shares .records
    scene_stride = max(1, num_scenes // eval_max_samples)

so every "eval" number ever reported is a stride-sample of the TRAINING scenes,
and which scenes it lands on is a function of len(records).  When the manifest
grew 113 -> 502 scenes on 2026-09-05 the stride went 3 -> 15 and the eval set
silently became a disjoint scene set -- see report_20260905_t0 §2, which is why
that report had to pin `manifest.jsonl.bak_113scenes_*` to stay comparable to the
b-sweep.  A split derived from a count is not a split; it is a coincidence.  This
writes an explicit list of scene_ids to a version-controlled file so that
growing the manifest again cannot move it.

A SCENE-LEVEL CRITERION, NOT A STRICTER VIEW THRESHOLD.  `min_view_valid_fraction`
filters views by the fraction of the frame carrying valid depth
(utils/compute_view_validity.py).  The first version of this script raised that
threshold to 0.8 for the hold-out; utils/visualize_bmvs_split.py showed that to be
wrong, and the reason is worth stating because it is not visible in any summary
statistic:

  * Per-view validity is largely a CAMERA-POSE proxy.  Within a scene, the angle
    between a view's optical axis and the scene-mean axis correlates -0.33
    (median; negative in 85% of the 495 measurable scenes) with validity, because
    oblique views frame sky and water while nadir views do not.  Median
    obliqueness by validity bin runs 74deg / 72 / 63 / 48 / 31.5.  So a strict
    per-view threshold silently selects nadir views, narrows the source/target
    baseline distribution, and makes the eval set EASIER and less discriminative
    -- the opposite of what a hold-out is for.
  * The confound actually worth removing is scene-level: object-centric turntable
    and tabletop captures, where the depth mask covers only the subject in EVERY
    view.  On those, `_splat_sources_pixel` splats almost nothing, so the
    covered/uncovered stratification the p0a and t0 diagnostics rest on reports
    near-zero coverage for reasons that have nothing to do with view overlap.
  * The two are separable by max-over-views validity: an aerial capture has at
    least one near-nadir, near-fully-valid view, a turntable never does.  The
    distribution is strongly bimodal -- 429 of 502 scenes reach max >= 0.99, and a
    tail of ~50 never passes 0.9.  Visual check at the boundary: every scene up to
    max=0.88 is object-centric bar one genuine aerial resort at 0.87.

  * The gate is COARSE, and that is as good as it gets from validity alone.  It
    removes the 36 captures where no view is ever well-labelled, but a turntable
    shot close enough to fill the frame passes it (a Nikon camera at max=0.963).
    Of the 96 hold-out scenes, 6 sit below max=0.99 and were inspected one by one
    (logs/viz_bmvs_split/holdout_leak_check.png): 1 is plainly object-centric, 3
    are single-subject but well-labelled, 2 are genuinely aerial.  So ~1/64 of val
    is object-centric.  Two mechanical fixes were tried and both FAIL -- do not
    retry them:
      - border validity (fraction of frame-border pixels labelled, max over the
        5 best views): the camera close-up scores 0.894, ABOVE the genuine aerial
        tower scene at 0.692, because a tight close-up labels its borders too.
      - median view validity: the camera scores 0.844 and a reclining statue
        0.915, above EVERY genuine aerial scene measured (0.696 / 0.709 / 0.719 /
        0.733), because aerial captures lose half the frame to sky and water.
        It is anti-correlated with the thing we want.
    Raising the max gate to 0.99 also fails: it would drop real aerial captures at
    0.93-0.97 while still admitting the camera.  The residual therefore has to be
    handled at ANALYSIS time -- coverage-stratified numbers should condition on the
    measured per-sample source validity, which is per-sample knowable, rather than
    assume the scene gate made the hold-out homogeneous.

  * TRAINING needs no filter at all beyond the existing 0.5.  The depth loss is
    already masked per pixel (train_with_accelerate_rgbd_nvs.py: loss_masks[:, 0]
    is the per-pixel valid mask, nearest-downsampled to latent resolution, and
    _masked_mse divides by mask.sum()), so invalid regions contribute exactly zero
    gradient and the remaining valid pixels are renormalised back to full scale.
    RGB is valid everywhere regardless of depth coverage.  Object-centric scenes
    therefore stay in train; they are only barred from the hold-out.

So: ONE view threshold everywhere (0.5), plus a scene-level eligibility gate on
max-over-views validity for val/test.  The split file records both numbers.

SIZING.  From the two paired N=8 posterior cells (96 views / 32 scenes, same
scenes both arms), the per-scene std of the PAIRED PSNR difference is 0.414 dB.
Pairing is what makes this small -- the per-view std of PSNR itself is 2.865 dB,
dominated by scene difficulty, which cancels when both arms see the same scenes.
At 80% power / alpha=0.05 two-tailed the minimum detectable paired difference is

    32 scenes -> 0.21 dB      64 -> 0.15 dB     135 -> 0.10 dB     538 -> 0.05 dB

Today's 32-scene eval cannot resolve the differences the b-sweep was deciding on
(per_frame_timestep was +0.15 dB).  64 is the cheapest size that can.

LEGACY CONTINUITY.  The 32 scenes the historical eval set happened to land on
are forced into the hold-out, so old numbers stay readable going forward.  23 of
them are hold-out eligible; the other 9 are object-centric captures (verified
visually -- a basketball, a bronze ding, several statues) and are held out too but
listed separately, under "val_legacy_low_validity", so they can reproduce the
historical number without polluting the primary val metric.  Those 9 were 28% of
the historical eval set, which is worth remembering when reading any coverage-
stratified number from before this split existed.

Usage:
  python -m tools.make_bmvs_split                      # writes configs/bmvs_split_v1.json
  python -m tools.make_bmvs_split --dry_run            # print the tables only
"""
import argparse
import json
import os
import statistics as st


def load(manifest_path, validity_path):
    records = []
    with open(manifest_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    with open(validity_path, encoding="utf-8") as handle:
        validity = json.load(handle)["scenes"]
    return records, validity


def good_views(record, validity, threshold):
    fracs = validity.get(str(record["scene_id"]), {})
    return [
        view for view in record["views"]
        if fracs.get(str(view.get("view_id", "")), 0.0) >= threshold
    ]


def legacy_eval_scenes(manifest_path, validity_path, num_views, threshold,
                       eval_max_samples, samples_per_scene):
    """The scene_ids the stride-derived eval set landed on, on the OLD manifest.

    Reproduces frozen_eval_indices() exactly: filter, then stride over scenes and
    take the first sample slot of each.  samples_per_scene only scales the raw
    dataset index, so it does not affect which scenes are chosen -- it is taken
    as an argument anyway so a future change to it is visibly accounted for.
    """
    records, validity = load(manifest_path, validity_path)
    kept = [
        str(r["scene_id"]) for r in records
        if len(good_views(r, validity, threshold)) >= num_views
    ]
    stride = max(1, len(kept) // eval_max_samples)
    del samples_per_scene
    return [kept[i] for i in range(0, len(kept), stride)][:eval_max_samples], len(kept)


def describe(name, scene_ids, stats):
    rows = [stats[s] for s in scene_ids if s in stats]
    if not rows:
        return f"{name:<24}{0:>5}"
    counts = sorted(r["n_train"] for r in rows)
    deciles = st.quantiles(counts, n=10) if len(counts) >= 10 else [counts[0]] * 9
    n_old = sum(r["old"] for r in rows)
    return (
        f"{name:<24}{len(rows):>5}{sum(counts):>9}"
        f"{deciles[0]:>8.0f}{st.median(counts):>7.0f}{deciles[8]:>8.0f}"
        f"{st.median([r['median_validity'] for r in rows]):>10.3f}"
        f"{n_old:>7}{100.0 * n_old / len(rows):>7.0f}%"
    )


HEADER = (
    f"{'subset':<24}{'scenes':>5}{'views':>9}"
    f"{'v/s p10':>8}{'med':>7}{'p90':>8}{'med_val':>10}{'old':>7}{'':>8}"
)


def main():
    parser = argparse.ArgumentParser()
    default_root = os.path.join(os.environ.get("DVD_DATA_ROOT", ""), "datasets", "BlendedMVS")
    parser.add_argument("--bmvs_root", default=default_root)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--validity", default=None)
    parser.add_argument("--legacy_manifest", default=None,
                        help="Old manifest whose stride-derived eval set is forced into "
                             "the hold-out. Pass empty to skip legacy continuity.")
    parser.add_argument("--legacy_validity", default=None)
    parser.add_argument("--num_views", type=int, default=7,
                        help="num_source_views + num_target_views; a scene needs this "
                             "many surviving views to form one tuple.")
    parser.add_argument("--view_threshold", type=float, default=0.5,
                        help="min_view_valid_fraction, applied identically to train and "
                             "hold-out. Do NOT raise this for the hold-out: it is a "
                             "camera-pose filter, see the module docstring.")
    parser.add_argument("--holdout_scene_max_validity", type=float, default=0.9,
                        help="A scene is hold-out eligible only if some view reaches this "
                             "validity. Separates object-centric captures from aerial ones "
                             "without touching the per-view pose distribution.")
    parser.add_argument("--n_val", type=int, default=64)
    parser.add_argument("--n_test", type=int, default=32)
    parser.add_argument("--out", default="configs/bmvs_split_v1.json")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    manifest = args.manifest or os.path.join(args.bmvs_root, "manifest.jsonl")
    validity = args.validity or os.path.join(args.bmvs_root, "manifest.view_validity.json")
    legacy_manifest = args.legacy_manifest
    if legacy_manifest is None:
        legacy_manifest = os.path.join(
            args.bmvs_root, "manifest.jsonl.bak_113scenes_20260905_112542")
    legacy_validity = args.legacy_validity or os.path.join(
        args.bmvs_root, "manifest.view_validity.json.bak_113scenes_20260905_112542")

    records, val_map = load(manifest, validity)
    stats = {}
    for record in records:
        scene_id = str(record["scene_id"])
        kept = good_views(record, val_map, args.view_threshold)
        if len(kept) < args.num_views:
            continue
        fracs = val_map.get(scene_id, {})
        # max over ALL views, not just the kept ones: eligibility asks "does this
        # capture ever see a fully-labelled frame", which is a property of the
        # capture, not of whatever the view threshold happened to leave behind.
        all_fracs = [fracs[str(v["view_id"])] for v in record["views"]
                     if str(v["view_id"]) in fracs]
        stats[scene_id] = {
            "n_train": len(kept),
            "max_validity": max(all_fracs) if all_fracs else 0.0,
            "strict_ok": bool(all_fracs)
            and max(all_fracs) >= args.holdout_scene_max_validity,
            "median_validity": st.median(
                [fracs.get(str(v["view_id"]), 0.0) for v in kept]),
            "old": False,
        }

    legacy, legacy_pool = [], 0
    if legacy_manifest and os.path.isfile(legacy_manifest):
        legacy, legacy_pool = legacy_eval_scenes(
            legacy_manifest, legacy_validity, args.num_views,
            args.view_threshold, args.n_test, 1)
        old_records, _ = load(legacy_manifest, legacy_validity)
        for record in old_records:
            scene_id = str(record["scene_id"])
            if scene_id in stats:
                stats[scene_id]["old"] = True

    missing = [s for s in legacy if s not in stats]
    if missing:
        raise SystemExit(f"legacy eval scenes absent from the new manifest: {missing}")

    # Hold-out membership.  Legacy scenes are pinned in regardless of threshold so
    # historical numbers stay readable; the ones that fail the strict threshold are
    # held out but quarantined out of the primary val metric.
    legacy_strict = [s for s in legacy if stats[s]["strict_ok"]]
    legacy_low = [s for s in legacy if not stats[s]["strict_ok"]]

    # Stratified systematic sample over the strict-eligible remainder.  Sorting by
    # (old, n_views) and taking evenly spaced ranks balances both axes at once:
    # the new 389 scenes are systematically EASIER than the old 113 (median
    # validity 0.997 vs 0.974, median 188 views vs 96), so a hold-out drawn
    # without stratifying on origin would be optimistically biased.
    pool = sorted(
        (s for s, v in stats.items() if v["strict_ok"] and s not in set(legacy)),
        key=lambda s: (stats[s]["old"], stats[s]["n_train"], s),
    )
    need = (args.n_val - len(legacy_strict)) + args.n_test
    if need > len(pool):
        raise SystemExit(f"need {need} strict-eligible scenes, pool has {len(pool)}")
    step = len(pool) / float(need)
    picks = [pool[min(len(pool) - 1, int((i + 0.5) * step))] for i in range(need)]
    picks = list(dict.fromkeys(picks))
    # Deal into val/test in proportion to what each still needs, walking picks in
    # stratified order.  A fixed stride (say picks[2::3]) would give test the
    # wrong count whenever n_val - len(legacy_strict) != 2 * n_test, and would
    # hand val and test different slices of the sort key.
    need_val = args.n_val - len(legacy_strict)
    extra_val, test = [], []
    for scene_id in picks:
        take_val = (len(extra_val) + 1) / max(need_val, 1) <= (len(test) + 1) / max(args.n_test, 1)
        if take_val and len(extra_val) < need_val:
            extra_val.append(scene_id)
        elif len(test) < args.n_test:
            test.append(scene_id)
        elif len(extra_val) < need_val:
            extra_val.append(scene_id)
    val = sorted(set(legacy_strict) | set(extra_val))
    test = sorted(test)
    heldout = set(val) | set(test) | set(legacy_low)
    train = sorted(s for s in stats if s not in heldout)

    assert not (set(val) & set(test)), "val/test overlap"
    assert not (set(train) & heldout), "train/holdout overlap"

    print(f"manifest        {os.path.basename(manifest)}  {len(records)} records")
    print(f"view threshold  {args.view_threshold} (train AND hold-out)  -> "
          f"{len(stats)} scenes usable")
    print(f"hold-out gate   max view validity >= {args.holdout_scene_max_validity}  -> "
          f"{sum(v['strict_ok'] for v in stats.values())} scenes eligible, "
          f"{sum(not v['strict_ok'] for v in stats.values())} object-centric "
          f"(train-only)")
    print(f"legacy eval set {len(legacy)} scenes (from a {legacy_pool}-scene pool); "
          f"{len(legacy_strict)} hold-out eligible, {len(legacy_low)} object-centric "
          f"({100.0 * len(legacy_low) / max(len(legacy), 1):.0f}% of the historical eval set)")
    print()
    print(HEADER)
    print("-" * len(HEADER))
    for name, ids in (("ALL usable", sorted(stats)), ("train", train), ("val", val),
                      ("test", test), ("val_legacy_low_validity", legacy_low)):
        print(describe(name, ids, stats))
    print("-" * len(HEADER))
    print(f"held out: {len(heldout)} scenes "
          f"({100.0 * len(heldout) / len(stats):.1f}% of usable)")

    payload = {
        "version": 1,
        "created": "2026-09-05",
        "manifest": os.path.basename(manifest),
        "num_views": args.num_views,
        "view_threshold": args.view_threshold,
        "holdout_scene_max_validity": args.holdout_scene_max_validity,
        "note": "Scene-level split. Use min_view_valid_fraction=view_threshold for "
                "EVERY split -- raising it for the hold-out biases the camera-pose "
                "distribution toward nadir views (see tools/make_bmvs_split.py). "
                "val/test were restricted to scenes with some view reaching "
                "holdout_scene_max_validity, which excludes object-centric "
                "captures; those stay in train. val + val_legacy_low_validity "
                "reproduces the historical 32-scene eval set.",
        "train": train,
        "val": val,
        "test": test,
        "val_legacy_low_validity": legacy_low,
        "legacy_eval_32": legacy,
    }
    if args.dry_run:
        print("\n--dry_run: nothing written")
        return
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)
        handle.write("\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
