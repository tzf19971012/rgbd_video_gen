"""C-sweep reading harness: window means, the paired-seed t-test, and the
DiD-vs-coverage correlation that `sharedrope` is actually judged on.

Reads the per_view_metrics.jsonl the training loop already writes, so nothing
here needs a GPU.

THREE THINGS, because the plan asks for three different readings and conflating
them is how the b-sweep produced a conclusion it had to walk back:

1. WINDOW MEANS + PAIRED-SEED t-TEST (plan §1).  Each (arm, seed) cell is the
   mean over the window's eval steps; d_i is the two seeds' difference for arm i;
   s_p^2 = sum(d_i^2) / (2 * n_arms) estimates the variance of ONE cell with
   n_arms df, and every contrast shares that one s_p.  With 5 arms, df = 5:
   |t| > 2.571 single, |t| > 3.81 Bonferroni at 4 contrasts.  The reason this is
   2 seeds x 5 arms and not a 2^3 factorial is a8-vs-b1 in the b-sweep: two
   byte-identical configs differed by 0.0099 LPIPS, ~5x the SE the factorial
   design assumed.

2. DiD vs COVERAGE (plan §3.3).  `sharedrope`'s acceptance criterion is NOT the
   mean.  The claim is that aligning the two modalities' RoPE frame ids lets the
   model convert "a source pixel reprojects here" into actual margin, which shows
   up as DiD rising with coverage.  Per view,
       DiD = (psnr_covered - psnr_copy_covered)
           - (psnr_uncovered - psnr_copy_uncovered)
   subtracts the copy baseline separately in each region, so it is a margin over
   difficulty rather than over absolute PSNR -- the covered region is easier for
   everyone, which is exactly what a raw cov-unc gap confounds.  Reported as the
   per-view Pearson r and OLS slope against coverage (binning first throws away
   power; the binned table is printed for readability only).

3. The same for every arm, because a positive slope is only evidence if base's
   is not equally positive.

Usage:
  python tools/analyze_c_sweep_window.py --steps 3500
  python tools/analyze_c_sweep_window.py --steps 2000 2250 2500 --label secondary
"""
import argparse
import json
import math
import os
from collections import defaultdict

ARMS = ["base", "sigma", "cleansrcd", "sharedrope", "cleansrcd_dwarp"]
SEEDS = [42, 1337]
# Which arm each factor is contrasted against.  cleansrcd_dwarp's control is
# cleansrcd, not base: it adds depth_warp_block ON TOP of clean_source_depth
# (plan §3.4), so against base it would measure two factors at once.
CONTRASTS = [
    ("sigma", "base"),
    ("cleansrcd", "base"),
    ("sharedrope", "base"),
    ("cleansrcd_dwarp", "cleansrcd"),
]
HEADLINE = ["tgt_rgb_psnr", "tgt_rgb_lpips", "tgt_depth_delta1", "tgt_depth_abs_rel"]
# Derived per-cell scalars that go through the SAME paired-seed test as the
# headline means, so `sharedrope`'s criterion is testable rather than eyeballed.
# did_slope is the criterion itself (plan §3.3 wants it POSITIVE); did_mean is the
# overall margin, which can improve without the slope turning around at all.
DERIVED = ["did_mean", "did_slope"]
COVERAGE_BINS = [(0.0, 0.20), (0.20, 0.35), (0.35, 0.50), (0.50, 0.65), (0.65, 1.01)]
# Two-sided t critical values, alpha=0.05 and alpha=0.05/n_contrasts.  Module
# level so analyze_c_sweep_log.py runs the identical test on the metrics that
# only exist in train.log.
T05 = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}
TBONF = {(4, 3): 3.96, (4, 4): 4.18, (5, 3): 3.53, (5, 4): 3.81, (6, 4): 3.52}


def finite(x):
    return x is not None and isinstance(x, (int, float)) and not math.isnan(x)


def mean(vals):
    vals = [v for v in vals if finite(v)]
    return float("nan") if not vals else sum(vals) / len(vals)


def load_cell(run_dir, steps):
    """Rows for one (arm, seed) over the requested eval steps.

    Deduplicates on (step, sample_id, view_index): a re-run appends to
    per_view_metrics.jsonl rather than truncating it, so a repeated key is the
    same frozen input scored twice -- keep the last.  Returns None if any
    requested step is missing or short of the full 192 rows, so a half-written
    eval can never be silently averaged in as if it were a complete one.
    """
    by_key = {}
    for step in steps:
        path = os.path.join(run_dir, f"eval_step_{step:06d}", "per_view_metrics.jsonl")
        if not os.path.isfile(path):
            return None, f"missing {path}"
        n = 0
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                by_key[(step, row.get("sample_id"), row.get("view_index"))] = row
                n += 1
        if n < 192:
            return None, f"{path} has {n}/192 rows (eval still running?)"
    return list(by_key.values()), None


def did_rows(rows):
    """(coverage, DiD) pairs for views where all four PSNR terms are finite."""
    out = []
    for r in rows:
        cov, unc = r.get("tgt_rgb_psnr_covered"), r.get("tgt_rgb_psnr_uncovered")
        ccov, cunc = r.get("tgt_rgb_psnr_copy_covered"), r.get("tgt_rgb_psnr_copy_uncovered")
        c = r.get("tgt_rgb_splat_coverage")
        if all(finite(v) for v in (cov, unc, ccov, cunc, c)):
            out.append((c, (cov - ccov) - (unc - cunc)))
    return out


def pearson_ols(pairs):
    """Pearson r, OLS slope (dB per unit coverage), and a two-sided t on r."""
    n = len(pairs)
    if n < 3:
        return float("nan"), float("nan"), float("nan"), n
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in pairs)
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return float("nan"), float("nan"), float("nan"), n
    r = sxy / math.sqrt(sxx * syy)
    slope = sxy / sxx
    # t = r*sqrt(n-2)/sqrt(1-r^2), df = n-2.  Views within a scene are not
    # independent, so treat this as descriptive, not a p-value.
    t = float("nan") if abs(r) >= 1 else r * math.sqrt(n - 2) / math.sqrt(1 - r * r)
    return r, slope, t, n


def fmt(x, w=8, p=3):
    return f"{'--':>{w}}" if not finite(x) else f"{x:>{w}.{p}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--log_root", type=str, default="logs")
    parser.add_argument("--arms", type=str, nargs="+", default=ARMS)
    # Any per-view key can be substituted for the headline four and goes through
    # the identical paired-seed test.  Needed for the SOURCE-view depth metrics:
    # clean_source_depth drops the source-disparity frames from the depth loss
    # (WanRGBDNVSTrainingModule.py:1222-1225), so if that lost supervision is what
    # ails `cleansrcd`, the damage must be worse on src_depth_* than on tgt_depth_*.
    # Random seed noise has no reason to distinguish source from target, so this is
    # the sharpest cheap discriminator between the two explanations.
    parser.add_argument("--metrics", type=str, nargs="+", default=None,
                        help="Override the headline metrics, e.g. "
                             "--metrics src_depth_abs_rel src_depth_delta1 "
                             "tgt_depth_abs_rel tgt_depth_delta1")
    args = parser.parse_args()

    headline = args.metrics or HEADLINE
    # did_mean/did_slope are derived from four fixed PSNR keys, so they are only
    # meaningful alongside the default headline; drop them under --metrics.
    derived = [] if args.metrics else list(DERIVED)

    label = args.label or "+".join(str(s) for s in args.steps)
    print(f"=== C-sweep window reading: steps={args.steps}  label={label} ===\n")

    cells = {}      # (arm, seed) -> rows
    skipped = []
    for arm in args.arms:
        for seed in SEEDS:
            run = os.path.join(args.log_root, f"bmvs_c_{arm}_s{seed}")
            rows, err = load_cell(run, args.steps)
            if rows is None:
                skipped.append((arm, seed, err))
                continue
            cells[(arm, seed)] = rows
    for arm, seed, err in skipped:
        print(f"[skip] {arm} s{seed}: {err}")
    if skipped:
        print()

    metrics = list(headline) + list(derived)

    def cell_value(arm, seed, m):
        rows = cells[(arm, seed)]
        if m in headline:
            return mean([r.get(m) for r in rows])
        pairs = did_rows(rows)
        if m == "did_mean":
            return mean([d for _, d in pairs])
        return pearson_ols(pairs)[1]      # did_slope

    # ---------------------------------------------------------------- 1. means
    print("--- window means (per arm, seeds averaged) ---")
    print(f"{'arm':<18}" + "".join(f"{m.replace('tgt_','') :>20}" for m in metrics))
    per_arm = {}
    per_cell = {}
    for arm in args.arms:
        vals = []
        for m in metrics:
            seed_means = []
            for seed in SEEDS:
                if (arm, seed) in cells:
                    v = cell_value(arm, seed, m)
                    per_cell[(arm, seed, m)] = v
                    seed_means.append(v)
            vals.append(mean(seed_means) if seed_means else float("nan"))
        per_arm[arm] = dict(zip(metrics, vals))
        print(f"{arm:<18}" + "".join(fmt(v, 20, 4) for v in vals))

    print(f"\n{'arm':<18}" + "".join(f"{m.replace('tgt_','')+' d(seed)':>20}" for m in metrics))
    diffs = defaultdict(list)
    for arm in args.arms:
        row = []
        for m in metrics:
            a, b = per_cell.get((arm, 42, m)), per_cell.get((arm, 1337, m))
            d = a - b if finite(a) and finite(b) else float("nan")
            row.append(d)
            if finite(d):
                diffs[m].append(d)
        print(f"{arm:<18}" + "".join(fmt(v, 20, 4) for v in row))

    # ------------------------------------------------------------ 2. t-test
    print("\n--- paired-seed pooled-variance t-test ---")
    sp = {}
    for m in metrics:
        d = diffs[m]
        n = len(d)
        sp[m] = (math.sqrt(sum(x * x for x in d) / (2 * n)), n) if n else (float("nan"), 0)
    for m in metrics:
        s, n = sp[m]
        print(f"  s_p({m}) = {s:.5f}   from {n} arms, df = {n}")
    n_arms = max((n for _, n in sp.values()), default=0)
    n_ctr = sum(1 for a, c in CONTRASTS if a in per_arm and c in per_arm)
    print(f"  df={n_arms}: |t|>{T05.get(n_arms, float('nan')):.3f} single, "
          f"|t|>{TBONF.get((n_arms, n_ctr), float('nan')):.2f} Bonferroni "
          f"({n_ctr} contrasts, alpha={0.05/max(n_ctr,1):.4f})")
    print(f"\n{'contrast':<30}" + "".join(f"{m.replace('tgt_','')+' t':>20}" for m in metrics))
    for arm, ctrl in CONTRASTS:
        if arm not in per_arm or ctrl not in per_arm:
            continue
        row = []
        for m in metrics:
            s, _ = sp[m]
            delta = per_arm[arm][m] - per_arm[ctrl][m]
            row.append(delta / s if finite(delta) and finite(s) and s > 0 else float("nan"))
        print(f"{arm + ' - ' + ctrl:<30}" + "".join(fmt(v, 20, 2) for v in row))

    print(f"\n{'contrast':<30}" + "".join(f"{m.replace('tgt_','')+' delta':>20}" for m in metrics))
    for arm, ctrl in CONTRASTS:
        if arm not in per_arm or ctrl not in per_arm:
            continue
        row = [per_arm[arm][m] - per_arm[ctrl][m] for m in metrics]
        print(f"{arm + ' - ' + ctrl:<30}" + "".join(fmt(v, 20, 4) for v in row))

    # -------------------------------------------- 3. DiD vs coverage (plan §3.3)
    print("\n--- DiD vs coverage: sharedrope's actual acceptance criterion ---")
    print("    DiD = (cov - copy_cov) - (unc - copy_unc), per view, in dB.")
    print("    A POSITIVE slope means the model turns 'evidence exists here' into margin.")
    print(f"\n{'arm':<18}{'seed':>6}{'n':>6}{'DiD mean':>10}{'r':>9}{'slope':>9}{'t(r)':>8}")
    arm_pairs = defaultdict(list)
    for arm in args.arms:
        for seed in SEEDS:
            if (arm, seed) not in cells:
                continue
            pairs = did_rows(cells[(arm, seed)])
            arm_pairs[arm].extend(pairs)
            r, slope, t, n = pearson_ols(pairs)
            dm = mean([p[1] for p in pairs])
            print(f"{arm:<18}{seed:>6}{n:>6}{fmt(dm,10,3)}{fmt(r,9,3)}{fmt(slope,9,3)}{fmt(t,8,2)}")
    print(f"\n{'arm (both seeds)':<18}{'':>6}{'n':>6}{'DiD mean':>10}{'r':>9}{'slope':>9}{'t(r)':>8}")
    for arm in args.arms:
        if arm not in arm_pairs:
            continue
        r, slope, t, n = pearson_ols(arm_pairs[arm])
        dm = mean([p[1] for p in arm_pairs[arm]])
        print(f"{arm:<18}{'':>6}{n:>6}{fmt(dm,10,3)}{fmt(r,9,3)}{fmt(slope,9,3)}{fmt(t,8,2)}")

    print(f"\n--- DiD by coverage bin (readability; the slope above is the test) ---")
    print(f"{'arm':<18}" + "".join(f"{f'[{lo:.2f},{hi:.2f})':>16}" for lo, hi in COVERAGE_BINS))
    for arm in args.arms:
        if arm not in arm_pairs:
            continue
        row = []
        for lo, hi in COVERAGE_BINS:
            row.append(mean([d for c, d in arm_pairs[arm] if lo <= c < hi]))
        print(f"{arm:<18}" + "".join(fmt(v, 16, 3) for v in row))
    print(f"{'n (per bin)':<18}" + "".join(
        f"{sum(1 for c, _ in arm_pairs[args.arms[0]] if lo <= c < hi):>16}"
        for lo, hi in COVERAGE_BINS) if args.arms[0] in arm_pairs else "")

    # -------------------------------------------- 4. headline metrics by coverage
    # Not DiD: the raw mean of each headline metric, binned by the view's own
    # splat coverage.  DiD already isolates the covered-region MARGIN over the
    # copy floor; this answers the plainer question the DiD table doesn't --
    # whether each metric is simply harder or easier as coverage rises, and
    # whether an arm's ranking holds inside every bin or is carried by one of
    # them.  tgt_rgb_splat_coverage is a per-view geometric covariate fixed by
    # the frozen eval set, so its bin populations are identical across arms;
    # printed once under the last metric row for that reason.
    print(f"\n--- headline metrics by coverage bin (pooled seeds, full-frame mean) ---")
    print(f"{'arm':<18}{'metric':<16}" + "".join(f"{f'[{lo:.2f},{hi:.2f})':>12}"
                                                  for lo, hi in COVERAGE_BINS))
    arm_rows = {}
    for arm in args.arms:
        rows = []
        for seed in SEEDS:
            if (arm, seed) in cells:
                rows.extend(cells[(arm, seed)])
        if not rows:
            continue
        arm_rows[arm] = rows
        for m in headline:
            row = []
            for lo, hi in COVERAGE_BINS:
                sel = [r.get(m) for r in rows
                       if finite(r.get("tgt_rgb_splat_coverage"))
                       and lo <= r["tgt_rgb_splat_coverage"] < hi]
                row.append(mean(sel))
            print(f"{arm:<18}{m.replace('tgt_', ''):<16}"
                  + "".join(fmt(v, 12, 3) for v in row))
    if arm_rows:
        rows = next(iter(arm_rows.values()))
        print(f"{'n (per bin)':<18}{'':<16}" + "".join(
            f"{sum(1 for r in rows if finite(r.get('tgt_rgb_splat_coverage')) and lo <= r['tgt_rgb_splat_coverage'] < hi):>12}"
            for lo, hi in COVERAGE_BINS))


if __name__ == "__main__":
    main()
