"""Cross-arm reading of the C-sweep posterior-mean diagnostic (N=8, val split).

WHY THIS EXISTS ON TOP OF analyze_posterior_mean.py.  That tool reports ONE run
in depth (coverage strata, the k-sweep, the k->inf fit).  The c-sweep question is
comparative and has to survive the same seed noise as every other reading in
report_20260907_c_sweep, so this collapses each run to a few scalars and
pushes them through the IDENTICAL paired-seed pooled-variance t-test used in §4:
d_i is the two seeds' difference for arm i, s_p^2 = sum(d_i^2)/(2*n_arms), and
every contrast shares that one s_p with n_arms df.

THE THREE THINGS IT ANSWERS.

  1. DOES THE RANKING SURVIVE THE BIAS/VARIANCE SPLIT?  Single-draw PSNR is
     b^2 + v; the shift knob and (plausibly) some factors move v without moving
     b^2 at all (t0 §4.2: b3's b^2 changed -0.14% while v fell 7.9%).  psnr_inf
     is -10log10(b^2), so a factor that only sharpens samples gains on psnr_1 and
     nothing on psnr_inf.  A factor that actually learned geometry gains on both.

  2. IS THE ARM MERELY MIS-CALIBRATED?  v/e1 is 50% when calibrated.  Reading it
     per arm says whether a factor moved the model along the bias axis or just
     re-tuned how much it gambles.

  3. WHERE DOES THE SPREAD LIVE?  std_covered vs std_uncovered per view.  A model
     that fuses evidence is confident where evidence exists, so u/c > 1; u/c
     collapsing toward 1 means the covered region is being guessed too.  This is
     an independent read on §7.4's claim about clean_source_depth, because it does
     not go through the DiD regression that §6.3 declared unusable this round.

COMMON VIEW SET.  Cells are compared on the INTERSECTION of (sample_id,
view_index) across every loaded cell, not on each cell's own mean.  The eval set
is frozen and identical by construction, so while the jobs are still running the
intersection is the prefix all cells have reached; without it a cell that is 96
views in would be compared against one that is 36 views in, and the frozen views
differ wildly in difficulty (15.7 dB in the lowest coverage bin vs 15.4 in the
highest, non-monotonically).  --require 192 refuses to print until complete.

THE HARNESS CROSSCHECK IS NOT OPTIONAL.  The first posterior run built the
`sharedrope` cells with shared_rope_frame_ids=False, because
diag_posterior_mean.py did not forward the flag; the parameter shapes are
unchanged, so the checkpoint loaded silently and the cell read 1.45 dB below its
own train.log.  Nothing inside the posterior arithmetic can notice that.  What
notices is comparing psnr_single against tgt_rgb_psnr from the SAME frozen views
in eval_step_*/per_view_metrics.jsonl: a different noise draw moves that by
~0.2 dB, a different MODEL by over a dB.  So it runs by default and any cell over
--crosscheck_tol is reported as suspect before a single number is interpreted.

Usage:
  python tools/analyze_c_posterior_arms.py
  python tools/analyze_c_posterior_arms.py --require 192
"""
import argparse
import json
import math
import os

from analyze_c_sweep_window import (ARMS, CONTRASTS, SEEDS, T05, TBONF, finite,
                                    fmt, mean)
from analyze_posterior_mean import extrapolate, load_rows, max_k

# Order matters only for the printed table.  psnr_inf is the headline: it is the
# bias term, i.e. the part of the score that averaging cannot buy.
METRICS = ["psnr_1", "psnr_N", "psnr_inf", "v_over_e1",
           "std_cov", "std_unc", "u_over_c", "d_gain",
           "lpips_1", "lpips_N"]


def cell_scalars(rows, n):
    """Collapse one run's per-view rows to the comparable scalars.

    Per-view first, then averaged, wherever a difference is involved: the covered
    and uncovered columns are NaN on views with an empty mask, so differencing
    column means would compare two different view populations.
    """
    out = {}
    m1 = mean([r.get("psnr_mean_1") for r in rows])
    mn = mean([r.get(f"psnr_mean_{n}") for r in rows])
    out["psnr_1"], out["psnr_N"] = m1, mn
    minf, frac = extrapolate(m1, mn, n)
    out["psnr_inf"] = minf
    out["v_over_e1"] = frac * 100.0 if finite(frac) else float("nan")
    sc = mean([r.get("std_covered") for r in rows])
    su = mean([r.get("std_uncovered") for r in rows])
    out["std_cov"], out["std_unc"] = sc, su
    out["u_over_c"] = su / sc if finite(sc) and sc > 0 else float("nan")
    # How much MORE dB averaging buys where the splat gives no evidence.  Formed
    # per view so a view missing one region drops out of both terms together.
    d = []
    for r in rows:
        cg = [r.get("covered_mean_1"), r.get(f"covered_mean_{n}")]
        ug = [r.get("uncovered_mean_1"), r.get(f"uncovered_mean_{n}")]
        if all(finite(v) for v in cg + ug):
            d.append((ug[1] - ug[0]) - (cg[1] - cg[0]))
    out["d_gain"] = mean(d)
    out["lpips_1"] = mean([r.get("lpips_single") for r in rows])
    out["lpips_N"] = mean([r.get("lpips_mean") for r in rows])
    return out


def training_eval_psnr(arm, seed, step, log_root):
    """{(sample_id, view_index): tgt_rgb_psnr} from the in-training eval."""
    path = os.path.join(log_root, f"bmvs_c_{arm}_s{seed}",
                        f"eval_step_{step:06d}", "per_view_metrics.jsonl")
    if not os.path.isfile(path):
        return None
    out = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                r = json.loads(line)
                out[(r.get("sample_id"), r.get("view_index"))] = r.get("tgt_rgb_psnr")
    return out


def median(vals):
    vals = sorted(v for v in vals if finite(v))
    if not vals:
        return float("nan")
    m = len(vals) // 2
    return vals[m] if len(vals) % 2 else 0.5 * (vals[m - 1] + vals[m])


def crosscheck(raw, common, step, log_root, tol, advisory=False):
    """Per-cell psnr_single vs the in-training eval on the same views.

    Returns the set of cells that look like a different MODEL rather than a
    different noise draw.  Deliberately not silenceable by a flag that also hides
    the number: the deviation is always printed.

    advisory=True keeps the arithmetic and drops the verdict.  It is for cells
    re-scored at a DIFFERENT eval_flow_shift than the run trained/evaluated with
    (--suffix), where the deviation from train.log is the effect being measured,
    not a defect.  Only the sampler may differ that way -- the model identity gate
    in c_diag_posterior.sbatch still covers the shape-preserving flags.
    """
    rows = []
    for (arm, seed), table in sorted(raw.items()):
        ref = training_eval_psnr(arm, seed, step, log_root)
        if ref is None:
            rows.append((arm, seed, None, None, None, "no in-training eval"))
            continue
        keys = [k for k in common if k in ref and finite(ref[k])
                and finite(table[k].get("psnr_single"))]
        if not keys:
            rows.append((arm, seed, None, None, None, "no overlapping views"))
            continue
        rows.append((arm, seed, len(keys),
                     mean([table[k]["psnr_single"] for k in keys]),
                     mean([ref[k] for k in keys]), None))

    # Centred on the median because two different things live in this delta.  The
    # two harnesses draw their noise from different generators, so any common
    # offset is a property of the COMPARISON, not of any arm (measured at +0.01 dB
    # on the c-sweep run once enough views had landed -- small, but a 36-view
    # prefix put it at -0.17, so it is worth removing rather than assuming away).
    # The bug this catches is per-cell by nature -- one arm's flag was dropped --
    # so the residual after removing the shared offset is the statistic with the
    # signal in it.  Both are printed: a large common offset would itself mean the
    # two harnesses disagree about something, which is also worth knowing.
    offset = median([p - t for _, _, n, p, t, _ in rows if n])
    print("--- harness crosscheck: psnr_single vs train-time tgt_rgb_psnr ---")
    if advisory:
        print(f"    shared offset (median over cells) = {offset:+.3f} dB; ADVISORY ONLY "
              f"(--suffix set: the sampler differs from train.log by design, so delta "
              f"is the eval-shift effect, not a defect)")
    else:
        print(f"    shared offset (median over cells) = {offset:+.3f} dB; a cell is SUSPECT "
              f"when |delta - offset| > {tol} dB")
    suspect = set()
    for arm, seed, n, post, train, err in rows:
        if err:
            print(f"  {arm} s{seed}: {err}")
            continue
        d = post - train
        res = d - offset
        bad = (not advisory) and finite(res) and abs(res) > tol
        if bad:
            suspect.add((arm, seed))
        print(f"  {arm} s{seed}: n={n:>3}  posterior={post:7.3f}  train={train:7.3f}"
              f"  delta={d:+.3f}  residual={res:+.3f}"
              f"{'   <-- SUSPECT: wrong model or wrong sampler' if bad else ''}")
    if suspect:
        print(f"\n  !! {len(suspect)} cell(s) suspect. Do NOT read those rows as a result "
              f"until either\n     the residual shrinks on the full 192 views (draw noise) "
              f"or diag_posterior_mean.py\n     is found to drop a shape-preserving flag "
              f"(shared_rope_frame_ids, clean_source_depth,\n     depth_warp_block, "
              f"per_frame_timestep).")
    print()
    return suspect


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="logs/c_diag_posterior")
    parser.add_argument("--arms", type=str, nargs="+", default=ARMS)
    parser.add_argument("--step", type=int, default=3500)
    parser.add_argument("--suffix", type=str, default="",
                        help="Cell-dir suffix, e.g. '_shift0.25' for a grid re-scored "
                             "at another eval_flow_shift (EVAL_FLOW_SHIFT in "
                             "c_diag_posterior.sbatch). Setting it also downgrades the "
                             "harness crosscheck to advisory, because those cells are "
                             "MEANT to disagree with train.log.")
    parser.add_argument("--require", type=int, default=0,
                        help="Refuse to print unless the common view set has at "
                             "least this many views (192 = a finished eval).")
    parser.add_argument("--log_root", type=str, default="logs",
                        help="Where bmvs_c_<arm>_s<seed>/ live, for the crosscheck.")
    parser.add_argument("--crosscheck_tol", type=float, default=0.5,
                        help="dB. A different noise draw moves psnr_single by ~0.2 dB; "
                             "a different model by over 1 dB.")
    args = parser.parse_args()

    print(f"=== C-sweep posterior mean, cross-arm: {args.root} step {args.step} ===")

    raw, ns = {}, {}
    for arm in args.arms:
        for seed in SEEDS:
            d = os.path.join(args.root,
                             f"{arm}_s{seed}_step{args.step}{args.suffix}")
            rows, paths = load_rows(d)
            if not rows:
                print(f"[skip] {arm} s{seed}: no per_view_posterior.jsonl under {d}")
                continue
            # A re-run appends rather than truncating, so keep the last row per key.
            raw[(arm, seed)] = {(r.get("sample_id"), r.get("view_index")): r
                                for r in rows}
            ns[(arm, seed)] = max_k(rows)

    if not raw:
        print("Nothing loaded.")
        return
    if len(set(ns.values())) > 1:
        print(f"[warn] cells disagree on N: {sorted(set(ns.values()))} -- the k->inf "
              f"fit is per cell, so this is legal but worth knowing.")

    common = set.intersection(*(set(v) for v in raw.values()))
    print(f"    {len(raw)} cells, N={sorted(set(ns.values()))}, "
          f"common view set = {len(common)}/192")
    for (arm, seed), v in sorted(raw.items()):
        if len(v) != len(common):
            print(f"    [partial] {arm} s{seed}: {len(v)} views written, "
                  f"{len(common)} used")
    if len(common) < args.require:
        print(f"\nSTOP: common view set {len(common)} < --require {args.require}. "
              f"Jobs still running.")
        return
    print()

    suspect = crosscheck(raw, common, args.step, args.log_root, args.crosscheck_tol,
                         advisory=bool(args.suffix))

    per_cell = {}
    for key, table in raw.items():
        rows = [table[k] for k in sorted(common)]
        per_cell[key] = cell_scalars(rows, ns[key])

    print("--- per arm (seeds averaged; * = a seed failed the crosscheck) ---")
    print(f"{'arm':<18}" + "".join(f"{m:>13}" for m in METRICS))
    per_arm = {}
    for arm in args.arms:
        vals = []
        for m in METRICS:
            vals.append(mean([per_cell[(arm, s)][m] for s in SEEDS
                              if (arm, s) in per_cell]))
        per_arm[arm] = dict(zip(METRICS, vals))
        if any(finite(v) for v in vals):
            tag = arm + ("*" if any((arm, s) in suspect for s in SEEDS) else "")
            print(f"{tag:<18}" + "".join(fmt(v, 13, 4) for v in vals))

    print(f"\n{'arm':<18}" + "".join(f"{m + ' d(sd)':>13}" for m in METRICS))
    diffs = {m: [] for m in METRICS}
    for arm in args.arms:
        if (arm, SEEDS[0]) not in per_cell or (arm, SEEDS[1]) not in per_cell:
            continue
        row = []
        for m in METRICS:
            d = per_cell[(arm, SEEDS[0])][m] - per_cell[(arm, SEEDS[1])][m]
            row.append(d)
            if finite(d):
                diffs[m].append(d)
        print(f"{arm:<18}" + "".join(fmt(v, 13, 4) for v in row))

    print("\n--- paired-seed pooled-variance t-test (same construction as §4) ---")
    sp = {}
    for m in METRICS:
        d = diffs[m]
        sp[m] = (math.sqrt(sum(x * x for x in d) / (2 * len(d))), len(d)) if d \
            else (float("nan"), 0)
        print(f"  s_p({m}) = {sp[m][0]:.5f}   from {sp[m][1]} arms, df = {sp[m][1]}")
    n_arms = max((n for _, n in sp.values()), default=0)
    n_ctr = sum(1 for a, c in CONTRASTS if a in per_arm and c in per_arm)
    print(f"  df={n_arms}: |t|>{T05.get(n_arms, float('nan')):.3f} single, "
          f"|t|>{TBONF.get((n_arms, n_ctr), float('nan')):.2f} Bonferroni "
          f"({n_ctr} contrasts)")

    for title, as_t in (("t", True), ("delta", False)):
        print(f"\n{'contrast':<30}" + "".join(f"{m + ' ' + title:>13}" for m in METRICS))
        for arm, ctrl in CONTRASTS:
            if arm not in per_arm or ctrl not in per_arm:
                continue
            row = []
            for m in METRICS:
                delta = per_arm[arm][m] - per_arm[ctrl][m]
                s = sp[m][0]
                row.append(delta / s if as_t and finite(s) and s > 0 else delta)
            print(f"{arm + ' - ' + ctrl:<30}"
                  + "".join(fmt(v, 13, 2 if as_t else 4) for v in row))

    print("\nHOW TO READ psnr_1 vs psnr_inf: a factor that gains on psnr_1 and NOT on\n"
          "psnr_inf only narrowed posterior variance -- it did not move the bias, so the\n"
          "gain is a sampling artefact under single-draw scoring and will not survive a\n"
          "better sampler. A factor that gains on psnr_inf learned something.")


if __name__ == "__main__":
    main()
