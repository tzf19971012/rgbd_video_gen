"""Stratify target-view RGB metrics by splat coverage and by view difficulty.

Answers the question §7 of report_20260904_b_sweep_2x2x2.md could not:
the model loses to a zero-parameter forward splat on full-frame PSNR, but only
~45% of target pixels have any source pixel reprojecting into them.  A
full-frame average therefore mixes a *determined* region (the answer is in the
source views) with an *unobserved* one (the answer is a guess), and a single
number cannot tell a geometry failure from an impossible task.

Reads the per-view JSONL that WanRGBDNVSTrainingModule.evaluate() writes and
reports:

  1. covered vs uncovered PSNR against the copy floor and the VAE ceiling
  2. the same, binned by per-view splat coverage
  3. the same, binned by the angle to the best-aligned source view

Usage:
  python tools/analyze_coverage_strat.py runs/p0a/b3_step4500
  python tools/analyze_coverage_strat.py runs/p0a/*            # one row per arm
"""
import argparse
import glob
import json
import math
import os
import sys


def load_rows(root):
    """Read every per_view_metrics.jsonl under root, deduplicated.

    The harness scores the frozen eval set in chunks that each land in their own
    eval_step_* directory, so a cell's rows are spread over several files; a
    re-run appends to those files rather than truncating them.  Within one file,
    (sample_id, view_index) is the identity of a scored view and the eval inputs
    are frozen by eval_view_seed, so a duplicate key is a repeat of the same
    measurement -- keep the last, which is also the one carrying any metric keys
    added since the first pass.

    The file path has to be part of the key.  The eval set is frozen across arms
    and checkpoints by design, so (sample_id, view_index) is IDENTICAL in every
    cell -- pooling nine cells under one root with a path-free key silently
    collapsed 864 rows to 96 and reported the last cell as if it were all of
    them.  Keying on the path dedups the in-file repeat, which is the actual
    problem this guards against, while keeping distinct cells distinct.
    """
    by_key = {}
    files = sorted(glob.glob(os.path.join(root, "**", "per_view_metrics.jsonl"),
                             recursive=True))
    for path in files:
        # Cell = the (arm, step, sampler) directory holding the eval_step_* chunks,
        # i.e. everything above eval_step_*/per_view_metrics.jsonl.
        cell = os.path.relpath(os.path.dirname(os.path.dirname(path)), root)
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row["_cell"] = cell
                by_key[(path, row.get("sample_id"), row.get("view_index"))] = row
    return list(by_key.values()), files


def mean(values):
    vals = [v for v in values if v is not None and not math.isnan(v)]
    return float("nan") if not vals else sum(vals) / len(vals)


def col(rows, key):
    return mean([r.get(key) for r in rows])


def fmt(x, width=7, prec=2):
    return f"{'--':>{width}}" if math.isnan(x) else f"{x:>{width}.{prec}f}"


HEAD = (
    f"{'stratum':<22}{'n':>5}{'cover':>7}"
    f"{'PSNR':>8}{'cov':>8}{'unc':>8}"
    f"{'copy':>8}{'cp_cov':>8}{'splat':>8}{'sp_cov':>8}{'VAEcov':>8}{'LPIPS':>8}"
)


def line(label, rows):
    if not rows:
        return f"{label:<22}{0:>5}" + " " * (len(HEAD) - 27)
    return (
        f"{label:<22}{len(rows):>5}"
        + fmt(col(rows, "tgt_rgb_splat_coverage"), 7, 3)
        + fmt(col(rows, "tgt_rgb_psnr"), 8)
        + fmt(col(rows, "tgt_rgb_psnr_covered"), 8)
        + fmt(col(rows, "tgt_rgb_psnr_uncovered"), 8)
        + fmt(col(rows, "tgt_rgb_psnr_copy"), 8)
        + fmt(col(rows, "tgt_rgb_psnr_copy_covered"), 8)
        + fmt(col(rows, "tgt_rgb_psnr_splat"), 8)
        + fmt(col(rows, "tgt_rgb_psnr_splat_valid"), 8)
        + fmt(col(rows, "tgt_rgb_psnr_vae_covered"), 8)
        + fmt(col(rows, "tgt_rgb_lpips"), 8, 4)
    )


def paired(rows, *keys):
    """Per-view values of `keys`, keeping only rows where all of them are finite.

    Differences must be formed per view and then averaged, not averaged and then
    subtracted: `_psnr_masked` returns NaN on an empty mask, so the covered and
    uncovered columns are supported on slightly different view sets and a
    difference of column means would silently compare two different populations.
    """
    out = []
    for row in rows:
        vals = [row.get(k) for k in keys]
        if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in vals):
            continue
        out.append(vals)
    return out


# The difficulty-normalised margin.  `covered - uncovered` is not a clean readout
# of "how much the model gains from having correspondence available", because the
# two strata are not equally hard: covered pixels are by construction the ones
# visible from a source view -- often textured foreground -- while uncovered ones
# are disocclusions and frame borders, frequently smooth background that is
# *easier* to guess.  Both effects push in opposite directions and partially
# cancel, so a flat `cov - unc` is consistent with a real information advantage
# hidden under a difficulty gradient.
#
# `tgt_rgb_psnr_copy_covered` / `_copy_uncovered` measure exactly that difficulty
# gradient: the copy baseline does zero geometry, so whatever it gains on covered
# pixels is stratum difficulty and nothing else.  Subtracting it twice gives a
# difference-in-differences,
#     DiD = (model_cov - copy_cov) - (model_unc - copy_unc)
# which is scored on identical pixel sets on both sides of each inner difference,
# so stratum difficulty cancels by construction.  If DiD grows with coverage the
# reprojection reading of §3.2 survives; if it is still flat, the flatness is a
# property of the model and no longer explainable by which pixels landed in which
# bin.  This costs no inference -- it re-reads the 864 rows already on disk.
MHEAD = (
    f"{'stratum':<22}{'n':>5}{'cover':>7}"
    f"{'cov':>8}{'cp_cov':>8}{'m_cov':>8}"
    f"{'unc':>8}{'cp_unc':>8}{'m_unc':>8}"
    f"{'DiD':>8}{'DiD>0':>7}{'ceil_gap':>9}"
)


def mline(label, rows):
    quad = paired(rows, "tgt_rgb_psnr_covered", "tgt_rgb_psnr_copy_covered",
                  "tgt_rgb_psnr_uncovered", "tgt_rgb_psnr_copy_uncovered")
    if not quad:
        return f"{label:<22}{len(rows):>5}" + " " * (len(MHEAD) - 27)
    m_cov = [c - cc for c, cc, u, cu in quad]
    m_unc = [u - cu for c, cc, u, cu in quad]
    did = [a - b for a, b in zip(m_cov, m_unc)]
    wins = sum(1 for d in did if d > 0) / len(did)
    ceil = [v - c for v, c in paired(rows, "tgt_rgb_psnr_vae_covered",
                                    "tgt_rgb_psnr_covered")]
    return (
        f"{label:<22}{len(quad):>5}"
        + fmt(col(rows, "tgt_rgb_splat_coverage"), 7, 3)
        + fmt(mean([c for c, cc, u, cu in quad]), 8)
        + fmt(mean([cc for c, cc, u, cu in quad]), 8)
        + fmt(mean(m_cov), 8)
        + fmt(mean([u for c, cc, u, cu in quad]), 8)
        + fmt(mean([cu for c, cc, u, cu in quad]), 8)
        + fmt(mean(m_unc), 8)
        + fmt(mean(did), 8)
        + f"{wins * 100:>6.0f}%"
        + fmt(mean(ceil), 9)
    )


def report_margin(rows):
    print(f"\n{'-' * len(MHEAD)}")
    print("difficulty-normalised margin: DiD = (cov - copy_cov) - (unc - copy_unc)")
    print(MHEAD)
    print("-" * len(MHEAD))
    print(mline("ALL", rows))
    print("-" * len(MHEAD))
    for (lo, hi), sel in bin_by(rows, "tgt_rgb_splat_coverage",
                                [0.0, 0.2, 0.35, 0.5, 0.65, 1.01]):
        print(mline(f"coverage [{lo:.2f},{hi:.2f})", sel))
    print("-" * len(MHEAD))
    for (lo, hi), sel in bin_by(rows, "tgt_view_min_src_angle_deg",
                                [0.0, 10.0, 20.0, 30.0, 45.0, 60.0, 181.0]):
        print(mline(f"angle [{lo:.0f},{hi:.0f})deg", sel))


def bin_by(rows, key, edges):
    """Group rows into [edges[i], edges[i+1]) buckets on `key`."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [
            r for r in rows
            if r.get(key) is not None and not math.isnan(r[key]) and lo <= r[key] < hi
        ]
        out.append(((lo, hi), sel))
    return out


def report(root, rows, files):
    print(f"\n{'=' * len(HEAD)}")
    print(f"{root}   {len(rows)} target views   {len(files)} jsonl file(s)")
    print(f"{'=' * len(HEAD)}")
    print(HEAD)
    print("-" * len(HEAD))
    print(line("ALL", rows))

    print("-" * len(HEAD))
    for (lo, hi), sel in bin_by(rows, "tgt_rgb_splat_coverage",
                                [0.0, 0.2, 0.35, 0.5, 0.65, 1.01]):
        print(line(f"coverage [{lo:.2f},{hi:.2f})", sel))

    print("-" * len(HEAD))
    for (lo, hi), sel in bin_by(rows, "tgt_view_min_src_angle_deg",
                                [0.0, 10.0, 20.0, 30.0, 45.0, 60.0, 181.0]):
        print(line(f"angle [{lo:.0f},{hi:.0f})deg", sel))

    # The headline comparison, stated as the two gaps that matter.  On covered
    # pixels the source views determine the answer, so the VAE reconstruction is
    # the real ceiling; on uncovered pixels there is nothing to reproject and
    # only a generative prior can help.
    cov = col(rows, "tgt_rgb_psnr_covered")
    unc = col(rows, "tgt_rgb_psnr_uncovered")
    vae = col(rows, "tgt_rgb_psnr_vae_covered")
    spv = col(rows, "tgt_rgb_psnr_splat_valid")
    spv_pred = col(rows, "tgt_rgb_psnr_splat_pred_valid")
    print("-" * len(HEAD))
    print(f"  covered   {cov:6.2f}   vs splat-valid(GT depth)   {spv:6.2f} "
          f"(gap {cov - spv:+6.2f})   vs VAE ceiling {vae:6.2f} (gap {cov - vae:+6.2f})")
    if not math.isnan(spv_pred):
        # Same comparison with the confound closed: this splat uses the
        # model's own predicted source depth, the only baseline it actually
        # has to beat at deployment (GT source depth is never available then
        # either).
        print(f"            vs splat-valid(pred depth) {spv_pred:6.2f} "
              f"(gap {cov - spv_pred:+6.2f})")
    print(f"  uncovered {unc:6.2f}   -- no source pixel reprojects here; "
          f"splat has no number to give")
    report_margin(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+",
                        help="eval output directories (globs expanded by the shell)")
    args = parser.parse_args()

    found = False
    for root in args.roots:
        rows, files = load_rows(root)
        if not rows:
            print(f"[skip] no per_view_metrics.jsonl under {root}", file=sys.stderr)
            continue
        found = True
        report(root, rows, files)
    if not found:
        sys.exit(1)


if __name__ == "__main__":
    main()
