"""Analyse the per-view JSONL written by tools/diag_posterior_mean.py.

The driver prints the full-frame table; this reports the two things it does not,
both of which bear on report_20260904_p0a §3.2:

  1. The SAMPLING PENALTY SPLIT BY THE GT-SPLAT PARTITION.  The covered-vs-
     uncovered margin, and the difference-in-differences built on it, are
     computed from SINGLE DRAWS.  If averaging recovers more dB on uncovered
     pixels than on covered ones -- which is what a prior doing the guessing
     where there is no evidence looks like -- then the single-draw margin
     UNDERSTATES the true one, and the flat DiD is partly a measurement artefact
     after all.  If the two recover equally, the flatness survives and the defect
     is in evidence fusion.

  2. The k->inf EXTRAPOLATION, done right.  A k-draw Monte-Carlo mean still
     carries 1/k of a single draw's posterior variance, so the measured gain is
     not the posterior-mean gain.  Writing the single-draw MSE as
         e1 = b^2 + v          (bias^2 + posterior variance)
     the k-average has  ek = b^2 + v/k, so two measurements identify b^2 and v
     without assuming calibration:
         v = (e1 - ek) / (1 - 1/k),    b^2 = e1 - v.
     The calibrated case is v = b^2, i.e. exactly the 3.01 dB textbook figure;
     reading off v/e1 says how far from calibrated this model actually is.

Usage:
  python tools/analyze_posterior_mean.py logs/t0_posterior/posterior_b3_n8_shift5
  python tools/analyze_posterior_mean.py logs/t0_posterior/posterior_*   # one block each
"""
import argparse
import glob
import json
import math
import os
import re
import sys


def load_rows(root):
    rows = []
    paths = sorted(glob.glob(os.path.join(root, "**", "per_view_posterior.jsonl"),
                             recursive=True))
    if not paths and os.path.isfile(root):
        paths = [root]
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows, paths


def mean(values):
    vals = [v for v in values if v is not None and not math.isnan(v)]
    return float("nan") if not vals else sum(vals) / len(vals)


def fmt(x, width=8, prec=2):
    return f"{'--':>{width}}" if math.isnan(x) else f"{x:>{width}.{prec}f}"


def bin_by(rows, key, edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        out.append(((lo, hi), [
            r for r in rows
            if r.get(key) is not None and not math.isnan(r[key]) and lo <= r[key] < hi
        ]))
    return out


def max_k(rows):
    ks = set()
    for row in rows:
        for key in row:
            m = re.fullmatch(r"psnr_mean_(\d+)", key)
            if m:
                ks.add(int(m.group(1)))
    return max(ks) if ks else 0


def paired_gain(rows, prefix, n):
    """Per-view (k=1, k=n) pairs for one region, keeping only finite pairs.

    Formed per view then averaged: _psnr_masked returns NaN on an empty mask, so
    the covered and uncovered columns are supported on different view sets and a
    difference of column means would compare two different populations.
    """
    out = []
    for row in rows:
        a, b = row.get(f"{prefix}_1"), row.get(f"{prefix}_{n}")
        if a is None or b is None or math.isnan(a) or math.isnan(b):
            continue
        out.append((a, b))
    return out


def extrapolate(m1, mn, n):
    """(k->inf PSNR, v/e1 fraction) from the k=1 and k=n means. NaN if unfittable."""
    if n < 2 or math.isnan(m1) or math.isnan(mn):
        return float("nan"), float("nan")
    e1, en = 10 ** (-m1 / 10), 10 ** (-mn / 10)
    v = (e1 - en) / (1.0 - 1.0 / n)
    b2 = e1 - v
    if b2 <= 0:
        return float("nan"), v / e1
    return -10 * math.log10(b2), v / e1


HEAD = (
    f"{'stratum':<22}{'n':>5}{'cover':>7}"
    f"{'full1':>8}{'fullN':>8}{'gain':>7}"
    f"{'cov1':>8}{'covN':>8}{'gain':>7}"
    f"{'unc1':>8}{'uncN':>8}{'gain':>7}"
    f"{'d_gain':>8}{'std_c':>8}{'std_u':>8}{'u/c':>6}"
)


def line(label, rows, n):
    if not rows:
        return f"{label:<22}{0:>5}"
    cells = f"{label:<22}{len(rows):>5}" + fmt(
        mean([r["tgt_rgb_splat_coverage"] for r in rows]), 7, 3)
    gains = {}
    for prefix in ("psnr_mean", "covered_mean", "uncovered_mean"):
        pairs = paired_gain(rows, prefix, n)
        a, b = mean([p[0] for p in pairs]), mean([p[1] for p in pairs])
        gains[prefix] = mean([q - p for p, q in pairs])
        cells += fmt(a) + fmt(b) + fmt(gains[prefix], 7)
    # The headline: how much MORE the average buys where there is no evidence.
    cells += fmt(gains["uncovered_mean"] - gains["covered_mean"], 8)
    sc = mean([r["std_covered"] for r in rows])
    su = mean([r["std_uncovered"] for r in rows])
    cells += fmt(sc, 8, 4) + fmt(su, 8, 4)
    cells += fmt(su / sc if sc > 0 and not math.isnan(sc) else float("nan"), 6)
    return cells


def report(root, rows, paths):
    n = max_k(rows)
    print(f"\n{'=' * len(HEAD)}")
    print(f"{root}   {len(rows)} target views   N={n}   {len(paths)} file(s)")
    print(f"{'=' * len(HEAD)}")
    print(HEAD)
    print("-" * len(HEAD))
    print(line("ALL", rows, n))
    print("-" * len(HEAD))
    for (lo, hi), sel in bin_by(rows, "tgt_rgb_splat_coverage",
                                [0.0, 0.2, 0.35, 0.5, 0.65, 1.01]):
        print(line(f"coverage [{lo:.2f},{hi:.2f})", sel, n))

    print("-" * len(HEAD))
    print("k-sweep (full frame), and the k->inf fit from each (1,k) pair:")
    m1 = mean([r["psnr_mean_1"] for r in rows])
    for k in range(2, n + 1):
        mk = mean([r[f"psnr_mean_{k}"] for r in rows])
        minf, frac = extrapolate(m1, mk, k)
        note = "" if not math.isnan(minf) else "  (unfittable: fitted bias^2 <= 0)"
        print(
            f"  k={k:<2} mean {mk:6.2f} dB  gain {mk - m1:+5.2f}"
            f"   -> k->inf {fmt(minf, 6)} dB  gain {fmt(minf - m1, 5)}"
            f"   v/e1 {frac * 100:4.0f}% (calibrated 50%){note}"
        )
    # A stable fit across k is the check that the two-parameter model holds; a
    # v/e1 that drifts with k means the draws are correlated (shared sampler
    # trajectory) and the extrapolation is an upper bound, not an estimate.
    lp1 = mean([r["lpips_single"] for r in rows])
    lpn = mean([r["lpips_mean"] for r in rows])
    print(f"  LPIPS single {lp1:.4f} -> mean-of-{n} {lpn:.4f}  ({lpn - lp1:+.4f}; "
          f"positive = averaging costs sharpness, the expected trade)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+")
    args = parser.parse_args()
    found = False
    for root in args.roots:
        rows, paths = load_rows(root)
        if not rows:
            print(f"[skip] no per_view_posterior.jsonl under {root}", file=sys.stderr)
            continue
        found = True
        report(root, rows, paths)
    if not found:
        sys.exit(1)


if __name__ == "__main__":
    main()
