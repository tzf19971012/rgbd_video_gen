"""Read the c-sweep eval_flow_shift scan (sbatch/c_sigma_shift_scan.sbatch).

WHAT THIS ANSWERS.  `sigma` (rgb_flow_shift 5.0 -> 1.0) lost 1.30-1.86 dB, but
all ten arms were scored on ONE pinned sampler (eval_flow_shift 0.5, 25 steps)
that leaves only 3 of 25 Euler steps above sigma 0.8 -- where base puts 59.3% of
its gradient and `sigma` only 8.5%.  So the reading confounds "low-sigma training
is harmful" with "low-sigma training is mismatched to THIS grid".  Scanning shift
on each arm's own checkpoint separates them, and base is the CONTROL: every
checkpoint improves somewhat under a shift scan (t0 §3: b3 gained 0.435 dB going
5 -> 0.5), so only whether the GAP closes matters.

WHY IT PARSES THE LOG.  In sweep mode eval_rgbd_nvs.py writes visualisations per
grid point but no per_view_metrics.jsonl, so the metrics live only in the
`[batch N]` lines under each `--- eval_flow_shift=S steps=N ---` header.  With
eval_max_samples=64 and batch_size=4 every batch is exactly 4 views, so the
unweighted mean over batch lines is the grid point's true mean.  Reading the log
also means partial results are visible while the job is still running: a grid
point with fewer than --min_batches lines is reported as incomplete rather than
averaged in.

READ THE RESULT WITH t0 §4.2 IN HAND.  Single-draw PSNR and LPIPS both reward
sharpness, and the shift knob buys sharpness by narrowing posterior variance
without moving the bias term (b3: b^2 -0.14%, v -7.9%).  So this scan locates the
argmax; the VERDICT needs sbatch/c_diag_posterior.sbatch re-run there.

Usage:
  python tools/analyze_c_shift_scan.py
  python tools/analyze_c_shift_scan.py --metrics tgt_rgb_psnr tgt_rgb_lpips
"""
import argparse
import glob
import math
import os
import re

from analyze_c_sweep_window import SEEDS, fmt, mean

HEADER = re.compile(r"^---\s*eval_flow_shift=([0-9.]+)\s+steps=(\d+)")
BATCH = re.compile(r"^\[batch (\d+)\]\s*(.*)$")
PAIR = re.compile(r"([A-Za-z_0-9]+)=(-?[0-9.]+(?:[eE][-+]?\d+)?)")
DEFAULT_METRICS = ["tgt_rgb_psnr", "tgt_rgb_lpips", "tgt_depth_delta1", "tgt_depth_abs_rel"]
# Lower is better for these, so the argmax search must flip sign.
LOWER_BETTER = {"tgt_rgb_lpips", "tgt_depth_abs_rel"}


def parse(path, min_batches):
    """{(shift, steps): {metric: mean}} for grid points with enough batch lines.

    Returns (complete, partial) so an in-flight job reports honestly instead of
    averaging a half-scored grid point into the table.
    """
    sections = {}
    key = None
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            h = HEADER.match(line)
            if h:
                key = (float(h.group(1)), int(h.group(2)))
                sections.setdefault(key, [])
                continue
            b = BATCH.match(line)
            if b and key is not None:
                sections[key].append({k: float(v) for k, v in PAIR.findall(b.group(2))})
    complete, partial = {}, {}
    for k, rows in sections.items():
        target = complete if len(rows) >= min_batches else partial
        target[k] = ({m: mean([r.get(m) for r in rows]) for m in set().union(*rows)}
                     if rows else {})
        target[k]["_n"] = len(rows)
    return complete, partial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="logs/c_sigma_shift")
    parser.add_argument("--arms", type=str, nargs="+", default=["base", "sigma"])
    parser.add_argument("--metrics", type=str, nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--min_batches", type=int, default=16,
                        help="Batch lines required to call a grid point complete "
                             "(eval_max_samples/batch_size, i.e. 64/4).")
    args = parser.parse_args()

    print(f"=== eval_flow_shift scan: {args.root} ===")
    print("    base is the CONTROL: every checkpoint gains under a shift scan, so read "
          "the GAP, not the levels.\n")

    cells = {}
    for arm in args.arms:
        for seed in SEEDS:
            log = os.path.join(args.root, f"{arm}_s{seed}_step3500.log")
            if not os.path.isfile(log):
                hits = glob.glob(os.path.join(args.root, f"{arm}_s{seed}_step*.log"))
                if not hits:
                    print(f"[skip] {arm} s{seed}: no log under {args.root}")
                    continue
                log = hits[0]
            done, part = parse(log, args.min_batches)
            cells[(arm, seed)] = done
            for k, v in sorted(part.items()):
                print(f"[running] {arm} s{seed} shift={k[0]} steps={k[1]}: "
                      f"{v['_n']}/{args.min_batches} batches")
    if any(cells.values()):
        print()

    shifts = sorted({k[0] for d in cells.values() for k in d})
    if not shifts:
        print("No complete grid point yet.")
        return

    for m in args.metrics:
        print(f"--- {m} (rows: arm/seed, cols: eval_flow_shift) ---")
        print(f"{'cell':<22}" + "".join(f"{s:>11}" for s in shifts))
        arm_means = {}
        for arm in args.arms:
            per_shift = {s: [] for s in shifts}
            for seed in SEEDS:
                d = cells.get((arm, seed), {})
                row = []
                for s in shifts:
                    v = next((d[k].get(m) for k in d if k[0] == s), float("nan"))
                    row.append(v)
                    if v is not None and not math.isnan(v):
                        per_shift[s].append(v)
                if any(v is not None and not math.isnan(v) for v in row):
                    print(f"{arm + ' s' + str(seed):<22}"
                          + "".join(fmt(v, 11, 4) for v in row))
            arm_means[arm] = {s: mean(per_shift[s]) for s in shifts}
            if all(len(per_shift[s]) == len(SEEDS) for s in shifts):
                print(f"{arm + ' (both)':<22}"
                      + "".join(fmt(arm_means[arm][s], 11, 4) for s in shifts))
        # The gap is the whole point: sigma - base at each shift, plus each arm's
        # own argmax, so "sigma recovers at its own optimum" is checkable directly.
        if len(args.arms) == 2:
            a, b = args.arms[1], args.arms[0]
            gap = [arm_means[a][s] - arm_means[b][s] for s in shifts]
            print(f"{a + ' - ' + b:<22}" + "".join(fmt(v, 11, 4) for v in gap))
        for arm in args.arms:
            vals = [(s, arm_means[arm][s]) for s in shifts
                    if not math.isnan(arm_means[arm][s])]
            if vals:
                best = (min if m in LOWER_BETTER else max)(vals, key=lambda p: p[1])
                print(f"    {arm} argmax: shift={best[0]}  {m}={best[1]:.4f}")
        print()

    print("REMINDER: this is SINGLE-DRAW. A shift that 'recovers' PSNR may only have "
          "narrowed\nposterior variance (t0 §4.2). Re-run c_diag_posterior.sbatch at the "
          "argmax before concluding.")


if __name__ == "__main__":
    main()
