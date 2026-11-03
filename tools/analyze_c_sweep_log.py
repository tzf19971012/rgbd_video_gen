"""C-sweep paired-seed test on metrics that only exist in train.log.

WHY A SECOND HARNESS.  analyze_c_sweep_window.py reads
eval_step_*/per_view_metrics.jsonl, which carries 30 keys and every one of them
is a TARGET-view metric -- there is no src_* in that file.  The `[eval step N]`
summary lines in train.log carry the source-view metrics too.  That matters for
exactly one question:

    clean_source_depth drops the source-disparity frames from the depth loss on
    clean_source_depth_prob of samples (WanRGBDNVSTrainingModule.py:1222-1225).
    If that lost supervision is what ails `cleansrcd`, the damage has to be
    WORSE on src_depth_* than on tgt_depth_*.  Seed noise has no reason to tell
    source from target, so the src-vs-tgt ratio discriminates the mechanism from
    the null in a way no target-only metric can.

WHAT IS LOST.  train.log gives the eval-set MEAN, already aggregated, so there
is no per-view resolution and nothing here can do the DiD/coverage regression.
That is fine: the paired-seed test needs one scalar per (arm, seed) cell and the
aggregate IS that scalar.  Run --crosscheck to confirm the two harnesses agree
on a key both of them have, i.e. that this parser reads what it thinks it does.

Usage:
  python tools/analyze_c_sweep_log.py --steps 3500 \
      --metrics src_depth_abs_rel src_depth_delta1 tgt_depth_abs_rel tgt_depth_delta1
  python tools/analyze_c_sweep_log.py --steps 2000 2250 2500 --crosscheck
"""
import argparse
import math
import os
import re

from analyze_c_sweep_window import ARMS, CONTRASTS, SEEDS, T05, TBONF, fmt, mean

EVAL_LINE = re.compile(r"^\[eval step (\d+)\]\s*(.*)$")
PAIR = re.compile(r"([A-Za-z_0-9]+)=(-?[0-9.]+(?:[eE][-+]?\d+)?)")
DEFAULT_METRICS = [
    "src_depth_abs_rel", "src_depth_delta1",
    "tgt_depth_abs_rel", "tgt_depth_delta1",
]


def load_log(run_dir, steps):
    """{step: {metric: value}} for the requested steps, or (None, reason).

    Keeps the LAST occurrence of a step: a resumed run re-evaluates the step it
    restarted from and appends, so a repeated header is the same frozen eval
    scored twice, not two different measurements.
    """
    path = os.path.join(run_dir, "train.log")
    if not os.path.isfile(path):
        return None, f"missing {path}"
    want = set(steps)
    found = {}
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            m = EVAL_LINE.match(line.strip())
            if not m:
                continue
            step = int(m.group(1))
            if step in want:
                found[step] = {k: float(v) for k, v in PAIR.findall(m.group(2))}
    missing = sorted(want - set(found))
    if missing:
        return None, f"{path}: no [eval step] line for {missing}"
    return found, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--metrics", type=str, nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--log_root", type=str, default="logs")
    parser.add_argument("--arms", type=str, nargs="+", default=ARMS)
    parser.add_argument("--crosscheck", action="store_true",
                        help="Also print tgt_rgb_psnr/tgt_rgb_lpips so the numbers can be "
                             "compared against analyze_c_sweep_window.py on the same window.")
    args = parser.parse_args()

    metrics = list(args.metrics)
    if args.crosscheck:
        for m in ("tgt_rgb_psnr", "tgt_rgb_lpips"):
            if m not in metrics:
                metrics.append(m)

    print(f"=== C-sweep train.log reading: steps={args.steps} ===")
    print("    source: [eval step N] lines (eval-set means, already aggregated)\n")

    per_cell = {}
    for arm in args.arms:
        for seed in SEEDS:
            run = os.path.join(args.log_root, f"bmvs_c_{arm}_s{seed}")
            found, err = load_log(run, args.steps)
            if found is None:
                print(f"[skip] {arm} s{seed}: {err}")
                continue
            for m in metrics:
                vals = [found[s].get(m) for s in args.steps]
                if any(v is None for v in vals):
                    continue
                per_cell[(arm, seed, m)] = mean(vals)

    print("--- window means (seeds averaged) ---")
    print(f"{'arm':<18}" + "".join(f"{m:>22}" for m in metrics))
    per_arm = {}
    for arm in args.arms:
        row = []
        for m in metrics:
            vals = [per_cell[(arm, s, m)] for s in SEEDS if (arm, s, m) in per_cell]
            row.append(mean(vals) if vals else float("nan"))
        per_arm[arm] = dict(zip(metrics, row))
        print(f"{arm:<18}" + "".join(fmt(v, 22, 4) for v in row))

    diffs = {m: [] for m in metrics}
    print(f"\n{'arm':<18}" + "".join(f"{m + ' d(seed)':>22}" for m in metrics))
    for arm in args.arms:
        row = []
        for m in metrics:
            a, b = per_cell.get((arm, 42, m)), per_cell.get((arm, 1337, m))
            d = a - b if a is not None and b is not None else float("nan")
            row.append(d)
            if not math.isnan(d):
                diffs[m].append(d)
        print(f"{arm:<18}" + "".join(fmt(v, 22, 4) for v in row))

    print("\n--- paired-seed pooled-variance t-test (same construction as "
          "analyze_c_sweep_window.py) ---")
    sp = {}
    for m in metrics:
        d = diffs[m]
        sp[m] = (math.sqrt(sum(x * x for x in d) / (2 * len(d))), len(d)) if d else (float("nan"), 0)
        print(f"  s_p({m}) = {sp[m][0]:.5f}   from {sp[m][1]} arms, df = {sp[m][1]}")
    n_arms = max((n for _, n in sp.values()), default=0)
    n_ctr = sum(1 for a, c in CONTRASTS if a in per_arm and c in per_arm)
    print(f"  df={n_arms}: |t|>{T05.get(n_arms, float('nan')):.3f} single, "
          f"|t|>{TBONF.get((n_arms, n_ctr), float('nan')):.2f} Bonferroni ({n_ctr} contrasts)")

    for title, as_t in (("t", True), ("delta", False)):
        print(f"\n{'contrast':<30}" + "".join(f"{m + ' ' + title:>22}" for m in metrics))
        for arm, ctrl in CONTRASTS:
            if arm not in per_arm or ctrl not in per_arm:
                continue
            row = []
            for m in metrics:
                delta = per_arm[arm][m] - per_arm[ctrl][m]
                s = sp[m][0]
                row.append(delta / s if as_t and s and s > 0 else delta)
            print(f"{arm + ' - ' + ctrl:<30}"
                  + "".join(fmt(v, 22, 2 if as_t else 4) for v in row))


if __name__ == "__main__":
    main()
