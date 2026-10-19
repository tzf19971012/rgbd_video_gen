"""D-sweep train.log reading harness: window means vs the sharedrope anchor.

WHY train.log AND NOT per_view_metrics.jsonl.  Same reason as
analyze_c_sweep_log.py: per_view_metrics.jsonl carries 30 keys and every one of
them is a target-view metric, so there is no src_depth_* in it. The `[eval step
N]` summary lines in train.log carry BOTH src_* and tgt_* -- already aggregated
over the full frozen val split (64 samples x 3 target views = 192 target views,
6 source views x 64 samples = 384 source views), which is what this script reads.

WHY --anchor_steps AND --d_steps ARE SEPARATE ARGS, EVEN THOUGH THEY DEFAULT TO
THE SAME WINDOW NOW.  The anchor (`bmvs_c_sharedrope_s*`) originally stopped at
step 3500 while every D-round arm ran to step 4000
(sbatch/bmvs_d_sweep.sbatch's `max_steps=4000`); report_20260912's first cut
compared anchor@[3250,3500] against D-round@[3750,4000] and read `droce`'s
apparent RGB win off that mismatched pair -- which is indistinguishable from
"droce just trained 500 steps longer" until report_20260912 §0.1 resumed the
anchor to 4000 (`RESUME_STEP=3500 MAX_STEPS=4000`, jobs 10916/10917) and found
its own RGB numbers barely moved (+0.017dB) while `droce`'s did. Keeping the
two knobs separate is what made that gap visible in the first place; collapsing
them into one shared window argument would make the next such mismatch silent
again.

WHY A WINDOW AT ALL.  report_20260907_c_sweep.md already paid for this lesson:
a single eval point can flip a ranking between arms of similar quality (t0
Sec.4.2's b7 example). Averaging the last two eval points does not fix that, but
it roughly halves the single-point noise (see --spread), which is enough to tell
`droce`'s +0.23dB from `d4`'s +0.02dB.

WHY THE SEED SPREAD MATTERS MORE THAN THE STEP SPREAD.  Two arms
(`d3rot`/`d3pose`) only have one seed. Everything else has two: the seed-to-seed
spread on the SAME window mean is the cleanest single noise estimate this data
gives, and it is printed unconditionally so a delta can be eyeballed against it
without re-deriving the pooled variance.

Usage:
  python tools/analyze_d_sweep_log.py
  python tools/analyze_d_sweep_log.py --d_steps 3250 3500 3750 4000 --anchor_steps 3000 3250 3500
"""
import argparse
import math
import os
import re

EVAL_LINE = re.compile(r"^\[eval step (\d+)\]\s*(.*)$")
PAIR = re.compile(r"([A-Za-z_0-9]+)=(-?[0-9.]+(?:[eE][-+]?\d+)?)")

# arm -> (run-dir template, seeds).  d3rot/d3pose never got a second seed
# (plan's 4-arm design counted d1/d2/d4 x 2 seeds as the cheap factors; d3rot/
# d3pose are the continuous-pose arms and were read single-seed against d1).
ARMS = {
    "d1": ("bmvs_d_d1_s{seed}", (42, 1337)),
    "d2": ("bmvs_d_d2_s{seed}", (42, 1337)),
    "d3rot": ("bmvs_d_d3rot_s{seed}", (42,)),
    "d3pose": ("bmvs_d_d3pose_s{seed}", (42,)),
    "droce": ("bmvs_d_droce_s{seed}", (42, 1337)),
    "d4": ("bmvs_d_d4_s{seed}", (42, 1337)),
    "drocefp32": ("bmvs_d_drocefp32_s{seed}", (42, 1337)),
}
ANCHOR = ("bmvs_c_sharedrope_s{seed}", (42, 1337))

DEFAULT_METRICS = [
    "src_depth_abs_rel", "src_depth_delta1", "src_depth_struct_corr",
    "tgt_depth_abs_rel", "tgt_depth_delta1", "tgt_depth_struct_corr",
    "tgt_rgb_psnr", "tgt_rgb_lpips",
]
# True = higher is better, for the delta sign printed alongside each number.
HIGHER_BETTER = {
    "src_depth_abs_rel": False, "src_depth_delta1": True, "src_depth_struct_corr": True,
    "tgt_depth_abs_rel": False, "tgt_depth_delta1": True, "tgt_depth_struct_corr": True,
    "tgt_rgb_psnr": True, "tgt_rgb_lpips": False,
}


def finite(x):
    return x is not None and isinstance(x, (int, float)) and not math.isnan(x)


def mean(vals):
    vals = [v for v in vals if finite(v)]
    return float("nan") if not vals else sum(vals) / len(vals)


def load_steps(run_dir, steps):
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


def window_mean(run_dir, steps, metrics):
    found, err = load_steps(run_dir, steps)
    if found is None:
        return None, err
    return {m: mean([found[s].get(m) for s in steps]) for m in metrics}, None


def fmt_delta(v, ref, higher_better):
    d = v - ref
    good = (d >= 0) == higher_better
    mark = "+" if good else ""  # sign is already in the number; mark just flags direction
    return f"{v:.4f}({'+' if d >= 0 else ''}{d:.4f}{'' if good else ' !'})"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log_root", default="logs")
    p.add_argument("--anchor_steps", type=int, nargs="+", default=[3750, 4000])
    p.add_argument("--d_steps", type=int, nargs="+", default=[3750, 4000])
    p.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    p.add_argument("--spread", action="store_true",
                    help="also print the within-window (step-to-step) and seed-to-seed spread")
    args = p.parse_args()
    metrics = args.metrics

    def cell(template, seed, steps):
        run_dir = os.path.join(args.log_root, template.format(seed=seed))
        wm, err = window_mean(run_dir, steps, metrics)
        if wm is None:
            print(f"[skip] {template.format(seed=seed)}: {err}")
        return wm

    anchor_cells = {s: cell(ANCHOR[0], s, args.anchor_steps) for s in ANCHOR[1]}
    anchor = {m: mean([anchor_cells[s][m] for s in ANCHOR[1] if anchor_cells[s]]) for m in metrics}

    print(f"=== D-sweep train.log window means ===")
    print(f"    anchor window (sharedrope): {args.anchor_steps}")
    print(f"    D-round window: {args.d_steps}\n")

    print(f"{'arm':<10}" + "".join(f"{m:>26}" for m in metrics))
    print(f"{'anchor':<10}" + "".join(f"{anchor[m]:>26.4f}" for m in metrics))

    arm_means = {}
    for arm, (template, seeds) in ARMS.items():
        cells = {s: cell(template, s, args.d_steps) for s in seeds}
        cells = {s: c for s, c in cells.items() if c is not None}
        if not cells:
            continue
        am = {m: mean([cells[s][m] for s in cells]) for m in metrics}
        arm_means[arm] = am
        row = "".join(
            f"{fmt_delta(am[m], anchor[m], HIGHER_BETTER[m]):>26}" for m in metrics
        )
        print(f"{arm:<10}" + row)

    if args.spread:
        print("\n--- seed-to-seed spread on the window mean (|s42 - s1337|) ---")
        for arm, (template, seeds) in [("anchor", ANCHOR)] + list(ARMS.items()):
            if len(seeds) < 2:
                continue
            cells = {s: cell(template, s, args.anchor_steps if arm == "anchor" else args.d_steps)
                     for s in seeds}
            if any(v is None for v in cells.values()):
                continue
            row = "".join(f"{abs(cells[42][m] - cells[1337][m]):>26.4f}" for m in metrics)
            print(f"{arm:<10}" + row)

        print("\n--- within-window spread (max eval point vs min, per cell) ---")
        for arm, (template, seeds) in [("anchor", ANCHOR)] + list(ARMS.items()):
            steps = args.anchor_steps if arm == "anchor" else args.d_steps
            for s in seeds:
                run_dir = os.path.join(args.log_root, template.format(seed=s))
                found, err = load_steps(run_dir, steps)
                if found is None:
                    continue
                row = "".join(
                    f"{max(found[st][m] for st in steps) - min(found[st][m] for st in steps):>26.4f}"
                    for m in metrics
                )
                print(f"{arm + '_s' + str(s):<10}" + row)


if __name__ == "__main__":
    main()
