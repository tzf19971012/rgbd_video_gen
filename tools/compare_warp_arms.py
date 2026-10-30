#!/usr/bin/env python
"""对比两个训练 run（如 nearest vs warpbi 两个 warp arm）的 eval 指标。

读取各自 run 目录下的 eval_step_*/per_view_metrics.jsonl，对每个指标按
per-view 行取均值（跳过 NaN），输出：
  * 逐 eval step 的两 arm 指标对照表；
  * 共同 step 上 pooled（全部 per-view 行合并）的汇总差值表，标出哪个 arm 更优。

用法：
  python tools/compare_warp_arms.py \
      --baseline logs/re10k13b_rand20261023_011303 \
      --arm logs/re10k13b_randbi20261024_XXXXXX

可选：
  --last N            汇总只统计两边都存在的最后 N 个 eval step（默认全部）
  --steps 1500,2000   只看指定 step
  --metrics a,b,c     只看指定指标
  --json PATH         额外把汇总结果写成 JSON

注意：只有两 arm 同 seed（eval_view_seed/eval_sample_seed）且配置单变量时，
差值才可归因于该变量（这里即 warp_splat_footprint: nearest -> bilinear）。
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

# 指标方向：higher-is-better 缺省；这里显式列出 lower-is-better 的。
LOWER_IS_BETTER = {
    "tgt_rgb_lpips",
    "tgt_rgb_lpips_splat",
    "tgt_rgb_lpips_splat_pred",
    "tgt_depth_abs_rel",
    "tgt_depth_abs_rel_clamp100",
    "tgt_depth_disp_l1",
    "src_depth_abs_rel",
    "src_depth_abs_rel_clamp100",
    "src_depth_disp_l1",
}

# 头部指标（对比时最关心的先打印）；其余指标按名字排序跟在后面。
HEADLINE_METRICS = [
    "tgt_rgb_psnr",
    "tgt_rgb_lpips",
    "tgt_depth_abs_rel",
    "tgt_depth_delta1",
    "tgt_depth_disp_l1",
    "tgt_rgb_psnr_splat",
    "tgt_rgb_psnr_splat_pred",
    "tgt_rgb_splat_coverage",
    "tgt_rgb_splat_pred_coverage",
    "src_depth_abs_rel",
    "src_depth_delta1",
]

EVAL_DIR_RE = re.compile(r"^eval_step_(\d+)$")


def _is_finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def load_eval_steps(run_dir):
    """返回 {step: {metric: [per-view values(有限值)]}}。"""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        sys.exit(f"run dir 不存在: {run_dir}")
    steps = {}
    for d in sorted(run_dir.iterdir()):
        m = EVAL_DIR_RE.match(d.name)
        if not m or not d.is_dir():
            continue
        jf = d / "per_view_metrics.jsonl"
        if not jf.is_file():
            continue
        rows = []
        with jf.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        metrics = {}
        for row in rows:
            for k, v in row.items():
                if k in ("sample_id", "target_index", "view_index"):
                    continue
                metrics.setdefault(k, []).append(v)
        steps[int(m.group(1))] = metrics
    if not steps:
        sys.exit(f"{run_dir} 下没有找到 eval_step_*/per_view_metrics.jsonl")
    return steps


def mean(values):
    vals = [v for v in values if _is_finite(v)]
    return sum(vals) / len(vals) if vals else None


def step_means(steps_data):
    """{step: {metric: mean}}。"""
    return {
        step: {k: mean(vs) for k, vs in metrics.items()}
        for step, metrics in steps_data.items()
    }


def pooled_means(steps_data):
    """把所有 step 的 per-view 行合并后取均值：{metric: mean}。"""
    merged = {}
    for metrics in steps_data.values():
        for k, vs in metrics.items():
            merged.setdefault(k, []).extend(vs)
    return {k: mean(vs) for k, vs in merged.items()}


def fmt(v, width=8):
    return f"{v:>{width}.4f}" if v is not None else f"{'n/a':>{width}}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baseline", required=True, help="基线 run 目录（nearest）")
    ap.add_argument("--arm", required=True, help="对照 run 目录（warpbi）")
    ap.add_argument("--last", type=int, default=0, help="汇总只统计最后 N 个共同 step（0=全部）")
    ap.add_argument("--steps", default="", help="逗号分隔 step 列表，只看这些 step")
    ap.add_argument("--metrics", default="", help="逗号分隔指标名，只看这些指标")
    ap.add_argument("--json", dest="json_path", default="", help="汇总结果写入 JSON")
    args = ap.parse_args()

    base_steps = load_eval_steps(args.baseline)
    arm_steps = load_eval_steps(args.arm)

    if args.steps:
        want = {int(s) for s in args.steps.split(",") if s.strip()}
        base_steps = {s: v for s, v in base_steps.items() if s in want}
        arm_steps = {s: v for s, v in arm_steps.items() if s in want}

    common = sorted(set(base_steps) & set(arm_steps))
    only_base = sorted(set(base_steps) - set(arm_steps))
    only_arm = sorted(set(arm_steps) - set(base_steps))
    if only_base:
        print(f"[warn] 仅基线有的 eval step（不参与对比）: {only_base}")
    if only_arm:
        print(f"[warn] 仅对照 arm 有的 eval step（不参与对比）: {only_arm}")
    if not common:
        sys.exit("两 arm 没有共同 eval step，无法对比")

    base_means = step_means(base_steps)
    arm_means = step_means(arm_steps)

    all_metrics = sorted(
        {k for d in list(base_means.values()) + list(arm_means.values()) for k in d}
    )
    if args.metrics:
        keep = {s.strip() for s in args.metrics.split(",") if s.strip()}
        all_metrics = [k for k in all_metrics if k in keep]

    ordered = [k for k in HEADLINE_METRICS if k in all_metrics]
    ordered += [k for k in all_metrics if k not in ordered]

    b_tag = Path(args.baseline).name
    a_tag = Path(args.arm).name

    # ---------- 逐 step 对照表 ----------
    for step in common:
        print(f"\n=== eval step {step} ===")
        print(f"{'metric':<34}{'base':>10}{'arm':>10}{'delta':>10}  better")
        print("-" * 78)
        for k in ordered:
            b, a = base_means[step].get(k), arm_means[step].get(k)
            if b is None and a is None:
                continue
            delta = None if (b is None or a is None) else a - b
            mark = ""
            if delta is not None and delta != 0:
                arm_better = delta < 0 if k in LOWER_IS_BETTER else delta > 0
                mark = "ARM" if arm_better else "base"
            print(f"{k:<34}{fmt(b)}{fmt(a)}{fmt(delta)}  {mark}")

    # ---------- pooled 汇总 ----------
    summary_steps = common[-args.last:] if args.last > 0 else common
    pooled_b = pooled_means({s: base_steps[s] for s in summary_steps})
    pooled_a = pooled_means({s: arm_steps[s] for s in summary_steps})

    print(f"\n=== pooled summary over common steps {summary_steps[0]}..{summary_steps[-1]} "
          f"(n={len(summary_steps)}) ===")
    print(f"baseline: {b_tag}\narm:      {a_tag}")
    print(f"{'metric':<34}{'base':>10}{'arm':>10}{'delta':>10}{'delta%':>9}  better")
    print("-" * 88)
    summary = {}
    for k in ordered:
        b, a = pooled_b.get(k), pooled_a.get(k)
        if b is None and a is None:
            continue
        delta = None if (b is None or a is None) else a - b
        pct = None
        if delta is not None and b not in (None, 0):
            pct = 100.0 * delta / abs(b)
        mark = ""
        if delta is not None and delta != 0:
            arm_better = delta < 0 if k in LOWER_IS_BETTER else delta > 0
            mark = "ARM" if arm_better else "base"
        print(f"{k:<34}{fmt(b)}{fmt(a)}{fmt(delta)}"
              f"{('%+.2f%%' % pct) if pct is not None else '      n/a':>9}  {mark}")
        summary[k] = {"baseline": b, "arm": a, "delta": delta, "delta_pct": pct,
                      "better": mark or None}

    wins_arm = sum(1 for v in summary.values() if v["better"] == "ARM")
    wins_base = sum(1 for v in summary.values() if v["better"] == "base")
    print(f"\n胜场（不含平局/无值）: arm={wins_arm}  base={wins_base}  "
          f"平局或n/a={len(summary) - wins_arm - wins_base}")

    if args.json_path:
        out = {
            "baseline": str(args.baseline), "arm": str(args.arm),
            "steps": summary_steps,
            "per_step": {str(s): {k: {"baseline": base_means[s].get(k),
                                      "arm": arm_means[s].get(k)}
                                  for k in ordered}
                         for s in common},
            "pooled": summary,
        }
        Path(args.json_path).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\n汇总已写入: {args.json_path}")


if __name__ == "__main__":
    main()
