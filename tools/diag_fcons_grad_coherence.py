"""Is the f-consist gradient a COHERENT bias or incoherent noise, per term?

The measurement this exists for.  report_20260916_pass1_prior_diag.md §12e found
that `fconsS` (scale term alone, weight 0.25) does 2-3x the damage of `fconsR`
(repro term alone, weight 0.1), which tools/diag_fcons_gt_floor.py then showed
cannot be a magnitude effect -- `repro`'s gradient norm is 8-15x `scale`'s, so
even after the weights `repro` pushes ~3.4x harder and still hurts less.  It also
cannot be a scale effect: evaluate() median-scale-aligns EVERY source view
independently before abs_rel/delta1 (WanRGBDNVSTrainingModule.py:2497, 2624), so
a per-view OR a global disparity rescaling is invisible to exactly the metrics
that collapsed, while the raw unaligned `disp_l1` barely moved.

What is left is DIRECTION.  A gradient that points the same way every time it is
sampled accumulates over 1750 steps; one that points a fresh random way averages
out and only adds variance.  The training loop resamples both the view triple
(randperm of 6, k=3) and the sigma-gated sample every step, so the relevant
question is how much of each term's gradient survives that resampling.

Reported per term, on GT disparity and on an 8x-lowpassed copy of it (the cheap
stand-in for the frozen VAE's bandwidth, which is the field the model can
actually produce):

  * cos_pairs   mean cosine between the gradient fields from two DIFFERENT random
                view triples on the same sample.  High = coherent bias.
  * cos_scale   cosine against the pure per-view rescaling direction (= the
                disparity field itself), as a control: this is the component the
                aligned metrics are blind to, so a term whose gradient is mostly
                this is harmless to delta1 however large it is.
  * cos_terms   cosine between `scale`'s and `repro`'s gradient on one triple, to
                check the two terms are not just the same push twice.

  python -m tools.diag_fcons_grad_coherence --run logs/bmvs_f_fconsS_s42_ft4000 -n 12
"""
import argparse
import itertools
import json
import os
import statistics

import torch
from omegaconf import OmegaConf

from tools.diag_fcons_gt_floor import _lowpass, summarise
from tools.vis_rgb_warp_pred_depth import build_dataset
from utils.depth_consistency import ordered_pairs, pairwise_consistency


def term_grad(disp, valid, K, c2w, views, kw, term):
    """d term / d disp for the chosen views, returned on the FULL view stack so
    gradients from different triples live in the same space and are comparable."""
    d = disp.detach().clone().requires_grad_(True)
    got = pairwise_consistency(
        d[views], valid[views], K[views], c2w[views],
        ordered_pairs(range(len(views))), **kw
    )
    if got is None or not got[term].requires_grad:
        return None
    got[term].backward()
    return d.grad.detach().clone()


def cos(a, b, mask):
    x, y = a[mask], b[mask]
    nx, ny = x.norm(), y.norm()
    if float(nx) == 0.0 or float(ny) == 0.0:
        return float("nan")
    return float((x * y).sum() / (nx * ny))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("-n", "--num-samples", type=int, default=12)
    ap.add_argument("--num-views", type=int, default=3)
    ap.add_argument("--triples", type=int, default=4,
                    help="independent view triples per sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = OmegaConf.load(os.path.join(args.run, "args.yaml"))
    OmegaConf.resolve(cfg)
    dataset = build_dataset(cfg)
    num_src = int(cfg.num_source_views)
    k = min(args.num_views, num_src)
    kw = dict(
        min_support=float(cfg.get("depth_consist_min_support", 0.02)),
        ratio_clip=float(cfg.get("depth_consist_ratio_clip", 4.0)),
        trim_quantile=float(cfg.get("depth_consist_trim_quantile", 0.9)),
    )

    gen = torch.Generator().manual_seed(args.seed)
    fields = {"gt": 0, "lp8": 8}
    acc = {(f, t, m): [] for f in fields for t in ("scale", "repro")
           for m in ("cos_pairs", "cos_scale", "norm")}
    acc_terms = {f: [] for f in fields}
    rows = []

    for idx in range(min(args.num_samples, len(dataset))):
        item = dataset[idx]
        disp0 = item["disparity"][:num_src, 0].float()
        valid = item["loss_masks"][0, :num_src, 0] > 0.5
        K = item["K"][:num_src].float()
        c2w = item["c2w"][:num_src].float()
        triples = [torch.randperm(num_src, generator=gen)[:k].tolist()
                   for _ in range(args.triples)]
        row = {"idx": idx, "sample_id": item["sample_id"], "triples": triples}

        for fname, factor in fields.items():
            disp = _lowpass(disp0, factor) if factor else disp0
            grads = {}
            for term in ("scale", "repro"):
                gs = []
                for views in triples:
                    g = term_grad(disp, valid, K, c2w, views, kw, term)
                    if g is not None:
                        gs.append((set(views), g))
                grads[term] = [g for _, g in gs]
                if not gs:
                    continue
                # Coherence across resampled view triples, compared ONLY on the
                # views the two triples share and only when they share some but
                # not all of them.  Without this the number is an artefact: two
                # identical view SETS give cosine exactly 1 whatever the term
                # does, and two disjoint sets give exactly 0 because the supports
                # do not overlap.  Both artefacts scored 0.71 for both terms.
                pair_cos = []
                for (sa, a), (sb, b) in itertools.combinations(gs, 2):
                    shared = sorted(sa & sb)
                    if not shared or sa == sb:
                        continue
                    m = torch.zeros_like(valid)
                    m[shared] = valid[shared]
                    c = cos(a, b, m)
                    if c == c:
                        pair_cos.append(c)
                # The rescaling direction: d(k*disp)/dk = disp.  Per view, because
                # evaluate() aligns per view.
                scale_cos = [cos(g, disp, valid) for _, g in gs]
                if pair_cos:
                    acc[(fname, term, "cos_pairs")].append(statistics.fmean(pair_cos))
                    row.setdefault(fname, {})[f"{term}_cos_pairs"] = \
                        statistics.fmean(pair_cos)
                acc[(fname, term, "cos_scale")].append(statistics.fmean(scale_cos))
                acc[(fname, term, "norm")].append(
                    statistics.fmean(float(g[valid].norm()) for _, g in gs))
                row.setdefault(fname, {})[f"{term}_cos_scale"] = \
                    statistics.fmean(scale_cos)
            if grads["scale"] and grads["repro"]:
                acc_terms[fname].append(
                    cos(grads["scale"][0], grads["repro"][0], valid))
        rows.append(row)

    print(f"\n{args.run}  k={k} of {num_src} views, {args.triples} triples/sample, "
          f"{len(rows)} samples")
    print("\nHigh cos_pairs = the term pushes the SAME way every time it is sampled,")
    print("so it accumulates over steps.  Low = it averages out into variance.")
    for fname in fields:
        print(f"\n-- field: {fname}")
        for term in ("scale", "repro"):
            cp = summarise(acc[(fname, term, "cos_pairs")])
            cs = summarise(acc[(fname, term, "cos_scale")])
            nm = summarise(acc[(fname, term, "norm")])
            if not cp.get("n"):
                continue
            print(f"   {term:6s} cos_pairs {cp['mean']:+.3f} (median {cp['median']:+.3f}, "
                  f"n={cp['n']})   cos_scale {cs['mean']:+.3f}   |grad| {nm['mean']:.3e}")
        ct = summarise([c for c in acc_terms[fname] if c == c])
        if ct.get("n"):
            print(f"   cos(scale_grad, repro_grad) {ct['mean']:+.3f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        print(f"\nper-sample rows -> {args.out}")


if __name__ == "__main__":
    main()
