"""Is pass-1's predicted SOURCE depth multi-view scale-INCONSISTENT, and does
that inconsistency explain the corrupted splats?

Motivation.  report_20260916_pass1_prior_diag.md §8a concluded that the
"several source views disagree at one target pixel" hypothesis was falsified:
only 7.0% of covered target pixels have >=2 surviving source views, so the
last-writer-wins overwrite in utils/warp.py throws away almost nothing.  That
conclusion is correct for the quantity it measured and BLIND to the failure
mode it was aimed at.  A per-view depth SCALE error does not put two values on
one target pixel; it puts the same 3D structure at two DIFFERENT target pixels
-- displaced duplication, which registers as exactly zero same-pixel variance.
b05/t2 is the worked example: frac_multi=0.026 and same-pixel disagreement
AUC=0.546, yet splat_err_gt=0.2587 against a 0.1689 mean.

The dataset makes multi-view consistency expressible, so an inconsistency is a
real error and not a parameterisation artefact:
kubric4d_rgbd_nvs_dataset.py:_shared_disparity_scale picks ONE scale s per
sample from the pooled SOURCE inverse depths and sets disp = (1/d)/s for every
view, scale-only with no shift, rescaling the camera translations by the same s.
The same 3D point therefore has a consistent disparity value in every view.
Nothing in the training objective rewards that consistency, though: the depth
loss is an independent per-frame MSE (train_with_accelerate_rgbd_nvs.py:665).

Three blocks, per sample unless stated:

  1a  GT-based, decisive for the mechanism.  Fit disparity -> GT per view
      (6 scales) and once for all views pooled (1 scale).  The residual GAP
      between the two fits is the multi-view inconsistency, and the spread of
      the per-view scales is its size.  Guarded by a within-view half-split
      noise floor, because a spread smaller than the fit noise is not a
      finding.

  1b  GT-FREE, and the direct measurement of the duplication mechanism.  Splat
      view i's predicted depth into view j and compare the disparity it implies
      there against view j's OWN prediction.  Fitting one scale a_ij and then
      looking at the residual decomposes the pairwise inconsistency into a
      SCALE component and a STRUCTURE (displacement) component -- a pure
      disparity scale error in view i moves points along i's rays, which as
      seen from j is both a depth change and a parallax shift.  Unlike any
      target-side pairwise statistic this has full support: neighbouring source
      views overlap almost completely, whereas frac_multi=0.026 starves the
      target-side version.

  1c  Cheap side question from the same session: is the target-side splat error
      diluted by MEAN pooling?  A localised structural corruption over 3% of
      pixels at error 0.3 moves a 0.15 mean by 0.009.  Reported as mean vs
      top-10% vs top-5% pooled, scored the same way as everything else.

Labels (per-pair lpips of p1 and p2same) come from an existing run's
records.jsonl, joined on (b, t) and hard-gated on reproducing the stored
lpips(p1).

Note on n: the 1a/1b features are SOURCE-side, hence one value per sample,
repeated across that sample's 3 targets.  Per-pair statistics over 192 rows are
therefore clustered and their confidence is overstated; the analysis reports a
per-sample (n=64, mean delta) version alongside.

Run as a MODULE from the project root, one GPU:
  python -m tools.diag_src_depth_scale_consistency \
      --run logs/bmvs_d_drocefp32_s42 --step 3750 \
      --labels tools/pass1_prior_a2_bmvs_d_drocefp32_s42_step3750/records.jsonl

Re-analyse without a GPU:
  python -m tools.diag_src_depth_scale_consistency --from-features <dir>/features.jsonl
"""
import argparse
import json
import math
import os
import time

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Subset

from examples.wanvideo.model_training.train_with_accelerate_rgbd_nvs import custom_collate_fn
from tools.vis_rgb_warp_pred_depth import build_model, build_dataset
from utils.warp import splat_sources_to_targets

EPS = 1e-3
# GT disparity is clamp(0,1)'d by _disparity_from_shared_scale, so pixels at the
# rails carry no scale information and would bias every fit toward them.
GT_LO, GT_HI = 1e-3, 0.999

# 1a + 1b are source-side: ONE value per sample, repeated across its targets.
SAMPLE_FEATS = [
    "scale_spread", "scale_ratio", "res_gap_scale", "res_gap_affine",
    "halfsplit_dev", "spread_over_noise",
    "pair_scale_dev", "pair_scale_dev_top", "pair_scale_max",
    "pair_res_after_scale", "pair_photo",
    # Support, carried as features on purpose: BMVS source views overlap
    # sparsely, so any pairwise reading has to be checked against how many pairs
    # it rests on before it can be called an inconsistency.
    "n_pairs", "support_mean",
]
# 1c is per (sample, target view).
PAIR_FEATS = ["splat_err_mean", "splat_err_top10", "splat_err_top5", "splat_cov"]
REF_FEATS = ["src_disp_l1_gt"]


# ----------------------------------------------------------------------------
# fits
# ----------------------------------------------------------------------------
def fit_scale(pred, gt):
    """min_a ||a*pred - gt||^2  ->  a = <pred,gt>/<pred,pred>, in float64.

    Scale-only, not affine, because the GT convention IS scale-only: under
    disparity_norm_mode='shared_scale' the target is disp = (1/d)/s with one s
    for every view, so a globally consistent prediction needs exactly one
    scale and a per-view SHIFT is not even in the model class.  (The affine
    variant is reported too, since evaluate()'s abs_rel/delta1 align affinely
    and would hide a shift-shaped error.)
    """
    p, g = pred.double(), gt.double()
    return float((p * g).sum() / (p * p).sum().clamp_min(1e-12))


def fit_affine_ab(pred, gt):
    """min_{a,b} ||a*pred + b - gt||^2, closed form in float64."""
    p, g = pred.double(), gt.double()
    n = p.numel()
    sp, spp, sg, spg = p.sum(), (p * p).sum(), g.sum(), (p * g).sum()
    det = n * spp - sp * sp
    if abs(float(det)) < 1e-12:
        return 1.0, 0.0
    return float((n * spg - sp * sg) / det), float((spp * sg - sp * spg) / det)


def l1_after_scale(pred, gt, a):
    return float((a * pred.double() - gt.double()).abs().mean())


def l1_after_affine(pred, gt, a, b):
    return float((a * pred.double() + b - gt.double()).abs().mean())


# ----------------------------------------------------------------------------
# 1a: per-view vs shared scale
# ----------------------------------------------------------------------------
def scale_consistency(pred_views, gt_views, masks):
    """pred_views/gt_views: lists of [H,W]; masks: list of bool [H,W].

    Returns a dict.  The two headline numbers are `res_gap_scale` (how much
    worse ONE shared scale fits than SIX per-view scales -- i.e. the part of
    the depth error that is pure inter-view disagreement) and `scale_spread`
    (the size of that disagreement), plus `halfsplit_dev` as its noise floor.
    """
    ps, gs, a_v, res_v, half = [], [], [], [], []
    for pred, gt, m in zip(pred_views, gt_views, masks):
        sel = m.bool() & (gt > GT_LO) & (gt < GT_HI) & (pred > EPS)
        if int(sel.sum()) < 256:
            continue
        p, g = pred[sel], gt[sel]
        a = fit_scale(p, g)
        a_v.append(a)
        res_v.append(l1_after_scale(p, g, a))
        # Within-view fit noise: the same estimator on two disjoint halves of
        # the SAME view.  An across-view spread below this is fit noise.
        idx = torch.arange(p.numel(), device=p.device)
        ae, ao = fit_scale(p[idx % 2 == 0], g[idx % 2 == 0]), fit_scale(p[idx % 2 == 1], g[idx % 2 == 1])
        half.append(abs(ae - ao) / max(abs(a), 1e-9))
        ps.append(p)
        gs.append(g)
    if len(a_v) < 2:
        return None

    p_all, g_all = torch.cat(ps), torch.cat(gs)
    a_sh = fit_scale(p_all, g_all)
    res_shared = l1_after_scale(p_all, g_all, a_sh)
    # Per-view residual pooled over the same pixels, so the two are comparable.
    res_per = float(sum(r * p.numel() for r, p in zip(res_v, ps)) / p_all.numel())

    a_sh_aff, b_sh_aff = fit_affine_ab(p_all, g_all)
    res_shared_aff = l1_after_affine(p_all, g_all, a_sh_aff, b_sh_aff)
    res_per_aff = 0.0
    for p, g in zip(ps, gs):
        aa, bb = fit_affine_ab(p, g)
        res_per_aff += l1_after_affine(p, g, aa, bb) * p.numel()
    res_per_aff /= p_all.numel()

    mean_a = sum(a_v) / len(a_v)
    std_a = math.sqrt(sum((a - mean_a) ** 2 for a in a_v) / len(a_v))
    noise = sum(half) / len(half)
    spread = std_a / max(abs(mean_a), 1e-9)
    return {
        "scale_spread": spread,
        "scale_ratio": max(a_v) / max(min(a_v), 1e-9),
        "res_gap_scale": res_shared - res_per,
        "res_gap_affine": res_shared_aff - res_per_aff,
        "halfsplit_dev": noise,
        "spread_over_noise": spread / max(noise, 1e-9),
        "_a_v": a_v,
        "_res_shared": res_shared,
        "_res_per": res_per,
    }


# ----------------------------------------------------------------------------
# 1b: pairwise source->source, GT-free
# ----------------------------------------------------------------------------
@torch.no_grad()
def pairwise_src_consistency(src_rgb, disp, masks, pixel_K, c2w, num_src,
                             geometry_fp32=False, ref_disp=None, min_support=0.02):
    """Splat each source view into every other and compare there.

    `disp` [S,H,W] is the disparity whose consistency is under test (pass-1's
    prediction, or the GT for the self-check).  `ref_disp` is what to compare
    against in the destination view; defaults to `disp` itself, which is what
    makes this GT-FREE -- view j's own prediction is the reference, no ground
    truth anywhere.

    For each ordered pair (i != j) fits one scale a_ij mapping i's implied
    disparity at j onto j's reference, then keeps the residual.  A pure
    disparity scale error in view i slides its points along i's rays, which
    from j is both a depth change (the scale part) and a parallax shift (the
    residual part) -- so the split separates "wrong scale" from "wrong place".

    Support is the catch, and it is scene geometry rather than a threshold
    choice: BMVS source views are spread widely enough that a given ordered pair
    often barely overlaps.  Measured on the GT depth of three val samples, the
    number of ordered pairs clearing even 256 pixels was 6 / 10 / 12 out of 30.
    So `min_support` is a FRACTION of the frame (a scale fit on 0.1% of the
    pixels in one corner is noise), every aggregate is weighted by support, and
    `n_pairs` / `support_mean` are reported as features so that a reader can see
    whether a "pairwise inconsistency" reading is really a proxy for how much
    the views happen to overlap.

    Returns (dict of aggregates, list of per-pair a_ij).
    """
    ref = disp if ref_disp is None else ref_disp
    min_px = max(256, int(min_support * disp.shape[-1] * disp.shape[-2]))
    devs, ress, photos, wts = [], [], [], []
    valid = disp > EPS
    depth = torch.where(valid, 1.0 / disp.clamp_min(EPS), torch.zeros_like(disp))
    for i in range(num_src):
        warped, mask, tdisp = splat_sources_to_targets(
            src_rgb=src_rgb[None, i:i + 1].float(),
            src_depth=depth[None, i:i + 1].float(),
            src_valid=valid[None, i:i + 1],
            src_K=pixel_K[None, i:i + 1], src_c2w=c2w[None, i:i + 1],
            tgt_K=pixel_K[None, :num_src], tgt_c2w=c2w[None, :num_src],
            return_target_disparity=True, geometry_fp32=geometry_fp32,
        )
        for j in range(num_src):
            if j == i:
                continue
            sel = (mask[0, j, 0] > 0.5) & masks[j].bool() & (tdisp[0, j, 0] > EPS) & (ref[j] > EPS)
            n_sel = int(sel.sum())
            if n_sel < min_px:
                continue
            p, g = tdisp[0, j, 0][sel], ref[j][sel]
            a = fit_scale(p, g)
            devs.append(abs(a - 1.0))
            # Residual normalised by the reference's own level, so it is
            # comparable across samples with different disparity magnitudes.
            ress.append(l1_after_scale(p, g, a) / max(float(g.mean()), 1e-9))
            photos.append(float((warped[0, j] - src_rgb[j]).abs().mean(dim=0)[sel].mean()))
            wts.append(n_sel)
    if not devs:
        return None, []
    tot = float(sum(wts))
    wmean = lambda xs: float(sum(x * w for x, w in zip(xs, wts)) / tot)
    return {
        "pair_scale_dev": wmean(devs),
        "pair_scale_max": max(devs),
        # The single best-supported pair: the most trustworthy reading, and a
        # check that the weighted mean is not being carried by thin pairs.
        "pair_scale_dev_top": devs[max(range(len(wts)), key=wts.__getitem__)],
        "pair_res_after_scale": wmean(ress),
        "pair_photo": wmean(photos),
        "n_pairs": float(len(devs)),
        "support_mean": tot / len(devs) / (disp.shape[-1] * disp.shape[-2]),
    }, devs


# ----------------------------------------------------------------------------
# stats
# ----------------------------------------------------------------------------
def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    d = math.sqrt(sxx * syy)
    return float("nan") if d < 1e-18 else sxy / d


def auc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rp = sum(r for r, l in zip(ranks, labels) if l)
    return (rp - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def analyse(rows, out_path=None):
    lines = []

    def emit(s):
        print(s, flush=True)
        lines.append(s)

    delta = [r["lpips_p2"] - r["lpips_p1"] for r in rows]
    label = [1 if d > 0 else 0 for d in delta]
    n = len(rows)
    emit(f"=== {n} (sample, target) pairs; {sum(label)} regress ({sum(label)/n:.1%}) ===")
    emit("")

    emit("--- 1a/1b source-side (per SAMPLE, repeated over its targets) + 1c target-side ---")
    emit(f"{'feature':>22} {'mean':>10} {'min':>9} {'max':>9} {'pearson':>8} {'AUC':>6}")
    for f in SAMPLE_FEATS + PAIR_FEATS + REF_FEATS:
        xs = [r["feat"][f] for r in rows if f in r["feat"] and math.isfinite(r["feat"][f])]
        if len(xs) < n:
            emit(f"{f:>22}   ({n - len(xs)} missing/non-finite of {n})")
            if not xs:
                continue
        ys = [d for r, d in zip(rows, delta) if f in r["feat"] and math.isfinite(r["feat"][f])]
        ls = [l for r, l in zip(rows, label) if f in r["feat"] and math.isfinite(r["feat"][f])]
        emit(f"{f:>22} {sum(xs)/len(xs):10.4f} {min(xs):9.4f} {max(xs):9.4f} "
             f"{pearson(xs, ys):+8.3f} {auc(xs, ls):6.3f}")
    emit("")

    # Per-sample view: the source-side features have n=64, not 192, and the
    # per-pair table above treats 3 clustered rows as 3 independent ones.
    by_sample = {}
    for r, d in zip(rows, delta):
        by_sample.setdefault(r["b"], []).append((r, d))
    emit(f"--- source-side features at the SAMPLE level (n={len(by_sample)}, "
         f"label = mean delta over that sample's targets) ---")
    emit(f"{'feature':>22} {'pearson':>8} {'AUC':>6}")
    s_delta = [sum(d for _, d in v) / len(v) for v in by_sample.values()]
    s_label = [1 if d > 0 else 0 for d in s_delta]
    for f in SAMPLE_FEATS + REF_FEATS:
        xs, ys, ls = [], [], []
        for (v, sd, sl) in zip(by_sample.values(), s_delta, s_label):
            val = v[0][0]["feat"].get(f)
            if val is not None and math.isfinite(val):
                xs.append(val)
                ys.append(sd)
                ls.append(sl)
        if len(xs) < 4:
            continue
        emit(f"{f:>22} {pearson(xs, ys):+8.3f} {auc(xs, ls):6.3f}")
    emit("")

    emit("--- group means: regressing vs improving pairs ---")
    emit(f"{'feature':>22} {'regress':>10} {'improve':>10} {'ratio':>7}")
    reg = [r for r, l in zip(rows, label) if l]
    imp = [r for r, l in zip(rows, label) if not l]
    for f in SAMPLE_FEATS + PAIR_FEATS + REF_FEATS:
        a = [r["feat"][f] for r in reg if f in r["feat"] and math.isfinite(r["feat"][f])]
        b = [r["feat"][f] for r in imp if f in r["feat"] and math.isfinite(r["feat"][f])]
        if not a or not b:
            continue
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        emit(f"{f:>22} {ma:10.4f} {mb:10.4f} {ma/mb if abs(mb) > 1e-12 else float('nan'):7.2f}")
    emit("")

    # 1b's own confound, checked rather than assumed: overlap tracks baseline
    # width, and baseline width is §8b's strongest label predictor, so a
    # pairwise reading that merely tracks support would be circular.
    emit("--- 1b is pair_scale_dev a proxy for how much the views overlap? ---")
    have = [r for r in rows if math.isfinite(r["feat"].get("n_pairs", float("nan")))]
    if have:
        npair = [r["feat"]["n_pairs"] for r in have]
        n_ord = have[0].get("n_ordered_pairs", 0)
        emit(f"  n_pairs: mean={sum(npair)/len(npair):.1f}"
             f"{f' of {n_ord} ordered pairs' if n_ord else ''}, "
             f"min={min(npair):.0f} max={max(npair):.0f}; "
             f"{len(rows)-len(have)} of {len(rows)} rows had none")
        for f in ("pair_scale_dev", "pair_scale_dev_top", "pair_res_after_scale"):
            xs = [r["feat"][f] for r in have]
            emit(f"  pearson({f}, n_pairs)={pearson(xs, npair):+.3f}  "
                 f"pearson({f}, support_mean)="
                 f"{pearson(xs, [r['feat']['support_mean'] for r in have]):+.3f}")
    emit("")

    # 1c's actual question, stated as a comparison rather than left to the eye.
    emit("--- 1c does top-k pooling beat mean pooling on the splat error? ---")
    for f in ("splat_err_mean", "splat_err_top10", "splat_err_top5"):
        xs = [r["feat"][f] for r in rows if math.isfinite(r["feat"].get(f, float("nan")))]
        ls = [l for r, l in zip(rows, label) if math.isfinite(r["feat"].get(f, float("nan")))]
        if xs:
            emit(f"  {f:>18}  AUC={auc(xs, ls):.3f}")
    emit("  (report_20260916 §8b measured the mean-pooled GT splat error at AUC 0.757)")

    if out_path:
        with open(out_path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"\nwrote {out_path}", flush=True)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--step", type=int)
    ap.add_argument("--labels", default=None,
                    help="records.jsonl from a diag_pass1_prior run, for the p1/p2 labels")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--max-chunks", type=int, default=None)
    ap.add_argument("--min-support", type=float, default=0.02,
                    help="1b: minimum fraction of the destination frame an ordered "
                         "source pair must cover to be fitted at all.  BMVS source "
                         "views overlap sparsely, so this is the difference between a "
                         "scale fit and a corner of noise")
    ap.add_argument("--gt-pair-tol", type=float, default=0.15,
                    help="self-check: splatting GT depth between source views must "
                         "recover a scale of 1 to within this, or the shared-scale "
                         "assumption (or this script's geometry) is wrong")
    ap.add_argument("--from-features", default=None)
    args = ap.parse_args()

    if args.from_features:
        rows = [json.loads(l) for l in open(args.from_features)]
        analyse(rows, os.path.join(os.path.dirname(args.from_features), "analysis.txt"))
        return
    if not (args.run and args.step and args.labels):
        raise SystemExit("--run, --step and --labels are required unless --from-features")

    out_dir = args.out_dir or (
        f"tools/src_scale_consist_{os.path.basename(args.run)}_step{args.step}")
    os.makedirs(out_dir, exist_ok=True)

    labels = {}
    for line in open(args.labels):
        r = json.loads(line)
        labels[(r["b"], r["t"])] = {
            "lpips_p1": r["variants"]["p1"]["lpips"],
            "lpips_p2": r["variants"]["p2same"]["lpips"],
            "sample_id": r["sample_id"],
        }
    print(f"loaded {len(labels)} labelled pairs from {args.labels}", flush=True)

    cfg = OmegaConf.load(os.path.join(args.run, "args.yaml"))
    OmegaConf.resolve(cfg)
    if cfg.get("disparity_norm_mode", "shared_scale") != "shared_scale":
        raise SystemExit("this diagnostic assumes disparity_norm_mode=shared_scale: "
                         "under per_view normalisation the GT itself is not "
                         "multi-view consistent and 1a has no meaning")
    set_seed(0)
    accelerator = Accelerator(mixed_precision=cfg.mixed_precision)
    model = build_model(cfg, accelerator)
    ckpt = os.path.join(args.run, f"checkpoint-step-{args.step}")
    nfiles = len(os.listdir(ckpt))
    if nfiles != 11:
        raise SystemExit(f"{ckpt} has {nfiles} files, expected 11 -- write was interrupted")
    missing, unexpected = model.load_state_dict(
        load_file(os.path.join(ckpt, "model.safetensors"), device="cpu"), strict=False)
    print(f"loaded {ckpt}: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model.to(accelerator.device)
    model.eval()
    device = model.pipe.device
    lpips_net = model._get_lpips(device)
    if lpips_net is None:
        raise SystemExit("lpips is required (the label join is gated on it)")

    def lpips(a, b):
        with torch.no_grad():
            return float(lpips_net(a.float() * 2 - 1, b.float() * 2 - 1).mean())

    dataset = build_dataset(cfg)
    eval_max_samples = int(cfg.get("eval_max_samples", 2))
    eval_batch_size = max(1, min(eval_max_samples, int(cfg.get("eval_batch_size", 4))))
    samples_per_scene = cfg.get("samples_per_scene", 1)
    eval_indices = [i * samples_per_scene for i in range(len(dataset.records))][:eval_max_samples]
    loader = DataLoader(Subset(dataset, eval_indices), batch_size=eval_batch_size,
                        shuffle=False, num_workers=0, collate_fn=custom_collate_fn,
                        drop_last=False)

    num_src = model.num_source_views
    num_tgt = model.num_target_views
    num_steps = model.eval_num_inference_steps
    seed = model.eval_sample_seed
    geom_fp32 = model.camera_geometry_fp32
    autocast_ctx = torch.autocast(device_type="cuda", dtype=model.pipe.torch_dtype, enabled=True)

    feats_path = os.path.join(out_dir, "features.jsonl")
    feats_file = open(feats_path, "w")
    rows = []
    t_start = time.time()

    for chunk_idx, batch in enumerate(loader):
        if args.max_chunks is not None and chunk_idx >= args.max_chunks:
            break
        with torch.no_grad(), autocast_ctx:
            model._eval_inputs = True
            try:
                inputs = model.forward_preprocess(batch)
            finally:
                model._eval_inputs = False

            # Pass 1 exactly as diag_pass1_prior.py's p1, so the label join is
            # comparing like with like.
            blind = dict(inputs)
            blind["warp_block"] = torch.zeros_like(inputs["warp_block"])
            if inputs.get("depth_warp_block") is not None:
                blind["depth_warp_block"] = torch.zeros_like(inputs["depth_warp_block"])
            g = torch.Generator(device=device)
            g.manual_seed(seed)
            dep1, rgb1 = model._sample_rgb_once(blind, num_steps, g)
            if model.latent_scale != 1.0:
                rgb1 = rgb1 / model.latent_scale

            rgb_px = batch["images"].to(device).float().clamp(0.0, 1.0)     # [B,V,3,H,W]
            disp_gt = batch["disparity"].to(device).float()                 # [B,V,3,H,W]
            loss_masks = inputs["loss_masks"]
            pixel_K = inputs["pixel_K"].float()
            c2w = inputs["c2w"].float()
            batch_size = rgb_px.shape[0]

            # RAW-latent convention: _warp_block_from_predicted_depth divides by
            # depth_latent_scale internally, so the decode here must too.
            # Source views only: 1a/1b are entirely source-side, and the target
            # disparity would cost 3 more VAE decodes per chunk for nothing.
            lat = dep1 / model.depth_latent_scale
            p1_disp = torch.stack([
                model.pipe.vae.decode(lat[:, :, v:v + 1].to(model.pipe.torch_dtype),
                                      device=device)[:, :, 0].float().mean(dim=1)
                for v in range(num_src)], dim=1).clamp(0.0, 1.0)             # [B,S,H,W]
            p1_tgt_rgb = {
                v: model.pipe.vae.decode(
                    (rgb1[:, :, v:v + 1]).to(model.pipe.torch_dtype),
                    device=device)[:, :, 0].float().clamp(0, 1)
                for v in range(num_src, num_src + num_tgt)}

            # The target splat pass 2 actually saw, for 1c.
            warped_rgb, warp_mask = model._splat_sources_pixel(
                rgb_px, p1_disp, pixel_K, c2w)

            for b in range(batch_size):
                b_global = chunk_idx * eval_batch_size + b
                masks_src = [loss_masks[b, 0, v, 0] for v in range(num_src)]

                # --- 1a
                sc = scale_consistency(
                    [p1_disp[b, v] for v in range(num_src)],
                    [disp_gt[b, v, 0] for v in range(num_src)],
                    masks_src)
                if sc is None:
                    print(f"  b{b_global}: too few valid source pixels, skipped", flush=True)
                    continue
                if sc["_res_shared"] < sc["_res_per"] - 1e-9:
                    raise SystemExit("self-check FAILED: a shared-scale fit beat the "
                                     "per-view fits, which is impossible -- fit bug")

                # --- 1b GT-free
                pw, _ = pairwise_src_consistency(
                    rgb_px[b], p1_disp[b], masks_src, pixel_K[b], c2w[b], num_src,
                    geometry_fp32=geom_fp32, min_support=args.min_support)
                if pw is None:
                    # No pair overlaps enough for a meaningful fit.  1a and 1c
                    # are unaffected, so keep the row and let the analysis skip
                    # the non-finite 1b entries rather than losing the sample.
                    print(f"  b{b_global}: no source pair clears "
                          f"{args.min_support:.1%} support; 1b -> nan", flush=True)
                    pw = {k: float("nan") for k in SAMPLE_FEATS if k.startswith(("pair_", "n_", "support"))}

                # --- self-check: the SAME machinery on GT depth must recover
                # scale 1, or the shared-scale premise / geometry here is wrong.
                if chunk_idx == 0 and b == 0:
                    pw_gt, devs_gt = pairwise_src_consistency(
                        rgb_px[b], disp_gt[b, :, 0], masks_src, pixel_K[b], c2w[b],
                        num_src, geometry_fp32=geom_fp32, min_support=args.min_support)
                    if pw_gt is None:
                        raise SystemExit("self-check could not run: no source pair on the "
                                         "first sample clears the support floor")
                    print(f"  self-check GT pairwise |a-1|: mean={pw_gt['pair_scale_dev']:.4f} "
                          f"max={pw_gt['pair_scale_max']:.4f} over {len(devs_gt)} of "
                          f"{num_src*(num_src-1)} ordered pairs", flush=True)
                    if pw_gt["pair_scale_dev"] > args.gt_pair_tol:
                        raise SystemExit(
                            f"self-check FAILED: GT depth splatted between source views "
                            f"gives mean |a-1|={pw_gt['pair_scale_dev']:.4f} > "
                            f"{args.gt_pair_tol}; the shared-scale assumption or this "
                            f"script's geometry is wrong, so 1b would be meaningless")

                src_l1 = []
                for v in range(num_src):
                    m = masks_src[v].bool()
                    if int(m.sum()) >= 16:
                        src_l1.append(float((p1_disp[b, v][m] - disp_gt[b, v, 0][m]).abs().mean()))
                feat_sample = {k: v for k, v in sc.items() if not k.startswith("_")}
                feat_sample.update(pw)
                feat_sample["src_disp_l1_gt"] = sum(src_l1) / max(len(src_l1), 1)

                for t in range(num_tgt):
                    vi = num_src + t
                    key = (b_global, t)
                    if key not in labels:
                        raise SystemExit(f"no label for {key}; --labels does not match this run")
                    lab = labels[key]
                    chk = lpips(p1_tgt_rgb[vi][b:b + 1], rgb_px[b:b + 1, vi])
                    if abs(chk - lab["lpips_p1"]) > 2e-3:
                        raise SystemExit(
                            f"join FAILED at b={b_global} t={t}: recomputed lpips(p1)="
                            f"{chk:.4f} vs stored {lab['lpips_p1']:.4f}; the two scripts "
                            f"are not looking at the same sample")

                    m = (warp_mask[b, t, 0] > 0.5) & loss_masks[b, 0, vi, 0].bool()
                    feat = dict(feat_sample)
                    if int(m.sum()) >= 256:
                        e = (warped_rgb[b, t] - rgb_px[b, vi]).abs().mean(dim=0)[m]
                        k10 = max(1, int(e.numel() * 0.10))
                        k5 = max(1, int(e.numel() * 0.05))
                        feat["splat_err_mean"] = float(e.mean())
                        feat["splat_err_top10"] = float(torch.topk(e, k10).values.mean())
                        feat["splat_err_top5"] = float(torch.topk(e, k5).values.mean())
                        feat["splat_cov"] = float(warp_mask[b, t].mean())

                    rec = {"b": b_global, "t": t, "sample_id": lab["sample_id"],
                           "lpips_p1": lab["lpips_p1"], "lpips_p2": lab["lpips_p2"],
                           "per_view_scales": sc["_a_v"],
                           "n_ordered_pairs": num_src * (num_src - 1), "feat": feat}
                    rows.append(rec)
                    feats_file.write(json.dumps(rec) + "\n")
            feats_file.flush()
        print(f"[chunk {chunk_idx}] {time.time()-t_start:.0f}s, {len(rows)} rows", flush=True)

    feats_file.close()
    print(f"\nwrote {feats_path}\n", flush=True)
    analyse(rows, os.path.join(out_dir, "analysis.txt"))


if __name__ == "__main__":
    main()
