"""Does the 19.3% of (sample, target) pairs where pass 2 is WORSE than pass 1
share a detectable, GT-free signature -- one an inference-time gate could use?

Motivation.  report_20260916_pass1_prior_diag.md established that 37/192 pairs
regress, that pass-1 source-depth error explains only r=+0.273 of it, and that
warp coverage predicts pass 2's *margin* but not the *occurrence* of a
regression.  This script tests a different family of features, centred on the
one thing the splat provably throws away:

    utils/warp.py:168-179 keeps EVERY source view whose z is within
    z_tolerance=2% of the front-most surface, then writes colour by plain
    assignment.  When k>1 source views survive at a target pixel, the
    largest-s one silently overwrites the rest.  Their disagreement -- in
    colour and in z -- is never computed and never reaches the DiT.

So the headline features here are per-target-view multi-source-view statistics:
how many source views survive per pixel, and how much they disagree.  Those are
computable at inference from the source images and pass-1's predicted source
depth alone.

Feature groups (per (sample, target view)):
  geom     free from the poses alone (no model, no GT)
  gtfree   needs pass 1 and/or pass 2, but no ground truth -- deployable
  gt       needs ground truth; NOT deployable.  Included as an upper bound:
           if even the GT-using feature cannot predict the regression, no
           GT-free surrogate of it will either.

Labels come from an existing run's records.jsonl (per-pair lpips of p1 and
p2same), joined on (b, t).  Pass 1 is re-run here with the same seed and the
join is hard-gated on reproducing the stored lpips(p1), so a silent mismatch
between the two scripts' sample ordering cannot go unnoticed.

Analysis reported at the end: Pearson vs delta = lpips(p2)-lpips(p1), AUC for
the binary "p2 worse" label, and a gating simulation ("use p1 where flagged,
else p2") both in-sample (optimistic) and 2-fold cross-fitted, against the two
numbers that bound it: always-p2 and the perfect-selection oracle.

Run as a MODULE from the project root, one GPU:
  python -m tools.diag_p2_regress_features \
      --run logs/bmvs_d_drocefp32_s42 --step 3750 \
      --labels tools/pass1_prior_a2_bmvs_d_drocefp32_s42_step3750/records.jsonl

Re-analyse without a GPU:
  python -m tools.diag_p2_regress_features --from-features <out_dir>/features.jsonl
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

from diffsynth.models.wan_video_camera_controller import fp32_geometry
from examples.wanvideo.model_training.train_with_accelerate_rgbd_nvs import custom_collate_fn
from tools.vis_rgb_warp_pred_depth import build_model, build_dataset
from utils.warp import _pixel_rays_to_world, splat_sources_to_targets

EPS = 1e-3
Z_NEAR = 1e-3
Z_TOL = 0.02

# Which features need what.  Kept as data so the analysis can report the three
# groups separately -- mixing a GT-using feature into a "deployable gate" table
# is the easiest way to fool yourself here.
GEOM_FEATS = ["tgt_dist_min", "tgt_dist_mean", "tgt_dist_min_rel", "tgt_ang_min", "tgt_ang_mean"]
GTFREE_FEATS = [
    "cov", "nviews_mean", "frac_multi", "std_rgb_mean", "std_rgb_p90", "frac_std_hi",
    "relstd_z_mean", "frac_relstd_z_hi", "loo_src_err", "loo_src_err_near",
    "splat_vs_p1_l1", "splat_vs_p2_l1", "splat_vs_p2_lpips", "p1_vs_p2_l1",
]
GT_FEATS = ["splat_err_gt", "splat_lpips_gt", "src_disp_l1_gt"]


# ----------------------------------------------------------------------------
# instrumented splat: same keep-logic as utils/warp.py, but per source view
# ----------------------------------------------------------------------------
@torch.no_grad()
def splat_per_source_stats(src_rgb, src_depth, src_valid, src_K, src_c2w,
                           tgt_K, tgt_c2w, geometry_fp32=False, exclude_self=False):
    """Per-target-pixel statistics ACROSS the source views that survive the
    z-test, which `_splat_sources_to_targets` computes and then discards.

    The zbuf/keep logic below is copied line-for-line from
    utils/warp.py:140-179 so the surviving set is identical; the only change is
    that instead of `warped[...] = colour` (last writer wins) we accumulate
    sum/sumsq/count over the survivors.

    Within one source view several source pixels can land on the same target
    pixel; the original resolves that race by assignment order.  Here each
    source view contributes exactly one value per target pixel, the FRONT-MOST
    one -- well-defined, and equal to the original's choice except in the rare
    case of two source pixels of the same view at the same z.

    exclude_self: skip s == t (for a leave-one-out source-to-source check,
    where the target list IS the source list and s == t would be the identity).

    Returns dict of [B, T, H, W] float tensors:
        cnt        number of source views surviving at this pixel (0 = uncovered)
        std_rgb    std across surviving views of the splatted colour, mean over
                   the 3 channels; 0 where cnt < 2
        relstd_z   std/mean across surviving views of z; 0 where cnt < 2
        err_self   |front-most survivor colour - tgt_rgb_ref| (filled by caller)
        rgb_front  [B, T, 3, H, W] colour of the globally front-most survivor
    """
    with fp32_geometry(src_rgb.device, geometry_fp32):
        batch, num_src, _, height, width = src_rgb.shape
        num_tgt = tgt_K.shape[1]
        device = src_rgb.device
        num_pix = height * width

        world = _pixel_rays_to_world(
            src_depth.reshape(batch * num_src, height, width),
            src_valid.reshape(batch * num_src, height, width),
            src_K.reshape(batch * num_src, 3, 3),
            src_c2w.reshape(batch * num_src, 4, 4),
        ).reshape(batch, num_src, num_pix, 3)
        colors = src_rgb.float().reshape(batch, num_src, 3, num_pix).transpose(2, 3)
        valid = src_valid.reshape(batch, num_src, num_pix)

        out_cnt = torch.zeros(batch, num_tgt, num_pix, device=device)
        out_std_rgb = torch.zeros(batch, num_tgt, num_pix, device=device)
        out_relstd_z = torch.zeros(batch, num_tgt, num_pix, device=device)
        out_rgb_front = torch.zeros(batch, num_tgt, num_pix, 3, device=device)

        for t in range(num_tgt):
            w2c = torch.linalg.inv(tgt_c2w[:, t].float())
            K_t = tgt_K[:, t].float()

            projected = []
            zbuf = torch.full((batch, num_pix), float("inf"), device=device)
            for s in range(num_src):
                if exclude_self and s == t:
                    projected.append(None)
                    continue
                cam = world[:, s] @ w2c[:, :3, :3].transpose(1, 2) + w2c[:, :3, 3].unsqueeze(1)
                z = cam[..., 2]
                uvw = cam @ K_t.transpose(1, 2)
                col = torch.where(z > Z_NEAR, uvw[..., 0] / z.clamp_min(Z_NEAR),
                                  torch.full_like(z, -1.0))
                row = torch.where(z > Z_NEAR, uvw[..., 1] / z.clamp_min(Z_NEAR),
                                  torch.full_like(z, -1.0))
                xi = col.floor().long()
                yi = row.floor().long()
                in_view = ((xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
                           & (z > Z_NEAR) & valid[:, s])
                flat = torch.where(in_view, yi * width + xi, torch.zeros_like(xi))
                zbuf.scatter_reduce_(
                    1, flat, torch.where(in_view, z, torch.full_like(z, float("inf"))),
                    reduce="amin", include_self=True)
                projected.append((flat, z, in_view))

            cnt = torch.zeros(batch, num_pix, device=device)
            sum_rgb = torch.zeros(batch, num_pix, 3, device=device)
            sumsq_rgb = torch.zeros(batch, num_pix, 3, device=device)
            sum_z = torch.zeros(batch, num_pix, device=device)
            sumsq_z = torch.zeros(batch, num_pix, device=device)
            zfront = torch.full((batch, num_pix), float("inf"), device=device)

            for s in range(num_src):
                if projected[s] is None:
                    continue
                flat, z, in_view = projected[s]
                front = zbuf.gather(1, flat)
                keep = in_view & (z <= front * (1.0 + Z_TOL) + 1e-6)
                if not bool(keep.any()):
                    continue
                # one value per (target pixel, source view): the front-most.
                z_s = torch.full((batch, num_pix), float("inf"), device=device)
                z_s.scatter_reduce_(
                    1, flat, torch.where(keep, z, torch.full_like(z, float("inf"))),
                    reduce="amin", include_self=True)
                is_front_of_s = keep & (z <= z_s.gather(1, flat) * (1.0 + 1e-6))
                bidx = torch.arange(batch, device=device).unsqueeze(1).expand_as(flat)[is_front_of_s]
                pidx = flat[is_front_of_s]
                rgb_s = torch.zeros(batch, num_pix, 3, device=device)
                rgb_s[bidx, pidx] = colors[:, s][is_front_of_s]

                has_s = torch.isfinite(z_s)
                zv = torch.where(has_s, z_s, torch.zeros_like(z_s))
                cnt += has_s.float()
                sum_rgb += rgb_s
                sumsq_rgb += rgb_s * rgb_s
                sum_z += zv
                sumsq_z += zv * zv
                # globally front-most survivor's colour, for the composite
                newer = has_s & (z_s < zfront)
                zfront = torch.where(newer, z_s, zfront)
                out_rgb_front[:, t] = torch.where(newer.unsqueeze(-1), rgb_s, out_rgb_front[:, t])

            n = cnt.clamp_min(1.0)
            multi = cnt >= 2
            var_rgb = (sumsq_rgb / n.unsqueeze(-1) - (sum_rgb / n.unsqueeze(-1)) ** 2).clamp_min(0.0)
            out_std_rgb[:, t] = torch.where(multi, var_rgb.sqrt().mean(dim=-1),
                                            torch.zeros_like(cnt))
            mean_z = sum_z / n
            var_z = (sumsq_z / n - mean_z ** 2).clamp_min(0.0)
            out_relstd_z[:, t] = torch.where(
                multi, var_z.sqrt() / mean_z.clamp_min(1e-6), torch.zeros_like(cnt))
            out_cnt[:, t] = cnt

        shape = (batch, num_tgt, height, width)
        return {
            "cnt": out_cnt.reshape(shape),
            "std_rgb": out_std_rgb.reshape(shape),
            "relstd_z": out_relstd_z.reshape(shape),
            "rgb_front": out_rgb_front.permute(0, 1, 3, 2).reshape(
                batch, num_tgt, 3, height, width),
        }


# ----------------------------------------------------------------------------
# small stats helpers
# ----------------------------------------------------------------------------
def auc(scores, labels):
    """Rank-based AUC.  labels: 1 = positive (p2 worse)."""
    pairs = sorted(zip(scores, labels))
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # average ranks for ties
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    s_pos = sum(r for r, (_, lab) in zip(ranks, pairs) if lab)
    return (s_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return num / den if den > 0 else float("nan")


def gate_best(scores, lp1, lp2, idx_fit, idx_apply):
    """Best single threshold on `scores` fitted on idx_fit, evaluated on
    idx_apply.  Gate = "use p1 where score >= thr, else p2".  Both directions
    are tried, because for some features high means unreliable and for others
    low does; the chosen direction is part of what is fitted."""
    best = None
    for sign in (+1.0, -1.0):
        # +inf must be in the candidate set, otherwise "decline to gate, always
        # use p2" is outside the hypothesis space and a useless feature can be
        # reported as fitting to something worse than the trivial baseline.
        cand = sorted({sign * scores[i] for i in idx_fit})
        for thr in [-float("inf")] + cand + [float("inf")]:
            tot = sum(lp1[i] if sign * scores[i] >= thr else lp2[i] for i in idx_fit)
            if best is None or tot / len(idx_fit) < best[0]:
                best = (tot / len(idx_fit), sign, thr)
    _, sign, thr = best
    applied = [lp1[i] if sign * scores[i] >= thr else lp2[i] for i in idx_apply]
    n_flag = sum(1 for i in idx_apply if sign * scores[i] >= thr)
    return sum(applied) / len(applied), n_flag, sign, thr


def analyse(rows, out_path=None):
    lines = []

    def emit(s):
        print(s, flush=True)
        lines.append(s)

    lp1 = [r["lpips_p1"] for r in rows]
    lp2 = [r["lpips_p2"] for r in rows]
    delta = [b - a for a, b in zip(lp1, lp2)]
    label = [1 if d > 0 else 0 for d in delta]
    n = len(rows)
    always_p2 = sum(lp2) / n
    always_p1 = sum(lp1) / n
    oracle = sum(min(a, b) for a, b in zip(lp1, lp2)) / n

    emit(f"=== {n} (sample, target) pairs; {sum(label)} regress ({sum(label)/n:.1%}) ===")
    emit(f"always-p1={always_p1:.4f}  always-p2={always_p2:.4f}  "
         f"perfect-selection oracle={oracle:.4f}  (ceiling for ANY gate: "
         f"{always_p2 - oracle:.4f} lpips)")
    emit("")

    for group, feats in (("geom (poses only)", GEOM_FEATS),
                         ("gtfree (deployable)", GTFREE_FEATS),
                         ("gt (NOT deployable -- upper bound)", GT_FEATS)):
        emit(f"--- {group} ---")
        emit(f"{'feature':>20} {'mean':>9} {'pearson':>8} {'AUC':>6} "
             f"{'gate(in-samp)':>14} {'gate(2fold)':>12} {'flag%':>6}")
        scored = []
        for f in feats:
            xs = [r["feat"][f] for r in rows]
            if not all(math.isfinite(v) for v in xs):
                emit(f"{f:>20}   (non-finite values, skipped)")
                continue
            pr = pearson(xs, delta)
            a = auc(xs, label)
            # in-sample: fit and apply on everything (optimistic by construction)
            g_in, nf, sign, thr = gate_best(xs, lp1, lp2, range(n), range(n))
            # 2-fold cross-fitted, interleaved so both folds span all scenes
            fa = [i for i in range(n) if i % 2 == 0]
            fb = [i for i in range(n) if i % 2 == 1]
            ga, _, _, _ = gate_best(xs, lp1, lp2, fb, fa)
            gb, _, _, _ = gate_best(xs, lp1, lp2, fa, fb)
            g_cv = (ga * len(fa) + gb * len(fb)) / n
            scored.append((a, f))
            emit(f"{f:>20} {sum(xs)/n:9.4f} {pr:+8.3f} {a:6.3f} "
                 f"{g_in:14.4f} {g_cv:12.4f} {100*nf/n:5.1f}%")
        if scored:
            best_a, best_f = max(scored, key=lambda p: abs(p[0] - 0.5))
            emit(f"  strongest by |AUC-0.5|: {best_f} (AUC={best_a:.3f})")
        emit("")

    # multi-feature linear probe on the deployable set, cross-fitted
    try:
        import numpy as np
        feats = [f for f in GTFREE_FEATS
                 if all(math.isfinite(r["feat"][f]) for r in rows)]
        X = np.array([[r["feat"][f] for f in feats] for r in rows], dtype=np.float64)
        X = (X - X.mean(0)) / X.std(0).clip(1e-9)
        X = np.hstack([X, np.ones((n, 1))])
        y = np.array(delta)
        emit("--- multi-feature linear probe on the deployable set (predict delta) ---")
        fa = np.arange(n) % 2 == 0
        preds = np.zeros(n)
        for tr, te in ((fa, ~fa), (~fa, fa)):
            beta, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
            preds[te] = X[te] @ beta
        r_cv = float(np.corrcoef(preds, y)[0, 1])
        gated = np.where(preds > 0, np.array(lp1), np.array(lp2)).mean()
        emit(f"  cross-fitted pearson(pred_delta, delta) = {r_cv:+.3f}")
        emit(f"  cross-fitted AUC                        = {auc(list(preds), label):.3f}")
        emit(f"  gate on pred_delta>0 -> {gated:.4f}   "
             f"(always-p2 {always_p2:.4f}, oracle {oracle:.4f}), "
             f"flagged {100*float((preds>0).mean()):.1f}%")
        beta_full, *_ = np.linalg.lstsq(X, y, rcond=None)
        order = np.argsort(-np.abs(beta_full[:-1]))
        emit("  |standardised coef| ranking: "
             + ", ".join(f"{feats[i]}={beta_full[i]:+.4f}" for i in order[:6]))
    except ImportError:
        emit("(numpy unavailable, skipping the multi-feature probe)")
    emit("")

    # the sub-question: among regressing pairs only, what is different?
    emit("--- group means: regressing vs improving pairs ---")
    reg = [r for r, l in zip(rows, label) if l]
    imp = [r for r, l in zip(rows, label) if not l]
    emit(f"{'feature':>20} {'regress':>10} {'improve':>10} {'ratio':>7}")
    for f in GEOM_FEATS + GTFREE_FEATS + GT_FEATS:
        a = sum(r["feat"][f] for r in reg) / max(len(reg), 1)
        b = sum(r["feat"][f] for r in imp) / max(len(imp), 1)
        if not (math.isfinite(a) and math.isfinite(b)):
            continue
        emit(f"{f:>20} {a:10.4f} {b:10.4f} {a/b if b else float('nan'):7.2f}")

    if out_path:
        with open(out_path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"\nwrote {out_path}")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--step", type=int)
    ap.add_argument("--labels", default=None,
                    help="records.jsonl from tools/diag_pass1_prior.py (source of "
                         "the per-pair lpips(p1)/lpips(p2same) labels)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--max-chunks", type=int, default=None)
    ap.add_argument("--std-hi", type=float, default=0.10,
                    help="threshold for frac_std_hi (source-view colour disagreement)")
    ap.add_argument("--relstd-z-hi", type=float, default=0.05,
                    help="threshold for frac_relstd_z_hi (source-view z disagreement)")
    ap.add_argument("--from-features", default=None,
                    help="skip the GPU work and re-analyse an existing features.jsonl")
    args = ap.parse_args()

    if args.from_features:
        rows = [json.loads(l) for l in open(args.from_features)]
        analyse(rows, os.path.join(os.path.dirname(args.from_features), "analysis.txt"))
        return

    if not (args.run and args.step and args.labels):
        raise SystemExit("--run, --step and --labels are required unless --from-features")

    out_dir = args.out_dir or (
        f"tools/p2_regress_feats_{os.path.basename(args.run)}_step{args.step}")
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
        raise SystemExit("lpips is required for this diagnostic")

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

            # Pass 1 and pass 2 exactly as diag_pass1_prior.py's p1 / p2same:
            # one generator, pass 2 re-seeded so it reuses pass 1's noise draw.
            blind = dict(inputs)
            blind["warp_block"] = torch.zeros_like(inputs["warp_block"])
            if inputs.get("depth_warp_block") is not None:
                blind["depth_warp_block"] = torch.zeros_like(inputs["depth_warp_block"])
            g = torch.Generator(device=device)
            g.manual_seed(seed)
            dep1, rgb1 = model._sample_rgb_once(blind, num_steps, g)
            warp2, dwarp2 = model._warp_block_from_predicted_depth(inputs, dep1)
            second = dict(inputs)
            second["warp_block"] = warp2
            second["depth_warp_block"] = dwarp2
            csd = dep1[:, :, :num_src] if model.clean_source_depth else None
            g2 = torch.Generator(device=device)
            g2.manual_seed(seed)
            dep2, rgb2 = model._sample_rgb_once(second, num_steps, g2, clean_src_depth=csd)
            if model.latent_scale != 1.0:
                rgb1 = rgb1 / model.latent_scale
                rgb2 = rgb2 / model.latent_scale

            rgb_px = batch["images"].to(device).float().clamp(0.0, 1.0)   # [B,V,3,H,W]
            disp_gt = batch["disparity"].to(device).float()               # [B,V,3,H,W]
            loss_masks = inputs["loss_masks"]
            pixel_K = inputs["pixel_K"].float()
            c2w = inputs["c2w"].float()
            batch_size = rgb_px.shape[0]

            # pass-1 source disparity, RAW-latent convention (the one
            # _warp_block_from_predicted_depth uses internally)
            lat = dep1[:, :, :num_src] / model.depth_latent_scale
            p1_src_disp = torch.stack([
                model.pipe.vae.decode(lat[:, :, v:v + 1].to(model.pipe.torch_dtype),
                                      device=device)[:, :, 0].float().mean(dim=1)
                for v in range(num_src)], dim=1).clamp(0.0, 1.0)          # [B,S,H,W]

            def decode_tgt(latents):
                return {v: model.pipe.vae.decode(
                    latents[:, :, v:v + 1].to(model.pipe.torch_dtype),
                    device=device)[:, :, 0].float().clamp(0, 1)
                    for v in range(num_src, num_src + num_tgt)}
            p1_tgt = decode_tgt(rgb1)
            p2_tgt = decode_tgt(rgb2)

            # the real splat pass 2 saw
            warped, wmask = model._splat_sources_pixel(rgb_px, p1_src_disp, pixel_K, c2w)

            # instrumented version: same keep-logic, per-source-view stats
            valid_s = p1_src_disp > EPS
            depth_s = torch.where(valid_s, 1.0 / p1_src_disp.clamp_min(EPS),
                                  torch.zeros_like(p1_src_disp))
            st = splat_per_source_stats(
                rgb_px[:, :num_src], depth_s, valid_s, pixel_K[:, :num_src],
                c2w[:, :num_src], pixel_K[:, num_src:], c2w[:, num_src:],
                geometry_fp32=model.camera_geometry_fp32)
            # Hard gate: the surviving SET must be identical to the shipped
            # splat's.  (Colour can differ on ties, which is exactly the
            # ambiguity this script is measuring, so only the mask is checked.)
            if chunk_idx == 0:
                mism = float(((st["cnt"] >= 1).float() - wmask[:, :, 0]).abs().max())
                print(f"[self-check] instrumented splat coverage vs shipped: "
                      f"max_abs_diff={mism:.3e}", flush=True)
                if mism > 0:
                    raise SystemExit("self-check FAILED -- keep-logic diverged")

            # leave-one-out source -> source, using pass-1's predicted depth.
            # Fully GT-free: the source images are given clean as conditioning.
            st_loo = splat_per_source_stats(
                rgb_px[:, :num_src], depth_s, valid_s, pixel_K[:, :num_src],
                c2w[:, :num_src], pixel_K[:, :num_src], c2w[:, :num_src],
                geometry_fp32=model.camera_geometry_fp32, exclude_self=True)
            loo_cov = (st_loo["cnt"] >= 1).float()                         # [B,S,H,W]
            loo_err = ((st_loo["rgb_front"] - rgb_px[:, :num_src]).abs().mean(dim=2)
                       * loo_cov)                                          # [B,S,H,W]
            loo_per_src = loo_err.sum(dim=(2, 3)) / loo_cov.sum(dim=(2, 3)).clamp_min(1.0)

            # camera geometry, and a scene-scale normaliser that needs no GT
            centres = c2w[:, :, :3, 3]                                     # [B,V,3]
            fwd = c2w[:, :, :3, 2]
            fwd = fwd / fwd.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            med_depth = (1.0 / p1_src_disp.clamp_min(EPS)).flatten(1).median(dim=1).values

            for b in range(batch_size):
                for t in range(num_tgt):
                    vi = num_src + t
                    key = (chunk_idx * eval_batch_size + b, t)
                    if key not in labels:
                        continue
                    lab = labels[key]
                    gt = rgb_px[b:b + 1, vi]
                    m = wmask[b, t, 0].bool()                              # [H,W]
                    cnt = st["cnt"][b, t]
                    cov = m.float()
                    n_cov = cov.sum().clamp_min(1.0)
                    multi = (cnt >= 2).float()
                    n_multi = multi.sum().clamp_min(1.0)
                    sr = st["std_rgb"][b, t]
                    rz = st["relstd_z"][b, t]

                    d = centres[b, vi] - centres[b, :num_src]
                    dist = d.norm(dim=-1)
                    ang = torch.rad2deg(torch.acos(
                        (fwd[b, vi] * fwd[b, :num_src]).sum(-1).clamp(-1, 1)))
                    near3 = torch.topk(-dist, k=min(3, num_src)).indices

                    def masked_l1(a, bb):
                        return float(((a[0] - bb[0]).abs().mean(dim=0) * cov).sum() / n_cov)

                    def fill(a):
                        """Composite the splat over a reference, so a perceptual
                        metric only sees the covered region's disagreement."""
                        return warped[b:b + 1, t] * cov + a * (1 - cov)

                    mk = loss_masks[b, 0, :num_src, 0].bool()
                    sd = float((((p1_src_disp[b] - disp_gt[b, :num_src, 0]).abs() * mk)
                                .sum() / mk.sum().clamp_min(1)))

                    feat = {
                        "tgt_dist_min": float(dist.min()),
                        "tgt_dist_mean": float(dist.mean()),
                        "tgt_dist_min_rel": float(dist.min() / med_depth[b].clamp_min(1e-6)),
                        "tgt_ang_min": float(ang.min()),
                        "tgt_ang_mean": float(ang.mean()),
                        "cov": float(cov.mean()),
                        "nviews_mean": float((cnt * cov).sum() / n_cov),
                        "frac_multi": float(multi.sum() / n_cov),
                        "std_rgb_mean": float((sr * multi).sum() / n_multi),
                        "std_rgb_p90": float(torch.quantile(
                            sr[multi > 0].float(), 0.9)) if float(n_multi) > 1 else 0.0,
                        "frac_std_hi": float(((sr > args.std_hi).float() * multi).sum() / n_cov),
                        "relstd_z_mean": float((rz * multi).sum() / n_multi),
                        "frac_relstd_z_hi": float(
                            ((rz > args.relstd_z_hi).float() * multi).sum() / n_cov),
                        "loo_src_err": float(loo_per_src[b].mean()),
                        "loo_src_err_near": float(loo_per_src[b][near3].mean()),
                        "splat_vs_p1_l1": masked_l1(warped[b:b + 1, t], p1_tgt[vi][b:b + 1]),
                        "splat_vs_p2_l1": masked_l1(warped[b:b + 1, t], p2_tgt[vi][b:b + 1]),
                        "splat_vs_p2_lpips": lpips(fill(p2_tgt[vi][b:b + 1]),
                                                   p2_tgt[vi][b:b + 1]),
                        "p1_vs_p2_l1": float((p1_tgt[vi][b] - p2_tgt[vi][b]).abs().mean()),
                        "splat_err_gt": masked_l1(warped[b:b + 1, t], gt),
                        "splat_lpips_gt": lpips(fill(gt), gt),
                        "src_disp_l1_gt": sd,
                    }
                    row = {"b": key[0], "t": t, "sample_id": lab["sample_id"],
                           "lpips_p1": lab["lpips_p1"], "lpips_p2": lab["lpips_p2"],
                           "feat": feat}
                    # Join gate: pass 1 is deterministic given the seed, so the
                    # re-run must reproduce the stored lpips(p1).  A mismatch
                    # means the two scripts are not looking at the same pair.
                    chk = lpips(p1_tgt[vi][b:b + 1], gt)
                    if abs(chk - lab["lpips_p1"]) > 2e-3:
                        raise SystemExit(
                            f"join FAILED at b={key[0]} t={t}: recomputed lpips(p1)="
                            f"{chk:.4f} vs stored {lab['lpips_p1']:.4f}")
                    rows.append(row)
                    feats_file.write(json.dumps(row) + "\n")
            feats_file.flush()
        print(f"[chunk {chunk_idx}] {time.time()-t_start:.0f}s total, "
              f"{len(rows)} pairs", flush=True)

    feats_file.close()
    print(f"\nwrote {feats_path}\n", flush=True)
    analyse(rows, os.path.join(out_dir, "analysis.txt"))


if __name__ == "__main__":
    main()
