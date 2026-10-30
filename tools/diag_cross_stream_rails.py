"""At sigma=1, does holding one stream on the ideal trajectory help the other?

report_20260916_pass1_prior_diag.md §3c found the RGB and depth streams almost
perfectly decoupled: with oracle depth + pass-1 RGB the RGB lands at pass-1
level (lpips 0.6174 vs p1's 0.6235), and with oracle RGB + pass-1 depth the
depth lands at pass-1 level (abs_rel 0.1198 vs 0.1266).  Neither stream crosses
over.  That reading has ONE confound, which is the reason for this script: it
was measured only under a sigma=0.613 warm start, where both streams are
already locked onto pass-1's prior.  A prior-locked trajectory is a bad place to
look for coupling -- there is little left to influence.  §3c also quantified the
coupling with a control (a 15.6:1 asymmetry that survives the confound), but the
question "would a full sigma=1 run have used a good depth stream?" was never
asked, and it is exactly the question that decides whether an explicit
cross-stream consistency loss is worth a training run.

Rails, not pins.  The obvious instrument -- hold the depth frames clean at
timestep 0 -- is unusable here: the measured run has clean_source_depth=false
and one timestep per batch, so this checkpoint has NEVER seen a clean depth
frame, and pinning would put it off-distribution and confound the answer with
that.  Instead, after every Euler step the railed frames are overwritten with

    x_railed = (1 - sigma_{i+1}) * x0_target + sigma_{i+1} * eps_fixed

reusing the SAME noise draw the stream started from.  The railed frames then sit
at exactly the sigma the timestep vector claims, which is precisely the marginal
training sampled; the only thing changed is WHICH trajectory the stream rides.
At i=0 this is an identity (sigma=1 gives pure eps), and at the last step
sigma=0 gives x0_target exactly -- which doubles as a hard self-check.

Arms, all sharing one explicit noise draw so every comparison is same-noise
(§0's lesson: the production sampler silently hands pass 2 a fresh draw):

  p1              blind pass, supplies the warp and pass-1's own depth latent
  p2              the baseline, = `p2same` in the earlier report
  rail_dp1_tgt    depth railed to PASS 1's target prediction.  Control for
                  whether railing per se is harmful, and the low end of a
                  dose-response curve in depth quality.
  rail_dg_tgt     depth railed to the ORACLE target latent.  The headline arm:
                  if RGB does not move, no training-side cross-stream term can
                  pay off through this channel.
  rail_dg_all     oracle depth railed on ALL 9 frames.  Asks separately whether
                  RGB reads the SOURCE depth latents -- the warp is still built
                  from pass 1's bad source depth, so this isolates the latent
                  path from the warp path.
  rail_rg_tgt     RGB railed to the oracle.  The reverse direction: does good
                  RGB pull depth?
  warpg           the warp REBUILT FROM ORACLE SOURCE DEPTH, nothing railed.
  warpg_dg_all    oracle warp AND oracle depth latents on all 9 frames.
  warp_oscale     warp from the model's own disparity, per-view rescaled to GT.
  warp_orel       ditto but only the cross-view DISAGREEMENT removed.  Bounds
                  what any consistency loss could buy -- see the ARMS comment.
  warp_oshape     warp from GT disparity with the model's per-view scale error
                  put back.  The complement of warp_oscale.
  warp_mvs        warp from a GT-FREE 1-DOF photometric scale fit of pass 1's
                  own source disparity (report §16f-g): view j's RGB is a
                  conditioning input, not GT, so this is real, deployable
                  inference-time correction, unlike the three arms above it.
                  Scan + shrinkage k_hat**MVS_LAMBDA, both settled in §16 on
                  the SAME 64-sample eval set this script uses, from
                  tools/diag_mvs_scale_fit.py.  §16f-g's own measurement puts
                  the ceiling near warp_orel, not warp_oscale: 24-33% of
                  warpg's gap, with 15-17/64 SAMPLES made worse.

Read the dose-response (rail_dp1_tgt -> p2 -> rail_dg_tgt) rather than any
single arm, and read the worst-decile alongside the mean -- §3d is the standing
example of a bug that cost 0.0088 on the mean and 3x that in the tail.

Why the two `warpg` arms were added later (2026-09-18), and what they are FOR.
§11c(a) established that source geometry reaches RGB through the warp ONLY: the
latent path is dead (rail_dg_all beat rail_dg_tgt by 0.0011, under the 0.0012
sampling floor).  But every arm above builds the warp from `dep1`, so none of
them ever tested the one live path with a good source depth.  That leaves the
number that decides whether ANY of the source-depth work has a worthwhile
ceiling unmeasured: §6's pass 3 rebuilds the warp from pass 2's source depth for
-0.0091 lpips, which is only a lower bound because pass 2's source depth is
itself bad.  `warpg` replaces it with the oracle, so it IS that ceiling, and
`warpg_dg_all` opens the warp path and the latent path at once, so it is the
ceiling of the whole depth line of work.  The decision rule set before running:
if `warpg` lands near §11's 0.0090 (a perfect TARGET depth oracle) then the
depth-consistency programme is capped below the 0.019 eval-to-eval noise and
should stop; if it is several times that, the deployment parameters
report_20260916 §13 identified (sigma_max / num_views / max_samples) are worth a
training round.

The oracle warp is built from the GT depth LATENT, so it carries one VAE round
trip.  That is measured and negligible for this purpose: tools/diag_fcons_gt_floor.py
puts the round trip at disp_l1 0.0022 against the model's own 0.055 (§13).

Run as a MODULE from the project root, one GPU:
  python -m tools.diag_cross_stream_rails --run logs/bmvs_d_drocefp32_s42 --step 3750

Multi-GPU: chunks are assigned round-robin, one independent process per shard
(no cross-process communication -- eval is embarrassingly parallel per sample):
  python -m tools.diag_cross_stream_rails --run ... --step ... \
      --out-dir OUT/shard_0 --num-shards 8 --shard-idx 0   # x8, one per GPU
Then merge: cat OUT/shard_*/records.jsonl > OUT/records.jsonl and re-run
--from-records on the merged file (see sbatch/diag_cross_stream_rails_shard.sbatch,
a SLURM job array).  Sum, don't average, each shard's timing.json before
dividing -- see that sbatch's merge step.

Re-analyse without a GPU:
  python -m tools.diag_cross_stream_rails --from-records <dir>/records.jsonl
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
from tools.diag_mvs_scale_fit import ordered_pairs, scan_global
from tools.diag_pass1_prior import decode_disp, decode_rgb, depth_metrics, psnr_full
from tools.vis_rgb_warp_pred_depth import build_model, build_dataset

# report_20260916_pass1_prior_diag.md §16g: k = k_hat**MVS_LAMBDA.  0.5-0.8 is a
# wide plateau (29-33% reduction in |geomean(a)-1|), not a value tuned to death
# on this eval set; 0.6 sits in the middle of it.  MVS_GRID is the same
# log-spaced 1-DOF scan grid §16 settled on (symmetric so k=1 is exactly on it).
MVS_LAMBDA = 0.6
MVS_GRID = [math.exp(x) for x in
            torch.linspace(math.log(1 / 1.6), math.log(1.6), 33).tolist()]

# arm -> (warp_from, rail_stream, rail_x0_source, rail_scope)
# warp_from: None = blind (zeroed warp), "p1" = warp built from pass 1's source
# depth (what production does), "gt" = warp built from the ORACLE source depth,
# and the three DECOMPOSITION variants of the p1->gt gap described below.
ARMS = [
    ("p1", None, None, None, None),
    ("p2", "p1", None, None, None),
    ("rail_dp1_tgt", "p1", "depth", "p1", "tgt"),
    ("rail_dg_tgt", "p1", "depth", "gt", "tgt"),
    ("rail_dg_all", "p1", "depth", "gt", "all"),
    ("rail_rg_tgt", "p1", "rgb", "gt", "tgt"),
    ("warpg", "gt", None, None, None),
    ("warpg_dg_all", "gt", "depth", "gt", "all"),
    ("warp_oscale", "oscale", None, None, None),
    ("warp_orel", "orel", None, None, None),
    ("warp_oshape", "oshape", None, None, None),
    ("warp_mvs", "mvs", None, None, None),
]

# Splitting the p1->gt warp gap into SCALE and SHAPE (2026-09-18).
#
# `warpg` measured that gap at -0.1404 lpips (§14), but that is an upper bound
# using GT.  §10 named a specific, plausibly fixable component of it: the model's
# per-source-view depth scales disagree by 8.7%, which is 544x the within-view
# fitting noise and 26% of src_disp_l1, and whose predictive power survives
# controlling for the overall error magnitude.  These three arms turn that claim
# into a number by rebuilding the warp from a per-view rescaling of the model's
# OWN disparity, with a_v fitted per source view against GT:
#
#   warp_oscale   a_v * p_v            model's SHAPE, GT's per-view scale.
#                 Removes global and cross-view scale error, keeps shape error.
#   warp_orel     (a_v / geomean a) * p_v
#                 Removes ONLY the cross-view DISAGREEMENT, keeping the model's
#                 own global scale error.  This is the arm that matters most: a
#                 cross-view consistency loss (the whole f round, §12/§13) fits a
#                 ratio BETWEEN views and is therefore blind to global scale by
#                 construction, so `warp_orel` -- not `warp_oscale` -- is the
#                 ceiling of that entire idea.
#   warp_oshape   g_v / a_v            GT's SHAPE with the model's per-view scale
#                 error reimposed.  The complement: what survives of `warpg` when
#                 only scale is wrong.
#
# a_v uses the median ratio, matching _align_and_score's convention
# (WanRGBDNVSTrainingModule.py:2497) so these numbers are commensurable with
# every abs_rel/delta1 in the report rather than with the loss's least squares.
# Disparity is clamped to [0,1] as _warp_block_from_predicted_depth does; the
# clamped fraction is recorded per arm because a rescaling can push mass out of
# range, and an arm that clamps heavily is measuring the clamp, not the scale.
RGB_KEYS = ("psnr", "lpips")
DEP_KEYS = ("abs_rel", "delta1", "tgt_disp_l1")


@torch.no_grad()
def decode_src_disparity(model, latents):
    """Latent -> source disparity [B, S, H, W], the module's own readout.

    Copied from _warp_block_from_predicted_depth (WanRGBDNVSTrainingModule.py:2238)
    rather than called, because that method goes straight on to build the warp and
    the scale arms need to intervene on the disparity in between.  Any divergence
    here silently changes what the arms measure, so it is asserted against the
    method itself in main() before the arms run.
    """
    num_src = model.num_source_views
    lat = latents[:, :, :num_src] / model.depth_latent_scale
    decoded = [
        model.pipe.vae.decode(
            lat[:, :, v:v + 1].to(model.pipe.torch_dtype), device=model.pipe.device
        )[:, :, 0]
        for v in range(num_src)
    ]
    return torch.stack(decoded, dim=1).float().mean(dim=2).clamp(0.0, 1.0)


@torch.no_grad()
def per_view_scale(pred, gt, valid):
    """a_v = median(gt_v) / median(pred_v) per source view -> [B, S].

    _align_and_score's convention (WanRGBDNVSTrainingModule.py:2497).  Views with
    no valid pixels get 1.0, i.e. left alone.
    """
    b, s = pred.shape[:2]
    a = pred.new_ones((b, s))
    for i in range(b):
        for v in range(s):
            m = valid[i, v]
            if not bool(m.any()):
                continue
            p = pred[i, v][m].median()
            g = gt[i, v][m].median()
            if float(p) > 1e-8:
                a[i, v] = g / p
    return a


@torch.no_grad()
def scaled_disparity(kind, disp_p1, disp_gt, a):
    """The three decomposition arms' warp input, plus the clamped fraction."""
    if kind == "oscale":
        raw = disp_p1 * a[..., None, None]
    elif kind == "orel":
        # geometric mean over views, so the correction is scale-free: it removes
        # the DISAGREEMENT between views and leaves the overall level alone.
        geo = a.clamp_min(1e-8).log().mean(dim=1, keepdim=True).exp()
        raw = disp_p1 * (a / geo)[..., None, None]
    elif kind == "oshape":
        raw = disp_gt / a.clamp_min(1e-8)[..., None, None]
    else:
        raise ValueError(kind)
    clamped = float(((raw < 0.0) | (raw > 1.0)).float().mean())
    return raw.clamp(0.0, 1.0), clamped


@torch.no_grad()
def sample_rails(model, inputs, num_steps, eps_rgb, eps_depth,
                 rail_rgb_x0=None, rail_depth_x0=None, rail_views=None,
                 clean_src_depth=None):
    """`_sample_rgb_once` with explicit noise and optional trajectory rails.

    Rewritten rather than wrapped so the noise is an argument (every arm must
    share one draw) and so a stream can be railed.  With both rails off and the
    same noise this must reproduce `_sample_rgb_once` bit-exactly; main()
    asserts that before trusting anything else.

    `rail_*_x0` is the target latent in the SCALED space, i.e. the same space as
    inputs["rgb_gt"] / inputs["depth_gt"].  `rail_views` lists the frame indices
    to hold; None means all of them.

    Returns (depth_latent, rgb_latent), matching `_sample_rgb_once`'s order.
    """
    rgb_gt = inputs["rgb_gt"]
    sigmas = model.inference_sigmas(num_steps, device=rgb_gt.device).to(rgb_gt.dtype)
    x = eps_rgb.clone()
    x_depth = eps_depth.clone() if model.depth_flow_matching else None
    if (rail_depth_x0 is not None or rail_rgb_x0 is not None) and rail_views is None:
        rail_views = list(range(rgb_gt.shape[2]))

    s = model.num_source_views
    clean_mask = None
    if clean_src_depth is not None:
        if x_depth is None:
            raise RuntimeError("clean_src_depth requires depth_flow_matching")
        clean_src_depth = clean_src_depth.to(x_depth.dtype)
        x_depth[:, :, :s] = clean_src_depth
        clean_mask = torch.ones(1, dtype=torch.bool, device=x_depth.device)

    pred_depth = None
    for i in range(num_steps):
        timestep = (sigmas[i] * model.pipe.scheduler.num_train_timesteps).reshape(1).to(
            dtype=model.pipe.torch_dtype, device=rgb_gt.device)
        timestep = model._expand_timestep(timestep, clean_depth_mask=clean_mask)
        pred_depth, velocity = model._forward_model(
            inputs, model._build_flow_input(inputs, x, x_depth), timestep)
        dt = sigmas[i] - sigmas[i + 1]
        x = x - dt * velocity.to(x.dtype)
        if model.depth_flow_matching:
            x_depth = x_depth - dt * pred_depth.to(x_depth.dtype)
            if clean_src_depth is not None:
                x_depth[:, :, :s] = clean_src_depth
        # Put the railed frames back onto their own trajectory, at the sigma the
        # NEXT timestep will claim for them.
        sn = sigmas[i + 1]
        if rail_rgb_x0 is not None:
            for v in rail_views:
                x[:, :, v] = ((1.0 - sn) * rail_rgb_x0[:, :, v].to(x.dtype)
                              + sn * eps_rgb[:, :, v])
        if rail_depth_x0 is not None and x_depth is not None:
            for v in rail_views:
                x_depth[:, :, v] = ((1.0 - sn) * rail_depth_x0[:, :, v].to(x_depth.dtype)
                                    + sn * eps_depth[:, :, v])
    if model.depth_flow_matching:
        return x_depth, x
    return pred_depth, x


def st_mean(vals):
    vals = [v for v in vals if v == v]
    return sum(vals) / len(vals) if vals else float("nan")


def summarise(records, out_path=None):
    lines = []

    def emit(s):
        print(s, flush=True)
        lines.append(s)

    arms = [a[0] for a in ARMS]
    n = len(records)
    emit(f"=== {n} (sample, target) pairs ===")
    emit("")

    def col(arm, key):
        return [r["variants"][arm][key] for r in records
                if math.isfinite(r["variants"].get(arm, {}).get(key, float("nan")))]

    emit(f"{'arm':>14} {'psnr':>8} {'lpips':>8} {'abs_rel':>8} {'delta1':>8} "
         f"{'tgt_disp_l1':>11} {'src_disp_l1':>11}")
    for arm in arms:
        vals = []
        for k in RGB_KEYS + DEP_KEYS:
            c = col(arm, k)
            vals.append(sum(c) / len(c) if c else float("nan"))
        sc = [r["src_disp_l1"][arm] for r in records
              if math.isfinite(r["src_disp_l1"].get(arm, float("nan")))]
        emit(f"{arm:>14} {vals[0]:8.3f} {vals[1]:8.4f} {vals[2]:8.4f} {vals[3]:8.4f} "
             f"{vals[4]:11.4f} {sum(sc)/len(sc) if sc else float('nan'):11.4f}")
    emit("")

    # Paired deltas vs p2, with the tail: §3d is the standing case of a change
    # that looked like -0.0088 on the mean and had wrecked the worst decile.
    emit("--- paired vs p2 (negative = better), with worst decile ---")
    emit(f"{'arm':>14} {'d_lpips':>9} {'worst10%':>9} {'n_worse':>8} {'d_abs_rel':>10}")
    for arm in arms:
        # Arms added after a records.jsonl was written are simply absent from it;
        # --from-records on an older file must still summarise what IS there.
        if arm == "p2" or arm not in records[0]["variants"]:
            continue
        dl = [r["variants"][arm]["lpips"] - r["variants"]["p2"]["lpips"] for r in records]
        da = [r["variants"][arm]["abs_rel"] - r["variants"]["p2"]["abs_rel"]
              for r in records
              if math.isfinite(r["variants"][arm]["abs_rel"])
              and math.isfinite(r["variants"]["p2"]["abs_rel"])]
        k = max(1, len(dl) // 10)
        worst = sorted(dl)[-k:]
        emit(f"{arm:>14} {sum(dl)/len(dl):+9.4f} {sum(worst)/len(worst):+9.4f} "
             f"{sum(1 for d in dl if d > 0):8d} "
             f"{sum(da)/len(da) if da else float('nan'):+10.4f}")
    emit("")

    # The actual question, stated as the two dose-response curves rather than
    # left for the reader to assemble from the table above.
    emit("--- does depth quality move RGB?  (depth quality ascending, top 3 rows) ---")
    for arm in ("rail_dp1_tgt", "p2", "rail_dg_tgt", "rail_dg_all"):
        a = col(arm, "abs_rel")
        l = col(arm, "lpips")
        note = "  (same tgt depth as rail_dg_tgt; adds railed SOURCE depth)" \
            if arm == "rail_dg_all" else ""
        emit(f"  {arm:>14}  tgt abs_rel={sum(a)/len(a):.4f}  ->  "
             f"rgb lpips={sum(l)/len(l):.4f}{note}")
    emit("  (the oracle-railed arms' own tgt depth numbers are the VAE decode")
    emit("   ceiling, since their target latent IS the GT latent exactly)")
    emit("--- does RGB quality move depth?  (RGB ascending L->R) ---")
    for arm in ("p2", "rail_rg_tgt"):
        a = col(arm, "abs_rel")
        l = col(arm, "lpips")
        emit(f"  {arm:>14}  rgb lpips={sum(l)/len(l):.4f}  ->  tgt abs_rel={sum(a)/len(a):.4f}")
    emit("")
    emit("For reference, §3c under a sigma=0.613 warm start: oracle depth moved RGB")
    emit("lpips 0.6235 -> 0.6174 (0.0061) while oracle RGB moved it 0.6235 -> 0.3191.")
    emit("")

    # THE CEILING READ.  §11c(a) showed source geometry reaches RGB through the
    # warp only, and every arm above builds the warp from pass 1, so this is the
    # first measurement of that path with a good source depth.  Stated as the
    # explicit comparison against the two numbers that decide the question, so
    # the verdict cannot be quietly read off the wrong baseline.
    def mean_lpips(arm):
        c = col(arm, "lpips")
        return sum(c) / len(c) if c else float("nan")

    if all(a in records[0]["variants"] for a in ("warpg", "warpg_dg_all")):
        p2l = mean_lpips("p2")
        emit("--- CEILING of the source-depth line: oracle warp (negative = better) ---")
        for arm, what in (("warpg", "oracle SOURCE depth rebuilds the warp"),
                          ("warpg_dg_all", "oracle warp + oracle depth latents")):
            dl = [r["variants"][arm]["lpips"] - r["variants"]["p2"]["lpips"]
                  for r in records]
            k = max(1, len(dl) // 10)
            emit(f"  {arm:>14}  lpips={mean_lpips(arm):.4f}  "
                 f"d_vs_p2={sum(dl) / len(dl):+.4f}  "
                 f"worst10%={sum(sorted(dl)[-k:]) / k:+.4f}   {what}")
        emit(f"  baseline p2 lpips={p2l:.4f}")
        emit("  Compare against, in order of what each one settles:")
        emit("    -0.0091  §6 pass 3, the same warp rebuild from PASS 2's source")
        emit("             depth -- so warpg minus this is what better source depth")
        emit("             could still buy over an extra pass that costs no GT")
        emit("    -0.0090  §11 a perfect TARGET depth oracle, the other stream's")
        emit("             entire ceiling")
        emit("     0.019   the eval-to-eval lpips swing WITHIN one training run")
        emit("             (bmvs_f_sweep.sbatch header, ectl s42 5250->5500)")
        emit("  Rule set before the run: if |warpg| is near 0.009 the depth line is")
        emit("  capped below the noise it would have to be measured through, and the")
        emit("  deployment sweep (§13: sigma_max / num_views / max_samples) is not")
        emit("  worth a training round.  Several times that, and it is.")

    if "warp_orel" in records[0]["variants"]:
        # How much of the warpg gap is SCALE, and how much of THAT is the
        # cross-view disagreement a consistency loss could actually fix?
        gap = st_mean([r["variants"]["p2"]["lpips"] - r["variants"]["warpg"]["lpips"]
                       for r in records])
        emit("")
        emit("--- is the gap SCALE or SHAPE?  (share of the warpg gap recovered) ---")
        for arm, what in (
            ("warp_oscale", "model's shape + GT per-view scale"),
            ("warp_orel", "cross-view DISAGREEMENT removed, global scale left wrong"),
            ("warp_oshape", "GT shape + model's per-view scale error"),
        ):
            dl = [r["variants"][arm]["lpips"] - r["variants"]["p2"]["lpips"]
                  for r in records]
            m = st_mean(dl)
            cf = st_mean([r["clamp_frac"][arm.split("_")[1]] for r in records
                          if "clamp_frac" in r])
            emit(f"  {arm:>14}  d_vs_p2={m:+.4f}  = {100.0 * m / -gap:5.1f}% of warpg"
                 f"   clamped={100.0 * cf:.2f}%   {what}")
        emit(f"  warpg gap being divided up: {-gap:.4f}")
        sp = st_mean([r["a_spread"] for r in records if "a_spread" in r])
        ra = st_mean([r["a_ratio"] for r in records if "a_ratio" in r])
        av = st_mean([x for r in records for x in r.get("a_view", [])])
        emit(f"  the per-view factors removed: mean a={av:.4f}  spread(cv)={sp:.4f}  "
             f"max/min={ra:.4f}   (§10 measured spread 0.087, max/min 1.30)")
        emit("  `warp_orel` is the one that bounds a cross-view consistency loss:")
        emit("  such a loss fits ratios BETWEEN views, so global scale is invisible")
        emit("  to it by construction and `warp_oscale` overstates what it could do.")
        emit("  A high `clamped` fraction means the arm is measuring the [0,1] clamp.")

    if "warp_mvs" in records[0]["variants"]:
        # The one arm above that is GT-FREE and therefore actually deployable.
        # §16f-g's pre-registered read: land near warp_orel (the consistency-loss
        # ceiling) or better, land near warp_oscale/warpg (the anchor pays off in
        # full), or land near p2 (no anchor after all -- §16h's patch-NCC control
        # already showed the naive version does not clear this bar).
        dl_mvs = [r["variants"]["warp_mvs"]["lpips"] - r["variants"]["p2"]["lpips"]
                  for r in records]
        m_mvs = st_mean(dl_mvs)
        k = max(1, len(dl_mvs) // 10)
        worst_mvs = sorted(dl_mvs)[-k:]
        n_worse_mvs = sum(1 for d in dl_mvs if d > 0)
        cf_mvs = st_mean([r["clamp_frac"]["mvs"] for r in records if "clamp_frac" in r])
        emit("")
        emit("--- `warp_mvs`: the GT-FREE arm, against the oracle ceilings above ---")
        emit(f"  d_vs_p2={m_mvs:+.4f}  = {100.0 * m_mvs / -gap:5.1f}% of warpg   "
             f"worst10%={sum(worst_mvs)/len(worst_mvs):+.4f}  n_worse={n_worse_mvs}/{len(dl_mvs)}   "
             f"clamped={100.0 * cf_mvs:.2f}%")
        emit(f"  for reference: warp_orel (consistency-loss ceiling) = {-gap * 0.072:+.4f}  "
             f"warp_oscale (oracle per-view scale) = {-gap * 0.492:+.4f}  "
             f"warpg (full oracle) = {-gap:+.4f}")
        emit("  report §16f-g's own pre-run estimate (linear extrapolation from the")
        emit("  |geomean(a)-1| reduction, NOT a measurement): ~-0.019, i.e. near the")
        emit("  0.019 eval-to-eval noise floor -- read this arm's own number, not that")
        emit("  estimate; §15 already showed loss-unit/geometric extrapolation to")
        emit("  LPIPS is unreliable (21-43x loss headroom paid off as only 7% RGB).")

    if out_path:
        with open(out_path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"\nwrote {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--step", type=int)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--max-chunks", type=int, default=None)
    ap.add_argument("--oracle-abs-rel-max", type=float, default=0.06,
                    help="self-check: an arm railed to the oracle depth must reach "
                         "roughly the oracle's own depth accuracy (§3a's ws60_ir got "
                         "0.0556), else the rail did not take where it matters")
    ap.add_argument("--from-records", default=None)
    ap.add_argument("--num-shards", type=int, default=1,
                    help="split the eval chunks round-robin across this many "
                         "independent processes (one per GPU); each shard "
                         "writes its own records.jsonl/timing.json under "
                         "--out-dir, to be merged afterwards")
    ap.add_argument("--shard-idx", type=int, default=0,
                    help="this process's shard, in [0, num_shards)")
    args = ap.parse_args()
    if not (0 <= args.shard_idx < max(args.num_shards, 1)):
        raise SystemExit(f"--shard-idx {args.shard_idx} out of range for "
                         f"--num-shards {args.num_shards}")

    if args.from_records:
        recs = [json.loads(l) for l in open(args.from_records)]
        summarise(recs, os.path.join(os.path.dirname(args.from_records), "summary.txt"))
        return
    if not (args.run and args.step):
        raise SystemExit("--run and --step are required unless --from-records")

    out_dir = args.out_dir or (
        f"tools/cross_stream_rails_{os.path.basename(args.run)}_step{args.step}")
    os.makedirs(out_dir, exist_ok=True)

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
    if not model.depth_flow_matching:
        raise SystemExit("this diagnostic is about the depth stream; "
                         "depth_flow_matching is off in this run")
    lpips_net = model._get_lpips(device)

    def lpips(a, b):
        if lpips_net is None:
            return float("nan")
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
    num_views = num_src + num_tgt
    num_steps = model.eval_num_inference_steps
    seed = model.eval_sample_seed
    tgt_views = list(range(num_src, num_views))
    all_views = list(range(num_views))
    autocast_ctx = torch.autocast(device_type="cuda", dtype=model.pipe.torch_dtype, enabled=True)

    rec_path = os.path.join(out_dir, "records.jsonl")
    rec_file = open(rec_path, "w")
    records = []
    checked_equiv = False
    checked_disp = False
    t_start = time.time()
    # Per-sample timing the user asked for: pass 1 (blind sample), the GT-free
    # MVS scale fit, pass 2 (resample with the p1-built warp).  Each entry is
    # (elapsed_seconds, batch_size) since sampling is batched; per-sample
    # averages are sum(elapsed)/sum(batch_size), not a mean of per-batch means.
    pass1_times, mvs_times, pass2_times = [], [], []

    for chunk_idx, batch in enumerate(loader):
        if args.max_chunks is not None and chunk_idx >= args.max_chunks:
            break
        if args.num_shards > 1 and chunk_idx % args.num_shards != args.shard_idx:
            continue
        with torch.no_grad(), autocast_ctx:
            model._eval_inputs = True
            try:
                inputs = model.forward_preprocess(batch)
            finally:
                model._eval_inputs = False

            rgb_gt_lat = inputs["rgb_gt"]
            dep_gt_lat = inputs["depth_gt"]
            # ONE draw shared by every arm.  Drawn in the same order as
            # _sample_rgb_once (rgb then depth) so the bit-exactness check below
            # is meaningful.
            g = torch.Generator(device=device)
            g.manual_seed(seed)
            eps_rgb = torch.randn(rgb_gt_lat.shape, device=device,
                                  dtype=rgb_gt_lat.dtype, generator=g)
            eps_dep = torch.randn(dep_gt_lat.shape, device=device,
                                  dtype=dep_gt_lat.dtype, generator=g)

            blind = dict(inputs)
            blind["warp_block"] = torch.zeros_like(inputs["warp_block"])
            if inputs.get("depth_warp_block") is not None:
                blind["depth_warp_block"] = torch.zeros_like(inputs["depth_warp_block"])

            torch.cuda.synchronize()
            t0 = time.time()
            dep1, rgb1 = sample_rails(model, blind, num_steps, eps_rgb, eps_dep)
            torch.cuda.synchronize()
            pass1_times.append((time.time() - t0, rgb_gt_lat.shape[0]))

            # Hard gate: the rewritten sampler must be the module's sampler.  A
            # stale-kwargs harness artefact once manufactured a 1.45 dB
            # difference in this project, so this exits rather than warns.
            if not checked_equiv:
                g0 = torch.Generator(device=device)
                g0.manual_seed(seed)
                d_ref, r_ref = model._sample_rgb_once(blind, num_steps, g0)
                dd = float((d_ref - dep1).abs().max())
                dr = float((r_ref - rgb1).abs().max())
                print(f"  self-check vs _sample_rgb_once: depth_max_abs_diff={dd:.3e} "
                      f"rgb_max_abs_diff={dr:.3e}", flush=True)
                if dd > 0 or dr > 0:
                    raise SystemExit("self-check FAILED: sample_rails is not "
                                     "bit-identical to _sample_rgb_once with rails off")
                checked_equiv = True

            # One inputs dict per warp source.  "p1" is what production does;
            # "gt" swaps in the oracle source depth and is the only thing that
            # exercises the live source-geometry->RGB path (see the module
            # docstring on `warpg`).  Same helper, same call, only the depth
            # latent it splats differs, so the two are directly comparable.
            bases = {None: blind}
            csds = {None: None}
            for src, lat in (("p1", dep1), ("gt", dep_gt_lat)):
                wb, dwb = model._warp_block_from_predicted_depth(inputs, lat)
                b = dict(inputs)
                b["warp_block"] = wb
                b["depth_warp_block"] = dwb
                bases[src] = b
                csds[src] = lat[:, :, :num_src] if model.clean_source_depth else None

            # The scale/shape decomposition arms.  These intervene on DISPARITY,
            # so they cannot reuse _warp_block_from_predicted_depth; they call the
            # same _build_warp_block it calls, on a per-view rescaling of the
            # model's own disparity.  See the ARMS comment for what each is.
            # `sdisp_*`, not `disp_*`: `disp_gt` is rebound further down to the
            # PIXEL-space GT disparity [B, V, C, H, W] for the metrics.  Two
            # meanings for one name in one loop body is a trap for the next edit.
            sdisp_p1 = decode_src_disparity(model, dep1)
            sdisp_gt = decode_src_disparity(model, dep_gt_lat)
            src_valid = inputs["loss_masks"][:, 0, :num_src, 0] > 0.5
            a_view = per_view_scale(sdisp_p1, sdisp_gt, src_valid)
            if not checked_disp:
                # decode_src_disparity must be the module's readout, or the arms
                # are measuring a reimplementation.  Compared through the warp,
                # because the method's internal disparity is not returned.
                #
                # The comparison needs its OWN baseline: the first version of this
                # check demanded a bit-exact match and failed at 1.06, which says
                # nothing on its own because the warp block holds VAE latents of
                # O(100) that a sub-pixel splat shift moves a long way, and the
                # method rounds its result to the latent dtype.  So measure how
                # far the method is from ITSELF on two identical calls.
                #
                # 2026-09-19: comparing d_mine against a PER-RUN self-baseline
                # turned out to be the wrong design -- d_self is itself one noisy
                # draw of a max-of-noise statistic (job 12530 0.703, 12566 0.758,
                # 12567 0.609 with THREE repeats each time, no code change), and
                # its distribution overlaps d_mine's rather than bounding it: a
                # 25% margin over a 3-repeat self-baseline still failed twice
                # (ratios 1.04, then 1.37).  Switched to an absolute threshold on
                # d_mine/scale instead, calibrated against every run on record:
                # benign 0.152 (pre-fix), 0.188 (12566), 0.206 (12567); the
                # original re-implementation BUG measured 0.262 (job 12459,
                # d_mine=1.062 / scale~4.06).  0.24 sits with room on both sides.
                wb_ref, _ = model._warp_block_from_predicted_depth(inputs, dep1)
                wb_ref2, _ = model._warp_block_from_predicted_depth(inputs, dep1)
                wb_mine, _ = model._build_warp_block(
                    inputs["source_images"], sdisp_p1, inputs["pixel_K"],
                    inputs["c2w"], tuple(dep1.shape[-2:]))
                wb_mine = wb_mine.to(dtype=dep1.dtype)
                d_self = float((wb_ref - wb_ref2).abs().max())
                d_mine = float((wb_ref - wb_mine).abs().max())
                scale = float(wb_ref.abs().max())
                rel_mine = d_mine / max(scale, 1.0)
                print(f"  self-check decode_src_disparity: "
                      f"method_vs_itself={d_self:.3e}  mine_vs_method={d_mine:.3e} "
                      f"({rel_mine:.3f} of scale)  warp_absmax={scale:.3e}  "
                      f"dtypes ref={wb_ref.dtype} mine={wb_mine.dtype}", flush=True)
                if rel_mine > 0.24:
                    raise SystemExit(
                        f"self-check FAILED: decode_src_disparity differs from "
                        f"_warp_block_from_predicted_depth by {rel_mine:.3f} of scale, "
                        f"above the 0.24 threshold -- the scale arms would be measuring "
                        f"a reimplementation, not the module's readout")
                if model.clean_source_depth:
                    raise SystemExit("the scale arms define a disparity, not a latent, "
                                     "so they cannot supply clean_src_depth; this "
                                     "checkpoint has clean_source_depth=true")
                checked_disp = True
            clamp_frac = {}
            if chunk_idx == 0:
                # Early warning, printed before the 2h of sampling: if mean a is
                # far from 1 or the clamp bites, the scale arms are measuring the
                # [0,1] clamp instead of the scale and the run is wasted.
                print(f"  per-view scale a: mean={float(a_view.mean()):.4f} "
                      f"min={float(a_view.min()):.4f} max={float(a_view.max()):.4f} "
                      f"cv={float(a_view.std(dim=1).mean() / a_view.mean()):.4f} "
                      f"(§10 measured cv 0.087)", flush=True)
            for kind in ("oscale", "orel", "oshape"):
                dsp, clamp_frac[kind] = scaled_disparity(
                    kind, sdisp_p1, sdisp_gt, a_view)
                wb, dwb = model._build_warp_block(
                    inputs["source_images"], dsp, inputs["pixel_K"], inputs["c2w"],
                    tuple(dep_gt_lat.shape[-2:]))
                b = dict(inputs)
                b["warp_block"] = wb.to(dep_gt_lat.dtype)
                b["depth_warp_block"] = None if dwb is None else dwb.to(dep_gt_lat.dtype)
                bases[kind] = b
                csds[kind] = None
                if chunk_idx == 0:
                    print(f"    {kind:7s} clamped={100.0 * clamp_frac[kind]:.2f}% "
                          f"of disparity pixels", flush=True)

            # `warp_mvs`: the GT-FREE arm.  Fits one global disparity scale per
            # SAMPLE from a 1-DOF photometric scan against pass 1's own source
            # RGB/disparity (report §16f), then applies MVS_LAMBDA shrinkage
            # (§16g) before rebuilding the warp -- same disparity->warp path as
            # the oscale/orel/oshape arms above, but nothing here reads GT.
            src_K = inputs["pixel_K"][:, :num_src]
            src_c2w = inputs["c2w"][:, :num_src]
            src_rgb = inputs["source_images"].float()
            pairs = ordered_pairs(num_src)
            k_hats, k_shrunks, mvs_disps = [], [], []
            for bi in range(sdisp_p1.shape[0]):
                t0 = time.time()
                errs, _ = scan_global(src_rgb[bi], sdisp_p1[bi], src_valid[bi],
                                      src_K[bi], src_c2w[bi], pairs, MVS_GRID)
                mvs_times.append((time.time() - t0, 1))
                fin = [(e, k) for e, k in zip(errs, MVS_GRID) if e == e]
                k_hat = min(fin, key=lambda t: t[0])[1] if fin else 1.0
                k_shrunk = k_hat ** MVS_LAMBDA
                k_hats.append(k_hat)
                k_shrunks.append(k_shrunk)
                mvs_disps.append((sdisp_p1[bi] * k_shrunk).clamp(0.0, 1.0))
            k_shrunk_t = torch.tensor(k_shrunks, device=sdisp_p1.device).view(-1, 1, 1, 1)
            raw_mvs = sdisp_p1 * k_shrunk_t
            clamp_frac["mvs"] = float(((raw_mvs < 0.0) | (raw_mvs > 1.0)).float().mean())
            sdisp_mvs = torch.stack(mvs_disps, dim=0)
            wb, dwb = model._build_warp_block(
                inputs["source_images"], sdisp_mvs, inputs["pixel_K"], inputs["c2w"],
                tuple(dep_gt_lat.shape[-2:]))
            b = dict(inputs)
            b["warp_block"] = wb.to(dep_gt_lat.dtype)
            b["depth_warp_block"] = None if dwb is None else dwb.to(dep_gt_lat.dtype)
            bases["mvs"] = b
            csds["mvs"] = None
            if chunk_idx == 0:
                print(f"    mvs     k_hat mean={sum(k_hats)/len(k_hats):.4f} "
                      f"k_shrunk(^{MVS_LAMBDA}) mean={sum(k_shrunks)/len(k_shrunks):.4f} "
                      f"clamped={100.0 * clamp_frac['mvs']:.2f}% of disparity pixels "
                      f"(GT-free, §16f-g)", flush=True)

            lat_out = {"p1": (dep1, rgb1)}
            for name, warp_from, stream, source, scope in ARMS:
                if name == "p1":
                    continue
                base = bases[warp_from]
                rail_rgb = rail_dep = None
                views = None
                if stream is not None:
                    views = tgt_views if scope == "tgt" else all_views
                    if stream == "depth":
                        rail_dep = dep_gt_lat if source == "gt" else dep1
                    else:
                        rail_rgb = rgb_gt_lat if source == "gt" else rgb1
                if name == "p2":
                    torch.cuda.synchronize()
                    t0 = time.time()
                dep_o, rgb_o = sample_rails(
                    model, base, num_steps, eps_rgb, eps_dep,
                    rail_rgb_x0=rail_rgb, rail_depth_x0=rail_dep, rail_views=views,
                    clean_src_depth=csds[warp_from])
                if name == "p2":
                    torch.cuda.synchronize()
                    pass2_times.append((time.time() - t0, rgb_gt_lat.shape[0]))
                # sigma reaches exactly 0, so a railed frame must END at its x0.
                if rail_dep is not None:
                    d = float((dep_o[:, :, views] - rail_dep[:, :, views].to(dep_o.dtype))
                              .abs().max())
                    if d > 0:
                        raise SystemExit(f"self-check FAILED: arm {name} railed depth "
                                         f"frames end {d:.3e} away from their x0")
                if rail_rgb is not None:
                    d = float((rgb_o[:, :, views] - rail_rgb[:, :, views].to(rgb_o.dtype))
                              .abs().max())
                    if d > 0:
                        raise SystemExit(f"self-check FAILED: arm {name} railed rgb "
                                         f"frames end {d:.3e} away from their x0")
                lat_out[name] = (dep_o, rgb_o)

            rgb_px = batch["images"].to(device).float().clamp(0.0, 1.0)
            disp_gt = batch["disparity"].to(device).float()
            depth_metric = batch["depth_metric"].to(device).float()
            loss_masks = inputs["loss_masks"]
            batch_size = rgb_px.shape[0]

            dec_rgb, dec_disp = {}, {}
            for name, (dep_o, rgb_o) in lat_out.items():
                dec_rgb[name] = decode_rgb(model, rgb_o, tgt_views, device)
                dec_disp[name] = decode_disp(model, dep_o, all_views, device)

            for b in range(batch_size):
                b_global = chunk_idx * eval_batch_size + b
                src_l1 = {}
                for name in lat_out:
                    vals = []
                    for sv in range(num_src):
                        m = loss_masks[b, 0, sv, 0].bool()
                        if int(m.sum()) >= 16:
                            vals.append(float(
                                (dec_disp[name][sv][b][m] - disp_gt[b, sv, 0][m]).abs().mean()))
                    src_l1[name] = sum(vals) / len(vals) if vals else float("nan")

                for t in range(num_tgt):
                    vi = num_src + t
                    variants = {}
                    for name in lat_out:
                        pred = dec_rgb[name][vi][b:b + 1]
                        gt = rgb_px[b:b + 1, vi]
                        abs_rel, delta1, dl1 = depth_metrics(
                            dec_disp[name][vi][b], depth_metric[b, vi, 0],
                            disp_gt[b, vi, 0], loss_masks[b, 0, vi, 0])
                        variants[name] = {
                            "psnr": psnr_full(pred, gt), "lpips": lpips(pred, gt),
                            "abs_rel": abs_rel, "delta1": delta1, "tgt_disp_l1": dl1,
                        }
                    # a_view is §10's `scale_spread` measured directly: the
                    # per-view factors the decomposition arms divide out.  Kept
                    # per record so the arms' effect can be read against the size
                    # of the disagreement they removed.
                    av = a_view[b]
                    rec = {"b": b_global, "t": t, "sample_id": batch["sample_id"][b],
                           "variants": variants, "src_disp_l1": src_l1,
                           "a_view": [float(x) for x in av],
                           "a_spread": float(av.std() / av.mean().clamp_min(1e-8)),
                           "a_ratio": float(av.max() / av.min().clamp_min(1e-8)),
                           "clamp_frac": clamp_frac}
                    records.append(rec)
                    rec_file.write(json.dumps(rec) + "\n")
            rec_file.flush()

            # Cheap early warning: if the oracle-railed depth arm is not at
            # oracle accuracy, the rail is not reaching the decoded output and
            # every conclusion below is about nothing.
            if chunk_idx == 0:
                ar = [r["variants"]["rail_dg_tgt"]["abs_rel"] for r in records
                      if math.isfinite(r["variants"]["rail_dg_tgt"]["abs_rel"])]
                m = sum(ar) / max(len(ar), 1)
                print(f"  self-check rail_dg_tgt tgt abs_rel={m:.4f} "
                      f"(must be < {args.oracle_abs_rel_max})", flush=True)
                if m > args.oracle_abs_rel_max:
                    raise SystemExit(
                        f"self-check FAILED: rail_dg_tgt abs_rel={m:.4f} is not at "
                        f"oracle level; the rail is not affecting the decoded depth")
        print(f"[chunk {chunk_idx}] {time.time()-t_start:.0f}s, {len(records)} rows",
              flush=True)

    rec_file.close()
    print(f"\nwrote {rec_path}\n", flush=True)
    summarise(records, os.path.join(out_dir, "summary.txt"))

    def per_sample(times):
        tot_t = sum(t for t, _ in times)
        tot_n = sum(n for _, n in times)
        return tot_t / max(tot_n, 1), tot_n

    p1_s, p1_n = per_sample(pass1_times)
    mvs_s, mvs_n = per_sample(mvs_times)
    p2_s, p2_n = per_sample(pass2_times)
    timing_lines = [
        "",
        "--- per-sample wall time (this run's hardware, batched where sampling is) ---",
        f"  pass 1 (blind sample)         {p1_s:6.2f}s/sample  (n={p1_n}, batched)",
        f"  mvs fit (GT-free, §16f-g)     {mvs_s:6.2f}s/sample  (n={mvs_n}, per-sample)",
        f"  pass 2 (warp-conditioned)     {p2_s:6.2f}s/sample  (n={p2_n}, batched)",
        f"  mvs fit as a fraction of one full pass: {100.0 * mvs_s / max(p2_s, 1e-9):.2f}%",
    ]
    for line in timing_lines:
        print(line, flush=True)
    with open(os.path.join(out_dir, "summary.txt"), "a") as fh:
        fh.write("\n".join(timing_lines) + "\n")
    # Raw (elapsed, n) lists, not just this shard's own average -- a sharded
    # run's shards must be summed BEFORE dividing, not averaged as if each
    # shard's mean were one more sample.
    with open(os.path.join(out_dir, "timing.json"), "w") as fh:
        json.dump({"pass1": pass1_times, "mvs": mvs_times, "pass2": pass2_times}, fh)


if __name__ == "__main__":
    main()
