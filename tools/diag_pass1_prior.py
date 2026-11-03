"""Can pass-1's DISCARDED target RGBD be used as a prior for pass 2?

Five measurements in one sampling loop over a run's frozen eval set, because
they all need pass-1's full clean latents and rebuilding those is the expensive
part.  Naming follows the discussion they came out of:

  §0  the noise confound.  sample_rgb() hands the SAME generator object to both
      passes (WanRGBDNVSTrainingModule.py:1934/1959), and _sample_rgb_once draws
      x and x_depth from it at entry -- so pass 2 currently starts from a
      DIFFERENT noise draw than pass 1.  Any per-sample "pass 2 is worse"
      therefore mixes the warp's effect with a noise re-roll.  Variants:
        p1     pass 1, noise draw #1
        p1b    pass 1, a different seed          -> seed-variance floor
        p2cur  pass 2, noise draw #2             -> reproduces production
        p2same pass 2, noise draw #1             -> information strictly ⊇ p1
  A   SDEdit warm start: pass 2 starts from pass-1's clean target latent
      renoised to sigma_mid instead of from pure noise, integrating only
      sigma_mid -> 0 with the warp on.  Needs no training change (the model
      trained on every sigma) and is cheaper than a full pass.  Variants ws*.
  B   confidence-gated warp mask: pass 1 also predicts TARGET disparity, an
      estimate independent of the splat's target disparity (which comes from
      pass-1's SOURCE depth).  Where the two disagree the splat is probably
      misplaced, and a misplaced splat is what makes pass 2 worse than pass 1
      (mask=0 regions are in-distribution, wrong-valued ones are not).  This
      script does NOT gate; it measures whether the disagreement signal exists
      at all (AUC against the splat's true photometric error, plus what an
      oracle-free gate would buy).  The two estimates come from the same network
      on the same pass, so their errors are correlated -- AUC ~ 0.5 kills the idea.
  C   GT-free selection: the 6 source images ARE clean at inference.  Splat a
      variant's predicted target RGBD back into them and score photometrically.
      Reported as ranking agreement with true LPIPS, and as the LPIPS a
      per-sample selector driven by this score would achieve.
  D   one more iteration: rebuild the warp from PASS 2's source depth and sample
      again (p3).  Does not use pass-1's target RGBD, but it is the cheapest
      attack on the stated root cause and shares this loop.

Writes one JSON record per (sample, target view) to records.jsonl as each chunk
finishes, so a crash or a timeout still leaves usable partial data.

Run as a MODULE from the project root, one GPU:
  python -m tools.diag_pass1_prior \
      --run logs/bmvs_d_drocefp32_s42 --step 3750 \
      --out-dir tools/pass1_prior_bmvs_d_drocefp32_s42_step3750
"""
import argparse
import json
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Subset

from examples.wanvideo.model_training.train_with_accelerate_rgbd_nvs import custom_collate_fn
from tools.vis_rgb_warp_pred_depth import build_model, build_dataset
from utils.warp import splat_sources_to_targets

EPS = 1e-3

# Which stream a warm-start arm takes its x_0 prior from.  The oracle arms are
# the ablation's control: if even "ORACLE both" degrades, the warm start itself
# is off-distribution rather than pass-1's latent being a bad prior.
MODALITY_DESC = {
    "b": "p1 rgb + p1 depth",
    "r": "p1 rgb + ORACLE depth",
    "d": "ORACLE rgb + p1 depth",
    "g": "ORACLE both",
}


# ----------------------------------------------------------------------------
# sampling
# ----------------------------------------------------------------------------
def sample_once(model, inputs, num_steps, eps_rgb, eps_depth,
                x0_rgb=None, x0_depth=None, start_idx=0, clean_src_depth=None):
    """_sample_rgb_once with EXPLICIT noise and an optional warm start.

    Reimplemented rather than called because the two things this script needs --
    pinning which noise draw a pass gets (§0) and starting from a given clean
    latent at an intermediate sigma (A) -- are both fixed inside the module's
    version.  `--self-check` asserts this reproduces it bit-for-bit on the
    pure-noise path, since a silent drift here would fake every number below.

    Flow convention (from _forward_flow): x_sigma = (1-sigma)*x_0 + sigma*eps,
    velocity = eps - x_0, integrated by x -= (sigma_i - sigma_{i+1}) * velocity.
    """
    rgb_gt = inputs["rgb_gt"]
    sigmas = model.inference_sigmas(num_steps, device=rgb_gt.device).to(rgb_gt.dtype)
    s0 = sigmas[start_idx]
    if start_idx > 0 and (x0_rgb is None or (model.depth_flow_matching and x0_depth is None)):
        # There is no valid "pure noise" state at sigma < 1: eps alone has the
        # sigma=1 marginal, so handing it to the model at a smaller sigma is
        # simply the wrong distribution.  A stream that skips steps must be
        # given a prior.
        raise ValueError("start_idx > 0 requires a warm-start latent for every stream")

    x = eps_rgb.clone() if x0_rgb is None else (
        (1.0 - s0) * x0_rgb.to(rgb_gt.dtype) + s0 * eps_rgb)
    x_depth = None
    if model.depth_flow_matching:
        depth_gt = inputs["depth_gt"]
        x_depth = eps_depth.clone() if x0_depth is None else (
            (1.0 - s0) * x0_depth.to(depth_gt.dtype) + s0 * eps_depth)

    s = model.num_source_views
    clean_mask = None
    if clean_src_depth is not None:
        if x_depth is None:
            raise RuntimeError("clean_src_depth requires depth_flow_matching")
        clean_src_depth = clean_src_depth.to(x_depth.dtype)
        x_depth[:, :, :s] = clean_src_depth
        clean_mask = torch.ones(1, dtype=torch.bool, device=x_depth.device)

    pred_depth = None
    for i in range(start_idx, num_steps):
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
    if model.depth_flow_matching:
        return x_depth, x
    return pred_depth, x


def blind_inputs(inputs):
    """The pass-1 regime: warp zeroed, i.e. what warp_dropout_prob trains for."""
    blind = dict(inputs)
    blind["warp_block"] = torch.zeros_like(inputs["warp_block"])
    if inputs.get("depth_warp_block") is not None:
        blind["depth_warp_block"] = torch.zeros_like(inputs["depth_warp_block"])
    return blind


def with_warp(inputs, warp_block, depth_warp_block):
    second = dict(inputs)
    second["warp_block"] = warp_block
    second["depth_warp_block"] = depth_warp_block
    return second


# ----------------------------------------------------------------------------
# decoding
# ----------------------------------------------------------------------------
def decode_rgb(model, latents_raw, views, device):
    """[B,16,V,h,w] RGB latent in the SCALED space -> {view: [B,3,H,W] in [0,1]}."""
    lat = latents_raw / model.latent_scale if model.latent_scale != 1.0 else latents_raw
    out = {}
    for v in views:
        v_lat = lat[:, :, v:v + 1].to(device).to(model.pipe.torch_dtype)
        out[v] = model.pipe.vae.decode(v_lat, device=device)[:, :, 0].float().clamp(0, 1)
    return out


def decode_disp(model, latents_raw, views, device):
    """[B,16,V,h,w] depth latent in the SCALED space -> {view: [B,H,W] in [0,1]}.

    Divides by depth_latent_scale, which is the convention evaluate() applies
    before scoring (WanRGBDNVSTrainingModule.py:2195-2197).  Callers that hand a
    latent to _warp_block_from_predicted_depth must pass the RAW one instead --
    that function divides internally.
    """
    scale = model.depth_latent_scale
    lat = latents_raw / scale if scale != 1.0 else latents_raw
    out = {}
    for v in views:
        v_lat = lat[:, :, v:v + 1].to(device).to(model.pipe.torch_dtype)
        dec = model.pipe.vae.decode(v_lat, device=device)[:, :, 0]  # [B,3,H,W]
        out[v] = dec.float().mean(dim=1).clamp(0.0, 1.0)
    return out


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def psnr_full(pred_img, gt_img):
    mse = ((pred_img - gt_img) ** 2).mean().item()
    return 100.0 if mse < 1e-10 else -10 * math.log10(mse)


def fit_affine(p, g):
    """min_{a,b} ||a*p + b - g||^2, closed form in float64."""
    n = p.numel()
    sp, spp = p.sum(), (p * p).sum()
    sg, spg = g.sum(), (p * g).sum()
    det = n * spp - sp * sp
    if torch.abs(det) < 1e-12:
        return g.median() / p.median().clamp_min(1e-8), torch.zeros((), dtype=torch.float64, device=p.device)
    return (n * spg - sp * sg) / det, (spp * sg - sp * spg) / det


def depth_metrics(pred_disp_hw, gt_metric_hw, gt_disp_hw, mask_hw):
    """abs_rel / delta1 after affine alignment in DISPARITY space, plus the raw
    unaligned disp_l1 -- the same protocol as evaluate()'s _depth_metrics, minus
    the ramp/skill/struct_corr side numbers.  `pred_disp_hw` is already decoded."""
    nan = (float("nan"),) * 3
    gt = gt_metric_hw.float()
    pred = pred_disp_hw
    if pred.shape != gt.shape:
        pred = F.interpolate(pred[None, None], size=gt.shape, mode="bilinear",
                             align_corners=False)[0, 0]
    valid = mask_hw.bool() & torch.isfinite(gt) & (gt > 0)
    if int(valid.sum()) < 16:
        return nan
    disp_l1 = (pred[valid] - gt_disp_hw.float()[valid]).abs().mean().item()
    p = pred[valid].double()
    gt_v = gt[valid].double()
    g = 1.0 / gt_v.clamp_min(1e-8)
    a, b = fit_affine(p, g)
    floor = (g.min() * 0.5).clamp_min(1e-8)
    pred_m = 1.0 / (a * p + b).clamp_min(floor)
    abs_rel = ((pred_m - gt_v).abs() / gt_v).mean().item()
    ratio = torch.max(pred_m / gt_v, gt_v / pred_m)
    return abs_rel, (ratio < 1.25).double().mean().item(), disp_l1


def auc_binary(score, label, max_n=200000):
    """Mann-Whitney AUC of `score` as a predictor of boolean `label`."""
    n = score.numel()
    if n > max_n:
        idx = torch.randperm(n, device=score.device)[:max_n]
        score, label = score[idx], label[idx]
    n_pos, n_neg = int(label.sum()), int((~label).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(score)
    ranks = torch.empty(score.numel(), dtype=torch.float64, device=score.device)
    ranks[order] = torch.arange(1, score.numel() + 1, dtype=torch.float64, device=score.device)
    r_pos = ranks[label].sum()
    return float((r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def repro_score(model, tgt_rgb, tgt_disp, pixel_K, c2w, num_src, vi, gt_src_rgb,
                penalty=0.25):
    """C: GT-free quality score for one predicted target view.

    Splat the predicted target RGB, using the predicted target depth, back into
    all source views and measure photometric L1 against the CLEAN source images.
    splat_sources_to_targets is direction-agnostic: "src" is simply whatever we
    splat FROM, so the target view goes in that slot.

    Two scores, because the obvious one is gameable: `covered` averages only over
    pixels something landed on, so a variant that predicts near-zero (invalid)
    disparity splats almost nothing, divides by a tiny denominator and scores
    BEST -- it would win any selector built on it.  `penalized` charges every
    uncovered source pixel a flat `penalty` instead, so degenerate depth cannot
    buy a good score.  Both are recorded; the selector uses `penalized`.

    Returns (covered [B], coverage [B], penalized [B]); lower is better.
    """
    disp = tgt_disp[:, None]                              # [B,1,H,W]
    valid = disp > EPS
    depth = torch.where(valid, 1.0 / disp.clamp_min(EPS), torch.zeros_like(disp))
    warped, mask = splat_sources_to_targets(
        src_rgb=tgt_rgb[:, None].float(), src_depth=depth.float(), src_valid=valid,
        src_K=pixel_K[:, vi:vi + 1], src_c2w=c2w[:, vi:vi + 1],
        tgt_K=pixel_K[:, :num_src], tgt_c2w=c2w[:, :num_src],
        geometry_fp32=model.camera_geometry_fp32,
    )
    err = (warped - gt_src_rgb[:, :num_src]).abs().mean(dim=2, keepdim=True) * mask
    denom = mask.sum(dim=(2, 3, 4)).clamp_min(1.0)
    covered = (err.sum(dim=(2, 3, 4)) / denom).mean(dim=1)
    n_pix = mask.shape[-1] * mask.shape[-2]
    penalized = ((err + penalty * (1.0 - mask)).sum(dim=(2, 3, 4)) / n_pix).mean(dim=1)
    return covered, mask.mean(dim=(1, 2, 3, 4)), penalized


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--max-chunks", type=int, default=None,
                    help="stop after N chunks (smoke test); default = whole eval set")
    ap.add_argument("--ws-specs", default="0.8:i:b,0.6:i:b,0.4:i:b,0.2:i:b",
                    help="A: comma list of sigma[:noise[:modality]] warm-start arms.  "
                         "noise: i = renoise with an INDEPENDENT draw (real SDEdit), "
                         "s = reuse the draw that produced pass 1 (which lands back on "
                         "pass-1's own ODE trajectory instead of the training "
                         "distribution, where x_0 and eps are independent).  "
                         "modality -- which stream's prior comes from pass 1 and which "
                         "is replaced by the GT latent (an ORACLE, so the truncated "
                         "schedule stays well-posed; 'pure noise' is not a valid state "
                         "at sigma<1, so a stream cannot simply be left out): "
                         "b = both from pass 1, r = RGB from pass 1 + oracle depth, "
                         "d = depth from pass 1 + oracle RGB, g = both oracle.  "
                         "The g arm is the mechanism's ceiling: if even g degrades, the "
                         "warm start itself is off-distribution rather than pass-1's "
                         "latent being a bad prior.  Arm names encode all three, e.g. "
                         "ws80_ib, so no table row can be misattributed.")
    ap.add_argument("--refs", default="p1,p1b,p2cur,p2same,p3",
                    help="which reference (non-warm-start) variants to sample")
    ap.add_argument("--fig-bs", default="5,2,6,7",
                    help="b-indices to also dump a comparison figure for")
    ap.add_argument("--fig-target", type=int, default=2)
    ap.add_argument("--seed-b", type=int, default=None,
                    help="second seed for the p1b variance floor (default eval_sample_seed+1)")
    ap.add_argument("--bad-thresh", type=float, default=0.1,
                    help="B: splat photometric L1 above this counts as a misplaced pixel")
    ap.add_argument("--repro-penalty", type=float, default=0.25,
                    help="C: cost charged to source pixels the back-splat does not cover")
    ap.add_argument("--no-self-check", action="store_true")
    args = ap.parse_args()

    out_dir = args.out_dir or (
        f"tools/pass1_prior_{os.path.basename(args.run)}_step{args.step}")
    os.makedirs(out_dir, exist_ok=True)
    fig_bs = {int(x) for x in args.fig_bs.split(",") if x.strip() != ""}

    cfg = OmegaConf.load(os.path.join(args.run, "args.yaml"))
    OmegaConf.resolve(cfg)
    set_seed(0)
    accelerator = Accelerator(mixed_precision=cfg.mixed_precision)

    model = build_model(cfg, accelerator)
    ckpt = os.path.join(args.run, f"checkpoint-step-{args.step}")
    nfiles = len(os.listdir(ckpt))
    if nfiles != 11:
        raise SystemExit(f"{ckpt} has {nfiles} files, expected 11 -- write was interrupted")
    state = load_file(os.path.join(ckpt, "model.safetensors"), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"loaded {ckpt}: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model.to(accelerator.device)
    model.eval()
    device = model.pipe.device

    lpips_net = model._get_lpips(device)
    if lpips_net is None:
        print("WARNING: lpips not installed -- lpips columns will be NaN", flush=True)

    def lpips(pred_img, gt_img):
        if lpips_net is None:
            return float("nan")
        with torch.no_grad():
            return float(lpips_net(pred_img.float() * 2 - 1, gt_img.float() * 2 - 1).mean())

    dataset = build_dataset(cfg)
    eval_max_samples = int(cfg.get("eval_max_samples", 2))
    eval_batch_size = max(1, min(eval_max_samples, int(cfg.get("eval_batch_size", 4))))
    samples_per_scene = cfg.get("samples_per_scene", 1)
    eval_indices = [i * samples_per_scene for i in range(len(dataset.records))][:eval_max_samples]
    loader = DataLoader(
        Subset(dataset, eval_indices), batch_size=eval_batch_size, shuffle=False,
        num_workers=0, collate_fn=custom_collate_fn, drop_last=False,
    )

    num_src = model.num_source_views
    num_tgt = model.num_target_views
    num_steps = model.eval_num_inference_steps
    seed_a = model.eval_sample_seed
    seed_b = args.seed_b if args.seed_b is not None else seed_a + 1

    # A: map each requested sigma_mid onto an index of the PRODUCTION schedule,
    # so a warm start is exactly the tail of the same integration, not a new one.
    sigmas = model.inference_sigmas(num_steps, device=device).float()
    ws_plan = []
    for spec in [x.strip() for x in args.ws_specs.split(",") if x.strip() != ""]:
        parts = spec.split(":")
        target = float(parts[0])
        noise = parts[1] if len(parts) > 1 else "i"
        modality = parts[2] if len(parts) > 2 else "b"
        if noise not in ("i", "s") or modality not in ("b", "r", "d", "g"):
            raise SystemExit(f"bad --ws-specs entry {spec!r}")
        i0 = int(torch.argmin((sigmas[:num_steps] - target).abs()).item())
        name = f"ws{int(round(target * 100)):02d}_{noise}{modality}"
        ws_plan.append((name, i0, float(sigmas[i0]), num_steps - i0, noise, modality))
    print(f"schedule (shift={model.eval_flow_shift}): "
          + " ".join(f"{i}:{float(s):.3f}" for i, s in enumerate(sigmas)), flush=True)
    for name, i0, sig, nst, noise, modality in ws_plan:
        print(f"  A {name}: start_idx={i0} sigma={sig:.4f} -> {nst} steps "
              f"(noise={'independent' if noise == 'i' else 'pass-1 reuse'}, "
              f"prior={MODALITY_DESC[modality]})", flush=True)

    ref_order = [v for v in ("p1", "p1b", "p2cur", "p2same", "p3")
                 if v in {x.strip() for x in args.refs.split(",")}]
    if "p1" not in ref_order:
        raise SystemExit("--refs must include p1: every other variant depends on it")
    variants = [v for v in ref_order if v != "p3"] + [n for n, *_ in ws_plan] \
        + (["p3"] if "p3" in ref_order else [])
    print(f"variants: {variants}", flush=True)
    print(f"config: clean_source_depth={model.clean_source_depth} "
          f"depth_warp_block={model.depth_warp_block} "
          f"warp_two_pass={model.warp_two_pass} seeds=({seed_a},{seed_b})", flush=True)

    records_path = os.path.join(out_dir, "records.jsonl")
    records_file = open(records_path, "w")
    rows = []
    autocast_ctx = torch.autocast(device_type="cuda", dtype=model.pipe.torch_dtype, enabled=True)
    t_start = time.time()

    for chunk_idx, batch in enumerate(loader):
        if args.max_chunks is not None and chunk_idx >= args.max_chunks:
            break
        t_chunk = time.time()
        with torch.no_grad(), autocast_ctx:
            model._eval_inputs = True
            try:
                inputs = model.forward_preprocess(batch)
            finally:
                model._eval_inputs = False

            rgb_gt, depth_gt = inputs["rgb_gt"], inputs["depth_gt"]
            batch_size = rgb_gt.shape[0]

            # Noise draws, made explicit.  Draw #1/#2 replicate the order
            # sample_rgb's single shared generator produces them in.
            g = torch.Generator(device=device)
            g.manual_seed(seed_a)
            eps1_rgb = torch.randn(rgb_gt.shape, device=device, dtype=rgb_gt.dtype, generator=g)
            eps1_dep = torch.randn(depth_gt.shape, device=device, dtype=depth_gt.dtype, generator=g)
            eps2_rgb = torch.randn(rgb_gt.shape, device=device, dtype=rgb_gt.dtype, generator=g)
            eps2_dep = torch.randn(depth_gt.shape, device=device, dtype=depth_gt.dtype, generator=g)
            # Draw #3 exists only for A's renoising: it must be independent of
            # eps1 (which produced the latent being warmed from) AND of eps2 (so
            # a warm-start arm is never accidentally comparable to p2cur for the
            # wrong reason).
            eps3_rgb = torch.randn(rgb_gt.shape, device=device, dtype=rgb_gt.dtype, generator=g)
            eps3_dep = torch.randn(depth_gt.shape, device=device, dtype=depth_gt.dtype, generator=g)
            gb = torch.Generator(device=device)
            gb.manual_seed(seed_b)
            epsB_rgb = torch.randn(rgb_gt.shape, device=device, dtype=rgb_gt.dtype, generator=gb)
            epsB_dep = torch.randn(depth_gt.shape, device=device, dtype=depth_gt.dtype, generator=gb)

            blind = blind_inputs(inputs)
            pred = {}
            pred["p1"] = sample_once(model, blind, num_steps, eps1_rgb, eps1_dep)

            if chunk_idx == 0 and not args.no_self_check:
                # Guard against this file's sampler drifting from the module's.
                g_chk = torch.Generator(device=device)
                g_chk.manual_seed(seed_a)
                ref_d, ref_r = model._sample_rgb_once(blind, num_steps, g_chk)
                d_err = float((ref_d.float() - pred["p1"][0].float()).abs().max())
                r_err = float((ref_r.float() - pred["p1"][1].float()).abs().max())
                print(f"[self-check] sample_once vs _sample_rgb_once: "
                      f"depth_max_abs_diff={d_err:.3e} rgb_max_abs_diff={r_err:.3e}", flush=True)
                if max(d_err, r_err) > 1e-3:
                    raise SystemExit("self-check FAILED -- sampler reimplementation diverged")

            if "p1b" in variants:
                pred["p1b"] = sample_once(model, blind, num_steps, epsB_rgb, epsB_dep)

            warp2, dwarp2 = model._warp_block_from_predicted_depth(inputs, pred["p1"][0])
            if warp2 is None:
                raise SystemExit("warp rebuild returned None -- run has no source_images")
            in2 = with_warp(inputs, warp2, dwarp2)
            csd2 = pred["p1"][0][:, :, :num_src] if model.clean_source_depth else None

            if "p2cur" in variants:
                pred["p2cur"] = sample_once(model, in2, num_steps, eps2_rgb, eps2_dep,
                                            clean_src_depth=csd2)
            if "p2same" in variants:
                pred["p2same"] = sample_once(model, in2, num_steps, eps1_rgb, eps1_dep,
                                             clean_src_depth=csd2)
            for name, i0, _, _, noise, modality in ws_plan:
                ws_rgb, ws_dep = ((eps3_rgb, eps3_dep) if noise == "i"
                                  else (eps1_rgb, eps1_dep))
                pred[name] = sample_once(
                    model, in2, num_steps, ws_rgb, ws_dep,
                    x0_rgb=pred["p1"][1] if modality in ("b", "r") else inputs["rgb_gt"],
                    x0_depth=pred["p1"][0] if modality in ("b", "d") else inputs["depth_gt"],
                    start_idx=i0, clean_src_depth=csd2)

            if "p3" in variants:
                src_p3 = pred["p2cur"] if "p2cur" in pred else pred["p2same"]
                warp3, dwarp3 = model._warp_block_from_predicted_depth(inputs, src_p3[0])
                in3 = with_warp(inputs, warp3, dwarp3)
                csd3 = src_p3[0][:, :, :num_src] if model.clean_source_depth else None
                pred["p3"] = sample_once(model, in3, num_steps, eps1_rgb, eps1_dep,
                                         clean_src_depth=csd3)

            # ---- pixel-space GT and the splat pass 2 actually saw
            rgb_px = batch["images"].to(device).float().clamp(0.0, 1.0)      # [B,V,3,H,W]
            disp_gt = batch["disparity"].to(device).float()                  # [B,V,3,H,W]
            depth_metric = batch["depth_metric"].to(device).float()
            loss_masks = inputs["loss_masks"]
            pixel_K = inputs["pixel_K"].float()
            c2w = inputs["c2w"]

            # ---- decode every variant
            tgt_views = list(range(num_src, num_src + num_tgt))
            dec_rgb = {v: decode_rgb(model, pred[v][1], tgt_views, device) for v in variants}
            dec_disp = {v: decode_disp(model, pred[v][0], range(num_src + num_tgt), device)
                        for v in variants}

            # The splat pass 2 actually saw.  Built from the SAME decoded source
            # disparity as dec_disp["p1"] rather than decoding it a second time --
            # _warp_block_from_predicted_depth does its own decode internally from
            # the raw latent, and this must reproduce that, not diverge from it.
            p1_src_disp = torch.stack(
                [dec_disp["p1"][v] for v in range(num_src)], dim=1)
            warped_rgb, warp_mask, splat_tgt_disp = model._splat_sources_pixel(
                rgb_px, p1_src_disp, pixel_K, c2w, return_target_disparity=True)

            # ---- per-sample source-depth quality (the stated root cause)
            src_l1 = {}
            for v in variants:
                per_b = []
                for b in range(batch_size):
                    vals = []
                    for sv in range(num_src):
                        m = loss_masks[b, 0, sv, 0].bool()
                        if int(m.sum()) < 16:
                            continue
                        vals.append(float((dec_disp[v][sv][b][m] - disp_gt[b, sv, 0][m]).abs().mean()))
                    per_b.append(sum(vals) / len(vals) if vals else float("nan"))
                src_l1[v] = per_b

            # ---- C: one batched splat per (variant, target view).  The splat
            # loops over its 6 destination views internally, so calling it once
            # per sample instead would multiply this chunk's cost for nothing.
            repro = {}
            for v in variants:
                for t in range(num_tgt):
                    vi = num_src + t
                    repro[(v, t)] = repro_score(
                        model, dec_rgb[v][vi], dec_disp[v][vi], pixel_K, c2w,
                        num_src, vi, rgb_px, penalty=args.repro_penalty)

            # ---- per (sample, target view)
            for b in range(batch_size):
                b_global = chunk_idx * eval_batch_size + b
                sample_id = batch["sample_id"][b]
                for t in range(num_tgt):
                    vi = num_src + t
                    gt_px = rgb_px[b:b + 1, vi]
                    mask_hw = loss_masks[b, 0, vi, 0]
                    per_variant = {}
                    for v in variants:
                        img = dec_rgb[v][vi][b:b + 1]
                        abs_rel, delta1, disp_l1 = depth_metrics(
                            dec_disp[v][vi][b], depth_metric[b, vi, 0],
                            disp_gt[b, vi, 0], mask_hw)
                        rs, rcov, rpen = repro[(v, t)]
                        per_variant[v] = {
                            "psnr": psnr_full(img, gt_px),
                            "lpips": lpips(img, gt_px),
                            "abs_rel": abs_rel, "delta1": delta1, "tgt_disp_l1": disp_l1,
                            "src_disp_l1": src_l1[v][b],
                            "repro": float(rs[b]), "repro_cov": float(rcov[b]),
                            "repro_pen": float(rpen[b]),
                        }

                    # ---- B: does pass-1's target disparity flag the bad splats?
                    m = (warp_mask[b, t, 0] > 0.5) & mask_hw.bool()
                    bdiag = {"masked_frac": float(m.float().mean())}
                    if int(m.sum()) >= 256:
                        e = (warped_rgb[b, t] - gt_px[0]).abs().mean(dim=0)[m]      # [N]
                        d_splat = splat_tgt_disp[b, t, 0][m]
                        d_p1 = dec_disp["p1"][vi][b][m]
                        d_raw = (d_splat - d_p1).abs()
                        a, bb = fit_affine(d_splat.double(), d_p1.double())
                        d_aff = ((a * d_splat.double() + bb) - d_p1.double()).abs().float()
                        bad = e > args.bad_thresh
                        bdiag.update({
                            "bad_frac": float(bad.float().mean()),
                            "splat_err_mean": float(e.mean()),
                            "auc_raw": auc_binary(d_raw, bad),
                            "auc_affine": auc_binary(d_aff, bad),
                            "affine_a": float(a), "affine_b": float(bb),
                        })
                        for q in (0.1, 0.2, 0.3):
                            k = int(d_aff.numel() * (1.0 - q))
                            if k >= 16:
                                keep = torch.topk(-d_aff, k).indices
                                bdiag[f"gate{int(q*100)}_err"] = float(e[keep].mean())
                        # random-drop baseline equals the ungated mean in expectation
                        bdiag["gate_baseline_err"] = float(e.mean())

                    rec = {
                        "b": b_global, "t": t, "sample_id": sample_id,
                        "warp_cov": float(warp_mask[b, t].mean()),
                        "variants": per_variant, "B": bdiag,
                    }
                    rows.append(rec)
                    records_file.write(json.dumps(rec) + "\n")

                    if b_global in fig_bs and t == args.fig_target:
                        save_figure(out_dir, args, model, rec, variants, gt_px,
                                    {v: dec_rgb[v][vi][b:b + 1] for v in variants},
                                    warped_rgb[b, t], warp_mask[b, t, 0],
                                    splat_tgt_disp[b, t, 0], dec_disp["p1"][vi][b],
                                    b_global, t, sample_id)
            records_file.flush()

        done = min(len(rows) // num_tgt, eval_max_samples)
        print(f"[chunk {chunk_idx}] {time.time() - t_chunk:.0f}s "
              f"({done} samples, {time.time() - t_start:.0f}s total)", flush=True)
        for b in range(batch_size):
            b_global = chunk_idx * eval_batch_size + b
            r = [x for x in rows if x["b"] == b_global]
            cov = sum(x["warp_cov"] for x in r) / len(r)
            line = f"  b{b_global:02d} cov={cov:.3f} lpips " + " ".join(
                f"{v}={sum(x['variants'][v]['lpips'] for x in r) / len(r):.3f}" for v in variants)
            print(line, flush=True)

    records_file.close()
    summarize(out_dir, rows, variants, ws_plan)


def save_figure(out_dir, args, model, rec, variants, gt_px, imgs,
                warped_rgb, warp_mask, splat_disp, p1_disp, b_global, t, sample_id):
    """Row 1: GT + every variant's RGB.  Row 2: the warp diagnostics behind B."""
    def np_img(x):
        return x[0].permute(1, 2, 0).cpu().numpy().clip(0, 1)

    ncol = max(len(variants) + 1, 5)
    fig, axes = plt.subplots(2, ncol, figsize=(2.6 * ncol, 6.2))
    for ax in axes.ravel():
        ax.axis("off")
    axes[0, 0].imshow(np_img(gt_px))
    axes[0, 0].set_title(f"tgt{t+1} GT", fontsize=9)
    for i, v in enumerate(variants):
        m = rec["variants"][v]
        axes[0, i + 1].imshow(np_img(imgs[v]))
        axes[0, i + 1].set_title(
            f"{v}\npsnr={m['psnr']:.2f} lpips={m['lpips']:.3f}\n"
            f"abs_rel={m['abs_rel']:.3f} d1={m['delta1']:.3f}\nrepro={m['repro']:.4f}",
            fontsize=7.5)
    axes[1, 0].imshow(warped_rgb.permute(1, 2, 0).cpu().numpy().clip(0, 1))
    axes[1, 0].set_title(f"splat into tgt{t+1}\ncov={rec['warp_cov']:.3f}", fontsize=8)
    axes[1, 1].imshow(warp_mask.cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    axes[1, 1].set_title("warp mask", fontsize=8)
    axes[1, 2].imshow(splat_disp.cpu().numpy(), cmap="viridis")
    axes[1, 2].set_title("splat tgt disparity\n(from p1 SRC depth)", fontsize=8)
    axes[1, 3].imshow(p1_disp.cpu().numpy(), cmap="viridis")
    axes[1, 3].set_title("p1 tgt disparity\n(predicted directly)", fontsize=8)
    dis = (splat_disp - p1_disp).abs() * warp_mask
    axes[1, 4].imshow(dis.cpu().numpy(), cmap="magma")
    b = rec["B"]
    axes[1, 4].set_title(
        f"|disagreement|\nauc_raw={b.get('auc_raw', float('nan')):.3f} "
        f"auc_aff={b.get('auc_affine', float('nan')):.3f}", fontsize=8)

    fig.suptitle(f"{os.path.basename(args.run)} step{args.step}  {sample_id}  "
                 f"b{b_global:02d} target#{t}", fontsize=11)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.02, wspace=0.05, hspace=0.15)
    safe = sample_id.replace("/", "_")
    path = os.path.join(out_dir, f"b{b_global:02d}_t{t}_{safe}.png")
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  figure -> {path}", flush=True)


def summarize(out_dir, rows, variants, ws_plan):
    lines = []

    def emit(s=""):
        print(s, flush=True)
        lines.append(s)

    def mean(vals):
        vals = [v for v in vals if v is not None and not math.isnan(v)]
        return sum(vals) / len(vals) if vals else float("nan")

    n = len(rows)
    if n == 0:
        emit("no rows")
        return
    emit(f"\n=== {n} (sample, target) pairs ===")
    emit(f"{'variant':10s} {'psnr':>7s} {'lpips':>7s} {'abs_rel':>8s} {'delta1':>7s} "
         f"{'src_l1':>7s} {'repro':>8s}")
    for v in variants:
        emit(f"{v:10s} "
             f"{mean([r['variants'][v]['psnr'] for r in rows]):7.3f} "
             f"{mean([r['variants'][v]['lpips'] for r in rows]):7.4f} "
             f"{mean([r['variants'][v]['abs_rel'] for r in rows]):8.4f} "
             f"{mean([r['variants'][v]['delta1'] for r in rows]):7.4f} "
             f"{mean([r['variants'][v]['src_disp_l1'] for r in rows]):7.4f} "
             f"{mean([r['variants'][v]['repro'] for r in rows]):8.5f}")

    # ---- §0: how much of a pass1/pass2 gap is just the noise re-roll?
    emit("\n--- §0 noise confound ---")
    if "p1b" in variants:
        d_seed = [abs(r["variants"]["p1"]["lpips"] - r["variants"]["p1b"]["lpips"]) for r in rows]
        emit(f"  |lpips(p1) - lpips(p1b)|      mean={mean(d_seed):.4f}  "
             f"(pass-1 seed-variance floor, per sample)")
    if "p2cur" in variants and "p2same" in variants:
        d_pair = [abs(r["variants"]["p2cur"]["lpips"] - r["variants"]["p2same"]["lpips"])
                  for r in rows]
        emit(f"  |lpips(p2cur) - lpips(p2same)| mean={mean(d_pair):.4f}  "
             f"(same warp, different noise draw)")
    for v in ("p2cur", "p2same"):
        if v not in variants:
            continue
        worse = [r for r in rows if r["variants"][v]["lpips"] > r["variants"]["p1"]["lpips"]]
        emit(f"  {v} worse than p1 on {len(worse)}/{n} pairs "
             f"({100.0 * len(worse) / n:.1f}%), mean regression among those "
             f"{mean([r['variants'][v]['lpips'] - r['variants']['p1']['lpips'] for r in worse]):.4f}")

    # ---- A: warm start
    ref = "p2same" if "p2same" in variants else "p2cur"
    emit(f"\n--- A SDEdit warm start (vs {ref}) ---")
    base = mean([r["variants"][ref]["lpips"] for r in rows])
    for name, i0, sig, nst, _noise, _mod in ws_plan:
        m = mean([r["variants"][name]["lpips"] for r in rows])
        worse = sum(1 for r in rows
                    if r["variants"][name]["lpips"] > r["variants"]["p1"]["lpips"])
        tail = sorted(r["variants"][name]["lpips"] - r["variants"]["p1"]["lpips"] for r in rows)
        emit(f"  {name} (sigma={sig:.3f}, {nst} steps): lpips={m:.4f} "
             f"({m - base:+.4f} vs {ref}), worse-than-p1 {worse}/{n}, "
             f"worst-decile delta vs p1 {mean(tail[-max(1, n // 10):]):+.4f}")

    # ---- B: is the disagreement signal real?
    emit("\n--- B warp-mask confidence signal ---")
    auc_raw = mean([r["B"].get("auc_raw", float("nan")) for r in rows])
    auc_aff = mean([r["B"].get("auc_affine", float("nan")) for r in rows])
    emit(f"  AUC(|splat_disp - p1_tgt_disp|      -> splat err > thresh) = {auc_raw:.4f}")
    emit(f"  AUC(same, after per-view affine fit) = {auc_aff:.4f}   (0.5 = no signal)")
    b0 = mean([r["B"].get("gate_baseline_err", float("nan")) for r in rows])
    emit(f"  ungated splat photometric err = {b0:.4f}")
    for q in (10, 20, 30):
        g = mean([r["B"].get(f"gate{q}_err", float("nan")) for r in rows])
        emit(f"  drop top-{q}% by disagreement -> {g:.4f} ({g - b0:+.4f})")

    # ---- C: GT-free selection
    emit("\n--- C GT-free selection from the clean source images ---")
    emit(f"  mean repro_cov per variant: " + " ".join(
        f"{v}={mean([r['variants'][v]['repro_cov'] for r in rows]):.3f}" for v in variants))
    for sk in ("repro", "repro_pen"):
        agree = sum(1 for r in rows
                    if (r["variants"]["p1"][sk] < r["variants"]["p2cur"][sk])
                    == (r["variants"]["p1"]["lpips"] < r["variants"]["p2cur"]["lpips"]))
        emit(f"  [{sk}] p1-vs-p2cur ranking agreement with true lpips: {agree}/{n} "
             f"({100.0 * agree / n:.1f}%)")
    always = mean([r["variants"]["p2cur"]["lpips"] for r in rows])
    for pool_name, pool in (("{p1,p2cur}", ["p1", "p2cur"]), ("all variants", variants)):
        oracle = mean([min(r["variants"][v]["lpips"] for v in pool) for r in rows])
        emit(f"  pool {pool_name:14s}: always-p2cur={always:.4f}  oracle={oracle:.4f}")
        for sk in ("repro", "repro_pen"):
            sel = mean([r["variants"][min(pool, key=lambda v, k=sk: r["variants"][v][k])]["lpips"]
                        for r in rows])
            emit(f"      selector on {sk:10s} -> {sel:.4f}")

    # ---- D: extra iteration
    emit("\n--- D one more iteration ---")
    for v in ("p1", "p2same", "p2cur", "p3"):
        if v not in variants:
            continue
        emit(f"  {v}: lpips={mean([r['variants'][v]['lpips'] for r in rows]):.4f} "
             f"src_disp_l1={mean([r['variants'][v]['src_disp_l1'] for r in rows]):.4f} "
             f"abs_rel={mean([r['variants'][v]['abs_rel'] for r in rows]):.4f}")

    path = os.path.join(out_dir, "summary.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {path} and {os.path.join(out_dir, 'records.jsonl')}", flush=True)


if __name__ == "__main__":
    main()
