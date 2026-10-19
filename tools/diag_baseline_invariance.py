"""Is pass-1 depth prediction INVARIANT to the camera-baseline magnitude?

report_20260916 §15 found that 42% of the source-depth->RGB gap is a GLOBAL
disparity scale error (one scalar per sample, |geomean(a)-1| = 0.12) and §16
found a GT-free photometric fit can only partly recover it.  The question this
script settles is WHY the model gets that scalar wrong: it has, in principle,
two ways to know the absolute scale of the normalised world --

  (a) a statistical convention: the dataset puts the 98th percentile of source
      inverse depth at disp = 1/headroom = 0.667, and the model can learn to
      place its output there from monocular cues alone;
  (b) geometry: `c2w[:3,3]` is multiplied by the same `s` as the disparities
      (kubric4d_rgbd_nvs_dataset.py:571-589), so the baseline MAGNITUDE in
      normalised units, together with cross-view correspondence, pins absolute
      depth by triangulation.

Reading the code says (b) is unavailable BY CONSTRUCTION on these checkpoints:
`relative_c2w` -> `plucker_field` -> per-VIEW RMS normalisation of the moment
o x d (diffsynth/models/view_condition.py:149-159, `normalize_moment=True` in
both runs' args.yaml) divides out |o| -- exactly the quantity the class's own
docstring (:28-33) says is "proportional to the camera baseline".  RoCE
consumes the same normalised field (`roce_camera_latents`); CaPE is off
(`rope_pose_translation_scale: null`); the pass-1 warp is zeroed.  So scaling
every camera translation by lambda should leave the model's INPUT bit-identical.

This script measures that instead of asserting it.  For lambda in LAMBDAS it
scales `batch["c2w"][..., :3, 3]`, re-runs forward_preprocess and pass 1 with
the SAME noise draw, and reports max|delta| of the plucker block, the RoCE
latents and the pass-1 depth latent against lambda=1, plus the per-view median
ratio of decoded source disparity (route (b) would give 1/lambda; invariance
gives 1.0).  A POSITIVE CONTROL then flips `normalize_moment` off on the live
encoder and repeats one lambda: the plucker block must now differ, proving the
perturbation reaches the model and the invariance above is the normalisation,
not a dead hook.  (The control's depth output is NOT interpreted -- the model
was never trained on un-normalised moments.)

Consequence for the "randomise headroom as augmentation" idea: if invariance
holds, randomising `s` removes cue (a) while (b) does not exist, so it can only
hurt until a scale-carrying channel is restored.

Run as a MODULE from the project root, one GPU (~3 min for 2 chunks):
  python -m tools.diag_baseline_invariance --run logs/bmvs_d_droce_s1337 --step 4000 --max-chunks 2
"""
import argparse
import math
import os

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Subset

from examples.wanvideo.model_training.train_with_accelerate_rgbd_nvs import custom_collate_fn
from tools.diag_cross_stream_rails import decode_src_disparity, sample_rails
from tools.vis_rgb_warp_pred_depth import build_dataset, build_model

LAMBDAS = (0.5, 2.0)


def maxdiff(a, b):
    if a is None or b is None:
        return float("nan")
    return float((a.float() - b.float()).abs().max())


def per_view_median_ratio(disp_l, disp_ref, valid):
    """[S] median(disp_lambda_v) / median(disp_ref_v) over valid pixels, one sample."""
    out = []
    for v in range(disp_l.shape[0]):
        m = valid[v]
        if not bool(m.any()):
            out.append(float("nan"))
            continue
        r = disp_ref[v][m].median().clamp_min(1e-8)
        out.append(float(disp_l[v][m].median() / r))
    return out


def geomean(xs):
    xs = [x for x in xs if x == x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--max-chunks", type=int, default=2)
    args = ap.parse_args()

    cfg = OmegaConf.load(os.path.join(args.run, "args.yaml"))
    OmegaConf.resolve(cfg)
    set_seed(0)
    accelerator = Accelerator(mixed_precision=cfg.mixed_precision)
    model = build_model(cfg, accelerator)
    ckpt = os.path.join(args.run, f"checkpoint-step-{args.step}")
    missing, unexpected = model.load_state_dict(
        load_file(os.path.join(ckpt, "model.safetensors"), device="cpu"), strict=False)
    print(f"loaded {ckpt}: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model.to(accelerator.device)
    model.eval()
    device = model.pipe.device
    enc = model.view_condition_encoder
    print(f"normalize_moment={enc.normalize_moment}  roce={model.roce_mode}  "
          f"rope_frame_axis={model.rope_frame_axis}", flush=True)

    dataset = build_dataset(cfg)
    eval_max_samples = int(cfg.get("eval_max_samples", 2))
    eval_batch_size = max(1, min(eval_max_samples, int(cfg.get("eval_batch_size", 4))))
    samples_per_scene = cfg.get("samples_per_scene", 1)
    eval_indices = [i * samples_per_scene for i in range(len(dataset.records))][:eval_max_samples]
    loader = DataLoader(Subset(dataset, eval_indices), batch_size=eval_batch_size,
                        shuffle=False, num_workers=0, collate_fn=custom_collate_fn,
                        drop_last=False)

    num_src = model.num_source_views
    num_steps = model.eval_num_inference_steps
    seed = model.eval_sample_seed
    autocast_ctx = torch.autocast(device_type="cuda", dtype=model.pipe.torch_dtype, enabled=True)

    def preprocess(batch):
        with torch.no_grad(), autocast_ctx:
            model._eval_inputs = True
            try:
                return model.forward_preprocess(batch)
            finally:
                model._eval_inputs = False

    def blind_pass1(inputs, eps_rgb, eps_dep):
        blind = dict(inputs)
        blind["warp_block"] = torch.zeros_like(inputs["warp_block"])
        if inputs.get("depth_warp_block") is not None:
            blind["depth_warp_block"] = torch.zeros_like(inputs["depth_warp_block"])
        with torch.no_grad(), autocast_ctx:
            dep1, _ = sample_rails(model, blind, num_steps, eps_rgb, eps_dep)
        return dep1

    def scaled_batch(batch, lam):
        b = dict(batch)
        c2w = batch["c2w"].clone()
        c2w[..., :3, 3] = c2w[..., :3, 3] * lam
        b["c2w"] = c2w
        return b

    agg = {lam: {"plucker": 0.0, "roce": 0.0, "dep1": 0.0, "ratios": []} for lam in LAMBDAS}
    ctrl = {"plucker": 0.0, "dep1": 0.0}
    n_samples = 0
    for chunk_idx, batch in enumerate(loader):
        if chunk_idx >= args.max_chunks:
            break
        ref_in = preprocess(batch)
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        eps_rgb = torch.randn(ref_in["rgb_gt"].shape, device=device,
                              dtype=ref_in["rgb_gt"].dtype, generator=g)
        eps_dep = torch.randn(ref_in["depth_gt"].shape, device=device,
                              dtype=ref_in["depth_gt"].dtype, generator=g)
        dep1_ref = blind_pass1(ref_in, eps_rgb, eps_dep)
        sdisp_ref = decode_src_disparity(model, dep1_ref)
        valid = ref_in["loss_masks"][:, 0, :num_src, 0] > 0.5

        # Determinism floor: the same call twice.  Pass 1 has no splat, so this
        # should be 0; anything nonzero here is the bar every lambda must clear.
        dep1_rep = blind_pass1(ref_in, eps_rgb, eps_dep)
        floor = maxdiff(dep1_rep, dep1_ref)

        for lam in LAMBDAS:
            lam_in = preprocess(scaled_batch(batch, lam))
            a = agg[lam]
            a["plucker"] = max(a["plucker"], maxdiff(lam_in.get("plucker_block"), ref_in.get("plucker_block")))
            a["roce"] = max(a["roce"], maxdiff(lam_in.get("roce_camera_latents"), ref_in.get("roce_camera_latents")))
            dep1_l = blind_pass1(lam_in, eps_rgb, eps_dep)
            a["dep1"] = max(a["dep1"], maxdiff(dep1_l, dep1_ref))
            sdisp_l = decode_src_disparity(model, dep1_l)
            for b in range(sdisp_l.shape[0]):
                a["ratios"].append(geomean(per_view_median_ratio(sdisp_l[b], sdisp_ref[b], valid[b])))

        # POSITIVE CONTROL: same lambda=2.0 perturbation with normalisation off.
        enc.normalize_moment = False
        try:
            c_in = preprocess(scaled_batch(batch, 2.0))
            ctrl["plucker"] = max(ctrl["plucker"], maxdiff(c_in.get("plucker_block"), ref_in.get("plucker_block")))
            dep1_c = blind_pass1(c_in, eps_rgb, eps_dep)
            ctrl["dep1"] = max(ctrl["dep1"], maxdiff(dep1_c, dep1_ref))
        finally:
            enc.normalize_moment = True

        n_samples += sdisp_ref.shape[0]
        print(f"[chunk {chunk_idx}] determinism floor max|dep1 - dep1|={floor:.3e}  "
              + "  ".join(f"lam={lam}: d_plucker={agg[lam]['plucker']:.3e} d_roce={agg[lam]['roce']:.3e} "
                          f"d_dep1={agg[lam]['dep1']:.3e}" for lam in LAMBDAS)
              + f"  | CONTROL(norm off, lam=2): d_plucker={ctrl['plucker']:.3e} d_dep1={ctrl['dep1']:.3e}",
              flush=True)

    print(f"\n=== baseline invariance, {n_samples} samples ===")
    print(f"{'lambda':>7} {'max|d plucker|':>15} {'max|d roce|':>12} {'max|d dep1|':>12} "
          f"{'geomean disp ratio':>19} {'route(b) predicts':>18}")
    for lam in LAMBDAS:
        a = agg[lam]
        rs = [r for r in a["ratios"] if r == r]
        print(f"{lam:>7.2f} {a['plucker']:>15.3e} {a['roce']:>12.3e} {a['dep1']:>12.3e} "
              f"{sum(rs)/len(rs) if rs else float('nan'):>19.4f} {1.0/lam:>18.4f}")
    print(f"  positive control (normalize_moment=False, lambda=2.0): "
          f"max|d plucker|={ctrl['plucker']:.3e}  max|d dep1|={ctrl['dep1']:.3e}")
    print("  Read: with normalisation ON, d_plucker and d_roce must be ~0 and the")
    print("  disparity ratio ~1.0 for every lambda -- the model cannot see the")
    print("  baseline magnitude, so route (b) is unavailable and only the statistical")
    print("  convention (a) sets its global scale.  The control must be NONZERO or the")
    print("  hook is dead.  If d_plucker is ~0 but d_dep1 is NOT, some other input")
    print("  still leaks c2w into pass 1 and this reading is incomplete.")


if __name__ == "__main__":
    main()
