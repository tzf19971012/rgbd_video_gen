"""Isolate whether the pass1/pass2 noise-draw convention is the ONLY reason
tools/diag_cross_stream_rails.py's `p2` arm (mean lpips 0.5232 on
bmvs_d_droce_s1337 step4000) disagrees with the production `eval/tgt_rgb_lpips`
TensorBoard scalar at that step (0.5073).

Two candidate differences, isolated in one run:

  1. NOISE CONVENTION.  Calls `model.sample_rgb()` DIRECTLY -- the production
     function, whose pass 2 continues the generator rather than repeating
     pass 1's draw (WanRGBDNVSTrainingModule.py:2136-2196) -- instead of
     diag_cross_stream_rails.py's `sample_rails`, which deliberately shares one
     draw across arms.  First full run (job 12609) measured this alone:
     0.5232 -> 0.5194, i.e. only 0.0038 of the 0.0159 gap.  NOT the whole story.
  2. LPIPS REFERENCE.  `evaluate()` scores against the VAE-ROUND-TRIPPED GT
     (`_decode_view(rgb_gt)`, WanRGBDNVSTrainingModule.py:2925), not the raw
     pixel GT `batch["images"]` that every tools/diag_*.py script uses.  The
     round trip removes high-frequency detail the model cannot reproduce
     anyway, so LPIPS against it is systematically LOWER.  This run reports
     both references side by side so the split is measured, not inferred.

Note this only affects ABSOLUTE numbers.  diag_cross_stream_rails.py's paired
deltas (warp_mvs vs p2 etc.) use one reference for every arm, so they are
internally consistent either way.

Run as a MODULE from the project root, one GPU:
  python -m tools.check_p2_noise_convention --run logs/bmvs_d_droce_s1337 --step 4000
"""
import argparse
import os

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Subset

from examples.wanvideo.model_training.train_with_accelerate_rgbd_nvs import custom_collate_fn
from tools.diag_pass1_prior import decode_rgb, psnr_full
from tools.vis_rgb_warp_pred_depth import build_dataset, build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--max-chunks", type=int, default=None)
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
    lpips_net = model._get_lpips(device)

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
    tgt_views = list(range(num_src, num_src + num_tgt))
    num_steps = model.eval_num_inference_steps
    seed = model.eval_sample_seed
    autocast_ctx = torch.autocast(device_type="cuda", dtype=model.pipe.torch_dtype, enabled=True)

    all_lpips, all_lpips_vaegt = [], []
    for chunk_idx, batch in enumerate(loader):
        if args.max_chunks is not None and chunk_idx >= args.max_chunks:
            break
        with torch.no_grad(), autocast_ctx:
            model._eval_inputs = True
            try:
                inputs = model.forward_preprocess(batch)
            finally:
                model._eval_inputs = False

            # THE PRODUCTION PATH, exactly as `evaluate()` calls it: one
            # generator object, handed to sample_rgb, which internally calls
            # _sample_rgb_once TWICE -- pass 2's torch.randn draws continue
            # this generator's state rather than repeating pass 1's draw.
            gen = torch.Generator(device=device)
            gen.manual_seed(seed)
            pred_depth, pred_rgb = model.sample_rgb(inputs, num_steps, gen)

            dec_rgb = decode_rgb(model, pred_rgb, tgt_views, device)
            # evaluate()'s reference: the GT latent decoded back through the
            # VAE, exactly `_decode_view(rgb_gt[b:b+1], vi)` at :2925.
            dec_gt = decode_rgb(model, inputs["rgb_gt"], tgt_views, device)
            rgb_px = batch["images"].to(device).float().clamp(0.0, 1.0)

        for b in range(pred_rgb.shape[0]):
            for vi in tgt_views:
                pred = dec_rgb[vi][b:b + 1]
                all_lpips.append(lpips(pred, rgb_px[b:b + 1, vi]))
                all_lpips_vaegt.append(lpips(pred, dec_gt[vi][b:b + 1]))
        print(f"[chunk {chunk_idx}] running mean over {len(all_lpips)} pairs: "
              f"raw-GT {sum(all_lpips)/len(all_lpips):.4f}   "
              f"VAE-GT {sum(all_lpips_vaegt)/len(all_lpips_vaegt):.4f}", flush=True)

    n = len(all_lpips)
    print(f"\n=== production noise convention, n={n} pairs ===")
    print(f"  lpips vs RAW pixel GT (tools/diag_*.py convention):   {sum(all_lpips)/n:.4f}")
    print(f"  lpips vs VAE-round-tripped GT (evaluate() convention): {sum(all_lpips_vaegt)/n:.4f}")
    print("  compare: TensorBoard eval/tgt_rgb_lpips @ step "
          f"{args.step} = 0.5073 ; diag_cross_stream_rails.py shared-noise p2 = 0.5232")


if __name__ == "__main__":
    main()
