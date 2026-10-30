"""re10k consecutive-frame NVS inference demo.

Sources are 6 CONSECUTIVE views of one re10k scene (manifest order, i.e. the
already-5-frame-sampled view_id order); targets are the next 3 consecutive
views.  Reuses build_model / build_dataset / _run_window / _compose_video from
utils.infer_video_demo, forcing the selection through the same _select_views
monkeypatch that tool's dense mode uses.

Run from the project root:
  .venv/bin/python -u -m tools.infer_re10k_consecutive \
      --config logs/<exp>/args_re10k_infer.yaml \
      --checkpoint logs/<exp>/checkpoint-step-2700 \
      --output_dir logs/<exp>/infer_re10k_cons6
"""
import argparse
import os

import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from safetensors.torch import load_file

from utils.infer_video_demo import (
    _compose_video,
    _run_window,
    build_dataset,
    build_model,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--record_idx", type=int, default=0,
                        help="Index into the re10k manifest records.")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="First source position in the scene's view_id order.")
    parser.add_argument("--num_src", type=int, default=6)
    parser.add_argument("--num_tgt", type=int, default=3)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--framerate", type=int, default=1)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    from accelerate.utils import set_seed
    set_seed(args.seed)
    accelerator = Accelerator(mixed_precision=cfg.mixed_precision)
    device = accelerator.device

    num_steps = args.num_inference_steps or int(cfg.get("eval_num_inference_steps", 4))

    # num_src / num_tgt must match the model's slot layout (6 src + 3 tgt).
    cfg.num_source_views = args.num_src
    cfg.num_target_views = args.num_tgt
    model = build_model(cfg, accelerator, num_steps)

    weights = os.path.join(args.checkpoint, "model.safetensors")
    state = load_file(weights, device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    accelerator.print(f"Loaded {weights}: {len(state)} tensors, "
                      f"missing={len(missing)}, unexpected={len(unexpected)}")
    model.to(device)
    model.eval()

    dataset = build_dataset(cfg)
    record = dataset.records[args.record_idx % len(dataset.records)]
    views_sorted = sorted(record["views"], key=lambda v: v.view_id)

    src_pos = list(range(args.start_idx, args.start_idx + args.num_src))
    tgt_pos = list(range(args.start_idx + args.num_src,
                         args.start_idx + args.num_src + args.num_tgt))
    if tgt_pos[-1] > len(views_sorted) - 1:
        raise ValueError(
            f"scene {record['scene_id']} has only {len(views_sorted)} views; "
            f"src {src_pos} + tgt {tgt_pos} out of range. Lower --start_idx.")
    source_views = [views_sorted[p] for p in src_pos]
    target_views = [views_sorted[p] for p in tgt_pos]

    accelerator.print(
        f"[re10k-cons] scene={record['scene_id']} views={len(views_sorted)} "
        f"src_ids={[v.view_id for v in source_views]} "
        f"tgt_ids={[v.view_id for v in target_views]}")

    latent_scale = float(cfg.get("latent_scale", 1.0))
    depth_latent_scale = model.depth_latent_scale
    res = _run_window(model, dataset, record, source_views, target_views,
                      device, latent_scale, depth_latent_scale, num_steps,
                      accelerator)

    sample_dir = os.path.join(
        args.output_dir,
        f"rec{args.record_idx:05d}_{str(record['scene_id'])[:16]}_s{args.start_idx}")
    os.makedirs(sample_dir, exist_ok=True)
    timeline = sorted(res.items(), key=lambda kv: kv[0])
    mp4, n = _compose_video(sample_dir, timeline, args.framerate)
    tgt_psnrs = [e["psnr"] for _, e in timeline if e["role"] == "tgt"]
    accelerator.print(
        f"[re10k-cons] wrote {mp4} ({n} target frames, "
        f"psnr {min(tgt_psnrs):.2f}..{max(tgt_psnrs):.2f} dB, "
        f"mean {sum(tgt_psnrs) / len(tgt_psnrs):.2f} dB)")
    del state
    torch.npu.empty_cache()


if __name__ == "__main__":
    main()
