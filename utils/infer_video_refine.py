"""Multi-round frame-complete interpolation over one scene, as an mp4.

Starting from GT anchor frames, repeatedly pick up to 3 pairs of consecutive
known frames whose integer midpoint is still unknown, forward the model with
those 6 sources, and merge the 3 rendered midpoints back into the known set.
Frames predicted in earlier rounds are fed back as source conditioning in
later rounds (--refine_source pred), so the whole scene ends up covered:
GT at the anchors, prediction everywhere else.

Outputs per scene directory:
  full_video.mp4     every frame in view_id order (GT anchors + predictions)
  interp_compare.mp4 GT|PRED x RGB|depth grid of the interpolated frames
  frames_full/*.png  the raw full-timeline frames

Run as a MODULE from the project root:
  python -m utils.infer_video_refine \
      --config <run>/args.yaml \
      --checkpoint <run>/checkpoint-step-900 \
      --record_idx 5000 \
      --output_dir <run>/infer_refine
"""
import argparse
import os
import subprocess

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from examples.wanvideo.model_training.train_with_accelerate_rgbd_nvs import custom_collate_fn
from utils.infer_video_demo import (
    _compose_video,
    _ffmpeg_bin,
    _psnr,
    build_dataset,
    build_model,
)


def _load_batch(dataset, record, source_views, target_views):
    orig_select = dataset._select_views
    dataset._select_views = lambda record_, idx=0: (source_views, target_views)
    try:
        return custom_collate_fn([dataset[0]])
    finally:
        dataset._select_views = orig_select


def _run_round(model, dataset, record, source_views, target_views, device,
               latent_scale, depth_latent_scale, num_steps, pred_slots):
    """One forward; `pred_slots` maps source-list index -> (rgb_u8, disp HW)
    for source frames produced by earlier rounds (fed back as conditioning)."""
    batch = _load_batch(dataset, record, source_views, target_views)
    for slot, (rgb, disp) in (pred_slots or {}).items():
        batch["images"][0, slot] = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        d = torch.from_numpy(np.clip(disp, 0.0, 1.0).astype(np.float32))
        batch["disparity"][0, slot] = d.unsqueeze(0).repeat(3, 1, 1)

    with torch.autocast(device_type="npu", dtype=model.pipe.torch_dtype, enabled=True):
        model._eval_inputs = True
        try:
            inputs = model.forward_preprocess(batch)
        finally:
            model._eval_inputs = False
        gen = torch.Generator(device=device)
        gen.manual_seed(model.eval_sample_seed)
        pred_depth, pred_rgb = model.sample_rgb(
            inputs, num_steps=num_steps, generator=gen
        )

    if latent_scale != 1.0:
        pred_rgb = pred_rgb / latent_scale
    if depth_latent_scale != 1.0:
        pred_depth = pred_depth / depth_latent_scale

    gt_images = batch["images"][0].numpy()
    gt_disp = batch["disparity"][0].numpy()
    num_src = len(source_views)
    out = {}
    for slot, view in enumerate(source_views):
        out[view.view_id] = {
            "role": "src",
            "gt_rgb": (gt_images[slot].transpose(1, 2, 0) * 255).astype(np.uint8),
            "gt_disp": gt_disp[slot][0],
        }
    for t, view in enumerate(target_views):
        vi = num_src + t
        gt_rgb = (gt_images[vi].transpose(1, 2, 0) * 255).astype(np.uint8)
        rgb = model.pipe.vae.decode(
            pred_rgb[:, :, vi:vi + 1].to(device).to(model.pipe.torch_dtype), device=device
        )[0, :, 0].float().clamp(0, 1).cpu().numpy()
        raw = model.pipe.vae.decode(
            pred_depth[:, :, vi:vi + 1].to(device).to(model.pipe.torch_dtype), device=device
        )[0, :, 0].float().mean(0).cpu().numpy()
        pred_rgb_u8 = (rgb.transpose(1, 2, 0) * 255).astype(np.uint8)
        out[view.view_id] = {
            "role": "tgt",
            "gt_rgb": gt_rgb,
            "gt_disp": gt_disp[vi][0],
            "pred_rgb": pred_rgb_u8,
            "pred_disp": raw,
            "psnr": _psnr(pred_rgb_u8, gt_rgb),
        }
    del inputs, pred_rgb, pred_depth, batch
    torch.npu.empty_cache()
    return out


def _plan_round(known, num_src, num_tgt):
    """Pick one 6-source window (contiguous known frames) covering the most
    unknown frames, then up to num_tgt unknown positions inside its span,
    easiest-bracket first.  Targets may sit anywhere between the sources --
    the pair-midpoint convention is not required by the model.  Returns
    (src_positions, tgt_positions) or (None, None) when nothing is left."""
    ks = sorted(known)
    best_window, best_inside = None, []
    for i in range(len(ks) - num_src + 1):
        window = ks[i:i + num_src]
        inside = [u for u in range(window[0], window[-1] + 1) if u not in known]
        if len(inside) > len(best_inside):
            best_window, best_inside = window, inside
    if not best_inside:
        return None, None

    def bracket(u):
        lo = max(k for k in ks if k < u)
        hi = min(k for k in ks if k > u)
        return hi - lo

    tgt = sorted(sorted(best_inside, key=lambda u: (bracket(u), u))[:num_tgt])
    return best_window, tgt


def _encode_mp4(frame_arrays, path, framerate):
    tmp = path + ".frames"
    os.makedirs(tmp, exist_ok=True)
    for i, fr in enumerate(frame_arrays):
        im = Image.fromarray(fr)
        canvas = Image.new("RGB", (im.width + im.width % 2, im.height + im.height % 2), (0, 0, 0))
        canvas.paste(im, (0, 0))
        canvas.save(os.path.join(tmp, f"frame_{i:03d}.png"))
    subprocess.run(
        [_ffmpeg_bin(), "-y", "-framerate", str(framerate),
         "-i", os.path.join(tmp, "frame_%03d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", path],
        check=True, capture_output=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--record_idx", type=int, default=5000)
    parser.add_argument("--anchors", type=str, default="",
                        help="Comma-separated anchor positions (view_id order). "
                             "Default: 6 evenly spaced + the last two positions.")
    parser.add_argument("--refine_source", choices=["pred", "gt"], default="pred",
                        help="Feed earlier predictions (pred) or their GT (gt) "
                             "as source conditioning in later rounds.")
    parser.add_argument("--framerate", type=int, default=4)
    args = parser.parse_args()

    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from omegaconf import OmegaConf
    from PIL import Image, ImageDraw

    cfg = OmegaConf.load(args.config)
    set_seed(0)
    accelerator = Accelerator(mixed_precision=cfg.mixed_precision)
    device = accelerator.device
    num_steps = int(cfg.get("eval_num_inference_steps", 4))

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
    views = sorted(record["views"], key=lambda v: v.view_id)
    n = len(views)
    accelerator.print(f"[refine] scene={record['scene_id']} views={n}")

    if args.anchors:
        anchors = [int(v) for v in args.anchors.split(",") if v.strip()]
    else:
        anchors = [round(i * (n - 1) / 5) for i in range(6)] + [n - 2, n - 1]
    anchors = sorted(set(p for p in anchors if 0 <= p < n))

    known = {}  # position -> timeline entry
    for p in anchors:
        known[p] = {"role": "src"}

    latent_scale = float(cfg.get("latent_scale", 1.0))
    depth_latent_scale = model.depth_latent_scale
    num_src = cfg.num_source_views
    num_tgt = cfg.num_target_views
    sample_dir = os.path.join(
        args.output_dir, f"refine_r{args.record_idx}_{record['scene_id']}")
    os.makedirs(sample_dir, exist_ok=True)

    rnd = 0
    while any(p not in known for p in range(n)):
        src_pos, tgt_pos = _plan_round(set(known), num_src, num_tgt)
        if not src_pos:
            accelerator.print(f"[refine] no usable window left; "
                              f"{n - len(known)} frames uncovered")
            break
        rnd += 1
        accelerator.print(f"[refine] round {rnd}: src={src_pos} tgt={tgt_pos}")
        source_views = [views[p] for p in src_pos]
        target_views = [views[p] for p in tgt_pos]
        pred_slots = {}
        for slot, p in enumerate(src_pos):
            e = known.get(p)
            if (e is not None and e["role"] == "tgt"
                    and args.refine_source == "pred"):
                pred_slots[slot] = (e["pred_rgb"], e["pred_disp"])
        res = _run_round(model, dataset, record, source_views, target_views,
                         device, latent_scale, depth_latent_scale, num_steps,
                         pred_slots)
        for slot, p in enumerate(src_pos):
            if "gt_rgb" not in known[p]:
                known[p]["gt_rgb"] = res[views[p].view_id]["gt_rgb"]
                known[p]["gt_disp"] = res[views[p].view_id]["gt_disp"]
        for t, view in enumerate(target_views):
            p = tgt_pos[t]
            if p in known:
                continue
            known[p] = res[view.view_id]
        accelerator.print(f"[refine] round {rnd} done; "
                          f"{sum(p in known for p in range(n))}/{n} frames known")

    # GT pixels for anchors that never served as a source (e.g. odd tails)
    missing_src = [p for p, e in known.items()
                   if e["role"] == "src" and "gt_rgb" not in e]
    while missing_src:
        chunk = missing_src[:num_tgt]
        missing_src = missing_src[num_tgt:]
        filler = [p for p in sorted(known) if p not in chunk][:num_src]
        batch = _load_batch(dataset, record, [views[p] for p in filler],
                            [views[p] for p in chunk])
        gts = batch["images"][0].numpy()
        disps = batch["disparity"][0].numpy()
        for j, p in enumerate(chunk):
            vi = len(filler) + j
            known[p]["gt_rgb"] = (gts[vi].transpose(1, 2, 0) * 255).astype(np.uint8)
            known[p]["gt_disp"] = disps[vi][0]

    timeline = sorted(known.items(), key=lambda kv: kv[0])
    for _, e in timeline:
        e["caption"] = ("GT ANCHOR" if e["role"] == "src"
                        else f"PRED  psnr {e['psnr']:.2f} dB")

    mp4, n_cmp = _compose_video(sample_dir, timeline, args.framerate)
    accelerator.print(f"[refine] wrote {mp4} ({n_cmp} interpolated targets)")

    full_frames = []
    os.makedirs(os.path.join(sample_dir, "frames_full"), exist_ok=True)
    for vid, e in timeline:
        img = e["gt_rgb"] if e["role"] == "src" else e["pred_rgb"]
        head_colour = (30, 60, 120) if e["role"] == "src" else (120, 30, 30)
        tag = Image.new("RGB", (img.shape[1], 26), head_colour)
        ImageDraw.Draw(tag).text((8, 6), f"view {vid}  {e['caption']}",
                                 fill=(255, 255, 255))
        full_frames.append(np.concatenate([np.array(tag), img], axis=0))
        Image.fromarray(img).save(
            os.path.join(sample_dir, "frames_full", f"{vid}.png"))
    full_mp4 = os.path.join(sample_dir, "full_video.mp4")
    _encode_mp4(full_frames, full_mp4, args.framerate)
    psnrs = [e["psnr"] for _, e in timeline if e["role"] == "tgt"]
    accelerator.print(
        f"[refine] wrote {full_mp4}: {len(timeline)} frames "
        f"({len(anchors)} GT anchors, {len(psnrs)} interpolated, "
        f"mean psnr {np.mean(psnrs):.2f} dB)")


if __name__ == "__main__":
    main()
