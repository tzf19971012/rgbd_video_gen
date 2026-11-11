"""Turn the trajectory-inference frame dumps into mp4 videos.

Inputs: the per-run outputs of examples.wanvideo.model_training.infer_trajectory
(frames/<vid>_side.jpg strips of [gt rgb | pred rgb | gt depth | pred depth]
plus per_frame_metrics.jsonl).

Outputs (next to the frame dumps, per scene):
  video_<run>.mp4    rows [GT | PRED] rgb over [GT | PRED] depth, captioned
  video_all.mp4      columns GT / each run, rows rgb / depth -- one A/B video

Run from the repo root:
  python -m tools.make_traj_videos \
      --run_dirs logs/ovf_gae_re10k_aux0_patch1/logs/... (comma separated)
"""
import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw

_FONT_CACHE = {}


def _font(size):
    if size not in _FONT_CACHE:
        from matplotlib import font_manager
        from PIL import ImageFont

        _FONT_CACHE[size] = ImageFont.truetype(
            font_manager.findfont("DejaVu Sans", fallback_to_default=True), size)
    return _FONT_CACHE[size]


def _label(img, lines, title=None):
    """Top caption strip (title) + per-frame caption lines at the bottom."""
    f = _font(16)
    small = _font(13)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    top = (probe.textbbox((0, 0), title, font=f)[3] + 8) if title else 0
    bot = len(lines) * 18 + 8
    h, w = img.shape[:2]
    out = Image.new("RGB", (w, top + h + bot), (255, 255, 255))
    out.paste(Image.fromarray(img), (0, top))
    d = ImageDraw.Draw(out)
    if title:
        d.text((4, 2), title, fill=(0, 0, 0), font=f)
    y = top + h + 4
    for line in lines:
        d.text((4, y), line, fill=(30, 30, 30), font=small)
        y += 18
    return np.asarray(out)


def _split_side(fp):
    """frames/<vid>_side.jpg -> (gt_rgb, pred_rgb, gt_depth, pred_depth)."""
    strip = np.asarray(Image.open(fp))
    H, W = strip.shape[:2]
    tw = W // 4
    return [strip[:, i * tw:(i + 1) * tw] for i in range(4)]


def _sep_v(h, size=3, color=(255, 255, 255)):
    """Vertical separator (for horizontal stacking): (H, size, 3)."""
    return np.full((h, size, 3), color, np.uint8)


def _sep_h(w, size=3, color=(255, 255, 255)):
    """Horizontal separator (for vertical stacking): (size, W, 3)."""
    return np.full((size, w, 3), color, np.uint8)


def load_run(run_dir):
    scene_dirs = sorted(
        d for d in os.listdir(run_dir)
        if d.startswith("traj_step") and os.path.isdir(os.path.join(run_dir, d)))
    if not scene_dirs:
        raise FileNotFoundError(f"no traj_step* under {run_dir}")
    scene_dir = os.path.join(run_dir, scene_dirs[-1])
    scenes = sorted(
        s for s in os.listdir(scene_dir)
        if os.path.isdir(os.path.join(scene_dir, s)))
    scene = scenes[0]
    base = os.path.join(scene_dir, scene)
    rows = [json.loads(l) for l in open(os.path.join(base, "per_frame_metrics.jsonl"))]
    tiles = []
    for r in rows:
        fp = os.path.join(base, "frames", f"{r['view_id']}_side.jpg")
        gt_rgb, pred_rgb, gt_d, pred_d = _split_side(fp)
        tiles.append(dict(vid=r["view_id"], gt=gt_rgb, pred=pred_rgb,
                          gt_d=gt_d, pred_d=pred_d, m=r))
    step = scene_dirs[-1].replace("traj_step", "")
    return tiles, os.path.basename(run_dir.rstrip("/")), step, scene, base


def encode(frames, fps, path):
    """Write frames as an H.264 mp4 (yuv420p + faststart, browser-playable).

    Go through imageio_ffmpeg's bundled ffmpeg binary directly so the codec
    can't silently degrade to mpeg4/mp4v (which browsers can't play).
    """
    try:
        import imageio_ffmpeg
    except ImportError as e:
        raise RuntimeError(
            "imageio-ffmpeg is required to encode mp4 (pip install "
            "imageio-ffmpeg)") from e

    h, w = frames[0].shape[:2]
    if h % 2 or w % 2:
        w2, h2 = w - w % 2, h - h % 2
        frames = [np.asarray(Image.fromarray(f).resize(
            (w2, h2), Image.BILINEAR)) for f in frames]
        h, w = frames[0].shape[:2]

    writer = imageio_ffmpeg.write_frames(
        path, (w, h), fps=fps, codec="libx264", pix_fmt_out="yuv420p",
        output_params=["-crf", "18", "-movflags", "+faststart"],
        macro_block_size=1)
    writer.send(None)
    for f in frames:
        writer.send(np.ascontiguousarray(np.asarray(f), dtype=np.uint8))
    writer.close()
    print(f"wrote {path} ({len(frames)} frames)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dirs", type=str, required=True,
                        help="comma-separated run dirs, each with traj_step*/<scene>/")
    parser.add_argument("--fps", type=float, default=3.0)
    args = parser.parse_args()

    runs = [load_run(d.strip()) for d in args.run_dirs.split(",")]
    fps = args.fps

    # -- per-run videos: [GT|PRED] rgb row over [GT|PRED] depth row ----------
    for tiles, run_name, step, scene, base in runs:
        frames = []
        for t in tiles:
            m = t["m"]
            h = t["gt"].shape[0]
            rgb = np.concatenate(
                [t["gt"], _sep_v(h), t["pred"]], axis=1)
            dep = np.concatenate(
                [t["gt_d"], _sep_v(h), t["pred_d"]], axis=1)
            img = np.concatenate(
                [rgb, _sep_h(rgb.shape[1], 4), dep], axis=0)
            frames.append(_label(img, [
                f"v{t['vid']}  psnr={m['tgt_rgb_psnr']:.2f}  "
                f"lpips={m['tgt_rgb_lpips']:.3f}  d1={m['tgt_depth_delta1']:.3f}  "
                f"rel={m['tgt_depth_abs_rel_aligned']:.3f}  "
                f"(copy-src {m['nearest_src_psnr']:.2f})"],
                title=f"{run_name} @step{step}  {scene}"))
        encode(frames, fps, os.path.join(base, f"video_{run_name}_step{step}.mp4"))

    # -- combined A/B video: columns GT / run1 / run2 / ..., rgb + depth -----
    min_len = min(len(tiles) for tiles, *_ in runs)
    heads = [tiles[:min_len] for tiles, *_ in runs]
    labels = [f"{name} @s{step}" for _, name, step, _, _ in runs]

    col_w = heads[0][0]["gt"].shape[1]
    frames = []
    for i in range(min_len):
        gt = heads[0][i]["gt"]
        h, col_w = gt.shape[:2]
        rgb_cells = [gt] + [t[i]["pred"] for t in heads]
        dep_cells = [heads[0][i]["gt_d"]] + [t[i]["pred_d"] for t in heads]
        rgb = np.concatenate(
            sum([[c, _sep_v(h)] for c in rgb_cells], [])[:-1], axis=1)
        dep = np.concatenate(
            sum([[c, _sep_v(h)] for c in dep_cells], [])[:-1], axis=1)
        img = np.concatenate([rgb, _sep_h(rgb.shape[1], 4), dep], axis=0)
        m0 = heads[0][i]["m"]
        frames.append(_label(img, [
            f"v{m0['view_id']}  GT psnr vs each pred: "
            + "  ".join(
                f"{labels[j]}={heads[j][i]['m']['tgt_rgb_psnr']:.2f}"
                for j in range(len(heads)))],
            title=" | ".join(labels)))
    encode(frames, fps, os.path.join(
        os.path.dirname(runs[0][4]), "video_all.mp4"))


if __name__ == "__main__":
    main()
