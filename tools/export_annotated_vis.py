"""Export one checkpoint's eval visualisations with per-target metrics burned in.

SUPERSEDED FOR NEW RUNS.  evaluate() now draws this same caption strip itself
(_compose_eval_grid's src_captions/tgt_captions, wired up in evaluate() -- see
tools/smoke_eval_grid.py for a GPU-free check of that layout), plus source-view
abs_rel/delta1 under each source column, which this script does not have.  Any
eval_step_* directory written AFTER that change already has captions on its
jpgs; running this script on one of those draws a second, redundant strip below
the first.  This script still earns its keep on jpgs written BEFORE the change
(e.g. the c-sweep's logs/bmvs_c_*/eval_step_*, all captured under the old code),
which is the only case it should be pointed at.

Reuses the exact layout WanRGBDNVSTrainingModule.evaluate() already writes (3 rows
x (S + 2T) columns: rgb / depth gt / depth pred, src1..srcS then a gt/pred pair per
target -- see _compose_eval_grid) and appends ONE more strip below it, column-
aligned to each target's gt/pred pair, with that target's coverage and the four
headline metrics (tgt_rgb_psnr, tgt_rgb_lpips, tgt_depth_delta1, tgt_depth_abs_rel).
It does not touch the existing jpg -- the metrics were not baked into eval's own
render at the time these particular images were made, so this is a second pass
over the saved images plus per_view_metrics.jsonl.

WHY THE GEOMETRY IS RECOMPUTED, NOT GUESSED.  _compose_eval_grid downscales the
raw tiles, then _annotate_grid adds a left margin (row labels) and top margin
(column labels) sized from the label text at the SAME font.  Reusing
_label_font and the identical formulas (see geometry()) reconstructs tile_w,
tile_h, sep, left, top exactly -- geometry() asserts its own output against the
saved image's actual size, so a config mismatch (wrong num_source_views, wrong
resolution) fails loudly instead of drawing captions in the wrong place.

WHICH IMAGES EXIST.  Only `eval_vis_chunks * eval_batch_size` of the eval set's
64 scenes get a rendered image (VAE-decode cost); the ones that do are always
the FIRST N scenes in the frozen eval set's iteration order, i.e. the first N
distinct sample_id values in per_view_metrics.jsonl's file order (chunks are
appended in iteration order, and c-sweep never resumed so the file is clean).
filename b{k:02d}_step{step}.jpg <-> the (k+1)-th distinct sample_id.

Usage:
  python tools/export_annotated_vis.py --run logs/bmvs_c_sharedrope_s1337 --step 3500
  python tools/export_annotated_vis.py --run logs/bmvs_c_sharedrope_s1337 --step 3500 \
      --out reports/exports/sharedrope_s1337_step3500_annotated
"""
import argparse
import json
import os

import yaml
from PIL import Image, ImageDraw

from examples.wanvideo.model_training.WanRGBDNVSTrainingModule import _label_font

METRICS = [
    ("cov", "tgt_rgb_splat_coverage", "{:.3f}"),
    ("psnr", "tgt_rgb_psnr", "{:.2f} dB"),
    ("lpips", "tgt_rgb_lpips", "{:.3f}"),
    ("delta1", "tgt_depth_delta1", "{:.3f}"),
    ("abs_rel", "tgt_depth_abs_rel", "{:.3f}"),
]


def geometry(h, w, n_src, n_tgt, downscale, actual_size):
    """Reconstruct _compose_eval_grid/_annotate_grid's pixel layout.

    Mirrors the source formulas exactly (row/col separators are 4 px pre-
    downscale; label margins come from the SAME font at the SAME row labels).
    Asserts against `actual_size` = the saved image's real (width, height), so a
    cfg that does not match the run this image came from fails here, not as a
    silently misaligned caption.
    """
    n_cols = n_src + 2 * n_tgt
    grid_w = w * n_cols + 4 * (n_cols - 1)
    grid_h = 3 * h + 2 * 4
    new_w = grid_w // downscale
    new_h = grid_h // downscale
    tile_w = w * new_w / grid_w
    tile_h = h * new_h / grid_h
    sep = 4 * new_w / grid_w

    size = max(9, round(tile_h * 0.11))
    font = _label_font(size)
    gap = max(3, size // 3)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    row_labels = ["rgb", "depth gt", "depth pred"]
    left = max(probe.textbbox((0, 0), t, font=font)[2] - probe.textbbox((0, 0), t, font=font)[0]
              for t in row_labels) + 2 * gap
    top = size + 2 * gap
    width, height = round(left + new_w), round(top + new_h)

    if (width, height) != tuple(actual_size):
        raise ValueError(
            f"reconstructed grid geometry {(width, height)} != actual image size "
            f"{tuple(actual_size)} -- num_source_views/num_target_views/resolution/"
            f"eval_vis_downscale from args.yaml do not match this run; refusing to "
            f"draw captions in what would be the wrong place."
        )
    return dict(n_cols=n_cols, tile_w=tile_w, tile_h=tile_h, sep=sep, left=left, top=top)


def load_samples(per_view_path):
    """[(sample_id, {target_index: row}), ...] in first-occurrence (= vis) order."""
    order, by_sample = [], {}
    with open(per_view_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = row["sample_id"]
            if sid not in by_sample:
                by_sample[sid] = {}
                order.append(sid)
            by_sample[sid][row["target_index"]] = row
    return [(sid, by_sample[sid]) for sid in order]


def _centered_lines(draw, center_x, y, lines, font, line_gap, fill=(20, 20, 20)):
    for line in lines:
        x0, y0, x1, y1 = draw.textbbox((0, 0), line, font=font)
        draw.text((round(center_x - (x1 - x0) / 2.0 - x0), y), line, fill=fill, font=font)
        y += (y1 - y0) + line_gap
    return y


def annotate(img, rows_by_target, geo, n_tgt):
    """Append a metrics strip below `img`, one column-aligned block per target."""
    cap_size = max(10, round(geo["tile_h"] * 0.09))
    font = _label_font(cap_size)
    line_gap = max(2, cap_size // 4)
    n_lines = len(METRICS) + 1                          # +1 for the "tgtN" header
    block_h = n_lines * (cap_size + line_gap) + 2 * line_gap

    out = Image.new("RGB", (img.width, img.height + block_h), (255, 255, 255))
    out.paste(img, (0, 0))
    draw = ImageDraw.Draw(out)
    draw.line([(0, img.height), (img.width, img.height)], fill=(160, 160, 160), width=1)

    for t in range(n_tgt):
        gt_col = geo["n_cols"] - 2 * n_tgt + 2 * t
        pred_col = gt_col + 1
        x_left = geo["left"] + gt_col * (geo["tile_w"] + geo["sep"])
        x_right = geo["left"] + pred_col * (geo["tile_w"] + geo["sep"]) + geo["tile_w"]
        center_x = (x_left + x_right) / 2.0

        row = rows_by_target.get(t)
        lines = [f"tgt{t + 1}"]
        if row is None:
            lines.append("(no row)")
        else:
            for label, key, fmt in METRICS:
                v = row.get(key)
                lines.append(f"{label} " + (fmt.format(v) if v is not None else "--"))
        _centered_lines(draw, center_x, img.height + line_gap, lines, font, line_gap)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="e.g. logs/bmvs_c_sharedrope_s1337")
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--out", type=str, default=None,
                        help="Default: reports/exports/<run basename>_step<N>_annotated")
    args = parser.parse_args()

    with open(os.path.join(args.run, "args.yaml"), encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    h, w = cfg["resolution"]
    n_src, n_tgt = cfg["num_source_views"], cfg["num_target_views"]
    downscale = cfg.get("eval_vis_downscale", 2)
    fmt = cfg.get("eval_vis_format", "jpg")

    eval_dir = os.path.join(args.run, f"eval_step_{args.step:06d}")
    samples = load_samples(os.path.join(eval_dir, "per_view_metrics.jsonl"))

    out_dir = args.out or os.path.join(
        "reports", "exports",
        f"{os.path.basename(args.run.rstrip('/'))}_step{args.step}_annotated")
    os.makedirs(out_dir, exist_ok=True)

    n = 0
    for k, (sample_id, rows_by_target) in enumerate(samples):
        src = os.path.join(eval_dir, f"b{k:02d}_step{args.step:06d}.{fmt}")
        if not os.path.isfile(src):
            break                                        # only the first N scenes were rendered
        img = Image.open(src).convert("RGB")
        geo = geometry(h, w, n_src, n_tgt, downscale, img.size)
        out = annotate(img, rows_by_target, geo, n_tgt)
        safe_id = sample_id.replace("/", "__")
        dst = os.path.join(out_dir, f"b{k:02d}_step{args.step:06d}_{safe_id}.jpg")
        out.save(dst, quality=95, optimize=True)
        print(f"{dst}  ({sample_id})")
        n += 1

    print(f"\n{n} image(s) -> {out_dir}")


if __name__ == "__main__":
    main()
