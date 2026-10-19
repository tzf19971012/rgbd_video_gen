"""Build a flat `by_sample/` index of evaluate()'s rendered grids, named by
sample_id instead of by batch position.

WHY.  evaluate() names its images `b{local_b:02d}_step{global_step:06d}.jpg`
inside `OUT_DIR/shift<S>_steps<N>/eval_step_<global_step:06d>/`
(WanRGBDNVSTrainingModule.py:3077-3085).  Nothing in that filename says which
scene it is: recovering that means opening the sibling per_view_metrics.jsonl
and counting distinct sample_ids, because `b{k}` is the (k+1)-th DISTINCT
sample_id in that file's order (chunks append in iteration order -- the same
convention tools/export_annotated_vis.py documents).  This writes that mapping
out once as a directory of relative symlinks, so
`by_sample/57f8d9bbe73f6760f10e916a_tt0_s50.jpg` opens the scene you meant.

SYMLINKS, NOT COPIES.  The links are relative (`../shift<S>_steps<N>/...`), so
`by_sample/` costs ~12 KB next to 31 MB of images and the whole OUT_DIR tree
still tars/copies as one self-contained unit (`cp -r`, `rsync -a` and `tar czf`
all preserve them; the targets are inside the same tree).  Two consequences
worth knowing: copying `by_sample/` ALONE leaves 64 dangling links, and a
destination without symlink support (Windows, FAT/exFAT, some cloud-sync
folders) cannot hold it -- dereference with `tar czhf` / `cp -rL` for those.

SHARDING.  sbatch/eval_full_val_vis_shard.sbatch calls this at the end of every
array task with that task's own --num-shards/--shard-idx, so each shard indexes
only the eval_step_* directories it wrote.  That needs no cross-task
coordination and cannot race: link names are keyed by sample_id, and a sample
belongs to exactly one batch, hence to exactly one shard.  Run WITHOUT those
flags (the default) to index every eval_step_* present -- which is what to do
for an output directory produced before this script existed.

Run as a MODULE from the project root (no GPU, no checkpoint, no dataset):
  python -m tools.index_eval_vis_by_sample --out-dir logs/full_val_vis/<RUN>_step<STEP>
"""
import argparse
import glob
import json
import os
import re


def eval_step_dirs(cell_dir):
    """[(global_step, path)] for every eval_step_* in `cell_dir`, ascending."""
    out = []
    for path in sorted(glob.glob(os.path.join(cell_dir, "eval_step_*"))):
        m = re.search(r"eval_step_(\d+)$", path)
        if m and os.path.isdir(path):
            out.append((int(m.group(1)), path))
    return out


def samples_in_order(step_dir):
    """Distinct sample_ids of one eval_step dir, in per_view_metrics.jsonl order.

    Position k in this list is the `b{k:02d}` of the rendered filenames -- the
    mapping evaluate() never writes down anywhere else.
    """
    jsonl = os.path.join(step_dir, "per_view_metrics.jsonl")
    if not os.path.exists(jsonl):
        return None
    order = []
    with open(jsonl, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            sid = json.loads(line)["sample_id"]
            if sid not in order:
                order.append(sid)
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True,
                    help="An eval output root, i.e. the --output_dir given to "
                         "eval_rgbd_nvs.py (the directory CONTAINING shift<S>_steps<N>/).")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="Index only the batches this shard owns, using the same "
                         "round-robin rule as eval_rgbd_nvs.py. Default 1 = index "
                         "every eval_step_* found.")
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--num-batches", type=int, default=None,
                    help="Total batch count of the whole run, needed to reproduce the "
                         "round-robin assignment when --num-shards > 1. Without it the "
                         "shard would have to glob the other shards' directories, which "
                         "may not exist yet while the array is still running.")
    args = ap.parse_args()
    if args.num_shards < 1 or not (0 <= args.shard_idx < args.num_shards):
        ap.error(f"--shard-idx {args.shard_idx} out of range for --num-shards {args.num_shards}")
    if args.num_shards > 1 and args.num_batches is None:
        ap.error("--num-batches is required when --num-shards > 1")

    cells = sorted(d for d in glob.glob(os.path.join(args.out_dir, "shift*_steps*"))
                   if os.path.isdir(d))
    if not cells:
        raise SystemExit(f"no shift<S>_steps<N> directory under {args.out_dir} -- "
                         f"is this an eval output root?")
    if len(cells) > 1:
        # by_sample/ sits ABOVE the cell dir, so two cells would fight over the
        # same link names with no way to tell whose render won.  Refuse rather
        # than silently index one sampler's images under a shared name.
        raise SystemExit(
            f"{args.out_dir} holds {len(cells)} sampler cells "
            f"({', '.join(os.path.basename(c) for c in cells)}); a flat by_sample/ "
            f"cannot name both. Point --out-dir at a single-cell root, or index the "
            f"cell directories separately.")
    cell = cells[0]

    if args.num_shards > 1:
        owned = set(range(args.shard_idx, args.num_batches, args.num_shards))
    else:
        owned = None

    flat = os.path.join(args.out_dir, "by_sample")
    os.makedirs(flat, exist_ok=True)
    n_links, n_skipped = 0, 0
    for step, step_dir in eval_step_dirs(cell):
        if owned is not None and step not in owned:
            continue
        order = samples_in_order(step_dir)
        if order is None:
            print(f"  skip eval_step_{step:06d}: no per_view_metrics.jsonl")
            n_skipped += 1
            continue
        for b, sid in enumerate(order):
            matches = glob.glob(os.path.join(step_dir, f"b{b:02d}_step{step:06d}.*"))
            # save_vis=False scores a chunk without rendering it, so a missing
            # image is a legitimate state, not a corrupt directory.
            matches = [m for m in matches if not m.endswith(".jsonl")]
            if not matches:
                n_skipped += 1
                continue
            src = matches[0]
            link = os.path.join(flat, sid.replace("/", "_") + os.path.splitext(src)[1])
            if os.path.lexists(link):
                os.remove(link)
            os.symlink(os.path.relpath(src, flat), link)
            n_links += 1

    scope = (f"shard {args.shard_idx}/{args.num_shards}" if owned is not None else "all batches")
    print(f"by_sample index ({scope}): {n_links} links in {flat}"
          + (f"  ({n_skipped} unrendered/missing)" if n_skipped else ""))


if __name__ == "__main__":
    main()
