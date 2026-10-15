import multiprocessing
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .kubric4d_rgbd_nvs_dataset import (_disparity_from_shared_scale,
                                        _disparity_per_view,
                                        _inverse_depth,
                                        _load_rgb,
                                        _resize_crop_rgb_depth,
                                        _shared_disparity_scale)


def _quat_to_rotmat(quat):
    """(qx, qy, qz, qw) -> 3x3 rotation matrix."""
    x, y, z, w = quat
    n = torch.sqrt((x * x + y * y + z * z + w * w).clamp_min(1e-12))
    x, y, z, w = x / n, y / n, z / n, w / n
    return torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=torch.float64,
    )


def _parse_tartanair_pose(path):
    """pose_{left,right}.txt -> {frame: (t, quat)} with 'tx ty tz qx qy qz qw' per line."""
    poses = {}
    with open(path, "r", encoding="utf-8") as handle:
        for frame, line in enumerate(handle):
            parts = line.split()
            if len(parts) < 7:
                continue
            t = torch.tensor(list(map(float, parts[0:3])), dtype=torch.float64)
            quat = torch.tensor(list(map(float, parts[3:7])), dtype=torch.float64)
            poses[frame] = (t, quat)
    return poses


# TartanAir (AirSim) camera frame: x forward, y right, z down (verified against
# the stereo baseline: right camera sits at (0, +0.25, 0) in the left frame).
# ray_condition expects an OpenCV pinhole frame: x right, y down, z forward.
_TA_TO_CV = torch.tensor(
    [[0.0, 0.0, 1.0],
     [1.0, 0.0, 0.0],
     [0.0, 1.0, 0.0]],
    dtype=torch.float64,
)


def _c2w_from_tartanair_pose(t, quat):
    """camera-to-world in the OpenCV pinhole frame."""
    c2w = torch.eye(4, dtype=torch.float64)
    c2w[:3, :3] = _quat_to_rotmat(quat) @ _TA_TO_CV
    c2w[:3, 3] = t
    return c2w


def _load_tartanair_depth(path, invalid_threshold):
    """TartanAir depth .npy: float32 HxW, meters, no invalid sentinel.

    Sky / far pixels carry huge values (up to ~16000) instead of a sentinel;
    treat anything >= invalid_threshold as invalid.
    Returns (depth_meter [H, W] float32, valid_mask [H, W] bool).
    """
    depth = torch.from_numpy(np.load(path).astype(np.float32))
    valid = torch.isfinite(depth) & (depth > 0) & (depth < invalid_threshold)
    return depth, valid


class TartanAirPoseNVSDataset(Dataset):
    """TartanAir dataloader for pose-NVS training.

    Directory layout:
        <root>/<env>/<difficulty>/P<seq>/
            image_left/ 000000_left.png  (640x480 RGB)
            image_right/ 000000_right.png
            depth_left/ 000000_left_depth.npy  (float32, meters)
            depth_right/ 000000_right_depth.npy
            pose_left.txt / pose_right.txt  (tx ty tz qx qy qz qw per line, c2w)

    Poses are converted from the AirSim frame (x fwd, y right, z down) to the
    OpenCV pinhole frame expected by ray_condition.  Intrinsics default to the
    published TartanAir stereo calibration (fx=fy=320, cx=320, cy=240, i.e.
    90 deg FOV at 640x480, 0.25 m baseline) because the sample data ships no
    calib file; override via `intrinsics` if a full download provides them.

    Every (frame, camera) pair is one view.  A sample draws one anchor view
    (source slot 0), num_source_views-1 additional nearby source views and
    num_target_views distant views, all inside one (env, difficulty, seq).
    The output dict mirrors VKitti2PoseNVSDataset exactly, so
    WanPoseNVSTrainingModule consumes it unchanged.
    """

    CAMERAS = ("left", "right")

    def __init__(
        self,
        tartanair_root,
        resolution=(384, 672),
        num_source_views=9,
        num_target_views=3,
        src_frame_range=10,
        tgt_frame_min=15,
        tgt_frame_max=30,
        random_permute_source_views=True,
        disparity_norm_mode="shared_scale",
        disparity_norm_quantile=0.98,
        disparity_norm_headroom=1.5,
        depth_clamp_max=None,
        deterministic_view_seed=None,
        samples_per_scene=1,
        max_samples=None,
        envs=None,
        difficulties=None,
        intrinsics=(320.0, 320.0, 320.0, 240.0),
        invalid_threshold=1000.0,
        tgt_interpolate=False,
        src_min_offset=1,
        src_stratified=False,
        tgt_min_sep=0,
        tgt_center_frac=0.0,
        tgt_extrap_frac=0.0,
        tgt_extrap_max=5,
        tgt_extrap_warmup=0,
        src_order="none",
        **kwargs,
    ):
        # Source ordering within the sequence layout.  "none" keeps the legacy
        # behaviour (random_permute_source_views shuffles the non-anchor tail
        # every sample); "pose_chain" replaces the shuffle with a deterministic
        # nearest-neighbour chain over camera centres (start = source closest
        # to the target camera), mirroring the kubric4d dataset so adjacent
        # sequence slots hold spatially adjacent views for the video DiT.
        if src_order not in ("none", "pose_chain"):
            raise ValueError(
                f"src_order must be 'none' or 'pose_chain', got {src_order!r}"
            )
        self.src_order = src_order
        self.tartanair_root = tartanair_root
        self.resolution = tuple(resolution)
        self.num_source_views = int(num_source_views)
        self.num_target_views = int(num_target_views)
        self.num_views = self.num_source_views + self.num_target_views
        self.src_frame_range = int(src_frame_range)
        self.tgt_frame_min = int(tgt_frame_min)
        self.tgt_frame_max = int(tgt_frame_max)
        # tgt_interpolate: draw targets strictly INSIDE the source-view frame
        # span (min..max of the drawn source frames) instead of anchor-relative
        # far offsets, so the target is an interpolation task rather than an
        # extrapolation one.
        self.tgt_interpolate = bool(tgt_interpolate)
        # Anti-shortcut sampling constraints (all default to the old behaviour):
        # src_min_offset: lower bound on |src_frame - anchor| so the source cone
        #   is never a near-duplicate cluster (1 frame ~ 0.15 m displacement).
        # tgt_min_sep: min |tgt_frame - src_frame| for EVERY source frame, so an
        #   interpolated target is never a trivial copy of an adjacent source.
        # tgt_center_frac: shrink the interpolated target range inward by this
        #   fraction of the source span on each side, biasing the target toward
        #   the middle where it is genuinely "between" the sources.
        self.src_min_offset = min(int(src_min_offset), self.src_frame_range)
        self.src_stratified = bool(src_stratified)
        self.tgt_min_sep = int(tgt_min_sep)
        self.tgt_center_frac = float(tgt_center_frac)
        # Mild-extrapolation mix (training-time augmentation; eval keeps its own
        # fixed distribution): with probability tgt_extrap_frac per sample the
        # target is drawn just OUTSIDE the source cone, at an anchor offset in
        # (src_frame_range, src_frame_range + tgt_extrap_max] frames.  1 frame
        # ~ 0.15 m, so tgt_extrap_max=5 with src_frame_range=10 caps targets at
        # ~1.5x the source support -- beyond blending range (anti copy-shortcut)
        # but short of the 2-4x regime where most content is unobserved and the
        # loss is irreducible noise.  tgt_extrap_warmup ramps the effective
        # fraction 0 -> tgt_extrap_frac over that many sampled draws (curriculum;
        # 0 = no ramp).  Each sample carries a `tgt_extrapolated` flag so the
        # training loop can down-weight these samples.
        self.tgt_extrap_frac = float(tgt_extrap_frac)
        self.tgt_extrap_max = max(1, int(tgt_extrap_max))
        self.tgt_extrap_warmup = int(tgt_extrap_warmup)
        # Draw counter SHARED across DataLoader worker processes (they are
        # forked once per epoch, so a plain int would reset every epoch and the
        # warmup ramp would never progress).  Only the training dataset
        # increments it; seeded (eval) sampling never does.
        self._extrap_draws = multiprocessing.Value("l", 0)
        self._last_sample_extrapolated = False
        self.random_permute_source_views = bool(random_permute_source_views)
        if disparity_norm_mode not in ("shared_scale", "per_view"):
            raise ValueError(
                f"disparity_norm_mode must be 'shared_scale' or 'per_view', got {disparity_norm_mode!r}"
            )
        self.disparity_norm_mode = disparity_norm_mode
        self.disparity_norm_quantile = float(disparity_norm_quantile)
        self.disparity_norm_headroom = float(disparity_norm_headroom)
        # Metric depth clamp (metres) applied BEFORE taking the inverse for the
        # disparity representation -- mirrors the DVD-official preprocessing
        # (TartanAirDepthTransform clamps depth to d_max=50 first).  The raw
        # depth_metric kept in the sample is NOT clamped, so eval metric
        # bookkeeping is unaffected.
        self.depth_clamp_max = (
            None if depth_clamp_max is None else float(depth_clamp_max)
        )
        self.deterministic_view_seed = (
            None if deterministic_view_seed is None else int(deterministic_view_seed)
        )
        self.samples_per_scene = int(samples_per_scene)
        self.invalid_threshold = float(invalid_threshold)

        fx, fy, cx, cy = map(float, intrinsics)
        self.K = torch.tensor(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float64
        )

        # records: one entry per (env, difficulty, seq) with geometry tables
        self.records = []
        for env in sorted(os.listdir(self.tartanair_root)):
            env_dir = os.path.join(self.tartanair_root, env)
            if not os.path.isdir(env_dir):
                continue
            if envs is not None and env not in envs:
                continue
            for difficulty in sorted(os.listdir(env_dir)):
                diff_dir = os.path.join(env_dir, difficulty)
                if not os.path.isdir(diff_dir):
                    continue
                if difficulties is not None and difficulty not in difficulties:
                    continue
                for seq in sorted(os.listdir(diff_dir)):
                    seq_dir = os.path.join(diff_dir, seq)
                    if not os.path.isdir(seq_dir):
                        continue
                    record = self._build_record(env, difficulty, seq, seq_dir)
                    if record is not None:
                        self.records.append(record)
        if not self.records:
            raise ValueError(f"No TartanAir sequences found under {tartanair_root}")
        if max_samples is not None:
            self.records = self.records[: int(max_samples)]

        need = self.tgt_frame_max * 2 + 2
        for record in self.records:
            if len(record["frames"]) < need:
                print(
                    f"[TartanAirPoseNVSDataset] WARNING: {record['scene_id']}/{record['time_id']} "
                    f"has only {len(record['frames'])} complete frames (< {need}); target sampling "
                    f"will clamp to the available range."
                )

    def _build_record(self, env, difficulty, seq, seq_dir):
        pose = {}
        c2w = {}
        for cam in self.CAMERAS:
            pose_path = os.path.join(seq_dir, f"pose_{cam}.txt")
            if not os.path.isfile(pose_path):
                return None
            pose[cam] = _parse_tartanair_pose(pose_path)
        frames = sorted(
            f for f in pose["left"]
            if f in pose["right"]
            and all(
                os.path.isfile(
                    os.path.join(seq_dir, f"image_{cam}", f"{f:06d}_{cam}.png")
                )
                and os.path.isfile(
                    os.path.join(
                        seq_dir, f"depth_{cam}", f"{f:06d}_{cam}_depth.npy"
                    )
                )
                for cam in self.CAMERAS
            )
        )
        if not frames:
            return None
        for cam in self.CAMERAS:
            c2w[cam] = {
                f: _c2w_from_tartanair_pose(*pose[cam][f]) for f in frames
            }
        return {
            "scene_id": env,
            "time_id": f"{difficulty}/{seq}",
            "seq_dir": seq_dir,
            "frames": frames,
            "c2w": c2w,
        }

    def __len__(self):
        return len(self.records) * self.samples_per_scene

    def _rng(self, idx):
        if self.deterministic_view_seed is not None:
            return random.Random(self.deterministic_view_seed + int(idx))
        return random

    def _effective_extrap_frac(self, draws):
        """Curriculum ramp: 0 until tgt_extrap_warmup draws have elapsed, then
        linearly up to tgt_extrap_frac.  `draws` is the shared cross-process
        counter value; seed-based (eval) sampling never increments it and never
        takes the extrapolation branch, so frozen eval sets stay bit-identical."""
        if self.tgt_extrap_frac <= 0.0:
            return 0.0
        if self.tgt_extrap_warmup <= 0:
            return self.tgt_extrap_frac
        return self.tgt_extrap_frac * min(1.0, draws / float(self.tgt_extrap_warmup))

    def _select_views(self, record, idx=0):
        """-> (source_views, target_views), each a list of (frame, camera).

        Sources: anchor (first) + nearby frames; targets: distant frames.
        All views are distinct (frame, camera) pairs.
        """
        rng = self._rng(idx)
        frames = record["frames"]
        f_min, f_max = frames[0], frames[-1]
        span = len(frames) - 1
        used = set()

        # Stratified source offsets: split [src_min_offset, src_frame_range]
        # into num_source_views-1 equal bins and draw one offset per bin, so
        # the source cone spans the full +-src_frame_range window evenly
        # instead of clustering on random small offsets.
        stratified_offsets = []
        if self.src_stratified and self.num_source_views > 1:
            lo, hi = self.src_min_offset, self.src_frame_range
            n = self.num_source_views - 1
            if hi - lo + 1 >= n:
                edges = [lo + (hi - lo + 1) * i // n for i in range(n + 1)]
                stratified_offsets = [
                    rng.randint(edges[i], edges[i + 1] - 1) for i in range(n)
                ]
                rng.shuffle(stratified_offsets)

        def draw_near(anchor_frame):
            for _ in range(50):
                if stratified_offsets:
                    offset = stratified_offsets.pop()
                else:
                    offset = rng.randint(self.src_min_offset, self.src_frame_range)
                f = anchor_frame + rng.choice([-1, 1]) * offset
                f = min(max(f, f_min), f_max)
                c = rng.choice(self.CAMERAS)
                if (f, c) not in used:
                    used.add((f, c))
                    return (f, c)
            raise RuntimeError("could not sample distinct near source views")

        def draw_far(anchor_frame):
            lo = min(self.tgt_frame_min, max(1, span // 2))
            hi = min(self.tgt_frame_max, span)
            for _ in range(50):
                offset = rng.randint(lo, max(lo, hi))
                f = anchor_frame + rng.choice([-1, 1]) * offset
                f = min(max(f, f_min), f_max)
                c = rng.choice(self.CAMERAS)
                if (f, c) not in used:
                    used.add((f, c))
                    return (f, c)
            raise RuntimeError("could not sample distinct far target views")

        def draw_mild_extrapolated(anchor_frame):
            # Just outside the source cone: offset in
            # (src_frame_range, src_frame_range + tgt_extrap_max] from the
            # anchor.  tgt_min_sep (>=1) still applies so the target can never
            # duplicate a source frame even after sequence-edge clamping.
            lo = self.src_frame_range + 1
            hi = self.src_frame_range + self.tgt_extrap_max
            sep = max(self.tgt_min_sep, 1)
            for _ in range(50):
                offset = rng.randint(lo, hi)
                f = anchor_frame + rng.choice([-1, 1]) * offset
                f = min(max(f, f_min), f_max)
                c = rng.choice(self.CAMERAS)
                if (f, c) not in used and all(
                    abs(f - s) >= sep for s, _ in source_views
                ):
                    used.add((f, c))
                    return (f, c)
            raise RuntimeError("could not sample distinct extrapolated target views")

        def draw_interpolated():
            src_frames = [f for f, _ in source_views]
            lo_f, hi_f = min(src_frames), max(src_frames)
            # Center-bias: shrink the drawable range inward so the target sits
            # in the middle of the source span, not hugging one edge.
            margin = int(self.tgt_center_frac * (hi_f - lo_f))
            lo, hi = lo_f + margin, hi_f - margin
            if lo > hi:
                lo, hi = lo_f, hi_f
            for _ in range(50):
                f = rng.randint(lo, hi)
                c = rng.choice(self.CAMERAS)
                if (f, c) not in used and all(
                    abs(f - s) >= self.tgt_min_sep for s in src_frames
                ):
                    used.add((f, c))
                    return (f, c)
            # Constraints too tight for this span (e.g. a degenerate one-sided
            # cluster of sources has no point >= tgt_min_sep from all of them):
            # fall back to the frame that maximises the distance to the nearest
            # source (the deepest point of the widest gap) instead of crashing.
            best_f, best_d = lo_f, -1
            for f in range(lo_f, hi_f + 1):
                d = min(abs(f - s) for s in src_frames)
                if d > best_d:
                    best_f, best_d = f, d
            cands = [c for c in self.CAMERAS if (best_f, c) not in used]
            c = rng.choice(cands or list(self.CAMERAS))
            used.add((best_f, c))
            return (best_f, c)

        anchor_frame = rng.choice(frames)
        anchor_cam = rng.choice(self.CAMERAS)
        used.add((anchor_frame, anchor_cam))
        source_views = [(anchor_frame, anchor_cam)]
        for _ in range(self.num_source_views - 1):
            source_views.append(draw_near(anchor_frame))
        # draw_far needs the anchor frame; draw_interpolated derives its range
        # from the sampled source views and takes no argument.
        extrap_sample = False
        if self.deterministic_view_seed is None:
            # Training-style draw: count it and maybe take the extrapolation
            # branch.  Seeded draws (frozen eval / reproducible inference) skip
            # both, so their view assignment is identical to pre-extrap code.
            with self._extrap_draws.get_lock():
                self._extrap_draws.value += 1
                draws = self._extrap_draws.value
            frac = self._effective_extrap_frac(draws)
            if frac > 0 and rng.random() < frac:
                target_views = [
                    draw_mild_extrapolated(anchor_frame)
                    for _ in range(self.num_target_views)
                ]
                extrap_sample = True
        if not extrap_sample:
            if self.tgt_interpolate:
                target_views = [
                    draw_interpolated() for _ in range(self.num_target_views)
                ]
            else:
                target_views = [
                    draw_far(anchor_frame) for _ in range(self.num_target_views)
                ]
        self._last_sample_extrapolated = extrap_sample

        if self.src_order == "pose_chain":
            # Camera-trajectory order: greedy nearest-neighbour chain over
            # camera centres, starting from the source closest to the (mean)
            # target camera.  Deterministic given the selected set, so frozen
            # eval tuples stay comparable across runs and arms.  Supersedes
            # the per-sample tail shuffle (same contract as the kubric4d
            # dataset).
            tgt_center = torch.stack(
                [record["c2w"][c][f][:3, 3] for f, c in target_views]
            ).mean(dim=0)
            centers = {
                (f, c): record["c2w"][c][f][:3, 3] for f, c in source_views
            }
            chain = [
                min(
                    source_views,
                    key=lambda v: float((centers[v] - tgt_center).norm()),
                )
            ]
            pool = [v for v in source_views if v != chain[0]]
            while pool:
                last = centers[chain[-1]]
                nxt = min(pool, key=lambda v: float((centers[v] - last).norm()))
                chain.append(nxt)
                pool.remove(nxt)
            source_views = chain
        elif self.random_permute_source_views:
            tail = source_views[1:]
            rng.shuffle(tail)
            source_views = [source_views[0]] + tail
        return source_views, target_views

    def _load_view(self, record, frame, camera):
        seq_dir = record["seq_dir"]
        rgb_path = os.path.join(
            seq_dir, f"image_{camera}", f"{frame:06d}_{camera}.png"
        )
        depth_path = os.path.join(
            seq_dir, f"depth_{camera}", f"{frame:06d}_{camera}_depth.npy"
        )
        image = _load_rgb(rgb_path)
        depth_metric, valid_mask = _load_tartanair_depth(
            depth_path, self.invalid_threshold
        )
        c2w = record["c2w"][camera][frame]

        image, depth_metric, intrins = _resize_crop_rgb_depth(
            image=image,
            depth_metric=depth_metric,
            intrins=self.K.clone(),
            target_size=self.resolution,
        )
        valid_mask = F.interpolate(
            valid_mask.float().unsqueeze(0).unsqueeze(0),
            size=self.resolution,
            mode="nearest",
        ).squeeze(0).squeeze(0).bool()
        if self.depth_clamp_max is not None:
            depth_metric = depth_metric.clamp(max=self.depth_clamp_max)
        inv_depth, positive = _inverse_depth(depth_metric, valid_mask)
        depth_metric_3 = depth_metric.unsqueeze(0).repeat(3, 1, 1)
        return {
            "image": image.float().clamp(0.0, 1.0),
            "inv_depth": inv_depth.float(),
            "positive": positive,
            "depth_metric": depth_metric_3.float(),
            "K": intrins.float(),
            "c2w": c2w.float(),
            "valid_mask": valid_mask.float().unsqueeze(0).unsqueeze(0),
        }

    def __getitem__(self, idx):
        raw_idx = int(idx)
        record_idx = (raw_idx // self.samples_per_scene) % len(self.records)
        record = self.records[record_idx]
        source_views, target_views = self._select_views(record, idx=raw_idx)

        source_loaded = [self._load_view(record, f, c) for f, c in source_views]
        target_loaded = [self._load_view(record, f, c) for f, c in target_views]
        all_loaded = source_loaded + target_loaded

        # Shared disparity scale from SOURCE views only (target depth is
        # unobservable at inference), same contract as the kubric4d dataset:
        # camera translations are multiplied by the same factor.
        if self.disparity_norm_mode == "shared_scale":
            norm_scale = _shared_disparity_scale(
                [item["inv_depth"] for item in source_loaded],
                [item["positive"] for item in source_loaded],
                quantile=self.disparity_norm_quantile,
                headroom=self.disparity_norm_headroom,
            )
            for item in all_loaded:
                item["disparity"] = _disparity_from_shared_scale(item["inv_depth"], norm_scale)
                c2w = item["c2w"].clone()
                c2w[:3, 3] = c2w[:3, 3] * norm_scale
                item["c2w"] = c2w
        else:
            norm_scale = torch.tensor(1.0)
            for item in all_loaded:
                item["disparity"] = _disparity_per_view(item["inv_depth"], item["positive"])

        def _stack(key):
            return torch.stack([item[key] for item in all_loaded], dim=0)

        images = _stack("image")
        disparity = _stack("disparity")
        depth_metric = _stack("depth_metric")
        K = _stack("K")
        c2w = _stack("c2w")

        num_src = self.num_source_views
        depth_masks = torch.cat([item["valid_mask"] for item in all_loaded], dim=0)
        rgb_masks = torch.cat(
            [torch.zeros_like(item["valid_mask"]) for item in source_loaded]
            + [torch.ones_like(item["valid_mask"]) for item in target_loaded],
            dim=0,
        )
        loss_masks = torch.stack([depth_masks, rgb_masks], dim=0)

        view_roles = torch.tensor(
            [0] * num_src + [1] * self.num_target_views, dtype=torch.long
        )

        sample_id = f"{record['scene_id']}/{record['time_id']}/s{idx}"

        return {
            "sample_id": sample_id,
            "scene_id": record["scene_id"],
            "time_id": record["time_id"],
            # 0-dim float tensor (0.0/1.0) so custom_collate_fn stacks it into [B]
            "tgt_extrapolated": torch.tensor(
                1.0 if self._last_sample_extrapolated else 0.0
            ),
            "source_images": images[:num_src],
            "source_disparity": disparity[:num_src],
            "source_depth_metric": depth_metric[:num_src],
            "source_K": K[:num_src],
            "source_c2w": c2w[:num_src],
            "target_images": images[num_src:],
            "target_disparity": disparity[num_src:],
            "target_depth_metric": depth_metric[num_src:],
            "target_K": K[num_src:],
            "target_c2w": c2w[num_src:],
            "images": images,
            "disparity": disparity,
            "depth_metric": depth_metric,
            "K": K,
            "c2w": c2w,
            "view_roles": view_roles,
            "loss_masks": loss_masks,
            "depth_norm_scale": norm_scale.reshape(()).float(),
        }


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    data_root = "/mi/data2T/y00959808/data/sample_data/TartanAir"
    dataset = TartanAirPoseNVSDataset(
        tartanair_root=data_root,
        deterministic_view_seed=0,
    )
    print(f"records: {[(r['scene_id'], r['time_id'], len(r['frames'])) for r in dataset.records]}")
    print(f"Length of dataset: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for i, batch in enumerate(dataloader):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: {tuple(v.shape)}, range: {v.min():.4f} - {v.max():.4f}")
            else:
                print(f"{k}: {v[0] if isinstance(v, list) else v}")
        if i >= 1:
            break

    # visualize one sample
    import os

    os.makedirs("debug_vis_tartanair", exist_ok=True)
    batch = next(iter(dataloader))
    from torchvision.utils import save_image
    num_src = batch["source_images"].shape[1]
    for v in range(batch["images"].shape[1]):
        role = "src" if v < num_src else "tgt"
        save_image(batch["images"][0, v], f"debug_vis_tartanair/{role}_rgb_v{v}.png")
        save_image(batch["disparity"][0, v], f"debug_vis_tartanair/{role}_disp_v{v}.png")
        save_image(batch["loss_masks"][0, 0, v].unsqueeze(0), f"debug_vis_tartanair/{role}_valid_v{v}.png")
    print("saved visualization to debug_vis_tartanair/")
