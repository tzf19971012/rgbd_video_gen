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


def _parse_intrinsic(path):
    """intrinsic.txt -> {(frame, camera): 3x3 K}. Columns: frame cam fx fy cx cy."""
    K = {}
    with open(path, "r", encoding="utf-8") as handle:
        next(handle)  # header
        for line in handle:
            parts = line.split()
            if len(parts) < 6:
                continue
            frame, cam = int(parts[0]), int(parts[1])
            fx, fy, cx, cy = map(float, parts[2:6])
            K[(frame, cam)] = torch.tensor(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
            )
    return K


def _parse_extrinsic(path):
    """extrinsic.txt -> {frame: {camera: 4x4 world-to-camera}}.

    Verified against pose.txt: extrinsic is w2c (R @ world + t == camera_space).
    """
    w2c = {}
    with open(path, "r", encoding="utf-8") as handle:
        next(handle)  # header
        for line in handle:
            parts = line.split()
            if len(parts) < 18:
                continue
            frame, cam = int(parts[0]), int(parts[1])
            w2c.setdefault(frame, {})[cam] = torch.tensor(
                list(map(float, parts[2:18])), dtype=torch.float32
            ).reshape(4, 4)
    return w2c


def _load_vkitti2_depth(path, invalid_threshold=655.0):
    """vKitti2 depth PNG: uint16, depth_meters = value / 100 (centimetres).

    65535 is the invalid/sky sentinel -> valid_mask False there.
    Returns (depth_meter [H, W] float32, valid_mask [H, W] bool).
    """
    from PIL import Image

    raw = np.asarray(Image.open(path), dtype=np.float32) / 100.0
    depth = torch.from_numpy(raw)
    valid = torch.isfinite(depth) & (depth > 0) & (depth < invalid_threshold)
    return depth, valid


class VKitti2PoseNVSDataset(Dataset):
    """vKitti2 dataloader for pose-NVS training.

    Directory layout (vKitti2 sample data):
        <root>/<Scene>/<clone>/frames/rgb/Camera_{0,1}/rgb_XXXXX.jpg
        <root>/<Scene>/<clone>/frames/depth/Camera_{0,1}/depth_XXXXX.png
        <root>/<Scene>/<clone>/{intrinsic,extrinsic}.txt

    Every (frame, camera) pair is one view.  A sample draws one anchor view
    (source slot 0, world origin), num_source_views-1 additional nearby source
    views and num_target_views distant views, all inside one (scene, clone) so
    appearance stays consistent.  The output dict mirrors
    Kubric4DPoseNVSDataset exactly, so WanPoseNVSTrainingModule consumes it
    unchanged.
    """

    CAMERAS = (0, 1)
    DEPTH_SCALE = 100.0  # uint16 -> meters

    def __init__(
        self,
        vkitti2_root,
        resolution=(384, 672),
        num_source_views=9,
        num_target_views=3,
        # sources sampled within +/- this many frames of the anchor; keep
        # this small so the source set is densely spaced — large baselines
        # between source views are hard to learn early in training.
        src_frame_range=4,
        tgt_frame_min=40,         # targets at least this many frames away
        tgt_frame_max=150,        # ... and at most this many
        random_permute_source_views=True,
        disparity_norm_mode="shared_scale",
        disparity_norm_quantile=0.98,
        disparity_norm_headroom=1.5,
        deterministic_view_seed=None,
        samples_per_scene=1,
        max_samples=None,
        **kwargs,
    ):
        self.vkitti2_root = vkitti2_root
        self.resolution = tuple(resolution)
        self.num_source_views = int(num_source_views)
        self.num_target_views = int(num_target_views)
        self.num_views = self.num_source_views + self.num_target_views
        self.src_frame_range = int(src_frame_range)
        self.tgt_frame_min = int(tgt_frame_min)
        self.tgt_frame_max = int(tgt_frame_max)
        self.random_permute_source_views = bool(random_permute_source_views)
        if disparity_norm_mode not in ("shared_scale", "per_view"):
            raise ValueError(
                f"disparity_norm_mode must be 'shared_scale' or 'per_view', got {disparity_norm_mode!r}"
            )
        self.disparity_norm_mode = disparity_norm_mode
        self.disparity_norm_quantile = float(disparity_norm_quantile)
        self.disparity_norm_headroom = float(disparity_norm_headroom)
        self.deterministic_view_seed = (
            None if deterministic_view_seed is None else int(deterministic_view_seed)
        )
        self.samples_per_scene = int(samples_per_scene)

        # records: one entry per (scene, clone) with geometry tables
        self.records = []
        for scene in sorted(os.listdir(self.vkitti2_root)):
            scene_dir = os.path.join(self.vkitti2_root, scene)
            if not os.path.isdir(scene_dir):
                continue
            for clone in sorted(os.listdir(scene_dir)):
                clone_dir = os.path.join(scene_dir, clone)
                frames_dir = os.path.join(clone_dir, "frames")
                intrinsic_path = os.path.join(clone_dir, "intrinsic.txt")
                extrinsic_path = os.path.join(clone_dir, "extrinsic.txt")
                if not (os.path.isdir(frames_dir) and os.path.isfile(intrinsic_path)
                        and os.path.isfile(extrinsic_path)):
                    continue
                w2c = _parse_extrinsic(extrinsic_path)
                K = _parse_intrinsic(intrinsic_path)
                # frame ids that have rgb+depth+pose for every camera
                frames = sorted(
                    f for f in w2c
                    if all(
                        os.path.isfile(os.path.join(frames_dir, "rgb", f"Camera_{c}", f"rgb_{f:05d}.jpg"))
                        and os.path.isfile(os.path.join(frames_dir, "depth", f"Camera_{c}", f"depth_{f:05d}.png"))
                        and (f, c) in K
                        for c in self.CAMERAS
                    )
                )
                if not frames:
                    continue
                self.records.append(
                    {
                        "scene_id": scene,
                        "time_id": clone,
                        "clone_dir": clone_dir,
                        "frames": frames,
                        "K": K,
                        "w2c": w2c,
                    }
                )
        if not self.records:
            raise ValueError(f"No vKitti2 (scene, clone) records found under {vkitti2_root}")
        if max_samples is not None:
            self.records = self.records[: int(max_samples)]

        need = self.tgt_frame_max * 2 + 2
        for record in self.records:
            if len(record["frames"]) < need:
                print(
                    f"[VKitti2PoseNVSDataset] WARNING: {record['scene_id']}/{record['time_id']} "
                    f"has only {len(record['frames'])} complete frames (< {need}); target sampling "
                    f"will clamp to the available range."
                )

    def __len__(self):
        return len(self.records) * self.samples_per_scene

    def _rng(self, idx):
        if self.deterministic_view_seed is not None:
            return random.Random(self.deterministic_view_seed + int(idx))
        return random

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

        def draw_near(anchor_frame):
            for _ in range(50):
                offset = rng.randint(1, self.src_frame_range)
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

        anchor_frame = rng.choice(frames)
        anchor_cam = rng.choice(self.CAMERAS)
        used.add((anchor_frame, anchor_cam))
        source_views = [(anchor_frame, anchor_cam)]
        for _ in range(self.num_source_views - 1):
            source_views.append(draw_near(anchor_frame))
        target_views = [draw_far(anchor_frame) for _ in range(self.num_target_views)]

        if self.random_permute_source_views:
            tail = source_views[1:]
            rng.shuffle(tail)
            source_views = [source_views[0]] + tail
        return source_views, target_views

    def _load_view(self, record, frame, camera):
        clone_dir = record["clone_dir"]
        rgb_path = os.path.join(
            clone_dir, "frames", "rgb", f"Camera_{camera}", f"rgb_{frame:05d}.jpg"
        )
        depth_path = os.path.join(
            clone_dir, "frames", "depth", f"Camera_{camera}", f"depth_{frame:05d}.png"
        )
        image = _load_rgb(rgb_path)
        depth_metric, valid_mask = _load_vkitti2_depth(depth_path)
        K = record["K"][(frame, camera)]
        c2w = torch.linalg.inv(record["w2c"][frame][camera])

        image, depth_metric, intrins = _resize_crop_rgb_depth(
            image=image,
            depth_metric=depth_metric,
            intrins=K,
            target_size=self.resolution,
        )
        valid_mask = F.interpolate(
            valid_mask.float().unsqueeze(0).unsqueeze(0),
            size=self.resolution,
            mode="nearest",
        ).squeeze(0).squeeze(0).bool()
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
