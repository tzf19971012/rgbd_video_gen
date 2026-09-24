import json
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .tartanair_vid_dataset import TartanAirDepthTransform, read_numpy_file


def read_rgb_image(image_file):
    """
    return a uint8 RGB numpy array given the file path (PIL backend)
    """
    with Image.open(image_file) as img:
        return np.asarray(img.convert("RGB"))


class ManifestVIDDataset(Dataset):
    """Video depth dataset driven by a manifest.jsonl file.

    Each line of the manifest describes one scene:
        {"scene_id": ..., "time_id": ..., "views": [
            {"view_id": ..., "rgb_path": ..., "depth_path": ...,
             "K": ..., "c2w": ..., "depth_scale": ...}, ...]}

    `rgb_path` / `depth_path` are relative to `data_root`. Views are sorted
    by `view_id` and treated as an ordered frame sequence, so a contiguous
    clip of `num_frames` frames with a given stride can be sampled exactly
    like the old folder-walking datasets -- but initialization only parses
    one jsonl file instead of walking the whole dataset tree.
    """

    def __init__(
        self,
        data_root,
        manifest_path="manifest.jsonl",
        name=None,
        random_flip=True,
        norm_type="trunc_disparity",
        resolution=(480, 640),
        truncnorm_min=0.02,
        max_num_frame=45,
        min_num_frame=45,
        max_sample_stride=1,
        min_sample_stride=1,
        scene_filter=None,
        train_ratio=1.0,
        enlarge_factor=1,
        d_max=50,
    ):
        self.data_root = data_root
        self.name = name or os.path.basename(os.path.normpath(data_root))

        if not os.path.isfile(manifest_path):
            manifest_path = os.path.join(data_root, manifest_path)
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(
                f"[ManifestVIDDataset:{self.name}] manifest not found: {manifest_path}"
            )

        self.scene_list = self._load_manifest(
            manifest_path, scene_filter, min_num_frame
        )
        if train_ratio < 1.0:
            origin_len = len(self.scene_list)
            self.scene_list = self.scene_list[: int(origin_len * train_ratio)]
            print(
                f"ManifestVIDDataset({self.name}) use {len(self.scene_list)} scenes instead of {origin_len}..."
            )
        self.enlarge_factor = enlarge_factor
        self.scene_list = self.scene_list * enlarge_factor

        new_h, new_w = resolution
        self.new_h = new_h
        self.new_w = new_w
        self.transform = TartanAirDepthTransform(
            (new_h, new_w), random_flip, norm_type, truncnorm_min, d_max=d_max
        )
        self.max_num_frame = max_num_frame
        self.min_num_frame = min_num_frame
        self.max_sample_stride = max_sample_stride
        self.min_sample_stride = min_sample_stride
        self.num_frames = list(range(min_num_frame, max_num_frame + 1))
        self.strides = list(range(min_sample_stride, max_sample_stride + 1))

        print(
            f"ManifestVIDDataset({self.name}): {len(self.scene_list)} scenes "
            f"({len(self.scene_list) // max(enlarge_factor, 1)} unique x{enlarge_factor}), "
            f"resolution=({new_h}, {new_w}), frames=[{min_num_frame}, {max_num_frame}], "
            f"stride=[{min_sample_stride}, {max_sample_stride}]"
        )

    def _load_manifest(self, manifest_path, scene_filter, min_num_frame):
        scene_list = []
        with open(manifest_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                scene_id = record["scene_id"]
                if scene_filter is not None and scene_filter not in scene_id:
                    continue
                views = sorted(record["views"], key=lambda v: v["view_id"])
                if len(views) < min_num_frame:
                    print(
                        f"[ManifestVIDDataset:{self.name}] skip scene {scene_id}: "
                        f"{len(views)} views < min_num_frame={min_num_frame}"
                    )
                    continue
                for v in views:
                    v["depth_scale"] = float(v.get("depth_scale", 1.0))
                scene_list.append({"scene_id": scene_id, "views": views})

        scene_list.sort(key=lambda x: x["scene_id"])
        if len(scene_list) == 0:
            raise ValueError(
                f"[ManifestVIDDataset:{self.name}] no valid scene in {manifest_path} "
                f"(scene_filter={scene_filter}, min_num_frame={min_num_frame}). "
                f"Lower min_num_frame or fix scene_filter."
            )
        return scene_list

    def __len__(self):
        return len(self.scene_list)

    def __getitem__(self, idx):
        idx = idx % len(self.scene_list)
        try:
            scene = self.scene_list[idx]
            scene_name = scene["scene_id"]
            views = scene["views"]

            total_frames = len(views)
            if total_frames < self.min_num_frame:
                raise ValueError(
                    f"Not enough frames for scene {scene_name}. Need at least "
                    f"{self.min_num_frame} frames, but got {total_frames}."
                )

            _sample_idx = None
            for _ in range(1000):
                _stride = random.choice(self.strides)
                _num_frames = random.choice(self.num_frames)
                if _num_frames % 4 != 1:
                    continue
                _total_frames_req = _stride * (_num_frames - 1) + 1
                if _total_frames_req > total_frames:
                    continue
                start_idx = random.randint(0, total_frames - _total_frames_req)
                end_idx = start_idx + _total_frames_req
                _sample_idx = list(range(start_idx, end_idx, _stride))
                break
            if _sample_idx is None:
                raise ValueError(f"Fail to sample frames for scene {scene_name}.")

            img_list = []
            depth_list = []
            for vi in _sample_idx:
                view = views[vi]
                _rgb_path = os.path.join(self.data_root, view["rgb_path"])
                _depth_path = os.path.join(self.data_root, view["depth_path"])
                img_list.append(read_rgb_image(_rgb_path))
                dist = read_numpy_file(_depth_path).astype(np.float32)
                depth_list.append(dist * view["depth_scale"])

            img_np = np.array(img_list).astype(np.float32) / 255.0
            depth_np = np.array(depth_list)

            # Replace invalid depth (<=0 or non-finite) with the clip's far
            # plane, otherwise 1/depth in the transform produces inf/nan.
            _valid = np.isfinite(depth_np) & (depth_np > 0)
            if not _valid.all():
                _fill = float(depth_np[_valid].max()) if _valid.any() else 50.0
                depth_np = np.where(_valid, depth_np, _fill)

            image = torch.from_numpy(img_np).permute(0, 3, 1, 2)
            if torch.isnan(image).any() or torch.isinf(image).any():
                print(f"Error loading data at index {idx}: image is nan or inf")
                return self.__getitem__(idx + 1)

            image, depth, depth_raw = self.transform(image, depth_np)
            if torch.isnan(depth).any() or torch.isinf(depth).any():
                print(f"Error loading data at index {idx}: depth is nan or inf")
                return self.__getitem__(idx + 1)

            return {
                "sample_idx": torch.tensor(idx),
                "images": image,
                "disparity": depth,
                "depth_raw": depth_raw,
                "image_path": os.path.join(self.data_root, views[0]["rgb_path"]),
                "depth_path": os.path.join(self.data_root, views[0]["depth_path"]),
                "scene_name": scene_name,
                "cam": self.name,
            }
        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            # In case of error, return the next sample
            return self.__getitem__(idx + 1)
