import json
import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from utils.projection import intrins_crop, intrins_resize


def _read_json_or_jsonl(path):
    if os.path.isdir(path):
        for candidate in ("manifest.jsonl", "manifest.json"):
            candidate_path = os.path.join(path, candidate)
            if os.path.exists(candidate_path):
                path = candidate_path
                break

    if not os.path.exists(path):
        raise FileNotFoundError(f"Manifest not found: {path}")

    def _read_lines():
        records = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    # Dispatch on ".jsonl" anywhere in the name, not just as the final suffix:
    # dated backups are spelled manifest.jsonl.bak_<n>scenes_<stamp>, and an
    # extension-only test sends them down the whole-file json.load path where they
    # die on "Extra data: line 2".  Pinning an old manifest is exactly how an
    # earlier checkpoint's frozen eval set gets reproduced, so this path has to
    # work.  Fall back to line-by-line for any other spelling too.
    if ".jsonl" in os.path.basename(path):
        return _read_lines()

    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError:
        return _read_lines()

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if "records" in payload and isinstance(payload["records"], list):
            return payload["records"]
        return [payload]
    raise TypeError(f"Unsupported manifest payload type: {type(payload)}")


def _resolve_path(root, path):
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(root, path)


def _to_tensor(matrix, shape, dtype=torch.float32):
    tensor = torch.as_tensor(matrix, dtype=dtype)
    return tensor.reshape(shape).clone()


def _load_rgb(path):
    image = Image.open(path).convert("RGB")
    return torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255.0


def _load_pfm(path):
    """Read a PFM depth map and return a float32 H×W numpy array."""
    import re as _re
    with open(path, "rb") as f:
        header = f.readline().decode("utf-8").rstrip()
        color = header == "PF"
        if header not in ("PF", "Pf"):
            raise ValueError(f"Not a PFM file: {path!r}")
        dim_match = _re.match(r"^(\d+)\s(\d+)\s$", f.readline().decode("utf-8"))
        if not dim_match:
            raise ValueError(f"Malformed PFM header in {path!r}")
        width, height = map(int, dim_match.groups())
        scale = float(f.readline().decode("utf-8").rstrip())
        endian = "<" if scale < 0 else ">"
        dtype = np.dtype(endian + "f4")
        channels = 3 if color else 1
        data = np.frombuffer(f.read(), dtype=dtype)
        data = data.reshape((height, width, channels) if color else (height, width))
        data = np.flipud(data)
    return data.astype(np.float32)


def _load_depth(path, depth_scale=1.0):
    if path.endswith(".npy"):
        depth = np.load(path)
    elif path.endswith(".exr"):
        # Single-channel scanline EXR (e.g. MoGe-2 metric depth from DL3DV_WAI).
        try:
            import OpenEXR
            import Imath
        except ImportError as exc:
            raise ImportError("OpenEXR is required to read EXR depth maps.") from exc
        handle = OpenEXR.InputFile(path)
        header = handle.header()
        window = header["dataWindow"]
        width = window.max.x - window.min.x + 1
        height = window.max.y - window.min.y + 1
        channels = [c for c in header["channels"] if c in ("Y", "R", "depth", "Depth")]
        name = channels[0] if channels else sorted(header["channels"])[0]
        ptype = header["channels"][name].type.v
        dtype = {Imath.PixelType.HALF: np.float16, Imath.PixelType.FLOAT: np.float32}[ptype]
        depth = np.frombuffer(handle.channel(name, Imath.PixelType(ptype)), dtype=dtype)
        depth = depth.reshape(height, width)
    elif path.endswith(".npz"):
        data = np.load(path)
        if "depth" in data:
            depth = data["depth"]
        else:
            depth = data[data.files[0]]
    elif path.endswith((".h5", ".hdf5")):
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("h5py is required to read HDF5 depth maps.") from exc
        with h5py.File(path, "r") as handle:
            key = "depth" if "depth" in handle else list(handle.keys())[0]
            depth = np.array(handle[key])
    elif path.endswith(".pfm"):
        depth = _load_pfm(path)
    else:
        depth = np.asarray(Image.open(path), dtype=np.float32)

    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    if depth_scale != 1.0:
        depth = depth * float(depth_scale)
    return torch.from_numpy(depth)


def _inverse_depth(depth_metric, valid_mask, eps=1e-6):
    """Inverse depth (1/d) with invalid pixels set to 0, plus the positivity mask."""
    positive = valid_mask & torch.isfinite(depth_metric) & (depth_metric > 0)
    inv_depth = torch.zeros_like(depth_metric)
    if positive.any():
        inv_depth[positive] = 1.0 / depth_metric[positive].clamp_min(eps)
    return inv_depth, positive


def _disparity_per_view(inv_depth, positive, eps=1e-6):
    """LEGACY per-view affine normalisation of disparity.

    Each view gets its own (low, high) from its own 2%/98% inverse-depth
    quantiles.  This breaks cross-view geometric consistency and must NOT be
    used for multi-view / NVS training -- kept only to reproduce old runs.
    """
    disparity = torch.zeros_like(inv_depth)
    if positive.any():
        valid_values = inv_depth[positive]
        low = torch.quantile(valid_values, 0.02)
        high = torch.quantile(valid_values, 0.98)
        disparity = ((inv_depth - low) / (high - low + eps)).clamp(0.0, 1.0)
    return disparity.unsqueeze(0).repeat(3, 1, 1)


def _shared_disparity_scale(inv_depths, positives, quantile=0.98, headroom=1.0, eps=1e-8):
    """One scale factor `s` (units 1/length) shared by every view of a sample.

    `s` is `headroom` times the `quantile`-th percentile of the pooled
    *source-view* inverse depths, so that `disp = (1/d) / s` lands in [0, 1]
    for ~`quantile` of the observed pixels while remaining strictly
    proportional to 1/d.  `headroom > 1` leaves room for target views that are
    closer to the scene than any source view, at the cost of contrast.

    Scale-only (no shift) normalisation is what makes disparity comparable
    across views: `disp_i = 1 / (d_i * s)` for every view i, so the same 3D
    point yields consistent values everywhere.  The world is thereby rescaled
    by `s`, which is why camera translations must be multiplied by `s` too.
    """
    pooled = [inv[pos] for inv, pos in zip(inv_depths, positives) if pos.any()]
    if not pooled:
        return torch.tensor(1.0)
    pooled = torch.cat(pooled)
    scale = torch.quantile(pooled, quantile)
    if not torch.isfinite(scale) or scale <= eps:
        median = torch.median(pooled)
        scale = median if torch.isfinite(median) and median > eps else torch.tensor(1.0)
    return scale * headroom


def _disparity_from_shared_scale(inv_depth, scale, eps=1e-8):
    disparity = (inv_depth / scale.clamp_min(eps)).clamp(0.0, 1.0)
    return disparity.unsqueeze(0).repeat(3, 1, 1)


def _resize_crop_rgb_depth(image, depth_metric, intrins, target_size):
    target_height, target_width = target_size
    height, width = image.shape[-2:]
    scale = max(target_height / height, target_width / width)
    resized_height = max(1, int(round(height * scale)))
    resized_width = max(1, int(round(width * scale)))

    image = F.interpolate(
        image.unsqueeze(0), size=(resized_height, resized_width), mode="bilinear", align_corners=False
    ).squeeze(0)
    depth_metric = F.interpolate(
        depth_metric.unsqueeze(0).unsqueeze(0),
        size=(resized_height, resized_width),
        mode="nearest",
    ).squeeze(0).squeeze(0)

    intrins = intrins_resize(intrins.clone(), ratio_H=scale, ratio_W=scale)

    top = max(0, (resized_height - target_height) // 2)
    left = max(0, (resized_width - target_width) // 2)
    image = image[:, top:top + target_height, left:left + target_width]
    depth_metric = depth_metric[top:top + target_height, left:left + target_width]
    intrins = intrins_crop(intrins, crop_top=top, crop_left=left)
    return image, depth_metric, intrins


@dataclass
class _ViewRecord:
    scene_id: str
    time_id: Optional[str]
    view_id: str
    rgb_path: str
    depth_path: str
    K: torch.Tensor
    c2w: torch.Tensor
    depth_scale: float = 1.0


VIEW_ORDERS = ("random", "pose", "video")


def _linspace_span(items, count, rng):
    """Draw `count` items as an ordered sub-span of the list, evenly spaced.

    Port of VGGT-Diff vggtdiff/data.py:72-84 (the ordered branch of
    `_sample_nvs_targets`): span length uniform in [count, len], start uniform,
    then `linspace` picks `count` frames.  With `span_size >= count` the
    rounded positions are distinct, so no dedup is needed.  Returned in
    temporal order -- this is what makes the RoPE frame axis a TIME axis.
    """
    total = len(items)
    if total < count:
        return None
    if count == 1:
        return [items[rng.randrange(total)]]
    span_size = rng.randint(count, total)
    start = rng.randint(0, total - span_size)
    span = items[start:start + span_size]
    return [span[round(p * (span_size - 1) / (count - 1))] for p in range(count)]


def _video_sample(views, num_source, num_target, rng):
    """Video-temporal view sampling for ordered video clips (DL3DV).

    Port of VGGT-Diff vggtdiff/data.py `SceneDataset.__getitem__` NVS protocol
    ("expects six source views").  Requires view_id sort == temporal order,
    which holds for the DL3DV manifest (zero-padded frame_XXXXX names).
    Unlike `pose`, the SOURCE block is chosen here, not reordered afterwards;
    both blocks keep temporal order except the unordered-target branch.

    1.  SOURCE: ordered span sampling (`_linspace_span`) -- consecutive frame
        indices are consecutive timestamps, the video analogue of the
        view_order="pose" rationale.
    2.  TARGET candidates: everything else.  With probability 20% they shrink
        to the source time interval (probabilistic version of data.py:181-188
        where 20% of bundles are interpolation-heavy), falling back to the
        global pool when the interval is too thin.
    3.  TARGET: `_sample_nvs_targets` 50/50 -- ordered sub-span (temporal
        order preserved) vs unordered scatter (`rng.sample`).

    Returns (source_views, target_views) or None when the record cannot
    supply enough views; `_select_views` surfaces that to the caller.
    """
    ordered = sorted(views, key=lambda v: v.view_id)
    source_views = _linspace_span(ordered, num_source, rng)
    if source_views is None:
        return None

    # _ViewRecord's dataclass __eq__ compares tensors, so membership is keyed
    # on view_id (unique within a scene) instead of `in` / list.index.
    src_ids = {v.view_id for v in source_views}
    candidates = [v for v in ordered if v.view_id not in src_ids]
    if len(candidates) < num_target:
        return None

    if rng.random() < 0.2:
        indices = [i for i, v in enumerate(ordered) if v.view_id in src_ids]
        lo, hi = min(indices), max(indices)
        local = [v for v in ordered[lo:hi + 1] if v.view_id not in src_ids]
        if len(local) >= num_target:
            candidates = local

    if rng.random() < 0.5:
        target_views = _linspace_span(candidates, num_target, rng)
        if target_views is None:
            return None
    else:
        target_views = rng.sample(candidates, num_target)
    return source_views, target_views


def _order_source_views_by_pose(source_views, target_views):
    """Order the SOURCE block as a greedy nearest-neighbour path through camera
    space, ending at the source view closest to the target block.

    D2 of reports/plan_20260906_rope_geometry.md.  The RoPE frame axis spends 44
    of every head's 128 dims on "which frame is this", and with
    random_permute_source_views that index is a random label, so the pretrained
    "adjacent frames are more correlated" prior is systematically wrong.  This
    makes the label mean something: consecutive frame indices are consecutive
    camera positions.

    Three constraints shape the exact recipe:

    1.  ONLY THE SOURCE BLOCK IS REORDERED.  The layout is
        [src RGB xS, tgt RGB xT, src depth xS, tgt depth xT] and both the
        per-frame timestep and the clean-depth mask are sliced BY POSITION, so
        mixing the blocks would silently feed a target view through the source
        path.  Reordering the target block would be legal, but `target_index` is
        positional and is the key every paired per-target analysis uses
        (per_view_metrics.jsonl carries no view id), so it would break pairing
        against the c-sweep history this arm is read against for a second-order
        gain.  Targets stay in their rng.sample order.
    2.  THE PATH ENDS NEXT TO THE TARGETS.  Frame distance from a target at
        position S+j to a source at position i is S+j-i, so which source sits
        LAST is what decides whether the target's nearest neighbours are also its
        nearest frames.  The walk is therefore seeded at the source closest to
        the target centroid and reversed.
    3.  IT IS A PURE FUNCTION OF THE SET, not of the order it arrives in.  The
        incoming list comes out of rng.sample in a random order, so ties -- and
        BlendedMVS does contain near-coincident cameras -- would otherwise
        resolve differently on every draw, i.e. put a random label straight back
        on the frame axis.  Hence the view_id sort before the walk.

    Greedy rather than optimal on purpose: with 6 views a held-out TSP solution
    would differ from the greedy path rarely and is not worth a dependency.
    """
    def center(view):
        return view.c2w[:3, 3]

    pool = sorted(source_views, key=lambda v: v.view_id)
    anchor = torch.stack([center(v) for v in target_views]).mean(dim=0)

    chain = []
    while pool:
        nearest = min(range(len(pool)), key=lambda i: float((center(pool[i]) - anchor).norm()))
        chain.append(pool.pop(nearest))
        anchor = center(chain[-1])
    return chain[::-1]


class Kubric4DRGBDNVSDataset(Dataset):
    """Manifest-driven RGB-D NVS dataset.

    `kubric4d_root` and `manifest_path` each accept either a single string or a
    parallel list of strings, so multiple datasets (e.g. BlendedMVS + GTA-SfM +
    MatrixCity + TartanAir) can be merged without walking any directory --
    every record comes straight from the manifest(s). `scene_filter` likewise
    accepts a single substring or a per-dataset list (None keeps everything).
    Records are tagged with a `dataset` name (basename of their root).
    """

    def __init__(
        self,
        kubric4d_root,
        manifest_path,
        resolution=(384, 672),
        num_source_views=2,
        num_target_views=1,
        random_permute_source_views=True,
        view_order="random",
        return_metadata=True,
        disparity_norm_mode="shared_scale",
        disparity_norm_quantile=0.98,
        disparity_norm_headroom=1.5,
        deterministic_view_seed=None,
        min_view_valid_fraction=0.0,
        view_validity_path=None,
        samples_per_scene=1,
        split_path=None,
        split=None,
        scene_filter=None,
        **kwargs,
    ):
        roots = ([kubric4d_root] if isinstance(kubric4d_root, (str, os.PathLike))
                 else list(kubric4d_root))
        manifests = ([manifest_path] if isinstance(manifest_path, (str, os.PathLike))
                     else list(manifest_path))
        if len(roots) != len(manifests):
            if len(roots) == 1:
                roots = roots * len(manifests)
            else:
                raise ValueError(
                    f"kubric4d_root ({len(roots)} entries) and manifest_path "
                    f"({len(manifests)} entries) must be parallel lists"
                )
        if scene_filter is None or isinstance(scene_filter, str):
            scene_filters = [scene_filter] * len(manifests)
        else:
            scene_filters = list(scene_filter)
            if len(scene_filters) != len(manifests):
                raise ValueError(
                    f"scene_filter ({len(scene_filters)} entries) must match "
                    f"manifest_path ({len(manifests)} entries)"
                )

        self.kubric4d_root = kubric4d_root
        self.manifest_path = (manifest_path if isinstance(manifest_path, (str, os.PathLike))
                              else ", ".join(str(m) for m in manifest_path))
        self.resolution = tuple(resolution)
        self.num_source_views = int(num_source_views)
        self.num_target_views = int(num_target_views)
        self.num_views = self.num_source_views + self.num_target_views
        self.random_permute_source_views = random_permute_source_views
        if view_order not in VIEW_ORDERS:
            raise ValueError(
                f"view_order must be one of {VIEW_ORDERS}, got {view_order!r}"
            )
        # "pose" supersedes random_permute_source_views: both decide the order of
        # the source block, and this one is not random.  Leaving
        # random_permute_source_views=true in the config of a pose-ordered arm is
        # therefore expected, not a contradiction -- what it still controls is
        # nothing.
        self.view_order = view_order
        self.return_metadata = return_metadata
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
        self.min_view_valid_fraction = float(min_view_valid_fraction)
        self.view_validity_path = view_validity_path
        self.split_path = split_path
        self.split = split
        self.samples_per_scene = int(samples_per_scene)
        if self.samples_per_scene < 1:
            raise ValueError(
                f"samples_per_scene must be >= 1, got {samples_per_scene!r}"
            )
        view_validity = self._load_view_validity()

        raw_entries = []
        for root, manifest, filter_i in zip(roots, manifests, scene_filters):
            dataset_name = os.path.basename(os.path.normpath(str(root)))
            for record in _read_json_or_jsonl(_resolve_path(root, manifest)):
                raw_entries.append((root, dataset_name, filter_i, record))

        self.records = []
        dropped_views = 0
        dropped_scenes = 0
        for record_idx, (root, dataset_name, filter_i, record) in enumerate(raw_entries):
            scene_id = str(record.get("scene_id", record.get("scene", record_idx)))
            if filter_i is not None and filter_i not in scene_id:
                continue
            time_id = record.get("time_id", record.get("time", None))
            views = record.get("views")
            if views is None:
                views = [record]

            normalized_views = []
            for view_idx, view in enumerate(views):
                rgb_path = _resolve_path(root, view.get("rgb_path") or view.get("image_path"))
                depth_path = _resolve_path(root, view.get("depth_path") or view.get("metric_depth_path") or view.get("depth_metric_path"))
                if rgb_path is None or depth_path is None:
                    continue

                K_value = view.get("K", view.get("intrinsics", view.get("intrinsic")))
                c2w_value = view.get("c2w", view.get("camera_to_world", view.get("pose")))
                if K_value is None or c2w_value is None:
                    continue

                normalized_views.append(
                    _ViewRecord(
                        scene_id=scene_id,
                        time_id=str(time_id) if time_id is not None else None,
                        view_id=str(view.get("view_id", view.get("camera_id", view.get("cam_id", view_idx)))),
                        rgb_path=rgb_path,
                        depth_path=depth_path,
                        K=_to_tensor(K_value, (3, 3)),
                        c2w=_to_tensor(c2w_value, (4, 4)),
                        depth_scale=float(view.get("depth_scale", 1.0)),
                    )
                )

            if len(normalized_views) >= self.num_views:
                self.records.append(
                    {
                        "scene_id": scene_id,
                        "time_id": time_id,
                        "views": normalized_views,
                        "dataset": dataset_name,
                    }
                )

        if not self.records:
            raise ValueError(
                f"No Kubric-4D records with at least {self.num_views} views were found in {manifest_path}."
            )

        if view_validity is not None:
            # Keep only views whose depth covers at least min_view_valid_fraction
            # of the frame, then keep only scenes that still have enough of them
            # to form a full (source + target) tuple.  Filtering at VIEW level
            # rather than scene level matters: the median view is 97% valid, so a
            # scene-level average is dominated by its few bad views and discards
            # scenes that still have dozens of perfectly good ones.
            kept_records = []
            for record in self.records:
                scene_fracs = view_validity.get(record["scene_id"], {})
                good = [
                    view
                    for view in record["views"]
                    if scene_fracs.get(view.view_id, 0.0) >= self.min_view_valid_fraction
                ]
                dropped_views += len(record["views"]) - len(good)
                if len(good) >= self.num_views:
                    record["views"] = good
                    kept_records.append(record)
                else:
                    dropped_scenes += 1
            if not kept_records:
                raise ValueError(
                    f"min_view_valid_fraction={self.min_view_valid_fraction} left no scene with "
                    f"at least {self.num_views} views. Lower the threshold."
                )
            print(
                f"[Kubric4DRGBDNVSDataset] min_view_valid_fraction="
                f"{self.min_view_valid_fraction}: dropped {dropped_views} views and "
                f"{dropped_scenes} scenes; {len(kept_records)} scenes / "
                f"{sum(len(r['views']) for r in kept_records)} views remain."
            )
            self.records = kept_records

        self._apply_split()

    def _apply_split(self):
        """Restrict self.records to one named subset of a frozen scene-id split.

        Filtering by an EXPLICIT scene_id list, not by a stride or a slice, is the
        whole point.  The previous eval set was `range(0, num_scenes, num_scenes //
        eval_max_samples)`, which silently selected a disjoint set of scenes the
        moment the manifest grew 113 -> 502 (report_20260905_t0 §2).  A list
        cannot do that: scenes added later are simply absent from it, and a scene
        that disappears from the manifest raises below instead of shifting
        everything after it.

        Ordering is taken from the split file, so the k-th eval sample is the k-th
        listed scene regardless of manifest order.
        """
        if self.split is None:
            return
        if self.split_path is None:
            raise ValueError(f"split={self.split!r} requires split_path")
        with open(self.split_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if self.split not in payload:
            raise KeyError(
                f"split={self.split!r} not in {self.split_path}; "
                f"available: {sorted(k for k, v in payload.items() if isinstance(v, list))}"
            )
        wanted = [str(s) for s in payload[self.split]]
        by_id = {}
        for record in self.records:
            by_id.setdefault(record["scene_id"], []).append(record)
        missing = [s for s in wanted if s not in by_id]
        if missing:
            raise ValueError(
                f"split={self.split!r} names {len(missing)} scene(s) absent from "
                f"{os.path.basename(self.manifest_path)} after "
                f"min_view_valid_fraction={self.min_view_valid_fraction} filtering, e.g. "
                f"{missing[:3]}. The split was frozen against manifest="
                f"{payload.get('manifest')} with view_threshold="
                f"{payload.get('view_threshold')}; a min_view_valid_fraction above that "
                f"can starve a held-out scene of views -- and raising it is a camera-pose "
                f"filter, not a quality filter (see tools/make_bmvs_split.py)."
            )
        self.records = [r for s in wanted for r in by_id[s]]
        print(
            f"[Kubric4DRGBDNVSDataset] split={self.split!r} from "
            f"{os.path.basename(self.split_path)}: {len(self.records)} scenes / "
            f"{sum(len(r['views']) for r in self.records)} views."
        )

    def _load_view_validity(self):
        """scene_id -> {view_id: valid_fraction}, or None when filtering is off.

        The sidecar is precomputed by utils/compute_view_validity.py.  We refuse
        to compute it lazily here: it is a 17k-file scan that every DDP rank and
        every DataLoader worker would otherwise repeat.
        """
        if self.min_view_valid_fraction <= 0.0:
            return None
        path = self.view_validity_path
        if path is None:
            path = os.path.splitext(self.manifest_path)[0] + ".view_validity.json"
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"min_view_valid_fraction={self.min_view_valid_fraction} requires a view-validity "
                f"sidecar, but {path} does not exist. Generate it with:\n"
                f"  python utils/compute_view_validity.py "
                f"--manifest {self.manifest_path} --root {self.kubric4d_root}"
            )
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload.get("scenes", payload)

    def __len__(self):
        return len(self.records) * self.samples_per_scene

    def _select_views(self, record, idx=0):
        views = list(record["views"])
        if len(views) < self.num_views:
            raise ValueError("Not enough views in record.")

        # With deterministic_view_seed set, the source/target split is a pure
        # function of (seed, idx) instead of the global RNG.  This freezes the
        # evaluation set: the same records yield the same views on every eval,
        # every run and every A/B arm, independent of DataLoader worker seeding.
        rng = random if self.deterministic_view_seed is None else random.Random(
            self.deterministic_view_seed + int(idx)
        )

        # video mode selects WHICH views, not just their order: sources come
        # from a temporal span and targets from a 50/50 ordered/unordered draw
        # (see _video_sample).  It is its own early branch because the existing
        # pipeline below picks targets first and then orders the source block;
        # video mode must pick sources first (the target interval depends on
        # them).  It consumes only the `rng` above, so a frozen
        # deterministic_view_seed eval stays a pure function of (seed, idx).
        if self.view_order == "video":
            result = _video_sample(views, self.num_source_views, self.num_target_views, rng)
            if result is None:
                raise ValueError(
                    f"view_order='video' could not draw {self.num_source_views} source + "
                    f"{self.num_target_views} target views from {len(views)} views."
                )
            return result

        target_views = rng.sample(views, self.num_target_views)
        remaining = [view for view in views if view not in target_views]
        source_views = rng.sample(remaining, self.num_source_views)

        # Both branches consume no further randomness, and nothing below reads
        # `rng` -- so the SELECTION is bit-identical either way and the frozen
        # eval set (deterministic_view_seed) stays paired with every run that
        # predates view_order, down to which views are targets.  Only the order
        # inside the source block changes.
        if self.view_order == "pose":
            source_views = _order_source_views_by_pose(source_views, target_views)
        elif self.random_permute_source_views:
            rng.shuffle(source_views)

        return source_views, target_views

    def _load_view(self, view):
        image = _load_rgb(view.rgb_path)
        depth_metric = _load_depth(view.depth_path, depth_scale=view.depth_scale)
        valid_mask = torch.isfinite(depth_metric) & (depth_metric > 0)
        image, depth_metric, intrins = _resize_crop_rgb_depth(
            image=image,
            depth_metric=depth_metric,
            intrins=view.K,
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
            "c2w": view.c2w.float(),
            "valid_mask": valid_mask.float().unsqueeze(0).unsqueeze(0),
            "view_id": view.view_id,
        }

    def __getitem__(self, idx):
        # One record is a SCENE holding ~100+ views, so a single (source, target)
        # tuple per scene throws away almost all of the available combinations and
        # pins len(self) to the scene count -- which in turn caps the per-device
        # batch size once the dataset is sharded.  `samples_per_scene` draws
        # several independent view tuples per scene instead.  The seed uses the
        # UNwrapped index so the draws differ from each other.
        raw_idx = int(idx)
        record_idx = (raw_idx // self.samples_per_scene) % len(self.records)
        record = self.records[record_idx]
        source_views, target_views = self._select_views(record, idx=raw_idx)

        source_loaded = [self._load_view(view) for view in source_views]
        target_loaded = [self._load_view(view) for view in target_views]
        all_loaded = source_loaded + target_loaded

        # --- Sample-level depth/pose scale normalisation -------------------
        # A single scale `s` (derived from the SOURCE views only, since target
        # depth is unobservable at inference time) maps every view into a
        # common normalised space:  disp = (1/d) / s  ==  1 / (d * s).
        # The world is rescaled by `s`, so camera translations get multiplied
        # by `s` as well -- this is what ties disparity magnitude to camera
        # baseline and makes the NVS geometry well-posed.
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

        source_images = torch.stack([item["image"] for item in source_loaded], dim=0)
        source_disparity = torch.stack([item["disparity"] for item in source_loaded], dim=0)
        source_depth_metric = torch.stack([item["depth_metric"] for item in source_loaded], dim=0)
        source_K = torch.stack([item["K"] for item in source_loaded], dim=0)
        source_c2w = torch.stack([item["c2w"] for item in source_loaded], dim=0)

        target_images = torch.stack([item["image"] for item in target_loaded], dim=0)
        target_disparity = torch.stack([item["disparity"] for item in target_loaded], dim=0)
        target_depth_metric = torch.stack([item["depth_metric"] for item in target_loaded], dim=0)
        target_K = torch.stack([item["K"] for item in target_loaded], dim=0)
        target_c2w = torch.stack([item["c2w"] for item in target_loaded], dim=0)

        images = torch.cat([source_images, target_images], dim=0)
        disparity = torch.cat([source_disparity, target_disparity], dim=0)
        depth_metric = torch.cat([source_depth_metric, target_depth_metric], dim=0)
        K = torch.cat([source_K, target_K], dim=0)
        c2w = torch.cat([source_c2w, target_c2w], dim=0)

        depth_masks = torch.cat([item["valid_mask"] for item in all_loaded], dim=0)
        rgb_masks = torch.cat(
            [torch.zeros_like(item["valid_mask"]) for item in source_loaded]
            + [torch.ones_like(item["valid_mask"]) for item in target_loaded],
            dim=0,
        )
        loss_masks = torch.stack([depth_masks, rgb_masks], dim=0)

        view_roles = torch.tensor(
            [0] * self.num_source_views + [1] * self.num_target_views,
            dtype=torch.long,
        )

        sample_id = f"{record['scene_id']}"
        if record.get("time_id") is not None:
            sample_id = f"{sample_id}/t{record['time_id']}"
        if self.return_metadata:
            sample_id = f"{sample_id}/s{idx}"

        return {
            "sample_id": sample_id,
            "scene_id": record["scene_id"],
            "time_id": record.get("time_id"),
            "source_images": source_images,
            "source_disparity": source_disparity,
            "source_depth_metric": source_depth_metric,
            "source_K": source_K,
            "source_c2w": source_c2w,
            "target_images": target_images,
            "target_disparity": target_disparity,
            "target_depth_metric": target_depth_metric,
            "target_K": target_K,
            "target_c2w": target_c2w,
            "images": images,
            "disparity": disparity,
            "depth_metric": depth_metric,
            "K": K,
            "c2w": c2w,
            "view_roles": view_roles,
            "loss_masks": loss_masks,
            # disp = 1 / (depth_metric * depth_norm_scale); camera translations
            # in `c2w` have already been multiplied by this same factor.
            "depth_norm_scale": norm_scale.reshape(()).float(),
        }