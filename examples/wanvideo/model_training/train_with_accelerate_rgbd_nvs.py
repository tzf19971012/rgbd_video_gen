import argparse
import copy
import json
import math
import os
import shutil

# Ascend NPU support (must run before the first torch import): map
# torch.cuda -> torch.npu for the whole process so the hard-coded
# `device_type="cuda"` autocasts and `.to("cuda")` calls downstream run
# unchanged on NPU.  A no-op on CUDA machines where torch_npu is absent.
try:
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
except ImportError:
    pass

from datetime import timedelta
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, set_seed
from omegaconf import OmegaConf
from tqdm import tqdm

from examples.dataset import Kubric4DRGBDNVSDataset
from examples.wanvideo.model_training.WanRGBDNVSTrainingModule import (
    WanRGBDNVSTrainingModule,
    resolve_rope_frame_axis,
)


process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=30))

PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


# ---------------------------------------------------------------------------
# Run directory placement: keep bulk artefacts off /home
# ---------------------------------------------------------------------------

def load_dotenv(project_dir=PROJECT_DIR):
    """Populate os.environ from <project>/.env, never overriding a real env var.

    Deliberately a hand-rolled 6-line parser rather than a python-dotenv
    dependency: the file only ever holds `KEY=value` lines.
    """
    path = os.path.join(project_dir, ".env")
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def provision_output_path(output_path, project_dir=PROJECT_DIR):
    """Back `logs/<run>` with a directory on $DVD_LOG_ROOT via a symlink.

    /home is a small NFS share that cannot hold many runs' checkpoints, which is
    what forced `max_checkpoints: 2` and destroyed every chance of a
    step-matched retrospective comparison.  Redirecting the storage rather than
    the *path* means configs, resume logic, report links and the VS Code file
    tree all keep working on the `logs/<run>` spelling, so the returned value is
    always the input value.

    Only a plain `logs/<run>` is redirected; absolute or nested paths are left
    alone, and an existing real directory is never touched -- silently moving a
    live run's data is not this function's job.  Call on the main process only.
    """
    log_root = os.environ.get("DVD_LOG_ROOT")
    prefix = "logs" + os.sep
    if not log_root or os.path.isabs(output_path) or not output_path.startswith(prefix):
        return output_path
    run = output_path[len(prefix):].strip(os.sep)
    if not run or os.sep in run:
        return output_path

    link = os.path.join(project_dir, "logs", run)
    target = os.path.join(log_root, run)
    if os.path.islink(link):
        if os.path.realpath(link) != os.path.realpath(target):
            raise RuntimeError(
                f"{link} already points at {os.path.realpath(link)}, not {target}. "
                f"Refusing to repoint a run directory."
            )
        return output_path
    if os.path.exists(link):
        print(
            f"NOTE: {link} is a real directory, so this run stays on /home. "
            f"Stop the run and `mv` it to {target}, then symlink, to free space."
        )
        return output_path

    os.makedirs(target, exist_ok=True)
    os.makedirs(os.path.dirname(link), exist_ok=True)
    try:
        os.symlink(target, link)
    except FileExistsError:
        pass
    print(f"Run directory: {link} -> {target}")
    return output_path


# ---------------------------------------------------------------------------
# Checkpoint manager: keep at most N checkpoints, protect metric-best ones
# ---------------------------------------------------------------------------

class CheckpointManager:
    """
    Keeps at most `max_ckpts` checkpoint directories under `output_path`.
    Always protects the checkpoint with:
      - lowest  src_depth_abs_rel  (symlink → best_src_depth)
      - lowest  tgt_rgb_lpips      (symlink → best_tgt_rgbd)
    Oldest unprotected checkpoints are deleted when the limit is exceeded.
    State is persisted to checkpoint_manager.json for resumption.
    """

    # metric name → (protected_dir_name, higher_is_better)
    #
    # LPIPS, not PSNR.  PSNR is a monotone function of MSE, and MSE is
    # minimised by the conditional mean -- so wherever the target view is
    # genuinely ambiguous (disocclusions, sub-pixel misalignment) the highest
    # PSNR belongs to the blurriest sample, not the most faithful one.  The
    # A/B sweep made this concrete: a1/a2/a4 kept a respectable PSNR while
    # their LPIPS degraded monotonically from step 250 onward, so a
    # PSNR-selected "best" checkpoint was tracking the wrong direction.
    TRACKED = {
        "src_depth_abs_rel": ("best_src_depth", False),
        "tgt_rgb_lpips":     ("best_tgt_rgbd",  False),
    }

    def __init__(self, output_path, max_ckpts=5):
        self.output_path = output_path
        self.max_ckpts   = max_ckpts
        self.history     = []   # list of {"step": int, "path": str, "metrics": dict}
        self.best        = {k: {"step": None, "value": None}
                            for k in self.TRACKED}
        self._load_state()

    # ---- persistence -------------------------------------------------------

    def _state_path(self):
        return os.path.join(self.output_path, "checkpoint_manager.json")

    def _load_state(self):
        p = self._state_path()
        if os.path.isfile(p):
            with open(p) as f:
                s = json.load(f)
            self.history = s.get("history", [])
            # Merge rather than replace: a run started before TRACKED changed
            # has a state file missing the new keys, and add() would KeyError
            # on the first checkpoint after resuming.  Stale keys are dropped.
            saved = s.get("best", {})
            for k in self.best:
                if k in saved:
                    self.best[k] = saved[k]

    def _save_state(self):
        with open(self._state_path(), "w") as f:
            json.dump({"history": self.history, "best": self.best}, f, indent=2)

    # ---- symlink helpers ---------------------------------------------------

    def _update_symlink(self, link_name, target_path):
        link = os.path.join(self.output_path, link_name)
        if os.path.islink(link):
            os.remove(link)
        os.symlink(os.path.abspath(target_path), link)

    # ---- main interface ----------------------------------------------------

    def add(self, global_step, ckpt_path, metrics):
        """Register a new checkpoint; prune if over limit."""
        entry = {"step": global_step, "path": ckpt_path,
                 "metrics": metrics or {}}
        self.history.append(entry)

        # Update best checkpoints
        for metric_key, (link_name, higher_better) in self.TRACKED.items():
            val = (metrics or {}).get(metric_key)
            # `val != val` is the NaN test.  tgt_rgb_lpips is NaN when the
            # lpips package is missing; without this guard the first NaN would
            # be stored as the best value and every later comparison against it
            # would return False, freezing the symlink forever.
            if val is None or val != val:
                continue
            prev = self.best[metric_key]["value"]
            is_better = (prev is None
                         or (higher_better and val > prev)
                         or (not higher_better and val < prev))
            if is_better:
                self.best[metric_key] = {"step": global_step, "value": val}
                self._update_symlink(link_name, ckpt_path)

        # Prune oldest non-protected checkpoints beyond limit
        # Always protect the latest checkpoint in addition to the best ones.
        # max_ckpts == 1 means strictly ONE checkpoint on disk: only the latest
        # is protected and metric-best ones are pruned like any other (their
        # step/value survive in checkpoint_manager.json, the dangling symlink
        # is removed below).
        latest_step = max(e["step"] for e in self.history)
        protected_steps = {latest_step}
        if self.max_ckpts > 1:
            protected_steps |= {v["step"] for v in self.best.values()
                                if v["step"] is not None}
        while len(self.history) > self.max_ckpts:
            deleted = False
            for i, e in enumerate(self.history):
                if e["step"] not in protected_steps:
                    old_path = e["path"]
                    if os.path.isdir(old_path):
                        # On NFS (SFS Turbo) a deleted-but-still-open file is
                        # silly-renamed to .nfsXXX inside the directory, which
                        # makes the final rmdir fail with ENOTEMPTY.  A prune
                        # must never take the training run down, so swallow the
                        # error: pop the entry from history either way, and the
                        # leftover directory is cleaned by hand once the holder
                        # closes the file.
                        try:
                            shutil.rmtree(old_path)
                        except OSError as err:
                            print(
                                f"[CheckpointManager] WARNING could not fully "
                                f"delete {old_path}: {err}"
                            )
                    for link_name, _ in self.TRACKED.values():
                        link = os.path.join(self.output_path, link_name)
                        if os.path.islink(link) and \
                                os.path.realpath(link) == os.path.realpath(old_path):
                            os.remove(link)
                    self.history.pop(i)
                    deleted = True
                    break
            if not deleted:
                break   # all remaining are protected; stop

        self._save_state()


def custom_collate_fn(batch):
    collated = {}
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values)
        elif isinstance(values[0], str):
            collated[key] = values
        elif values[0] is None:
            collated[key] = None
        else:
            collated[key] = values
    return collated


def _resize_mask(mask, latent_size):
    batch_size, num_slots = mask.shape[:2]
    reshaped = mask.reshape(batch_size * num_slots, 1, mask.shape[-2], mask.shape[-1])
    resized = F.interpolate(reshaped.float(), size=latent_size, mode="nearest")
    return resized.reshape(batch_size, num_slots, 1, latent_size[0], latent_size[1])


def _masked_mse(pred, target, mask):
    """MSE averaged over *unmasked* elements only.

    F.mse_loss(pred * mask, target * mask) divides by the full element count,
    so a mask that only covers the target views (3 of 7) silently scales the
    RGB loss by 3/7 relative to the depth loss, breaking the intended
    depth:rgb weighting.  The mask broadcasts over the channel dim, so the
    denominator counts channels too.
    """
    diff2 = (pred - target).float().pow(2) * mask
    denom = mask.expand_as(diff2).sum().clamp_min(1.0)
    return diff2.sum() / denom


# Interior edges of the sigma buckets used for diagnostic logging.  Training
# draws sigma from logit-normal pushed through the shift map, which concentrates
# almost all mass at high sigma; these buckets make that visible and let us
# confirm that a change to the sigma distribution actually reaches the low-noise
# regime where fine detail is learned.
SIGMA_EDGES = (0.2, 0.4, 0.6, 0.8)
SIGMA_LABELS = ("0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0")


def _masked_mse_per_sample(pred, target, mask):
    """Same as _masked_mse but reduced per batch element, shape [B]."""
    diff2 = (pred - target).float().pow(2) * mask
    dims = tuple(range(1, diff2.dim()))
    denom = mask.expand_as(diff2).sum(dim=dims).clamp_min(1.0)
    return diff2.sum(dim=dims) / denom


def _accumulate_sigma_buckets(store, sigma, depth_ps, rgb_ps):
    """Bin per-sample losses by sigma into a [5, 3] GPU accumulator, in place.

    Stays on device so the diagnostic never forces a host sync inside the
    training step; the single transfer happens when the buckets are logged.
    """
    edges = torch.tensor(SIGMA_EDGES, device=sigma.device, dtype=sigma.dtype)
    idx = torch.bucketize(sigma, edges)
    store.index_add_(
        0, idx, torch.stack([torch.ones_like(depth_ps), depth_ps, rgb_ps], dim=1)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    # Any remaining `key=value` arguments override the config, so a single
    # sbatch template can launch short pre-flight runs without duplicating YAML.
    args, overrides = parser.parse_known_args()
    cfg = OmegaConf.load(args.config)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        kwargs_handlers=[process_group_kwargs],
    )
    accelerator.print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    # Must run before any rank calls makedirs, otherwise a rank that gets there
    # first creates a real logs/<run> directory and defeats the symlink.
    load_dotenv()
    if accelerator.is_main_process:
        provision_output_path(cfg.output_path)
    accelerator.wait_for_everyone()

    os.makedirs(cfg.output_path, exist_ok=True)
    if accelerator.is_main_process:
        with open(os.path.join(cfg.output_path, "args.yaml"), "w", encoding="utf-8") as handle:
            handle.write(OmegaConf.to_yaml(cfg))
    accelerator.wait_for_everyone()

    # Tee all accelerator.print output to cfg.output_path/train.log on main process
    _log_file = None
    if accelerator.is_main_process:
        import builtins
        _log_path = os.path.join(cfg.output_path, "train.log")
        _log_file = open(_log_path, "a", buffering=1, encoding="utf-8")
        _orig_print = builtins.print
        def _tee_print(*args, **kwargs):
            _orig_print(*args, **kwargs)
            kwargs.pop("file", None)
            _orig_print(*args, file=_log_file, **kwargs)
        builtins.print = _tee_print

    # The full config dump above went out before the tee was installed, so it only
    # reaches stdout.  Repeat the flags that change the model's behaviour without
    # changing any parameter shape -- the ones a sweep arm is defined by, and the
    # ones a mis-set eval would silently get wrong -- into train.log itself.
    accelerator.print(
        f"factors: token_layout={cfg.get('token_layout', 'channel')}  "
        f"warp_mode={cfg.get('warp_mode', 'none')}  "
        f"warp_splat_footprint={cfg.get('warp_splat_footprint', 'nearest')}  "
        f"per_frame_timestep={cfg.get('per_frame_timestep', False)}  "
        f"depth_warp_block={cfg.get('depth_warp_block', False)}  "
        f"clean_source_depth={cfg.get('clean_source_depth', False)}"
        f"@p={cfg.get('clean_source_depth_prob', 0.5)}  "
        f"shared_rope_frame_ids={cfg.get('shared_rope_frame_ids', False)}  "
        # RESOLVED, not cfg's raw value: the sharedrope arm and every D2 arm spell
        # this factor as the older bool and leave rope_frame_axis unset, so echoing
        # the raw `None` reads as "the frame axis is untouched" in exactly the arms
        # where it is not.
        f"rope_frame_axis={resolve_rope_frame_axis(cfg.get('rope_frame_axis', None), cfg.get('shared_rope_frame_ids', False))}  "
        f"rope_pose_translation_scale={cfg.get('rope_pose_translation_scale', None)}  "
        f"roce={cfg.get('roce', 'off')}  "
        f"camera_geometry_fp32={cfg.get('camera_geometry_fp32', False)}  "
        # The whole point of this line is that an arm can be identified from its
        # log alone; e-warpnoise degrades only the TRAINING warp, so it leaves no
        # other trace in any reported metric.
        f"warp_perturb_prob={cfg.get('warp_perturb_prob', 0.0)}"
        f"@blur={cfg.get('warp_perturb_blur_sigma', 2.0)}"
        f",gain={cfg.get('warp_perturb_gain_log_std', 0.34)}"
        f",shift={cfg.get('warp_perturb_shift_std', 0.059)}"
        f",lowfreq={cfg.get('warp_perturb_lowfreq_std', 0.05)}"
        f"/{cfg.get('warp_perturb_lowfreq_cells', 6)}"
        f",fill={cfg.get('warp_perturb_fill_invalid', True)}"
        f"x{cfg.get('warp_perturb_fill_scale', 1.0)}  "
        # Same trap as warp_perturb, one step worse: f-consist adds a loss TERM, so
        # it changes no shape, no parameter and no reported metric name -- an arm
        # launched with the weights at 0 would look identical to the arm that has
        # them set, in every file except this line.
        f"depth_consist={cfg.get('depth_consist_scale_weight', 0.0)}"
        f"/{cfg.get('depth_consist_repro_weight', 0.0)}"
        f"@sig<={cfg.get('depth_consist_sigma_max', 0.6)}"
        f",v={cfg.get('depth_consist_num_views', 3)}"
        f",n<={cfg.get('depth_consist_max_samples', 2)}"
        f",sup={cfg.get('depth_consist_min_support', 0.02)}"
        f",clip={cfg.get('depth_consist_ratio_clip', 4.0)}"
        f",trim={cfg.get('depth_consist_trim_quantile', 0.9)}"
        f",ckpt={cfg.get('depth_consist_vae_checkpoint', True)}  "
        f"view_order={cfg.get('view_order', 'random')}  "
        f"rgb_flow_shift={cfg.get('rgb_flow_shift', 5.0)}  "
        f"eval_flow_shift={cfg.get('eval_flow_shift', None)}  "
        f"seed={cfg.seed}"
    )

    tb_writer = None
    if accelerator.is_main_process:
        tb_log_dir = os.path.join(cfg.output_path, "tensorboard")
        tb_writer = SummaryWriter(log_dir=tb_log_dir)

    model = WanRGBDNVSTrainingModule(
        args=cfg,
        accelerator=accelerator,
        model_id_with_origin_paths=cfg.model_id_with_origin_paths,
        trainable_models=cfg.trainable_models,
        lora_rank=cfg.lora_rank,
        lora_base_model=cfg.lora_base_model,
        lora_target_modules=cfg.lora_target_modules,
        local_model_path=cfg.get("wan_model_root", "./models"),
        skip_download=cfg.get("skip_download", False),
        use_gradient_checkpointing=cfg.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=cfg.use_gradient_checkpointing_offload,
        num_source_views=cfg.num_source_views,
        num_target_views=cfg.num_target_views,
        vae_encoding_mode=cfg.vae_encoding_mode,
        init_dvd_ckpt=cfg.init_dvd_ckpt,
        rgb_loss_weight=cfg.rgb_loss_weight,
        depth_loss_weight=cfg.depth_loss_weight,
        view_condition_plucker_resolution=cfg.get("view_condition_plucker_resolution", "latent"),
        view_condition_pixel_scale=cfg.get("view_condition_pixel_scale", 8),
        view_condition_mode=cfg.get("view_condition_mode", "add"),
        view_condition_use_role=cfg.get("view_condition_use_role", True),
        view_condition_normalize_moment=cfg.get("view_condition_normalize_moment", False),
        latent_scale=cfg.get("latent_scale", 1.0),
        depth_latent_scale=cfg.get("depth_latent_scale", None),
        token_layout=cfg.get("token_layout", "channel"),
        warp_mode=cfg.get("warp_mode", "none"),
        warp_splat_footprint=cfg.get("warp_splat_footprint", "nearest"),
        warp_dropout_prob=cfg.get("warp_dropout_prob", 0.5),
        warp_two_pass=cfg.get("warp_two_pass", True),
        warp_perturb_prob=cfg.get("warp_perturb_prob", 0.0),
        warp_perturb_gain_log_std=cfg.get("warp_perturb_gain_log_std", 0.34),
        warp_perturb_shift_std=cfg.get("warp_perturb_shift_std", 0.059),
        warp_perturb_blur_sigma=cfg.get("warp_perturb_blur_sigma", 2.0),
        warp_perturb_lowfreq_std=cfg.get("warp_perturb_lowfreq_std", 0.05),
        warp_perturb_lowfreq_cells=cfg.get("warp_perturb_lowfreq_cells", 6),
        warp_perturb_fill_invalid=cfg.get("warp_perturb_fill_invalid", True),
        warp_perturb_fill_scale=cfg.get("warp_perturb_fill_scale", 1.0),
        depth_warp_block=cfg.get("depth_warp_block", False),
        depth_consist_scale_weight=cfg.get("depth_consist_scale_weight", 0.0),
        depth_consist_repro_weight=cfg.get("depth_consist_repro_weight", 0.0),
        depth_consist_sigma_max=cfg.get("depth_consist_sigma_max", 0.6),
        depth_consist_num_views=cfg.get("depth_consist_num_views", 3),
        depth_consist_max_samples=cfg.get("depth_consist_max_samples", 2),
        depth_consist_min_support=cfg.get("depth_consist_min_support", 0.02),
        depth_consist_ratio_clip=cfg.get("depth_consist_ratio_clip", 4.0),
        depth_consist_trim_quantile=cfg.get("depth_consist_trim_quantile", 0.9),
        depth_consist_vae_checkpoint=cfg.get("depth_consist_vae_checkpoint", True),
        per_frame_timestep=cfg.get("per_frame_timestep", False),
        clean_source_depth=cfg.get("clean_source_depth", False),
        clean_source_depth_prob=cfg.get("clean_source_depth_prob", 0.5),
        shared_rope_frame_ids=cfg.get("shared_rope_frame_ids", False),
        rope_frame_axis=cfg.get("rope_frame_axis", None),
        rope_pose_translation_scale=cfg.get("rope_pose_translation_scale", None),
        roce=cfg.get("roce", "off"),
        camera_geometry_fp32=cfg.get("camera_geometry_fp32", False),
        rgb_flow_matching=cfg.get("rgb_flow_matching", True),
        rgb_flow_shift=cfg.get("rgb_flow_shift", 5.0),
        rgb_flow_timestep_sampling=cfg.get("rgb_flow_timestep_sampling", "logit_normal"),
        rgb_flow_logit_mean=cfg.get("rgb_flow_logit_mean", 0.0),
        rgb_flow_logit_std=cfg.get("rgb_flow_logit_std", 1.0),
        depth_flow_matching=cfg.get("depth_flow_matching", False),
        depth_head_init=cfg.get("depth_head_init", "pretrained"),
        eval_num_inference_steps=cfg.get("eval_num_inference_steps", 4),
        eval_flow_shift=cfg.get("eval_flow_shift", None),
        eval_sample_seed=cfg.get("eval_sample_seed", 1234),
        eval_lpips_net=cfg.get("eval_lpips_net", "alex"),
        eval_vis_downscale=cfg.get("eval_vis_downscale", 2),
        eval_vis_format=cfg.get("eval_vis_format", "jpg"),
        eval_vis_depth_norm=cfg.get("eval_vis_depth_norm", "shared"),
    )
    model.set_training_param()

    # Weight-only finetune init: load `init_model_ckpt` (a directory containing
    # model.safetensors, e.g. an accelerate save_state checkpoint, or the
    # safetensors file itself) into the freshly constructed model.  Runs BEFORE
    # optimizer creation, so AdamW moments start fresh and global_step restarts
    # at 0 -- this is a finetune, not a resume.  Use cfg.resume +
    # cfg.training_state_dir instead to continue a run exactly (optimizer +
    # scheduler + step counter).
    if cfg.get("init_model_ckpt", None):
        from safetensors.torch import load_file
        _init = cfg.init_model_ckpt
        _init_w = (os.path.join(_init, "model.safetensors")
                   if os.path.isdir(_init) else _init)
        _state = load_file(_init_w, device="cpu")
        _missing, _unexpected = model.load_state_dict(_state, strict=False)
        accelerator.print(
            f"[init_model_ckpt] loaded {_init_w}: {len(_state)} tensors, "
            f"missing={len(_missing)}, unexpected={len(_unexpected)}"
        )
        if _missing or _unexpected:
            accelerator.print(
                f"[init_model_ckpt] WARNING structure mismatch: "
                f"missing={list(_missing)[:5]} unexpected={list(_unexpected)[:5]}"
            )

    train_dataset = Kubric4DRGBDNVSDataset(
        kubric4d_root=cfg.kubric4d_root,
        manifest_path=cfg.manifest_path,
        resolution=tuple(cfg.resolution),
        num_source_views=cfg.num_source_views,
        num_target_views=cfg.num_target_views,
        random_permute_source_views=cfg.random_permute_source_views,
        view_order=cfg.get("view_order", "random"),
        video_tgt_interval_prob=cfg.get("video_tgt_interval_prob", 0.2),
        video_max_src_span=cfg.get("video_max_src_span", None),
        disparity_norm_mode=cfg.get("disparity_norm_mode", "shared_scale"),
        disparity_norm_quantile=cfg.get("disparity_norm_quantile", 0.98),
        disparity_norm_headroom=cfg.get("disparity_norm_headroom", 1.5),
        min_view_valid_fraction=cfg.get("min_view_valid_fraction", 0.0),
        view_validity_path=cfg.get("view_validity_path", None),
        samples_per_scene=cfg.get("samples_per_scene", 1),
        split_path=cfg.get("split_path", None),
        split=cfg.get("train_split", None),
        scene_filter=cfg.get("scene_filter", None),
    )
    if cfg.get("num_train_samples", None) is not None:
        train_dataset.records = train_dataset.records[: int(cfg.num_train_samples)]

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.dataloader_num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    eval_max_samples = int(cfg.get("eval_max_samples", 2))
    # The frozen eval set is scored in CHUNKS.  A single batch of
    # `eval_max_samples` would have to fit alongside the training activations on
    # rank 0, so enlarging the eval set would OOM; the two knobs are therefore
    # independent and only `eval_batch_size` costs memory.
    eval_batch_size = max(1, min(eval_max_samples, int(cfg.get("eval_batch_size", 4))))
    eval_dataloader = None
    if accelerator.is_main_process and eval_max_samples > 0:
        # The eval set must be FROZEN: identical records AND identical
        # source/target view assignments on every validation step, every run and
        # every A/B arm.  Otherwise metrics that depend only on the ground truth
        # (e.g. delta1_ramp) drift between runs and nothing is comparable.
        #   - shallow copy shares `records` with train_dataset (so any
        #     num_train_samples truncation is inherited) without re-parsing the
        #     manifest;
        #   - deterministic_view_seed makes _select_views a pure function of the
        #     sample index, immune to DataLoader worker seeding;
        #   - a fixed, evenly spaced index list + shuffle=False pins which
        #     records are drawn.
        eval_split = cfg.get("eval_split", None)
        split_path = cfg.get("split_path", None)
        if eval_split is not None:
            # HELD-OUT path.  A separate dataset object, not a shallow copy: the
            # copy shared `records` with train_dataset, which meant every eval
            # number ever reported was measured on TRAINING scenes.
            #
            # `eval_min_view_valid_fraction` exists but should normally be left
            # unset, i.e. equal to training's.  Raising it for the hold-out looks
            # like a quality filter and is really a camera-pose filter: per-view
            # validity correlates -0.33 with obliqueness because oblique views
            # frame sky, so a strict threshold selects nadir views and shrinks the
            # baseline distribution the eval set is supposed to probe.  What keeps
            # object-centric captures out of the hold-out is the scene-level gate
            # in tools/make_bmvs_split.py, already baked into the split lists.
            eval_dataset = Kubric4DRGBDNVSDataset(
                kubric4d_root=cfg.kubric4d_root,
                manifest_path=cfg.manifest_path,
                resolution=tuple(cfg.resolution),
                num_source_views=cfg.num_source_views,
                num_target_views=cfg.num_target_views,
                random_permute_source_views=cfg.random_permute_source_views,
                # The factor has to be ON for the hold-out too: it changes what
                # the model sees, not just what it trains on.  It leaves the view
                # SELECTION untouched, so the eval set stays paired with the
                # random-order arms (_select_views' comment spells out why).
                view_order=cfg.get("view_order", "random"),
                video_tgt_interval_prob=cfg.get(
                    "video_tgt_interval_prob", 0.2),
                video_max_src_span=cfg.get("video_max_src_span", None),
                disparity_norm_mode=cfg.get("disparity_norm_mode", "shared_scale"),
                disparity_norm_quantile=cfg.get("disparity_norm_quantile", 0.98),
                disparity_norm_headroom=cfg.get("disparity_norm_headroom", 1.5),
                min_view_valid_fraction=cfg.get(
                    "eval_min_view_valid_fraction",
                    cfg.get("min_view_valid_fraction", 0.0)),
                view_validity_path=cfg.get("view_validity_path", None),
                samples_per_scene=cfg.get("samples_per_scene", 1),
                split_path=split_path,
                split=eval_split,
                scene_filter=cfg.get("scene_filter", None),
            )
            eval_dataset.deterministic_view_seed = int(cfg.get("eval_view_seed", 1234))
            train_ids = {r["scene_id"] for r in train_dataset.records}
            leaked = train_ids & {r["scene_id"] for r in eval_dataset.records}
            if leaked:
                raise ValueError(
                    f"eval_split={eval_split!r} shares {len(leaked)} scene(s) with the "
                    f"training split, e.g. {sorted(leaked)[:3]}. The hold-out is not held "
                    f"out; regenerate the split with tools/make_bmvs_split.py."
                )
            samples_per_scene = getattr(eval_dataset, "samples_per_scene", 1)
            num_scenes = len(eval_dataset.records)
            # One sample slot per held-out scene, in SPLIT-FILE order.  No stride:
            # a stride is what made the old eval set a function of len(records).
            eval_indices = [i * samples_per_scene for i in range(num_scenes)]
            if eval_max_samples < num_scenes:
                accelerator.print(
                    f"WARNING: eval_max_samples={eval_max_samples} < {num_scenes} scenes in "
                    f"split={eval_split!r}; scoring only the first {eval_max_samples}. "
                    f"Paired A/B power scales with the number of SCENES -- the per-scene std "
                    f"of the paired PSNR difference is 0.414 dB, so 32 scenes resolve ~0.21 dB "
                    f"and 64 resolve ~0.15 dB at 80% power / alpha=0.05. Raise "
                    f"eval_max_samples to {num_scenes} to score the whole hold-out."
                )
                eval_indices = eval_indices[:eval_max_samples]
        else:
            eval_dataset = copy.copy(train_dataset)
            eval_dataset.deterministic_view_seed = int(cfg.get("eval_view_seed", 1234))
            # Stride over SCENES, not over raw indices: with samples_per_scene > 1 a
            # raw-index stride would drift onto different scenes whenever that
            # multiplier changes, silently breaking A/B comparability.  Taking the
            # first sample slot of evenly spaced scenes keeps the eval set fixed.
            samples_per_scene = getattr(eval_dataset, "samples_per_scene", 1)
            num_scenes = len(eval_dataset.records)
            scene_stride = max(1, num_scenes // eval_max_samples)
            eval_indices = [
                scene_idx * samples_per_scene
                for scene_idx in range(0, num_scenes, scene_stride)
            ][:eval_max_samples]
        eval_dataloader = DataLoader(
            Subset(eval_dataset, eval_indices),
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=cfg.dataloader_num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=True,
            drop_last=False,
        )
        accelerator.print(
            f"Frozen eval set: {len(eval_indices)} samples in "
            f"{(len(eval_indices) + eval_batch_size - 1) // eval_batch_size} chunk(s) of "
            f"{eval_batch_size}  indices={eval_indices} "
            f"eval_view_seed={eval_dataset.deterministic_view_seed}"
        )

    # fused=True keeps the whole update on-device; the default (non-fused) path
    # calls .item() twice per parameter (~1200 stream syncs per step on NPU,
    # profiled 10-10), which serialises the entire pipeline behind the optimizer.
    try:
        optimizer = torch.optim.AdamW(model.trainable_modules(), lr=cfg.learning_rate, fused=True)
    except (RuntimeError, ValueError):
        optimizer = torch.optim.AdamW(model.trainable_modules(), lr=cfg.learning_rate)
    warmup_steps = int(cfg.get("warmup_steps", 0))
    # Cosine decay from `learning_rate` down to `lr_min` over `lr_decay_steps`
    # REAL optimizer steps AFTER warmup, then held flat at `lr_min` forever.
    # `lr_decay_steps <= 0` disables the decay entirely (constant lr after
    # warmup), which is what every arm up to a8 ran with.
    lr_decay_steps = int(cfg.get("lr_decay_steps", 0))
    lr_min = float(cfg.get("lr_min", 0.0))
    min_factor = min(1.0, lr_min / float(cfg.learning_rate)) if cfg.learning_rate else 0.0
    warmup_start_factor = 0.5
    # `accelerator.prepare` wraps the scheduler so that its `.step()` fires
    # `num_processes` times per optimizer step; every step count the lambda
    # compares against therefore has to be converted back to REAL steps first.
    world_size = accelerator.num_processes

    def _lr_factor(sched_step):
        step = sched_step / world_size
        if warmup_steps > 0 and step < warmup_steps:
            return warmup_start_factor + (1.0 - warmup_start_factor) * (step / warmup_steps)
        if lr_decay_steps <= 0:
            return 1.0
        progress = min(1.0, max(0.0, (step - warmup_steps) / lr_decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_factor + (1.0 - min_factor) * cosine

    # A single LambdaLR expresses warmup + decay + floor without SequentialLR,
    # whose milestones would need the same world-size scaling and silently
    # mis-fire if it were forgotten in one of the three places.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_factor)
    model, optimizer, scheduler, train_dataloader = accelerator.prepare(
        model, optimizer, scheduler, train_dataloader
    )

    # `__len__` of this dataset is the number of SCENES, which is small (~100
    # after view filtering).  Sharded over N processes with drop_last=True, a
    # per-device batch_size above len(dataset)/N yields ZERO batches per epoch:
    # the training loop then spins over empty epochs forever without erroring
    # and without logging a single step.  Fail loudly instead.
    if len(train_dataloader) == 0:
        raise ValueError(
            f"train_dataloader is empty: dataset has {len(train_dataset)} samples "
            f"split over {accelerator.num_processes} processes "
            f"(~{len(train_dataset) // accelerator.num_processes} each), which is "
            f"smaller than batch_size={cfg.batch_size} with drop_last=True. "
            f"Lower batch_size to at most "
            f"{max(1, len(train_dataset) // accelerator.num_processes)}, or raise "
            f"the dataset length."
        )

    if cfg.resume and cfg.training_state_dir not in (None, "na"):
        accelerator.load_state(cfg.training_state_dir)

    global_step = int(cfg.global_step)
    optimizer.zero_grad(set_to_none=True)
    model.train()

    # FACTOR f-consist: read once, so the hot loop cannot pay for an OmegaConf
    # lookup per step and cannot disagree with what the module was constructed with.
    consist_scale_weight = float(cfg.get("depth_consist_scale_weight", 0.0))
    consist_repro_weight = float(cfg.get("depth_consist_repro_weight", 0.0))

    ckpt_manager = CheckpointManager(
        output_path=cfg.output_path,
        max_ckpts=int(cfg.get("max_checkpoints", 5)),
    )

    # [5, 3] = per sigma bucket: count, depth_loss_sum, rgb_loss_sum.  Only rank
    # 0 reports, so only rank 0 needs to accumulate.  Drained on every log.
    sigma_buckets = (
        torch.zeros(len(SIGMA_LABELS), 3, device=accelerator.device)
        if accelerator.is_main_process
        else None
    )

    for epoch in range(cfg.num_epochs):
        for batch in tqdm(train_dataloader, disable=not accelerator.is_main_process):
            if global_step >= cfg.max_steps:
                break

            with accelerator.accumulate(model):
                res = model(batch)
                pred = res["pred"]
                if not isinstance(pred, tuple):
                    raise RuntimeError("NVS training expects (depth, rgb) predictions.")
                pred_depth, pred_rgb = pred

                depth_gt = res["depth_gt"]
                rgb_gt = res["rgb_gt"]
                loss_masks = res["loss_masks"]

                # masks: [B, V, 1, H, W] → after resize [B, V, 1, h, w]
                # pred/gt: [B, 16, V, h, w] → need mask as [B, 1, V, h, w] to broadcast
                depth_mask = _resize_mask(loss_masks[:, 0], pred_depth.shape[-2:]).to(pred_depth.dtype)
                depth_mask = depth_mask.permute(0, 2, 1, 3, 4)  # [B, 1, V, h, w]
                rgb_mask = _resize_mask(loss_masks[:, 1], pred_rgb.shape[-2:]).to(pred_rgb.dtype)
                rgb_mask = rgb_mask.permute(0, 2, 1, 3, 4)  # [B, 1, V, h, w]

                depth_loss = _masked_mse(pred_depth, depth_gt, depth_mask)
                rgb_loss = _masked_mse(pred_rgb, rgb_gt, rgb_mask)
                loss = cfg.depth_loss_weight * depth_loss + cfg.rgb_loss_weight * rgb_loss

                # FACTOR f-consist.  Zero-valued (and zero-cost) unless a weight is
                # set, so this line is a no-op for every historical arm.  Both
                # terms are means over the samples the module actually measured,
                # not over the batch -- see _depth_consistency_loss for why, and
                # for what that does to the effective weight.
                consist = res.get("consist")
                if consist is not None:
                    loss = (
                        loss
                        + consist_scale_weight * consist["scale"]
                        + consist_repro_weight * consist["repro"]
                    )

                if sigma_buckets is not None and "sigma" in res:
                    with torch.no_grad():
                        _accumulate_sigma_buckets(
                            sigma_buckets,
                            res["sigma"].detach().float(),
                            _masked_mse_per_sample(pred_depth.detach(), depth_gt, depth_mask),
                            _masked_mse_per_sample(pred_rgb.detach(), rgb_gt, rgb_mask),
                        )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if cfg.clip_grad_norm:
                        accelerator.clip_grad_norm_(accelerator.unwrap_model(model).trainable_modules(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    global_step += 1

                    if global_step % cfg.log_step == 0 and accelerator.is_main_process:
                        lr = scheduler.get_last_lr()[0]
                        accelerator.print(
                            f"step {global_step}: loss={loss.item():.6f} depth={depth_loss.item():.6f} rgb={rgb_loss.item():.6f} lr={lr:.8f}"
                        )
                        tb_writer.add_scalar("train/loss",       loss.item(),       global_step)
                        tb_writer.add_scalar("train/depth_loss", depth_loss.item(), global_step)
                        tb_writer.add_scalar("train/rgb_loss",   rgb_loss.item(),   global_step)
                        tb_writer.add_scalar("train/lr",         lr,                global_step)

                        if consist is not None and consist["n_samples"] > 0:
                            # `flat` is the degenerate-solution monitor: a constant
                            # disparity field scores zero on both terms, so if this
                            # collapses the weight is too high.  `n_samples` and
                            # `n_pairs` are the "is the factor firing at all"
                            # readout -- BMVS source views overlap sparsely enough
                            # that n_pairs of 30 is routinely single digits.
                            # .detach() before float(): both terms carry gradient at
                            # this point, and float() on such a tensor emits a
                            # UserWarning -- once per log_step, into train.log, which
                            # is the file the arm has to be readable from.
                            c_scale = float(consist["scale"].detach())
                            c_repro = float(consist["repro"].detach())
                            accelerator.print(
                                f"  consist  scale={c_scale:.5f}"
                                f" repro={c_repro:.5f}"
                                f" |a-1|={consist['scale_dev']:.4f}"
                                f" pairs={consist['n_pairs']:.1f}"
                                f" support={consist['support']:.3f}"
                                f" flat={consist['flat']:.4f}"
                                f" n={consist['n_samples']:.0f}"
                                # The sigma of the MEASURED samples.  The terms are
                                # strongly sigma-dependent -- x0_hat gets (1-sigma)
                                # of the truth handed to it through x_t -- so a
                                # |a-1| without a sigma beside it cannot be read
                                # against the diagnostic's inference-time 0.1309.
                                f" sig={consist['sigma']:.3f}"
                            )
                            for tag, val in (
                                ("scale", c_scale),
                                ("repro", c_repro),
                                ("scale_dev", consist["scale_dev"]),
                                ("n_pairs", consist["n_pairs"]),
                                ("support", consist["support"]),
                                ("flatness", consist["flat"]),
                                ("n_samples", consist["n_samples"]),
                                ("sigma", consist["sigma"]),
                            ):
                                tb_writer.add_scalar(f"consist/{tag}", val, global_step)

                        if sigma_buckets is not None:
                            # counts are rank-0 only; the shape of the histogram
                            # is what matters, not the absolute totals
                            stats = sigma_buckets.cpu()
                            sigma_buckets.zero_()
                            parts = []
                            for bi, tag in enumerate(SIGMA_LABELS):
                                n, dsum, rsum = (float(v) for v in stats[bi])
                                tb_writer.add_scalar(f"sigma/{tag}/count", n, global_step)
                                if n == 0:
                                    parts.append(f"{tag}:n=0")
                                    continue
                                tb_writer.add_scalar(f"sigma/{tag}/depth_loss", dsum / n, global_step)
                                tb_writer.add_scalar(f"sigma/{tag}/rgb_loss", rsum / n, global_step)
                                parts.append(f"{tag}:n={n:.0f} rgb={rsum / n:.4f} dep={dsum / n:.4f}")
                            accelerator.print("  sigma buckets  " + "  ".join(parts))

                    if global_step % cfg.validate_step == 0:
                        ckpt_path = os.path.join(cfg.output_path, f"checkpoint-step-{global_step}")
                        # save_state is collective — all ranks participate
                        accelerator.save_state(ckpt_path)

                        eval_metrics = None
                        if accelerator.is_main_process:
                            torch.save({"global_step": global_step},
                                       os.path.join(cfg.output_path, "trainer_state.pt"))

                            # Evaluate: decode predictions and compute metrics
                            eval_model = accelerator.unwrap_model(model)
                            eval_batches = [batch] if eval_dataloader is None else list(eval_dataloader)
                            # Every chunk writes to the same eval_step_* directory,
                            # so only the first `eval_vis_chunks` chunks render
                            # comparison images (default 1, i.e. eval_batch_size
                            # images total); later chunks skip the VAE-decode cost
                            # entirely. `vis_offset` keeps each chunk's filenames
                            # (b00, b01, ...) from colliding with the previous
                            # chunk's, so raising eval_vis_chunks actually adds
                            # images instead of overwriting them.
                            eval_vis_chunks = int(cfg.get("eval_vis_chunks", 1))
                            chunk_metrics = []
                            for chunk_idx, eval_batch in enumerate(eval_batches):
                                chunk_metrics.append(eval_model.evaluate(
                                    eval_batch,
                                    output_dir=cfg.output_path,
                                    global_step=global_step,
                                    max_samples=eval_batch_size,
                                    save_vis=(chunk_idx < eval_vis_chunks),
                                    vis_offset=chunk_idx * eval_batch_size,
                                ))
                            eval_metrics = {}
                            for k in {key for m in chunk_metrics for key in m}:
                                vals = [m[k] for m in chunk_metrics if k in m]
                                eval_metrics[k] = float(np.nanmean(vals))
                            # delta1_skill_agg is a ratio and must be rebuilt
                            # from the pooled delta1 / ramp -- averaging the
                            # per-chunk ratios would reintroduce exactly the
                            # mean-of-ratios bias this metric exists to avoid.
                            for side in ("src", "tgt"):
                                d1 = eval_metrics.get(f"{side}_depth_delta1")
                                ramp = eval_metrics.get(f"{side}_depth_delta1_ramp")
                                if d1 is None or ramp is None or 1.0 - ramp <= 1e-3:
                                    continue
                                eval_metrics[f"{side}_depth_delta1_skill_agg"] = (d1 - ramp) / (1.0 - ramp)
                            accelerator.print(
                                f"[eval step {global_step}] "
                                + "  ".join(f"{k}={v:.4f}" for k, v in eval_metrics.items())
                            )
                            for k, v in eval_metrics.items():
                                tb_writer.add_scalar(f"eval/{k}", v, global_step)

                            # Register with checkpoint manager (prunes old ones)
                            ckpt_manager.add(global_step, ckpt_path, eval_metrics)

        if global_step >= cfg.max_steps:
            break

    if tb_writer is not None:
        tb_writer.close()
    if _log_file is not None:
        _log_file.close()
    accelerator.end_training()


if __name__ == "__main__":
    main()