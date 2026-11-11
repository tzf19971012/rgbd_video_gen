"""GAE-latent NVS training (Phase 2) — single-metric loss trainer.

Same harness as train_with_accelerate_rgbd_nvs.py with the RGB-D specifics
removed: the only objective is the masked rectified-flow velocity MSE on the
standardized GAE latent of the target views.

Launch (rgbd venv, from rgbd_video_gen root):
  python examples/wanvideo/model_training/train_with_accelerate_gae_nvs.py \
      --config train_config/normal_config/gae_nvs_lora.yaml
"""
import argparse
import copy
import json
import math
import os

try:
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
except ImportError:
    pass

from datetime import timedelta

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    set_seed,
)
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from examples.dataset import Kubric4DRGBDNVSDataset
from examples.wanvideo.model_training.WanGAENVSTrainingModule import (
    WanGAENVSTrainingModule,
)
from examples.wanvideo.model_training.train_with_accelerate_rgbd_nvs import (
    CheckpointManager,
    _masked_mse,
    _resize_mask,
    custom_collate_fn,
    load_dotenv,
    provision_output_path,
)

# Some arms (e.g. flip-PE without condition dropout) leave registered
# trainable params out of the loss graph; DDP then aborts with "parameters
# that were not used in producing loss". Configs can opt into the scan via
# ddp_find_unused: true (small per-step overhead at 1.3B-LoRA scale).
process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=30))
PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

DATASET_CLASSES = {"kubric4d": Kubric4DRGBDNVSDataset}


def _dataset_class(cfg):
    return DATASET_CLASSES.get(cfg.get("dataset_class", "kubric4d"))


def build_dataset(cfg, split=None):
    return Kubric4DRGBDNVSDataset(
        kubric4d_root=cfg.kubric4d_root,
        manifest_path=cfg.manifest_path,
        resolution=tuple(cfg.resolution),
        num_source_views=cfg.num_source_views,
        num_target_views=cfg.num_target_views,
        random_permute_source_views=cfg.random_permute_source_views,
        view_order=cfg.get("view_order", "random"),
        video_tgt_interval_prob=cfg.get("video_tgt_interval_prob", 0.2),
        video_max_src_span=cfg.get("video_max_src_span", None),
        vggt_local_prob=cfg.get("vggt_local_prob", 0.2),
        vggt_window_lo=cfg.get("vggt_window_lo", 0.15),
        vggt_window_hi=cfg.get("vggt_window_hi", 0.4),
        vggt_min_local_views=cfg.get("vggt_min_local_views", 24),
        vggt_strata=cfg.get("vggt_strata", 12),
        # Anything but "shared_scale" leaves c2w (and depth) METRIC, which the
        # GAE ray field requires; the disparity it computes is unused here.
        disparity_norm_mode=cfg.get("disparity_norm_mode", "metric"),
        disparity_norm_quantile=cfg.get("disparity_norm_quantile", 0.98),
        disparity_norm_headroom=cfg.get("disparity_norm_headroom", 1.5),
        min_view_valid_fraction=cfg.get("min_view_valid_fraction", 0.0),
        view_validity_path=cfg.get("view_validity_path", None),
        samples_per_scene=cfg.get("samples_per_scene", 1),
        split_path=cfg.get("split_path", None),
        split=split,
        scene_filter=cfg.get("scene_filter", None),
    )


def restrict_views_to_cache(dataset, cache_root, dataset_name, num_views, accelerator):
    """Clip every record's view pool to the views whose GAE latent is cached.

    The offline cache stores a linspace subsample of each scene's views
    (cache_gae_latents.py --max-views-per-scene), while the manifest lists ALL
    of them -- a bmvs scene can carry 2000+ views against 24 cached.  Without
    this clamp the dataset happily samples an uncached view and the module's
    cache-miss check kills the run mid-epoch.  Scenes left with fewer than
    num_views admissible views are dropped entirely.
    """
    cache_dir = os.path.join(cache_root, dataset_name)
    kept, dropped = [], []
    for record in dataset.records:
        meta_fp = os.path.join(cache_dir, f"{record['scene_id']}.meta.json")
        if not os.path.exists(meta_fp):
            dropped.append(record["scene_id"])
            continue
        with open(meta_fp, "r", encoding="utf-8") as handle:
            allowed = {str(v) for v in json.load(handle)["views"]}
        views = [v for v in record["views"] if str(v.view_id) in allowed]
        if len(views) < num_views:
            dropped.append(record["scene_id"])
            continue
        record["views"] = views
        kept.append(record)
    dataset.records = kept
    accelerator.print(
        f"[gae] view pool clipped to cache: {len(kept)} scenes kept, "
        f"{len(dropped)} dropped ({dropped[:3]}), "
        f"views/scene min={min(len(r['views']) for r in kept)}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args, overrides = parser.parse_known_args()
    cfg = OmegaConf.load(args.config)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        kwargs_handlers=[
            process_group_kwargs,
            DistributedDataParallelKwargs(
                find_unused_parameters=bool(cfg.get("ddp_find_unused", False))),
        ],
    )
    accelerator.print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    load_dotenv()
    if accelerator.is_main_process:
        provision_output_path(cfg.output_path)
    accelerator.wait_for_everyone()

    os.makedirs(cfg.output_path, exist_ok=True)
    if accelerator.is_main_process:
        with open(os.path.join(cfg.output_path, "args.yaml"), "w", encoding="utf-8") as handle:
            handle.write(OmegaConf.to_yaml(cfg))
    accelerator.wait_for_everyone()

    _log_file = None
    if accelerator.is_main_process:
        import builtins

        _log_path = os.path.join(cfg.output_path, "train.log")
        _log_file = open(_log_path, "a", buffering=1, encoding="utf-8")
        _orig_print = builtins.print

        def _tee_print(*pargs, **pkwargs):
            _orig_print(*pargs, **pkwargs)
            pkwargs.pop("file", None)
            _orig_print(*pargs, file=_log_file, **pkwargs)

        builtins.print = _tee_print

    accelerator.print(
        f"factors: latent_backend=gae  dataset={cfg.get('gae_dataset_name')}  "
        f"cache={cfg.get('gae_cache_root')}  "
        f"stats={cfg.get('gae_stats_path')}  "
        f"rope_frame_axis={cfg.get('gae_rope_frame_axis', 'view')}  "
        f"rgb_flow_shift={cfg.get('rgb_flow_shift', None)} (auto={cfg.get('gae_auto_flow_shift', True)})  "
        f"rgb_aux={cfg.get('gae_rgb_loss_weight', 0.0)}"
        f"@t<={cfg.get('gae_rgb_t_max', 0.4)}"
        f",views={cfg.get('gae_rgb_max_views', 1)}  "
        f"patch_mode={cfg.get('gae_patch_mode', 'native')}  "
        f"prediction={cfg.get('gae_prediction', 'v')}  "
        f"lora={cfg.get('lora_base_model')} r={cfg.get('lora_rank')}  "
        f"train_all={cfg.get('train_all', False)}  seed={cfg.seed}"
    )

    tb_writer = None
    if accelerator.is_main_process:
        tb_writer = SummaryWriter(log_dir=os.path.join(cfg.output_path, "tensorboard"))

    model = WanGAENVSTrainingModule(
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
        init_dvd_ckpt=None,
        view_condition_plucker_resolution=cfg.get("view_condition_plucker_resolution", "latent"),
        view_condition_pixel_scale=cfg.get("view_condition_pixel_scale", 8),
        view_condition_mode=cfg.get("view_condition_mode", "add"),
        view_condition_use_role=cfg.get("view_condition_use_role", True),
        view_condition_normalize_moment=cfg.get(
            "view_condition_normalize_moment", False),
        camera_geometry_fp32=cfg.get("camera_geometry_fp32", False),
        rgb_flow_shift=cfg.get("rgb_flow_shift", 5.0),
        rgb_flow_timestep_sampling=cfg.get("rgb_flow_timestep_sampling", "logit_normal"),
        rgb_flow_logit_mean=cfg.get("rgb_flow_logit_mean", 0.0),
        rgb_flow_logit_std=cfg.get("rgb_flow_logit_std", 1.0),
        eval_num_inference_steps=cfg.get("eval_num_inference_steps", 4),
        eval_flow_shift=cfg.get("eval_flow_shift", None),
        eval_sample_seed=cfg.get("eval_sample_seed", 1234),
        eval_lpips_net=cfg.get("eval_lpips_net", "alex"),
        gae_cache_root=cfg.get("gae_cache_root"),
        gae_dataset_name=cfg.get("gae_dataset_name", "bmvs"),
        gae_stats_path=cfg.get("gae_stats_path", None),
        gae_prediction=cfg.get("gae_prediction", "v"),
        gae_rope_frame_axis=cfg.get("gae_rope_frame_axis", "view"),
        gae_auto_flow_shift=cfg.get("gae_auto_flow_shift", True),
        gae_codec_device=cfg.get("gae_codec_device", None),
        gae_rgb_loss_weight=cfg.get("gae_rgb_loss_weight", 0.0),
        gae_rgb_t_max=cfg.get("gae_rgb_t_max", 0.4),
        gae_rgb_max_views=cfg.get("gae_rgb_max_views", 1),
        gae_patch_mode=cfg.get("gae_patch_mode", "native"),
        v_loss_t_eps=cfg.get("v_loss_t_eps", 0.05),
        camera_drop_prob=cfg.get("camera_drop_prob", 0.0),
        ref_drop_prob=cfg.get("ref_drop_prob", 0.0),
        eval_cfg_scale=cfg.get("eval_cfg_scale", 1.0),
        gae_plucker_flip_pe=cfg.get("gae_plucker_flip_pe", False),
        gae_plucker_pe_mlp_hidden=cfg.get("gae_plucker_pe_mlp_hidden", 64),
        gae_plucker_pe_scale=cfg.get("gae_plucker_pe_scale", 1.0),
        gae_plucker_pe_log_scale_aug_prob=cfg.get(
            "gae_plucker_pe_log_scale_aug_prob", 0.3),
        gae_plucker_pe_log_scale_aug_range=cfg.get(
            "gae_plucker_pe_log_scale_aug_range", [-1.2, 1.6]),
    )
    model.set_training_param()

    train_dataset = build_dataset(cfg, split=cfg.get("train_split", None))
    num_views = int(cfg.num_source_views) + int(cfg.num_target_views)
    restrict_views_to_cache(
        train_dataset,
        cfg.get("gae_cache_root"),
        cfg.get("gae_dataset_name", "bmvs"),
        num_views,
        accelerator,
    )
    if cfg.get("num_train_samples", None) is not None:
        train_dataset.records = train_dataset.records[: int(cfg.num_train_samples)]

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.dataloader_num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    eval_max_samples = int(cfg.get("eval_max_samples", 2))
    eval_batch_size = max(1, min(eval_max_samples, int(cfg.get("eval_batch_size", 4))))
    eval_dataloader = None
    if accelerator.is_main_process and eval_max_samples > 0:
        eval_dataset = copy.copy(train_dataset)
        eval_dataset.deterministic_view_seed = int(cfg.get("eval_view_seed", 1234))
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
        accelerator.print(f"Frozen eval set: {len(eval_indices)} samples")

    # Fresh from-scratch layers (GAE patch_embedding / head) get their own LR
    # multiplier: they must learn a foreign 64-ch latent space from zero while
    # the LoRA adapters only retune a pretrained trunk, so a single shared LR
    # starves the new layers.  fresh_layer_lr_mult=1.0 keeps the old behavior.
    fresh_lr_mult = float(cfg.get("fresh_layer_lr_mult", 1.0))
    fresh_params = [p for p in model.pipe.dit.patch_embedding.parameters()
                    if p.requires_grad]
    fresh_params += [p for p in model.pipe.dit.head.parameters()
                     if p.requires_grad]
    # Plücker Flip PE branch: brand-new per-block modules, train alongside the
    # other fresh layers at the fresh LR multiplier.
    fresh_params += list(model.pipe.dit.plucker_pe_parameters())
    fresh_ids = {id(p) for p in fresh_params}
    other_params = [p for p in model.trainable_modules()
                    if id(p) not in fresh_ids]
    param_groups = [{"params": other_params, "lr": cfg.learning_rate}]
    if fresh_params:
        param_groups.append(
            {"params": fresh_params, "lr": cfg.learning_rate * fresh_lr_mult})
    accelerator.print(
        f"[opt] fresh_lr_mult={fresh_lr_mult}  fresh={sum(p.numel() for p in fresh_params)} "
        f"other={sum(p.numel() for p in other_params)}  "
        f"lr={cfg.learning_rate} / {cfg.learning_rate * fresh_lr_mult}")
    # Every trainable parameter must occur exactly once, including at 1x LR.
    grouped_ids = [id(p) for group in param_groups for p in group["params"]]
    expected_ids = {id(p) for p in model.trainable_modules()}
    if len(grouped_ids) != len(set(grouped_ids)) or set(grouped_ids) != expected_ids:
        raise RuntimeError("Optimizer groups omit or duplicate trainable parameters")
    try:
        optimizer = torch.optim.AdamW(param_groups, fused=True)
    except (RuntimeError, ValueError):
        optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    model, optimizer, scheduler, train_dataloader = accelerator.prepare(
        model, optimizer, scheduler, train_dataloader
    )

    if len(train_dataloader) == 0:
        raise ValueError("train_dataloader is empty (dataset too small for world size)")

    if cfg.resume and cfg.training_state_dir not in (None, "na"):
        accelerator.load_state(cfg.training_state_dir)

    global_step = int(cfg.global_step)
    optimizer.zero_grad(set_to_none=True)
    model.train()

    ckpt_manager = CheckpointManager(
        output_path=cfg.output_path,
        max_ckpts=int(cfg.get("max_checkpoints", 5)),
    )

    for epoch in range(cfg.num_epochs):
        for batch in tqdm(train_dataloader, disable=not accelerator.is_main_process):
            if global_step >= cfg.max_steps:
                break

            with accelerator.accumulate(model):
                res = model(batch)
                pred = res["pred"]                      # velocity [B, C, V, h, w]
                rgb_gt = res["rgb_gt"]                  # velocity target
                loss_masks = res["loss_masks"]

                rgb_mask = _resize_mask(
                    loss_masks[:, 1], pred.shape[-2:]).to(pred.dtype)
                rgb_mask = rgb_mask.permute(0, 2, 1, 3, 4)   # [B, 1, V, h, w]
                loss = _masked_mse(pred, rgb_gt, rgb_mask)
                # Per-element weight from the module (w(sigma)=1/max(sigma,eps)^2,
                # the GAE x-prediction equivalent).  With w == 1 this reduces
                # exactly to _masked_mse above.
                w = res.get("weight")
                if torch.is_tensor(w):
                    diff2 = (pred - rgb_gt).float().pow(2) * rgb_mask
                    dims = tuple(range(1, diff2.dim()))
                    num = diff2.sum(dim=dims)                        # [B]
                    denom = rgb_mask.expand_as(diff2).sum().clamp_min(1.0)
                    loss = (num * w.to(num.dtype).reshape(-1)).sum() / denom

                # GAE-style RGB aux loss: module returns the UNWEIGHTED tensor
                # (None on sigma-gated steps); the weight is applied here so it
                # lives in exactly one place -- the config.
                rgb_aux = res.get("rgb_aux")
                rgb_aux_weight = float(cfg.get("gae_rgb_loss_weight", 0.0))
                if rgb_aux is not None and rgb_aux_weight > 0:
                    loss = loss + rgb_aux_weight * rgb_aux

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if cfg.clip_grad_norm:
                        accelerator.clip_grad_norm_(
                            accelerator.unwrap_model(model).trainable_modules(),
                            max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    global_step += 1

                    if global_step % cfg.log_step == 0 and accelerator.is_main_process:
                        lr = scheduler.get_last_lr()[0]
                        aux_str = (
                            f" rgb_aux={float(rgb_aux.detach()):.4f}"
                            if rgb_aux is not None else " rgb_aux=--"
                        )
                        w_str = ""
                        if torch.is_tensor(res.get("weight")):
                            w_str = f" w_mean={float(res['weight'].mean()):.3f}"
                        accelerator.print(
                            f"step {global_step}: loss={loss.item():.6f} "
                            f"sigma_mean={float(res['sigma'].mean()):.3f}{aux_str}{w_str} "
                            f"lr={lr:.8f}"
                        )
                        tb_writer.add_scalar("train/loss", loss.item(), global_step)
                        tb_writer.add_scalar("train/lr", lr, global_step)
                        if torch.is_tensor(res.get("weight")):
                            tb_writer.add_scalar(
                                "train/w_mean", float(res["weight"].mean()), global_step)
                        if rgb_aux is not None:
                            tb_writer.add_scalar(
                                "train/rgb_aux", float(rgb_aux.detach()), global_step)

                    if global_step % cfg.validate_step == 0:
                        ckpt_path = os.path.join(
                            cfg.output_path, f"checkpoint-step-{global_step}")
                        accelerator.save_state(ckpt_path)
                        if accelerator.is_main_process:
                            torch.save({"global_step": global_step},
                                       os.path.join(cfg.output_path, "trainer_state.pt"))
                            eval_model = accelerator.unwrap_model(model)
                            eval_batches = (
                                [batch] if eval_dataloader is None else list(eval_dataloader))
                            chunk_metrics = []
                            for chunk_idx, eval_batch in enumerate(eval_batches):
                                chunk_metrics.append(eval_model.evaluate(
                                    eval_batch,
                                    output_dir=cfg.output_path,
                                    global_step=global_step,
                                    max_samples=eval_batch_size,
                                    save_vis=(chunk_idx < 1),
                                    vis_offset=chunk_idx * eval_batch_size,
                                ))
                            eval_metrics = {}
                            for k in {key for m in chunk_metrics for key in m}:
                                vals = [m[k] for m in chunk_metrics if k in m]
                                eval_metrics[k] = float(np.nanmean(vals))
                            accelerator.print(
                                f"[eval step {global_step}] "
                                + "  ".join(f"{k}={v:.4f}" for k, v in eval_metrics.items())
                            )
                            for k, v in eval_metrics.items():
                                tb_writer.add_scalar(f"eval/{k}", v, global_step)
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
