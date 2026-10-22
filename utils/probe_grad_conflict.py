"""Measure RGB-vs-depth gradient conflict on the shared DiT LoRA parameters.

For each probe batch: one forward pass, then torch.autograd.grad for
depth_loss and rgb_loss SEPARATELY against every trainable (LoRA) parameter.
The cosine between the two flattened gradient vectors is the classic
multi-task conflict signal:  ~+1 = the two tasks push the shared parameters
the same way (synergy),  ~0 = independent,  <0 = fighting over capacity.

Also reports the cosine bucketed by DiT block quartile so a localised
conflict (e.g. only late blocks near the output heads) is visible.

Usage (single NPU, ~model-load time + a few seconds per batch):
    python utils/probe_grad_conflict.py \
        --config logs/<run>/args.yaml \
        --ckpt logs/<run>/checkpoint-step-1350 \
        --num-batches 8
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "examples", "wanvideo", "model_training"))

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from omegaconf import OmegaConf

from examples.dataset import Kubric4DRGBDNVSDataset
from examples.wanvideo.model_training import train_with_accelerate_rgbd_nvs as train_script


def bucket_of(name, num_blocks):
    if ".blocks." in name:
        idx = int(name.split(".blocks.")[1].split(".")[0])
        q = 4 * idx // max(num_blocks, 1)
        return f"blocks_q{q}"
    if "head" in name.lower():
        return "heads:" + ("rgb" if "rgb" in name.lower() else ("depth" if "depth" in name.lower() else "other"))
    return "embed_other"


def flat_cos(g_d, g_r):
    d = g_d.float().flatten()
    r = g_r.float().flatten()
    n = d.norm()
    m = r.norm()
    if n < 1e-12 or m < 1e-12:
        return None
    return float(torch.nn.functional.cosine_similarity(d, r, dim=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True, help="accelerate checkpoint dir with model.safetensors")
    parser.add_argument("--num-batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--fullft", action="store_true",
                        help="probe ALL requires_grad params (default: LoRA lora_A/lora_B only)")
    args = parser.parse_args()

    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device
    cfg = OmegaConf.load(args.config)

    # Same dataset protocol as the run being probed.
    dataset = Kubric4DRGBDNVSDataset(
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
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=2,
        collate_fn=train_script.custom_collate_fn,
    )

    print("[probe] building model (14B load, takes a while) ...")
    model = train_script.WanRGBDNVSTrainingModule(
        args=cfg,
        accelerator=accelerator,
        model_id_with_origin_paths=cfg.model_id_with_origin_paths,
        trainable_models=cfg.trainable_models,
        lora_rank=cfg.lora_rank,
        lora_base_model=cfg.lora_base_model,
        lora_target_modules=cfg.lora_target_modules,
        local_model_path=cfg.get("wan_model_root", "./models"),
        skip_download=cfg.get("skip_download", False),
        strip_i2v_image_input=cfg.get("strip_i2v_image_input", False),
        use_gradient_checkpointing=False,   # double-backward needs a real graph
        use_gradient_checkpointing_offload=False,
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
        warp_dropout_prob=0.0,              # deterministic probe forward
        warp_two_pass=cfg.get("warp_two_pass", True),
        warp_perturb_prob=0.0,
        depth_warp_block=cfg.get("depth_warp_block", False),
        depth_consist_scale_weight=0.0,
        depth_consist_repro_weight=0.0,
        per_frame_timestep=cfg.get("per_frame_timestep", False),
        clean_source_depth=False,
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
        eval_num_inference_steps=4,
        eval_flow_shift=None,
        eval_sample_seed=1234,
        eval_lpips_net="alex",
        eval_vis_downscale=2,
        eval_vis_format="jpg",
        eval_vis_depth_norm="shared",
    )
    model.set_training_param()

    from safetensors.torch import load_file
    state = load_file(os.path.join(args.ckpt, "model.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[probe] loaded ckpt: matched={len(state)}, missing={len(missing)}, unexpected={len(unexpected)}")
    if unexpected:
        print(f"[probe] WARNING unexpected keys e.g. {list(unexpected)[:3]}")

    model.to(device)
    model.train()

    names, params = zip(*[(n, p) for n, p in model.named_parameters()
                          if p.requires_grad and (args.fullft or ("lora_" in n or "lora" in n.lower()))])
    if not params:
        raise RuntimeError("no trainable params matched the lora filter")
    num_blocks = sum(1 for n in names if f".blocks." in n and n.endswith("lora_A.default.weight"))
    buckets = sorted({bucket_of(n, num_blocks) for n in names})
    print(f"[probe] {len(params)} trainable params, buckets: {buckets}")

    accum = {b: [] for b in buckets}
    gaccum = []
    it = iter(dataloader)
    for step in range(args.num_batches):
        torch.manual_seed(args.seed + step)
        try:
            batch = next(it)
        except StopIteration:
            break
        res = model(batch)
        pred_depth, pred_rgb = res["pred"]
        depth_mask = train_script._resize_mask(
            res["loss_masks"][:, 0], pred_depth.shape[-2:]).to(pred_depth.dtype).permute(0, 2, 1, 3, 4)
        rgb_mask = train_script._resize_mask(
            res["loss_masks"][:, 1], pred_rgb.shape[-2:]).to(pred_rgb.dtype).permute(0, 2, 1, 3, 4)
        depth_loss = train_script._masked_mse(pred_depth, res["depth_gt"], depth_mask)
        rgb_loss = train_script._masked_mse(pred_rgb, res["rgb_gt"], rgb_mask)

        g_d = torch.autograd.grad(depth_loss, params, retain_graph=True, allow_unused=True)
        g_r = torch.autograd.grad(rgb_loss, params, allow_unused=True)

        per_bucket = {b: [] for b in buckets}
        for n, gd, gr in zip(names, g_d, g_r):
            if gd is None or gr is None:
                continue
            c = flat_cos(gd, gr)
            if c is None:
                continue
            per_bucket[bucket_of(n, num_blocks)].append(c)
        g_all_d = torch.cat([g.flatten().float() for g in g_d if g is not None])
        g_all_r = torch.cat([g.flatten().float() for g in g_r if g is not None])
        gcos = float(F.cosine_similarity(g_all_d, g_all_r, dim=0))

        line = f"batch {step}: depth={depth_loss.item():.4f} rgb={rgb_loss.item():.4f} | global cos={gcos:+.3f} |"
        for b in buckets:
            vals = per_bucket[b]
            if vals:
                m = sum(vals) / len(vals)
                accum[b].append(m)
                line += f" {b}={m:+.3f}"
        gaccum.append(gcos)
        print(line, flush=True)

    print("\n==== summary over", len(gaccum), "batches ====")
    print(f"global grad cos: mean={sum(gaccum)/len(gaccum):+.3f}  "
          f"neg_batches={sum(1 for c in gaccum if c < 0)}/{len(gaccum)}")
    for b in buckets:
        if accum[b]:
            v = accum[b]
            print(f"{b:<14} mean={sum(v)/len(v):+.3f}  neg={sum(1 for x in v if x < 0)}/{len(v)}")
    print("\nread: cos ~ +1 synergy, ~ 0 independent, < 0 conflict "
          "(PCGrad-style evidence that depth supervision fights RGB on shared capacity)")


if __name__ == "__main__":
    main()
