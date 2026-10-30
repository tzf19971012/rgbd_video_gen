"""Measure the scale of the VAE latents actually used by RGB-D NVS training.

The rectified-flow loss is only interpretable relative to the second moment of
its target ``v = noise - x0``:  ``E[v^2] = 1 + E[x0^2]``.  And the noise
schedule itself assumes ``x0`` is roughly unit-variance -- that is exactly what
the Wan VAE's per-channel ``(mean, 1/std)`` normalisation is calibrated for,
*for inputs in [-1, 1]*.

``WanRGBDNVSTrainingModule._encode_views`` calls ``pipe.vae.encode`` directly on
the dataset tensors, which live in [0, 1]; the pipeline's own inference path
goes through ``pipe.preprocess_video`` first, which maps [0, 1] -> [-1, 1].
This script measures how much that costs.

CPU-only, so keep ``--num-views`` small.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.dataset.kubric4d_rgbd_nvs_dataset import Kubric4DRGBDNVSDataset  # noqa: E402


def _stats(name, lat):
    m2 = lat.pow(2).mean().item()
    print(
        f"  {name:34s} mean={lat.mean().item():+.4f}  std={lat.std().item():.4f}  "
        f"E[x0^2]={m2:.4f}  ->  E[v^2]={1 + m2:.4f}"
    )
    return m2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae", required=True, help="path to Wan2.1_VAE.pth")
    parser.add_argument("--num-scenes", type=int, default=2)
    parser.add_argument("--num-views", type=int, default=2)
    args = parser.parse_args()

    data_root = os.environ["DVD_DATA_ROOT"]
    root = f"{data_root}/datasets/BlendedMVS/BlendedMVS"
    manifest = f"{data_root}/datasets/BlendedMVS/manifest.jsonl"

    from diffsynth.models.wan_video_vae import WanVideoVAE

    vae = WanVideoVAE()
    state = torch.load(args.vae, map_location="cpu", weights_only=True)
    missing, unexpected = vae.model.load_state_dict(state, strict=False)
    print(f"VAE loaded: missing={len(missing)} unexpected={len(unexpected)}")
    vae = vae.eval()

    dataset = Kubric4DRGBDNVSDataset(
        kubric4d_root=root,
        manifest_path=manifest,
        resolution=(384, 672),
        num_source_views=4,
        num_target_views=3,
        disparity_norm_mode="shared_scale",
        disparity_norm_headroom=1.5,
        min_view_valid_fraction=0.5,
        samples_per_scene=1,
    )
    dataset.deterministic_view_seed = 1234
    print(f"dataset: {len(dataset.records)} scenes")

    acc = {}
    stride = max(1, len(dataset.records) // args.num_scenes)
    scene_indices = list(range(0, len(dataset.records), stride))[: args.num_scenes]
    for scene_idx in scene_indices:
        sample = dataset[scene_idx]
        print(f"\nscene index {scene_idx} ({sample['sample_id']})")
        for key in ("images", "disparity"):
            views = sample[key][: args.num_views]  # [v, 3, H, W] in [0, 1]
            print(f" {key}: pixel range [{views.min():.3f}, {views.max():.3f}]")
            for tag, frames in (
                ("as-trained  [0,1]", views),
                ("pipeline    [-1,1]", views * 2.0 - 1.0),
            ):
                with torch.no_grad():
                    lat = vae.encode(
                        [frames[v].unsqueeze(1) for v in range(frames.shape[0])],
                        device="cpu",
                    )
                m2 = _stats(f"{key} {tag}", lat)
                acc.setdefault((key, tag), []).append(m2)

    print("\n=== summary (mean E[x0^2] over scenes) ===")
    for (key, tag), vals in acc.items():
        m2 = sum(vals) / len(vals)
        print(
            f"{key:10s} {tag:20s} E[x0^2]={m2:.4f}  "
            f"E[v^2]={1 + m2:.4f}   (loss of the trivial v_hat=0 predictor)"
        )


if __name__ == "__main__":
    main()
