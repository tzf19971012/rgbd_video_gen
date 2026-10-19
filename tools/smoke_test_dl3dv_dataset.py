"""Smoke test: load DL3DV records through Kubric4DRGBDNVSDataset and print shapes."""
import sys
sys.path.insert(0, "/mi/sfs_turbo/t00845730/9_22_code/DVD-main/DVD-main")
import torch

from examples.dataset.kubric4d_rgbd_nvs_dataset import Kubric4DRGBDNVSDataset

ds = Kubric4DRGBDNVSDataset(
    kubric4d_root="/mi/data2T/y00959808/data/sample_data/DL3DV_WAI",
    manifest_path="/mi/sfs_turbo/t00845730/dl3dv/manifest.jsonl",
    resolution=(384, 672),
    num_source_views=5,
    num_target_views=1,
    random_permute_source_views=False,
    view_order="random",
    disparity_norm_mode="shared_scale",
    disparity_norm_quantile=0.98,
    disparity_norm_headroom=1.0,
    min_view_valid_fraction=0.5,
    samples_per_scene=8,
)
print(f"dataset length: {len(ds)}")
print(f"records: {[len(r['views']) for r in ds.records]}")

for i in (0, 7):
    s = ds[i]
    print(f"\n--- sample {i} (scene {s['scene_id'][:16]}) ---")
    for k in ("images", "disparity", "depth_metric", "K", "c2w", "view_roles", "loss_masks"):
        v = s[k]
        print(f"  {k}: {tuple(v.shape)} {v.dtype} min={float(v.min()):.4f} max={float(v.max()):.4f}")
    print(f"  depth_norm_scale: {float(s['depth_norm_scale']):.4f}")
    print(f"  source_images: {tuple(s['source_images'].shape)}, target_images: {tuple(s['target_images'].shape)}")

print("\nSMOKE TEST OK")
