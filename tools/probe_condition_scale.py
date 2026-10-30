"""Measure the magnitude and the cross-view DISCRIMINABILITY of the camera
conditioning that the target views actually receive.

Why this matters more than the latent scale:

``_build_flow_input`` sets, for the target slots,

    model_in[:, :, s:] = noisy_rgb[:, :, s:] + view_condition[:, :, s:]

At the median training sigma (0.833 with shift=5, logit_mean=0) the noisy RGB
latent has std ~0.83.  The *only* per-view information distinguishing target
view 5 from target view 6 is ``view_condition``.  Two things can therefore kill
target-view synthesis outright, and neither shows up in the loss:

  1. MAGNITUDE -- if ``view_condition`` is tiny next to the noise it is added
     to, the pose signal is below the model's effective resolution.  The final
     conv is zero-initialised, so this quantity starts at exactly 0 and has to
     grow on its own.
  2. DISCRIMINABILITY -- if ``view_condition`` is nearly the SAME tensor for
     every target view, the model is being asked to render three different
     viewpoints from three identical prompts.  No amount of training fixes
     that; the task is not well posed.  A large overall std does not rule this
     out: a big constant offset (e.g. driven by the role embedding, which is
     spatially constant and identical for all target views) inflates the std
     while carrying zero per-view information.

So we decompose, over the target slots only,

    total variance = across-view variance + within-view variance

and report the across-view fraction.  A role-embedding-dominated encoder scores
near 0.

Only ``view_condition_encoder.*`` is pulled out of the checkpoint (63k params),
lazily via ``safe_open`` -- loading the full 3.4 GB state dict gets the process
killed on the login node.  CPU-only.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffsynth.models.view_condition import ViewConditionEncoder  # noqa: E402
from examples.dataset.kubric4d_rgbd_nvs_dataset import Kubric4DRGBDNVSDataset  # noqa: E402
from utils.projection import intrins_resize  # noqa: E402


def load_encoder_weights(checkpoint, encoder):
    """Pull only view_condition_encoder.* out of the safetensors shard."""
    from safetensors import safe_open

    path = os.path.join(checkpoint, "model.safetensors")
    prefix = "view_condition_encoder."
    state = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith(prefix):
                state[key[len(prefix):]] = f.get_tensor(key)
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    print(f"view_condition_encoder: loaded {len(state)} tensors, "
          f"missing={len(missing)}, unexpected={len(unexpected)}")
    if len(state) == 0:
        raise RuntimeError(f"no '{prefix}*' keys found in {path}")
    return encoder


def decompose(x):
    """Split [B, C, V, h, w] into the component SHARED by all views and the
    component that DIFFERS between views.

    ``x - x.mean(dim=2)`` is each view's deviation from the view-average, so its
    second moment is the view-discriminative variance; whatever is left of the
    total is common to every view and carries no per-view information.
    """
    view_mean = x.mean(dim=2, keepdim=True)
    var_differing = (x - view_mean).pow(2).mean()
    var_total = (x - x.mean()).pow(2).mean()
    var_shared = (var_total - var_differing).clamp_min(0.0)
    return var_shared.item(), var_differing.item()


def frac(shared, differing):
    """Discriminative fraction; nan when the encoder output is identically zero
    (the zero-initialised final conv, i.e. before any training)."""
    total = shared + differing
    return differing / total if total > 0.0 else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None,
                        help="accelerate save_state dir. Omit to measure the "
                             "zero-initialised encoder (sanity: should give 0).")
    parser.add_argument("--num-scenes", type=int, default=4)
    parser.add_argument("--num-source-views", type=int, default=4)
    parser.add_argument("--num-target-views", type=int, default=3)
    parser.add_argument("--mode", default="add", choices=["add", "concat"],
                        help="must match the arm's view_condition_mode, or the "
                             "state dict will not fit the module")
    parser.add_argument("--use-role", default="true", choices=["true", "false"])
    parser.add_argument("--normalize-moment", default="false",
                        choices=["true", "false"])
    parser.add_argument("--median-sigma", type=float, default=0.833,
                        help="median training sigma, for the SNR line")
    parser.add_argument("--latent-std", type=float, default=0.472,
                        help="measured RGB latent std (probe_latent_scale.py)")
    args = parser.parse_args()

    data_root = os.environ["DVD_DATA_ROOT"]
    root = f"{data_root}/datasets/BlendedMVS/BlendedMVS"
    manifest = f"{data_root}/datasets/BlendedMVS/manifest.jsonl"

    height, width = 384, 672
    latent_h, latent_w = height // 8, width // 8

    encoder = ViewConditionEncoder(
        out_channels=16,
        role_dim=8,
        hidden_channels=64,
        plucker_resolution="pixel",
        pixel_scale=8,
        mode=args.mode,
        use_role=args.use_role == "true",
        normalize_moment=args.normalize_moment == "true",
    ).eval()
    print(f"probing arm: mode={args.mode} use_role={args.use_role} "
          f"normalize_moment={args.normalize_moment} "
          f"out_channels={encoder.out_channels}")
    if args.checkpoint:
        load_encoder_weights(args.checkpoint, encoder)
    else:
        print("view_condition_encoder: RANDOM INIT (final conv zeroed)")

    dataset = Kubric4DRGBDNVSDataset(
        kubric4d_root=root,
        manifest_path=manifest,
        resolution=(height, width),
        num_source_views=args.num_source_views,
        num_target_views=args.num_target_views,
        disparity_norm_mode="shared_scale",
        disparity_norm_headroom=1.5,
        min_view_valid_fraction=0.5,
        samples_per_scene=1,
    )
    dataset.deterministic_view_seed = 1234

    s = args.num_source_views
    stride = max(1, len(dataset.records) // args.num_scenes)
    scene_indices = list(range(0, len(dataset.records), stride))[: args.num_scenes]

    agg = {"src": [], "tgt": [], "shared": [], "differing": [],
           "baseline": [], "raw_shared": [], "raw_differing": []}
    for scene_idx in scene_indices:
        sample = dataset[scene_idx]
        K = sample["K"].unsqueeze(0).float()
        c2w = sample["c2w"].unsqueeze(0).float()
        roles = sample["view_roles"].unsqueeze(0)

        c2w_rel = torch.linalg.inv(c2w[:, 0:1]) @ c2w
        latent_K = intrins_resize(K.clone(), ratio_H=latent_h / height,
                                  ratio_W=latent_w / width)

        with torch.no_grad():
            vc = encoder(latent_K, c2w_rel, roles,
                         latent_height=latent_h, latent_width=latent_w,
                         pixel_K=K, pixel_height=height, pixel_width=width)
            # CONTROL: the raw Plucker field the encoder is fed.  If the raw
            # field already fails to distinguish the target views then the
            # problem is upstream (pose normalisation); if it distinguishes
            # them and the encoder output does not, the encoder collapsed it.
            raw = encoder._plucker(K, c2w_rel, height, width)   # [B,V,6,H,W]
            raw = raw.permute(0, 2, 1, 3, 4)                    # [B,6,V,H,W]

        # vc: [B, C, V, h, w]
        src = vc[:, :, :s]
        tgt = vc[:, :, s:]
        shared, differing = decompose(tgt)
        raw_shared, raw_differing = decompose(raw[:, :, s:])

        # Scale reference: what the conditioning is being ADDED to.
        noisy_std = ((1 - args.median_sigma) ** 2 * args.latent_std ** 2
                     + args.median_sigma ** 2) ** 0.5

        print(f"\nscene {scene_idx} ({sample['sample_id']})")
        print(f"  view_condition std   source={src.std().item():.5f}   "
              f"target={tgt.std().item():.5f}")
        print(f"  target var  shared-by-all-views={shared:.3e}   "
              f"differing-between-views={differing:.3e}   "
              f"discriminative fraction={frac(shared, differing):.5f}")
        print(f"  raw plucker discriminative fraction="
              f"{frac(raw_shared, raw_differing):.5f}   "
              f"(control: is the INPUT view-distinctive?)")

        agg["src"].append(src.std().item())
        agg["tgt"].append(tgt.std().item())
        agg["shared"].append(shared)
        agg["differing"].append(differing)
        agg["raw_shared"].append(raw_shared)
        agg["raw_differing"].append(raw_differing)
        agg["baseline"].append(noisy_std)

    n = len(agg["tgt"])
    mean = lambda k: sum(agg[k]) / n  # noqa: E731
    shared, differing = mean("shared"), mean("differing")
    raw_frac = frac(mean("raw_shared"), mean("raw_differing"))
    base = mean("baseline")
    print("\n=== summary ===")
    print(f"  scenes                                 {n}")
    print(f"  view_condition std (source)            {mean('src'):.5f}")
    print(f"  view_condition std (target)            {mean('tgt'):.5f}")
    print(f"    source/target magnitude ratio        {mean('src') / mean('tgt'):.2f}x"
          if mean("tgt") > 0 else "    source/target magnitude ratio        n/a")
    if args.mode == "add":
        print(f"  std of the latent it is added to       {base:.5f}")
        print(f"  overall pose SNR (target)              {mean('tgt') / base:.5f}")
    print("  --- target-view conditioning decomposition ---")
    print(f"  variance shared by all target views    {shared:.3e}")
    print(f"  variance DIFFERING between views       {differing:.3e}")
    print(f"  view-discriminative fraction           {frac(shared, differing):.5f}")
    print(f"    -> view-discriminative std           {differing ** 0.5:.6f}")
    if args.mode == "add":
        print(f"    -> EFFECTIVE per-view SNR            {differing ** 0.5 / base:.6f}")
    print(f"  CONTROL raw plucker discr. fraction    {raw_frac:.5f}")
    if args.mode == "add":
        print("\n  'EFFECTIVE per-view SNR' is the fraction of the target-slot input")
        print("  that actually says WHICH target view to render.  If it is ~0 the")
        print("  three target views are near-identical prompts and the task is")
        print("  degenerate no matter how long training runs.")
    else:
        print("\n  concat mode: the conditioning is NOT added to the noisy latent,")
        print("  it is a separate patch_embedding input block, so there is no")
        print("  additive SNR to report.  The discriminative fraction should sit")
        print("  at the raw-plucker control value (no CNN to collapse it); the")
        print("  only way it can drop is if channel_scale zeroes out channels.")


if __name__ == "__main__":
    main()
