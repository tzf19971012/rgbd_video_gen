"""How much does each conditioning path actually move the DiT's first token?

CPU-only, weights + real camera geometry, no sampling.  Written after a
magnitude argument about these same numbers produced two OPPOSITE conclusions
in one session (report_20260910_pose_conditioning_magnitude.md §2), because
"how strong is the Plucker path" has no answer until you say what input
distribution you are averaging over.

`patch_embedding` is a Conv3d whose input channels are, in this order
(_build_flow_input's docstring is the authority):

    [ latent | depth (channel layout only) | plucker | warp | modality ]

In the c/D-round configuration that is 16 + 0 + 384 + 17 + 1 = 418.  Each path
is a disjoint slice of the SAME conv, so their contributions to the output token
are directly comparable: run the conv on one slice with the others zeroed and
take the RMS of the result.

THE TRAP.  The 384 Plucker channels are not 384 independent signals.  In
"pixel" mode the field is rasterised at 384x672 and folded down by
PixelUnshuffle(8), so channels p*64 .. p*64+63 are 64 neighbouring PIXELS of
the same physical quantity, and a ray field is smooth over 8 pixels.  Feeding
this path i.i.d. noise therefore understates it by roughly sqrt(64), which is
exactly how one gets "the Plucker path is 8-17x weaker than the warp path"
from a per-channel weight RMS.  This script reports both regimes side by side:

  iid   -- unit-variance Gaussian per channel.  The right model for the VAE
           latent blocks (latent_scale/depth_latent_scale exist precisely to
           make them unit variance, see tools/probe_latent_scale.py) and the
           WRONG model for the Plucker block.
  real  -- the actual field, rasterised from real dataset poses, times the
           trained `channel_scale`.  This is the only honest number for the
           Plucker path.

It also reports, per physical Plucker channel, the coherence of the 64
subposition weight columns

    coherence(S) = || sum_{i in S} c_i || / ( |S| * mean_i ||c_i|| )

where c_i is channel i's weight column flattened over (out, kernel taps).
1.0 means the conv adds the 64 subpositions up (linear superposition, so the
effective weight is 64x the per-channel one); 1/sqrt(|S|) is what independent
random columns would give.  Anything near 1.0 means per-channel weight RMS
cannot be read as path strength -- and, separately, that those 64 channels are
buying ~1 channel's worth of expressiveness, i.e. that
view_condition_plucker_resolution="latent" (6 channels) is a testable
equivalence rather than a downgrade.

  python -m tools.probe_condition_paths \
      --config logs/bmvs_c_sharedrope_s42/args.yaml \
      --checkpoint logs/bmvs_c_sharedrope_s42/checkpoint-step-3500 \
      --num_samples 8
"""
import argparse
import os

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from safetensors.torch import load_file

from diffsynth.models.view_condition import ViewConditionEncoder
from examples.dataset import Kubric4DRGBDNVSDataset
from examples.wanvideo.model_training.WanRGBDNVSTrainingModule import relative_c2w
from utils.projection import intrins_resize

PLUCKER_NAMES = ("moment_x", "moment_y", "moment_z", "dir_x", "dir_y", "dir_z")


def coherence(cols):
    """cols: [n, D] -- one flattened weight column per input channel."""
    norms = cols.norm(dim=1)
    denom = cols.shape[0] * norms.mean()
    if denom <= 0:
        return float("nan")
    return float(cols.sum(dim=0).norm() / denom)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=8,
                        help="Dataset samples used for the real Plucker field statistics. "
                             "The field is a deterministic function of the poses and the "
                             "spread across scenes is small, so 8 is plenty.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    torch.manual_seed(args.seed)

    state = load_file(os.path.join(args.checkpoint, "model.safetensors"), device="cpu")
    W = state["pipe.dit.patch_embedding.weight"].float()      # [1536, Cin, 1, 2, 2]
    channel_scale = state.get("view_condition_encoder.channel_scale")
    out_dim, c_in = W.shape[0], W.shape[1]

    # Rebuild the channel accounting rather than hardcoding 418: a config with
    # token_layout="channel" has a depth block and no modality flag, and reading
    # the plucker slice off by 16 channels would silently compare the wrong thing.
    plucker_ch = 384 if cfg.get("view_condition_plucker_resolution", "latent") == "pixel" else 6
    if cfg.get("view_condition_mode", "add") != "concat":
        plucker_ch = 0
    warp_ch = 17 if cfg.get("warp_mode", "none") == "splat" else 0
    seq = cfg.get("token_layout", "channel") == "sequence"
    base_ch = 16
    depth_ch = 0 if (seq or not cfg.get("depth_flow_matching", False)) else 16
    modality_ch = 1 if seq else 0
    expect = base_ch + depth_ch + plucker_ch + warp_ch + modality_ch
    if expect != c_in:
        raise SystemExit(
            f"channel accounting says {expect} but the checkpoint's conv has {c_in}; "
            "the layout in _build_flow_input must have changed."
        )

    off = 0
    blocks = {}
    for name, n in (("latent", base_ch), ("depth", depth_ch),
                    ("plucker", plucker_ch), ("warp", warp_ch),
                    ("modality", modality_ch)):
        if n:
            blocks[name] = (off, off + n)
        off += n

    print(f"checkpoint: {args.checkpoint}")
    print(f"patch_embedding.weight {tuple(W.shape)}   blocks: "
          + "  ".join(f"{k}[{a}:{b}]" for k, (a, b) in blocks.items()))
    if channel_scale is not None:
        cs = channel_scale.float()
        print(f"channel_scale: mean={cs.mean():.4f} min={cs.min():.4f} max={cs.max():.4f} "
              f"(init 1.0)")

    # ---------------------------------------------------------------- weights --
    print("\n=== per-channel weight magnitude (what a naive reading sees) ===")
    print(f"{'block':<10}{'n':>5}{'RMS(W)':>12}{'mean||col||':>14}{'coherence':>12}{'random':>10}")
    for name, (a, b) in blocks.items():
        sl = W[:, a:b].reshape(out_dim, b - a, -1)
        cols = sl.permute(1, 0, 2).reshape(b - a, -1)
        n = b - a
        print(f"{name:<10}{n:>5}{sl.pow(2).mean().sqrt():>12.6f}"
              f"{cols.norm(dim=1).mean():>14.6f}{coherence(cols):>12.4f}"
              f"{1.0 / n ** 0.5:>10.4f}")

    if "plucker" in blocks and plucker_ch == 384:
        a, _ = blocks["plucker"]
        print("\n=== Plucker: coherence of the 64 subposition columns, per physical channel ===")
        print("PixelUnshuffle(8) puts physical channel p at indices p*64 .. p*64+63.")
        print(f"{'channel':<10}{'mean||col||':>14}{'coherence':>12}{'random':>10}")
        for p in range(6):
            sl = W[:, a + p * 64:a + (p + 1) * 64].reshape(out_dim, 64, -1)
            cols = sl.permute(1, 0, 2).reshape(64, -1)
            print(f"{PLUCKER_NAMES[p]:<10}{cols.norm(dim=1).mean():>14.6f}"
                  f"{coherence(cols):>12.4f}{1.0 / 8:>10.4f}")

        # Effective dimension of what the conv reads out of those 384 channels:
        # participation ratio (sum l)^2 / sum l^2 of the Gram matrix's spectrum.
        cols = W[:, a:a + 384].reshape(out_dim, 384, -1).permute(1, 0, 2).reshape(384, -1)
        if channel_scale is not None:
            cols = cols * channel_scale.float().reshape(384, 1)
        lam = torch.linalg.eigvalsh(cols @ cols.T).clamp_min(0)
        pr = float(lam.sum() ** 2 / lam.pow(2).sum())
        print(f"\neffective dimension of the 384-channel read (participation ratio "
              f"of the Gram spectrum, channel_scale included): {pr:.2f}")
        print(f"top-6 eigenvalue share: {float(lam[-6:].sum() / lam.sum()):.4f}")

    # ------------------------------------------------------- real pose field ---
    enc = ViewConditionEncoder(
        out_channels=16,
        plucker_resolution=cfg.get("view_condition_plucker_resolution", "latent"),
        pixel_scale=cfg.get("view_condition_pixel_scale", 8),
        mode=cfg.get("view_condition_mode", "add"),
        use_role=cfg.get("view_condition_use_role", True),
        normalize_moment=cfg.get("view_condition_normalize_moment", False),
        geometry_fp32=cfg.get("camera_geometry_fp32", False),
    ).eval()

    dataset = Kubric4DRGBDNVSDataset(
        kubric4d_root=cfg.kubric4d_root,
        manifest_path=cfg.manifest_path,
        resolution=tuple(cfg.resolution),
        num_source_views=cfg.num_source_views,
        num_target_views=cfg.num_target_views,
        random_permute_source_views=cfg.random_permute_source_views,
        view_order=cfg.get("view_order", "random"),
        disparity_norm_mode=cfg.get("disparity_norm_mode", "shared_scale"),
        disparity_norm_headroom=cfg.get("disparity_norm_headroom", 1.5),
        deterministic_view_seed=int(cfg.get("eval_view_seed", 1234)),
        min_view_valid_fraction=cfg.get("min_view_valid_fraction", 0.0),
        view_validity_path=cfg.get("view_validity_path", None),
        samples_per_scene=cfg.get("samples_per_scene", 1),
        split_path=cfg.get("split_path", None),
        split=cfg.get("eval_split", None),
    )
    height, width = tuple(cfg.resolution)
    lat_h, lat_w = height // 8, width // 8

    fields = []
    for i in range(min(args.num_samples, len(dataset))):
        s = dataset[i]
        K = s["K"].float().unsqueeze(0)
        c2w = relative_c2w(s["c2w"].float().unsqueeze(0), cfg.get("camera_geometry_fp32", False))
        latent_K = intrins_resize(K.clone(), ratio_H=lat_h / height, ratio_W=lat_w / width)
        with torch.no_grad():
            fields.append(enc.plucker_field(
                latent_K, c2w, latent_height=lat_h, latent_width=lat_w,
                pixel_K=K, pixel_height=height, pixel_width=width,
            ))
    field = torch.cat(fields, dim=0)          # [N, V, C, h, w]
    print(f"\n=== real Plucker field over {field.shape[0]} samples x {field.shape[1]} views ===")
    if plucker_ch == 384:
        per_phys = field.reshape(-1, 6, 64, lat_h, lat_w)
        print(f"{'channel':<10}{'mean':>12}{'std':>12}{'rms':>12}{'subpos corr':>14}")
        for p in range(6):
            x = per_phys[:, p]                                  # [NV, 64, h, w]
            flat = x.permute(1, 0, 2, 3).reshape(64, -1)
            c = torch.corrcoef(flat)
            off_diag = (c.sum() - c.diagonal().sum()) / (64 * 63)
            print(f"{PLUCKER_NAMES[p]:<10}{x.mean():>12.4f}{x.std():>12.4f}"
                  f"{x.pow(2).mean().sqrt():>12.4f}{float(off_diag):>14.4f}")

    # ---------------------------------------------------- path contributions ---
    # Run the conv on one block at a time with the rest zeroed, and take the RMS
    # of the output.  Same conv, same taps, so the numbers are commensurate.
    print("\n=== contribution to the output token (RMS over 1536 dims) ===")
    print("iid  = unit-variance Gaussian input; right for VAE latents, wrong for Plucker.")
    print("real = the actual field x channel_scale; the only honest number for Plucker.")
    print(f"{'block':<10}{'n':>5}{'iid':>12}{'real':>12}")

    def conv_rms(block_input, a, b):
        w = W[:, a:b]
        out = F.conv3d(block_input, w, stride=(1, 2, 2))
        return float(out.pow(2).mean().sqrt())

    nv = 4
    for name, (a, b) in blocks.items():
        x = torch.randn(1, b - a, nv, lat_h, lat_w)
        iid = conv_rms(x, a, b)
        real = ""
        if name == "plucker":
            f = field.reshape(-1, plucker_ch, lat_h, lat_w)[:nv]
            f = f.permute(1, 0, 2, 3).unsqueeze(0)              # [1, C, V, h, w]
            if channel_scale is not None:
                f = f * channel_scale.float().reshape(1, -1, 1, 1, 1)
            real = f"{conv_rms(f, a, b):>12.4f}"
        print(f"{name:<10}{b - a:>5}{iid:>12.4f}{real:>12}")

    print("\nRead this table with report_20260910_pose_conditioning_magnitude.md §2: "
          "neither column predicts behaviour.  Only tools/diag_pose_sensitivity.py does.")


if __name__ == "__main__":
    main()
