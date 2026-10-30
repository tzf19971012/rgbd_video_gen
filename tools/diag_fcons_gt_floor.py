"""What does the f-consist loss score on GROUND-TRUTH disparity?

The question this answers, and why it was never asked.  tools/test_f_sweep_factors.py
asserts `a_consistent_plane_scores_zero_on_both_terms` -- but "consistent" there
means a synthetic plane whose depth is computed analytically from the same K and
c2w the loss then reads.  That test can only catch a bug in the loss's own
algebra.  It cannot catch the loss having a NONZERO floor on real data, because
there is no real data in it.

If GT scores far from zero, the loss's minimum is not the truth, and every
gradient step that lowers it past the floor must move the prediction AWAY from
the ground truth.  That would explain both f-round arms doing clean damage
(report_20260916_pass1_prior_diag.md §12) without any bug in the algebra, and it
would explain the asymmetry: `scale` is an UNTRIMMED least-squares fit while
`repro` is trimmed at q=0.9 and Charbonnier'd, so whatever contaminates the data
(occlusion surviving the loose ratio_clip=4.0, BMVS's MVS-reconstructed depth
noise, the [0,1] disparity clamp) lands on `scale` and is largely filtered out of
`repro`.

Everything here is GT, on the CPU or one GPU, no model and no checkpoint.  The
only thing loaded from a run directory is args.yaml, so the dataset is built with
the same normalisation the training run used.

  python -m tools.diag_fcons_gt_floor --run logs/bmvs_f_fconsS_s42_ft4000 -n 32
"""
import argparse
import json
import os
import statistics

import torch
from omegaconf import OmegaConf

from tools.vis_rgb_warp_pred_depth import build_dataset
from utils.depth_consistency import flatness, ordered_pairs, pairwise_consistency


def _lowpass(disp, factor):
    """Area-downsample by `factor` and bilinear back up: a stand-in for the VAE's
    spatial bottleneck, applied to the SAME field for every view so it cannot
    manufacture a cross-view disagreement out of nothing."""
    x = disp.unsqueeze(1)
    h, w = x.shape[-2:]
    small = torch.nn.functional.interpolate(
        x, size=(max(1, h // factor), max(1, w // factor)), mode="area"
    )
    up = torch.nn.functional.interpolate(
        small, size=(h, w), mode="bilinear", align_corners=False
    )
    return up.squeeze(1).clamp(0.0, 1.0)


def grad_profile(disp, valid, K, c2w, pairs, kw, nbins=5):
    """Where does each term's gradient LIVE, as a function of disparity?

    The hypothesis this tests: `depth = 1/disp.clamp_min(1e-3)` gives the map from
    disparity to the geometry a 1/disp^2 Jacobian, so the FAR tail dominates the
    gradient however little it contributes to the loss value -- and `repro`'s
    q=0.9 trim discards exactly those pixels while `scale`'s plain least squares
    does not.  Returns, per term, the mean |grad| per disparity quintile over
    valid pixels and the share of total gradient mass in the lowest quintile.
    """
    out = {}
    for term in ("scale", "repro"):
        d = disp.detach().clone().requires_grad_(True)
        got = pairwise_consistency(d, valid, K, c2w, pairs, **kw)
        if got is None:
            return None
        val = got[term]
        if not val.requires_grad:
            continue
        val.backward()
        g = d.grad.abs()[valid]
        vals = disp.detach()[valid]
        edges = torch.quantile(
            vals.float(), torch.linspace(0, 1, nbins + 1, device=vals.device)
        )
        per_bin, mass = [], []
        for b in range(nbins):
            lo, hi = edges[b], edges[b + 1]
            sel = (vals >= lo) & (vals <= hi if b == nbins - 1 else vals < hi)
            if int(sel.sum()) == 0:
                per_bin.append(float("nan"))
                mass.append(0.0)
                continue
            per_bin.append(float(g[sel].mean()))
            mass.append(float(g[sel].sum()))
        total = sum(mass) or 1.0
        out[term] = {
            "per_bin": per_bin,
            "edges": [float(e) for e in edges],
            "low_bin_mass_share": mass[0] / total,
            "low_over_high": (per_bin[0] / per_bin[-1]) if per_bin[-1] else float("nan"),
            "grad_norm": float(d.grad.norm()),
        }
    return out


def summarise(values):
    vals = [v for v in values if v == v]
    if not vals:
        return {"n": 0}
    out = {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "median": statistics.median(vals),
        "p90": sorted(vals)[int(0.9 * (len(vals) - 1))],
        "max": max(vals),
    }
    if len(vals) > 1:
        out["std"] = statistics.stdev(vals)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="a run dir, read only for args.yaml")
    ap.add_argument("-n", "--num-samples", type=int, default=32)
    ap.add_argument("--split", default=None,
                    help="override cfg.eval_split; 'train' to match training")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num-views", type=int, default=3,
                    help="k, as depth_consist_num_views; 0 means all source views")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vae", action="store_true",
                    help="add the real VAE round-trip arm (needs a GPU)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = OmegaConf.load(os.path.join(args.run, "args.yaml"))
    OmegaConf.resolve(cfg)
    if args.split is not None:
        cfg.eval_split = args.split
    dataset = build_dataset(cfg)
    num_src = int(cfg.num_source_views)
    k = num_src if args.num_views == 0 else min(args.num_views, num_src)

    # Exactly the training kwargs (WanRGBDNVSTrainingModule.depth_consist_kwargs),
    # plus two variants of the occlusion filter to test whether the floor IS
    # occlusion.  ratio_clip=4.0 is the shipped value.
    base = dict(
        min_support=float(cfg.get("depth_consist_min_support", 0.02)),
        ratio_clip=float(cfg.get("depth_consist_ratio_clip", 4.0)),
        trim_quantile=float(cfg.get("depth_consist_trim_quantile", 0.9)),
    )
    # Each arm is (pipeline applied to the GT disparity, pairwise kwargs).  The
    # clip* arms test whether the floor is occlusion contamination surviving the
    # loose ratio_clip.  The lp* arms exist because the disparity the LOSS sees is
    # not GT: it is GT after a round trip through the frozen VAE, whose spatial
    # bottleneck is 8x.  A model that predicted the depth latent PERFECTLY would
    # still be scored on the decode.  An 8x area-down/bilinear-up is a cheap CPU
    # proxy for that bottleneck; --vae measures the real thing.
    arms = {
        "shipped": (0, dict(base)),
        "clip1.5": (0, dict(base, ratio_clip=1.5)),
        "clip1.2": (0, dict(base, ratio_clip=1.2)),
        "lp8": (8, dict(base)),
        "lp16": (16, dict(base)),
    }

    w_scale = float(cfg.get("depth_consist_scale_weight", 0.0)) or 0.25
    w_repro = float(cfg.get("depth_consist_repro_weight", 0.0)) or 0.1

    # The decisive arm.  A model that predicted the depth LATENT perfectly is
    # still scored on `vae.decode` of it, so this is the floor the loss actually
    # imposes on the network -- and the network cannot lower it, because the VAE
    # is frozen.  The encode/decode path and the `disp > 1e-3` valid mask are the
    # training module's own, not a reimplementation.
    model = None
    if args.vae:
        from accelerate import Accelerator
        from tools.vis_rgb_warp_pred_depth import build_model
        model = build_model(cfg, Accelerator())
        arms["vae_rt"] = (0, dict(base))
        # Everything else in this script is placed on args.device; the module puts
        # its pipe wherever the Accelerator decided.  Follow the pipe rather than
        # guessing, or the round-trip arm compares a cuda tensor to a cpu one.
        args.device = model.pipe.device

    gen = torch.Generator().manual_seed(args.seed)
    acc = {name: {key: [] for key in
                  ("scale", "repro", "scale_dev", "n_pairs", "support", "flat", "loss")}
           for name in arms}
    per_pair_dev = {name: [] for name in arms}
    n_none = {name: 0 for name in arms}
    rows = []

    n = min(args.num_samples, len(dataset))
    for idx in range(n):
        item = dataset[idx]
        # [S, H, W]: the three disparity channels are identical by construction.
        disp = item["disparity"][:num_src, 0].to(args.device).float()
        valid = (item["loss_masks"][0, :num_src, 0] > 0.5).to(args.device)
        K = item["K"][:num_src].to(args.device).float()
        c2w = item["c2w"][:num_src].to(args.device).float()

        views = torch.randperm(num_src, generator=gen)[:k].tolist()
        d, v = disp[views], valid[views]
        row = {"idx": idx, "sample_id": item["sample_id"], "views": views,
               "flat": flatness(d, v),
               "disp_mean": float(d[v].mean()), "disp_min": float(d[v].min()),
               "disp_p01": float(torch.quantile(d[v].flatten().float(), 0.01)),
               "valid_frac": float(v.float().mean())}
        rt = None
        if model is not None:
            with torch.no_grad():
                lat, _ = model._encode_views(
                    item["disparity"][:num_src].unsqueeze(0).to(
                        model.pipe.device, dtype=model.pipe.torch_dtype
                    )
                )
                lat = lat * model.depth_latent_scale
                rt = model._decode_src_disparity_with_grad(
                    lat[:, :, views]
                ).float().to(d.device)
            row["vae_disp_l1"] = float((rt - d).abs()[v].mean())
            row["vae_flat"] = flatness(rt, v)

        for name, (factor, kw) in arms.items():
            if name == "vae_rt":
                # The valid mask is the training module's: dataset mask AND the
                # decoded disparity clearing 1e-3.
                dd, vv = rt, v & (rt > 1e-3)
            else:
                dd, vv = (_lowpass(d, factor) if factor else d), v
            got = pairwise_consistency(
                dd, vv, K[views], c2w[views], ordered_pairs(range(k)), **kw
            )
            if got is None:
                n_none[name] += 1
                continue
            loss = w_scale * float(got["scale"]) + w_repro * float(got["repro"])
            for key in ("scale", "repro", "scale_dev", "n_pairs", "support"):
                acc[name][key].append(float(got[key]))
            acc[name]["loss"].append(loss)
            acc[name]["flat"].append(row["flat"])
            row[name] = {key: float(got[key]) for key in
                         ("scale", "repro", "scale_dev", "n_pairs", "support")}
            row[name]["loss"] = loss
            if name in ("shipped", "lp8", "vae_rt"):
                prof = grad_profile(
                    dd, vv, K[views], c2w[views], ordered_pairs(range(k)), kw
                )
                if prof:
                    row[name]["grad"] = prof
        rows.append(row)

    print(f"\ndataset: {cfg.get('eval_split')!r} split, {len(dataset)} samples, "
          f"read {n};  k={k} of {num_src} source views")
    print(f"loss weights read from args.yaml: scale={w_scale} repro={w_repro}")
    print("\nGT DISPARITY, the loss's floor.  Training-time measured levels on the")
    print("model's own x0_hat were |a-1| 0.0518 and repro 0.0321 (report §12a).")
    for name in arms:
        s = acc[name]
        print(f"\n-- {name} (lowpass={arms[name][0]}x, "
              f"ratio_clip={arms[name][1]['ratio_clip']}), "
              f"{len(s['scale'])}/{n} samples had a usable pair, "
              f"{n_none[name]} dropped")
        for key in ("scale_dev", "repro", "scale", "loss", "n_pairs", "support"):
            st = summarise(s[key])
            if not st["n"]:
                continue
            extra = f" std {st['std']:.4f}" if "std" in st else ""
            print(f"   {key:10s} mean {st['mean']:.4f}  median {st['median']:.4f}"
                  f"  p90 {st['p90']:.4f}  max {st['max']:.4f}{extra}")

    if model is not None:
        st = summarise([r["vae_disp_l1"] for r in rows if "vae_disp_l1" in r])
        fl = summarise([r["vae_flat"] for r in rows if "vae_flat" in r])
        gf = summarise([r["flat"] for r in rows])
        print(f"\nVAE round trip on GT disparity, no model involved:")
        print(f"   disp_l1 vs GT  mean {st['mean']:.4f}  median {st['median']:.4f}"
              f"  p90 {st['p90']:.4f}  max {st['max']:.4f}")
        print(f"   -> the eval metric src_depth_disp_l1 reads ~0.055 for the model, "
              f"i.e. {0.055 / max(st['mean'], 1e-9):.2f}x this codec floor")
        print(f"   flatness  GT {gf['mean']:.4f}  after round trip {fl['mean']:.4f}")

    print("\nGRADIENT PROFILE: mean |d term / d disp| per disparity quintile,")
    print("quintile 1 = the FAR tail.  Also the share of total gradient mass in it.")
    for name in ("shipped", "lp8", "vae_rt"):
        for term in ("scale", "repro"):
            got = [r[name]["grad"][term] for r in rows
                   if name in r and "grad" in r[name] and term in r[name]["grad"]]
            if not got:
                continue
            nb = len(got[0]["per_bin"])
            bins = [statistics.fmean(g["per_bin"][b] for g in got) for b in range(nb)]
            share = statistics.fmean(g["low_bin_mass_share"] for g in got)
            gn = statistics.fmean(g["grad_norm"] for g in got)
            edges = [statistics.fmean(g["edges"][b] for g in got) for b in range(nb + 1)]
            print(f"   {name:8s} {term:6s} |grad| " +
                  " ".join(f"{b:.2e}" for b in bins) +
                  f"   far-tail mass {share * 100:5.1f}%   ||grad|| {gn:.3e}")
            print(f"   {'':8s} {'':6s} disp   " +
                  " ".join(f"{edges[b]:.2f}-{edges[b + 1]:.2f}" for b in range(nb)))

    ship = acc["shipped"]
    if ship["scale_dev"]:
        print("\nREAD:")
        dev = statistics.fmean(ship["scale_dev"])
        rep = statistics.fmean(ship["repro"])
        loss = statistics.fmean(ship["loss"])
        print(f"   GT |a-1| = {dev:.4f} vs the model's 0.0518 "
              f"-> the model sits at {0.0518 / max(dev, 1e-9):.2f}x the GT floor")
        print(f"   GT repro = {rep:.4f} vs the model's 0.0321 "
              f"-> {0.0321 / max(rep, 1e-9):.2f}x the GT floor")
        print(f"   GT loss  = {loss:.5f} vs the model's 0.0043 "
              f"-> {0.0043 / max(loss, 1e-9):.2f}x")
        print("   If these ratios are near 1, the term has no headroom above the")
        print("   data's own inconsistency and can only push the prediction off GT.")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        print(f"\nper-sample rows -> {args.out}")


if __name__ == "__main__":
    main()
