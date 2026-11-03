"""Pose-EXTRAPOLATION inference on the DL3DV overfit set, rendered as an mp4.

The interp tool (infer_dl3dv_interp_video.py) only ever renders targets that
are REAL views of the clip, so every result sits inside the known camera
trajectory and the pose change stays small.  This tool instead synthesises
FAKE target poses OUTSIDE the clip's trajectory (constant-velocity SE(3)
extrapolation of the real camera path, beyond both ends), hands them to the
model through the same 6-src + 3-tgt pipeline, and renders a PRED-only video
(there is no GT for a pose that was never captured).

Extrapolation model: the camera moves with constant per-frame velocity over
the `--est_span` frames adjacent to each end of the clip --
  rotation: axis-angle (w, theta/J) per frame, applied k times (Rodrigues);
  translation: per-frame linear velocity v = (t_b - t_a)/J.
A target k frames beyond the boundary view T_b is  A_k @ T_b  with
A_k = (R(k*theta/J), k*v).  Curved trajectories therefore deviate from the
real path the farther out they go -- which is exactly the point.

Frame metrics in the caption: dR = rotation angle between the fake pose and
the boundary view, dt = translation distance -- the "pose change" magnitude.

Sources are REAL views at --src_gap spacing ending (forward) / starting
(backward) at the boundary, in temporal order.  Fake target _ViewRecords
reuse the boundary view's rgb/depth paths only so the dataloader can load
slots; those images are never shown and never scored.

Two extrapolation modes (--look):

* forward (default) -- constant-velocity SE(3) continuation of the trajectory,
  the camera keeps looking along the path.
* down -- BEV-like: position continues along the trajectory (optionally rising
  via --rise_per_frame) but the camera PITCHES DOWN toward the ground by
  --pitch_deg around the trajectory heading.  The world up is estimated from
  the clip itself (mean camera up vector; c2w is OpenCV, y = down).

Run as a MODULE from the project root:
  python -m tools.infer_dl3dv_extrap_video \
      --config train_config/normal_config/dl3dv2_overfit_seq6s3t_video_warpbi.yaml \
      --checkpoint logs/dl3dv2_overfit_seq6s3t_video_warpbi/best_tgt_rgbd \
      --output_dir logs/dl3dv2_overfit_seq6s3t_video_warpbi/infer_video \
      --record_idx 0 --extrap 12,24,36,48,72,96
  # BEV-like:
  python -m tools.infer_dl3dv_extrap_video ... --look down --pitch_deg 60
"""
import argparse
import math
import os
import shutil
import subprocess

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from PIL import Image
from safetensors.torch import load_file

from examples.dataset.kubric4d_rgbd_nvs_dataset import _ViewRecord
from tools.infer_dl3dv_interp_video import (
    _plasma,
    _run_window,
    _tile,
    build_dataset,
    build_model,
)


# --- minimal SE(3) extrapolation math (numpy, float64) ----------------------

def _rodrigues(axis, angle):
    w = np.asarray(axis, dtype=np.float64)
    n = float(np.linalg.norm(w))
    if n < 1e-12:
        return np.eye(3)
    w = w / n
    K = np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])
    a = float(angle)
    return np.eye(3) + np.sin(a) * K + (1.0 - np.cos(a)) * (K @ K)


def _rot_to_axisangle(R):
    """R -> (unit axis, angle in [0, pi])."""
    c = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    theta = math.acos(c)
    if theta < 1e-8:
        return np.zeros(3), 0.0
    if math.pi - theta < 1e-5:
        # Near pi the cross-matrix form is degenerate; read the axis off the
        # symmetric part instead.
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(A), 0.0, None))
        k = int(np.argmax(axis))
        if axis[k] > 1e-12:
            axis = A[:, k] / axis[k]
            if np.dot(np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]), axis) < 0:
                axis = -axis
        return axis, theta
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return axis / (2.0 * math.sin(theta)), theta


def _motion_over_span(Ta, Tb, span):
    """Constant per-frame motion taking Ta to Tb over `span` frames.

    Returns (unit axis, angle-per-frame, translation-per-frame).
    """
    Rb, tb = Tb[:3, :3], Tb[:3, 3]
    axis, theta = _rot_to_axisangle(Rb @ Ta[:3, :3].T)
    return axis, theta / float(span), (tb - Ta[:3, 3]) / float(span)


def _extrap_c2w(boundary_c2w, motion, k, inverse=False):
    """A_k @ T_b with A_k the k-frame constant-velocity motion.

    inverse=True rolls the boundary BACK along that motion instead
    (A_k^-1 @ T_b): used to extrapolate before the clip start, where the
    motion is the forward motion measured at the head of the clip.
    """
    axis, theta_pf, v = motion
    Rk = _rodrigues(axis, k * theta_pf)
    if inverse:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = Rk.T @ boundary_c2w[:3, :3]
        T[:3, 3] = Rk.T @ (boundary_c2w[:3, 3] - k * v)
        return T
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rk @ boundary_c2w[:3, :3]
    T[:3, 3] = Rk @ boundary_c2w[:3, 3] + k * v
    return T


def _pose_delta(Ta, Tb):
    """(rotation angle in deg, translation norm) between two c2w matrices."""
    c = float(np.clip((np.trace(Ta[:3, :3] @ Tb[:3, :3].T) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(c)), float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3]))


def _estimate_world_up(views):
    """World up from the clip itself: mean camera up vector (-R[:,1], since
    the c2w convention is OpenCV with y pointing DOWN), normalised.  A level
    walking/driving trajectory makes this a solid gravity estimate without
    knowing the world frame's convention."""
    ups = np.stack([-np.asarray(v.c2w)[:3, 1] for v in views])
    up = ups.mean(axis=0)
    n = float(np.linalg.norm(up))
    if n < 1e-6:
        return np.array([0.0, 0.0, 1.0])
    return up / n


def _bev_c2w(boundary_c2w, v, k, up, pitch_deg, rise_per_frame):
    """Fake c2w at k frames beyond the boundary, LOOKING DOWN toward the
    ground (BEV-like).

    Position keeps the trajectory's per-frame velocity v (plus an optional
    drone-like rise along the world up).  Orientation: the horizontal heading
    of the trajectory pitched down by `pitch_deg` (0 = level, 90 = nadir).
    OpenCV convention (x right, y down, z forward): z = forward,
    x = normalize(z x up), y = z x x -- this makes y point away from `up`.
    """
    pitch = math.radians(pitch_deg)
    pos = boundary_c2w[:3, 3] + k * v + rise_per_frame * k * up

    # Heading: horizontal component of the trajectory velocity; fall back to
    # the boundary camera's forward when the motion is vertical/still.
    h = v - np.dot(v, up) * up
    if float(np.linalg.norm(h)) < 1e-6:
        zb = boundary_c2w[:3, 2]
        h = zb - np.dot(zb, up) * up
    h = h / float(np.linalg.norm(h))

    z = h * math.cos(pitch) - up * math.sin(pitch)
    if abs(np.dot(z, up)) > 0.999:  # nadir: x-hint from the heading
        x = h - np.dot(h, z) * z
    else:
        x = np.cross(z, up)
    x = x / float(np.linalg.norm(x))
    y = np.cross(z, x)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.stack([x, y, z], axis=1)
    T[:3, 3] = pos
    return T


def _fake_view_record(boundary_view, c2w_np, view_id):
    """A _ViewRecord whose pose is the fake one; rgb/depth paths point at the
    boundary view's files purely so _load_view can fill the slot (the loaded
    images are never rendered or scored for extrapolated targets)."""
    return _ViewRecord(
        scene_id=boundary_view.scene_id,
        time_id=boundary_view.time_id,
        view_id=view_id,
        rgb_path=boundary_view.rgb_path,
        depth_path=boundary_view.depth_path,
        K=boundary_view.K.clone(),
        c2w=torch.tensor(c2w_np, dtype=boundary_view.c2w.dtype),
        depth_scale=boundary_view.depth_scale,
    )


# --- composition ------------------------------------------------------------

def _compose_extrap_video(sample_dir, frames, framerate):
    """frames: ordered dicts with ref_rgb / pred_rgb / pred_disp / caption.

    Three columns: REF (the boundary source's GT, the closest real view) |
    PRED RGB | PRED depth.  No GT column exists for a fake pose.
    """
    grid_frames = []
    for f in frames:
        cap = f["caption"]
        cols = (
            _tile(f["ref_rgb"], "REF (boundary GT)", cap),
            _tile(f["pred_rgb"], "PRED", cap),
            _tile(_plasma(f["pred_disp"]), "PRED depth", cap),
        )
        grid_frames.append(np.concatenate([np.array(c) for c in cols], axis=1))

    grid = np.concatenate(grid_frames, axis=0)
    Image.fromarray(grid).save(os.path.join(sample_dir, "timeline_grid.png"))

    tmp = os.path.join(sample_dir, "frames")
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    for i, fr in enumerate(grid_frames):
        im = Image.fromarray(fr)
        canvas = Image.new("RGB", (im.width + im.width % 2, im.height + im.height % 2), (0, 0, 0))
        canvas.paste(im, (0, 0))
        canvas.save(os.path.join(tmp, f"frame_{i:03d}.png"))
    mp4 = os.path.join(sample_dir, "extrap_compare.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-framerate", str(framerate),
         "-i", os.path.join(tmp, "frame_%03d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", mp4],
        check=True, capture_output=True)
    return mp4, len(grid_frames)


def _chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--record_idx", type=int, default=0,
                        help="Scene index into the manifest.")
    parser.add_argument("--extrap", type=str, default="12,24,36,48,72,96",
                        help="Extrapolation distances in FRAMES beyond each end "
                             "of the clip.  Larger = larger pose change.")
    parser.add_argument("--src_gap", type=int, default=12,
                        help="Index gap between the 6 real source views.")
    parser.add_argument("--est_span", type=int, default=24,
                        help="Frames adjacent to each end used to estimate the "
                             "constant per-frame camera motion.")
    parser.add_argument("--look", type=str, default="forward", choices=["forward", "down"],
                        help="forward: keep looking along the trajectory (constant-"
                             "velocity SE(3) extrapolation).  down: fake poses that "
                             "keep the trajectory's position+heading but pitch toward "
                             "the ground (BEV-like), see --pitch_deg.")
    parser.add_argument("--pitch_deg", type=float, default=60.0,
                        help="look=down only: pitch below the horizon in degrees "
                             "(0 = level, 90 = straight-down nadir BEV).")
    parser.add_argument("--rise_per_frame", type=float, default=0.0,
                        help="look=down only: camera height gain per extrapolated "
                             "frame along the world up (drone-like), in world units.")
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--framerate", type=int, default=4)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    set_seed(args.seed)
    accelerator = Accelerator(mixed_precision=cfg.mixed_precision)
    device = accelerator.device
    num_steps = args.num_inference_steps or int(cfg.get("eval_num_inference_steps", 4))

    model = build_model(cfg, accelerator, num_steps)
    weights = os.path.join(args.checkpoint, "model.safetensors")
    state = load_file(weights, device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    accelerator.print(f"Loaded {weights}: {len(state)} tensors, "
                      f"missing={len(missing)}, unexpected={len(unexpected)}")
    model.to(device)
    model.eval()

    dataset = build_dataset(cfg)
    record = dataset.records[args.record_idx % len(dataset.records)]
    views_sorted = sorted(record["views"], key=lambda v: v.view_id)
    n = len(views_sorted)
    latent_scale = float(cfg.get("latent_scale", 1.0))
    depth_latent_scale = model.depth_latent_scale
    num_src = cfg.num_source_views
    num_tgt = cfg.num_target_views
    scene_tag = str(record["scene_id"]).split("/")[-1][:24]

    distances = sorted({int(v) for v in args.extrap.split(",") if v.strip()})
    if not distances:
        raise ValueError("--extrap gave no distances")

    # Per-end constant-velocity motion estimated from real views.
    fwd_b = views_sorted[n - 1]
    fwd_motion = _motion_over_span(
        views_sorted[n - 1 - args.est_span].c2w.numpy().astype(np.float64),
        fwd_b.c2w.numpy().astype(np.float64), args.est_span)
    bwd_b = views_sorted[0]
    bwd_motion = _motion_over_span(
        bwd_b.c2w.numpy().astype(np.float64),
        views_sorted[args.est_span].c2w.numpy().astype(np.float64), args.est_span)

    if args.look == "down":
        world_up = _estimate_world_up(views_sorted)
        accelerator.print(
            f"[extrap] look=down pitch={args.pitch_deg}deg "
            f"rise/frame={args.rise_per_frame} world_up={np.round(world_up, 3).tolist()}")

    accelerator.print(
        f"[extrap] scene={record['scene_id']} views={n} "
        f"est_span={args.est_span} src_gap={args.src_gap} "
        f"motion/deg-per-frame fwd={math.degrees(fwd_motion[1]):.3f} "
        f"bwd={math.degrees(bwd_motion[1]):.3f}; distances={distances}")

    # Video order: walk in from the far past to the far future
    # (-max ... -12, +12 ... +max).  Each forward's num_tgt target slots carry
    # DISTINCT extrapolation distances (repeating one pose across slots would
    # be off the training distribution, where the 3 targets differ).
    plan = []
    for d_chunk in _chunked(sorted(distances, reverse=True), num_tgt):
        plan.append(("bwd", [-d for d in sorted(d_chunk)]))
    for d_chunk in _chunked(distances, num_tgt):
        plan.append(("fwd", list(d_chunk)))

    frames = []
    accelerator.print("[extrap] target pose deltas vs boundary view:")
    for direction, signed_ks in plan:
        if direction == "fwd":
            boundary = fwd_b
            motion = fwd_motion
            src_pos = [n - 1 - (num_src - 1 - j) * args.src_gap for j in range(num_src)]
        else:
            boundary = bwd_b
            motion = bwd_motion
            src_pos = [j * args.src_gap for j in range(num_src)]
        source_views = [views_sorted[min(p, n - 1)] for p in src_pos]

        target_views = []
        for signed_k in signed_ks:
            k = abs(signed_k)
            if args.look == "down":
                # Position rolls the boundary back/forward along the SAME
                # trajectory velocity (bwd uses the negated velocity), then the
                # camera pitches down at the boundary's height (+ rise).
                v = motion[2] if direction == "fwd" else -motion[2]
                c2w_np = _bev_c2w(boundary.c2w.numpy().astype(np.float64),
                                  v, k, world_up, args.pitch_deg, args.rise_per_frame)
            else:
                c2w_np = _extrap_c2w(boundary.c2w.numpy().astype(np.float64), motion, k,
                                     inverse=(direction == "bwd"))
            sign = "+" if signed_k > 0 else "-"
            target_views.append(_fake_view_record(
                boundary, c2w_np, f"x{sign}{k:03d}"))
        accelerator.print(f"[extrap] window {direction}: "
                          f"tgt={[v.view_id for v in target_views]}")
        res = _run_window(model, dataset, record, source_views, target_views,
                          device, latent_scale, depth_latent_scale, num_steps,
                          accelerator)

        # Video order within the window: far past -> future, i.e. keep the
        # global plan order (bwd windows are already far->near).
        ref_rgb = res[source_views[-1].view_id]["gt_rgb"] if direction == "fwd" \
            else res[source_views[0].view_id]["gt_rgb"]
        bnd_np = boundary.c2w.numpy().astype(np.float64)
        for tgt_view in target_views:
            entry = res[tgt_view.view_id]
            sign = "+" if tgt_view.view_id[1] == "+" else "-"
            k = abs(int(tgt_view.view_id[2:]))
            if args.look == "down":
                v = motion[2] if direction == "fwd" else -motion[2]
                fake = _bev_c2w(bnd_np, v, k, world_up, args.pitch_deg,
                                args.rise_per_frame)
            else:
                fake = _extrap_c2w(bnd_np, motion, k,
                                   inverse=(direction == "bwd"))
            dR, dt = _pose_delta(fake, bnd_np)
            mode_tag = f"BEV pitch {args.pitch_deg:.0f}deg" if args.look == "down" \
                else "fake pose, no GT"
            frames.append({
                "ref_rgb": ref_rgb,
                "pred_rgb": entry["pred_rgb"],
                "pred_disp": entry["pred_disp"],
                "caption": [
                    f"EXTRAP {sign}{k}f ({mode_tag})  scene {scene_tag}",
                    f"dR {dR:.1f} deg   dt {dt:.3f} vs boundary  frame_{sign}{k:03d}",
                ],
            })
            accelerator.print(
                f"  {sign}{k:>4d}f  dR={dR:7.2f} deg  dt={dt:.4f}")

    sample_dir = os.path.join(
        args.output_dir,
        f"extrap_{args.look}_p{args.pitch_deg:.0f}"
        f"_r{args.record_idx}_g{args.src_gap}_e{args.est_span}_{scene_tag}")
    os.makedirs(sample_dir, exist_ok=True)
    mp4, nf = _compose_extrap_video(sample_dir, frames, args.framerate)
    accelerator.print(f"[extrap] wrote {mp4} ({nf} timeline frames)")


if __name__ == "__main__":
    main()
