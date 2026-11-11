"""BEV flyover demo for the GAE patch1 runs (rgbd_video_gen_gae analogue of
GAE-GeometricAutoEncoder-main's demo3view `bev` trajectory).

Pins the scene's 6 consecutive cached captures as sources, then renders a
straight-down (top-down) camera above each source centre plus constant-
velocity extrapolated waypoints -- all pitched 90 deg down with image-up
tracking the local heading.  Sources keep their cached GAE latents; the BEV
targets are virtual views (no GT) starting from noise.

Run from the repo root:
  ASCEND_RT_VISIBLE_DEVICES=0 python -m tools.infer_gae_bev_demo \
      --run_dir logs/<exp> --checkpoint logs/<exp>/checkpoint-step-8000 \
      [--manifest /mnt/.../manifest_test.jsonl --cache_name re10k_test]
"""
import argparse
import math
import os

try:
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
except ImportError:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
import torch
from accelerate import Accelerator
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from examples.wanvideo.model_training.infer_nvs_compare import (
    build_model,
    latest_checkpoint,
)
from examples.wanvideo.model_training.infer_trajectory import (
    _FILL_CTX,
    _install_zero_fill_shim,
)
from examples.wanvideo.model_training.infer_virtual_views import (
    VirtualView,
    _patch_load_view,
)
from examples.wanvideo.model_training.train_with_accelerate_gae_nvs import (
    build_dataset,
)


def _unit(v):
    return v / max(np.linalg.norm(v), 1e-9)


def _pose_at(eye, z, up):
    """OpenCV c2w: optical axis z, image-up as close to `up` as possible."""
    z = _unit(z)
    y = -_unit(up)                       # camera y is image-DOWN
    x = _unit(np.cross(y, z))
    y = np.cross(z, x)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, eye
    return m


def _heading(c2w, up):
    f = c2w[:3, 2]
    h = f - np.dot(f, up) * up
    return c2w[:3, 0] if np.linalg.norm(h) < 1e-6 else _unit(h)


def main():
    from pathlib import Path
    import json, inspect, time
    parser=argparse.ArgumentParser()
    for name in ['arm','run_dir','checkpoint','scene_id','job_dir','manifest','cache_name']:
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--num_inference_steps',type=int,default=25)
    parser.add_argument('--seed',type=int,default=1234)
    args=parser.parse_args();args.fps=2.;args.height_scale=.9
    directory=Path(args.job_dir);directory.mkdir(parents=True,exist_ok=True)
    if (directory/'complete.json').exists():return
    cfg=OmegaConf.load(Path(args.run_dir)/'args.yaml')
    if cfg.get('gae_stats_path') is None:cfg.gae_stats_path=os.path.join(cfg.gae_cache_root,cfg.gae_dataset_name,'latent_stats.pt')
    cfg.manifest_path=args.manifest;cfg.gae_dataset_name=args.cache_name
    torch.npu.set_compile_mode(jit_compile=False);torch.npu.config.allow_internal_format=False;torch.npu.set_per_process_memory_fraction(.35)
    accelerator=Accelerator(mixed_precision=cfg.mixed_precision)
    model=build_model('gae',cfg,accelerator);model.set_training_param();model.eval_num_inference_steps=25
    impl=inspect.getfile(type(model))
    if args.arm=='control':assert not hasattr(model,'projection_mode'),impl
    else:assert model.projection_mode==args.arm and impl.startswith(os.environ['PROJECTION_EVAL_CODE_ROOT']+os.sep),(impl,args.arm)
    from safetensors.torch import load_file
    state=load_file(os.path.join(args.checkpoint,'model.safetensors'),device='cpu')
    missing,unexpected=model.load_state_dict(state,strict=False)
    assert not missing and not unexpected,(missing,unexpected)
    print('STRICT LOAD PASS',impl,len(state),flush=True);del state
    model.to(accelerator.device);model.eval()
    step=int(args.checkpoint.rsplit('-',1)[1]);dataset=build_dataset(cfg,split=cfg.get('eval_split',None))
    args.output_dir=str(directory/'lateral')
    with torch.no_grad():run_probe(args,cfg,model,accelerator,dataset,step,'gae')
    import trajectory_impl as real
    model._latent_file_cache.clear();real.build_model=lambda *a,**k:model
    trajectory=str(directory/'trajectory')
    with torch.no_grad():real.run_scene(args.run_dir,args.checkpoint,args.scene_id,6,15,trajectory,downscale=2,shared_target_noise=True,manifest=args.manifest,cache_name=args.cache_name)
    real._synthesize_video(trajectory,os.path.basename(args.run_dir),step,args.scene_id,fps=3.)
    result=score_outputs(args.output_dir,trajectory)
    result.update(step=step,scene=args.scene_id,projection_mode=args.arm,checkpoint=args.checkpoint,implementation=impl,completed_at=time.time())
    target=directory/'complete.json';tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(target)

def run_probe(args,cfg,model,accelerator,dataset,step,kind):
    if args.scene_id is None:
        args.scene_id = dataset.records[0]["scene_id"]
    record = next(r for r in dataset.records if r["scene_id"] == args.scene_id)
    views = sorted(record["views"], key=lambda v: str(v.view_id))

    import json
    cache_dir = os.path.join(
        cfg.get("gae_cache_root"), cfg.get("gae_dataset_name", "bmvs"))
    with open(os.path.join(cache_dir, f"{args.scene_id}.meta.json")) as fh:
        cached = {str(v) for v in json.load(fh)["views"]}
    cached_views = [v for v in views if str(v.view_id) in cached]
    s = model.num_source_views
    if len(cached_views) < s:
        raise ValueError(f"only {len(cached_views)} cached views in {args.scene_id}")

    # middle window of consecutive captures keeps the flyover in video order
    lo = max(0, (len(cached_views) - s) // 2)
    ids = [str(v.view_id) for v in cached_views[::max(1,len(cached_views)//s)][:s]]
    src_views = [next(v for v in cached_views if str(v.view_id)==vid) for vid in ids]
    accelerator.print(
        f"{len(views)} frames, {len(cached_views)} cached -> sources "
        f"{[str(v.view_id) for v in src_views]}")

    H, W = int(cfg.resolution[0]), int(cfg.resolution[1])
    _patch_load_view(dataset, H, W)
    src_items = [dataset._load_view(v) for v in src_views]

    valid = [(it["depth_metric"][0] > 1e-3) & torch.isfinite(it["depth_metric"][0])
             for it in src_items]
    med_depth = float(np.median([
        float(it["depth_metric"][0][m].median())
        for it, m in zip(src_items, valid) if bool(m.any())]))
    src_k = torch.stack([it["K"] for it in src_items]).median(0).values

    c2ws = [v.c2w.double().numpy() for v in src_views]
    centers = np.stack([m[:3, 3] for m in c2ws])
    ups = np.stack([-m[:3, 1] for m in c2ws])            # re10k y-col points down
    up = _unit(ups.mean(0))
    heads = [_heading(m, up) for m in c2ws]
    vel = (centers[-1] - centers[0]) / (len(centers) - 1)

    # Isolate rotation and translation around an observed source camera.
    import json
    anchor = c2ws[0].copy()
    src_k = src_items[0]["K"].clone()
    specs = []
    bevs = []
    for deg in [-20, -15, -10, -5, 0, 5, 10, 15, 20]:
        a = np.deg2rad(deg)
        ry = np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
        pose = anchor.copy()
        pose[:3,:3] = anchor[:3,:3] @ ry
        bevs.append(pose)
        specs.append({"kind":"yaw", "degrees":deg})
    for fraction in [-.2,-.1,-.05,.05,.1,.2]:
        pose = anchor.copy()
        pose[:3,3] += anchor[:3,0] * (fraction * med_depth)
        bevs.append(pose)
        specs.append({"kind":"lateral", "depth_fraction":fraction})
    n_full = (len(bevs) // model.num_target_views) * model.num_target_views
    if n_full < len(bevs):
        accelerator.print(f"dropping {len(bevs) - n_full} trailing waypoint(s)")
    bevs = bevs[:n_full]
    accelerator.print(
        f"med_depth={med_depth:.2f} bev_height={args.height_scale * med_depth:.2f} "
        f"-> {n_full} top-down waypoints")

    tgt_views = [VirtualView(f"bev{i:02d}", src_k, torch.from_numpy(p).float())
                 for i, p in enumerate(bevs)]
    _FILL_CTX["scene"] = args.scene_id
    _FILL_CTX["fill_views"] = [str(v.view_id) for v in tgt_views]
    _install_zero_fill_shim()

    device = accelerator.device
    shot = "lateral_probe"
    output_dir = args.output_dir or os.path.join(
        args.run_dir, f"{shot}_step{step}", args.scene_id)
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    Image.fromarray((src_items[0]["image"].permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)).save(os.path.join(output_dir,"anchor_rgb.png"))
    with open(os.path.join(output_dir,"poses.json"),"w") as fh:
        json.dump({"specs":specs,"anchor":anchor.tolist(),"K":src_k.tolist(),"poses":[p.tolist() for p in bevs],"median_depth":med_depth},fh,indent=2)


    def _load_batch(tgt_chunk):
        items = src_items + [dataset._load_view(v) for v in tgt_chunk]
        depth_masks = torch.cat([it["valid_mask"] for it in items], dim=0)
        rgb_masks = torch.cat(
            [torch.zeros_like(it["valid_mask"]) for it in items[:s]]
            + [torch.ones_like(it["valid_mask"]) for it in items[s:]], dim=0)
        return {
            "scene_id": [args.scene_id],
            "source_view_ids": [[str(v.view_id) for v in src_views]],
            "target_view_ids": [[str(v.view_id) for v in tgt_chunk]],
            "images": torch.stack([it["image"] for it in items]).unsqueeze(0),
            "depth_metric": torch.stack(
                [it["depth_metric"] for it in items]).unsqueeze(0),
            "K": torch.stack([it["K"] for it in items]).unsqueeze(0),
            "c2w": torch.stack([it["c2w"] for it in items]).unsqueeze(0),
            "view_roles": torch.tensor(
                [0] * s + [1] * len(tgt_chunk)).unsqueeze(0),
            "loss_masks": torch.stack(
                [depth_masks, rgb_masks], dim=0).unsqueeze(0),
            "sample_id": [args.scene_id],
        }

    frames = []
    for ti in range(0, n_full, model.num_target_views):
        chunk = tgt_views[ti:ti + model.num_target_views]
        batch = _load_batch(chunk)
        model._eval_inputs = True
        try:
            inputs = model.forward_preprocess(batch)
        finally:
            model._eval_inputs = False
        gen = torch.Generator(device=device)
        gen.manual_seed(int(args.seed if args.seed is not None
                            else cfg.get("eval_sample_seed", 1234)))
        if kind == "geocolor_token":
            pred_geo, pred_color = model.sample_latents(inputs=inputs, generator=gen)
            pred_rgb = model._decode_color_views(
                pred_color[:, :, s:], (H, W), (W, H))
            pred_depth = model._decode_geo_depth(
                pred_geo[:, :, s:], (H, W), (W, H))
        else:
            pred_z = model.sample_latents(
                inputs, num_steps=args.num_inference_steps, generator=gen,
                shared_target_noise=True)
            pred_rgb, pred_depth = model._decode_views(pred_z[:, :, s:], H, W)

        for j, view in enumerate(chunk):
            rgb = (pred_rgb[0, j].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
                   * 255).astype(np.uint8)
            d = pred_depth[0, j].float().cpu().numpy()
            m = np.isfinite(d) & (d > 1e-3)
            lo_, hi_ = (np.percentile(d[m], 2), np.percentile(d[m], 98)
                        if bool(m.any()) else (0.0, 1.0))
            disp = (cm.plasma(np.clip((d - lo_) / max(hi_ - lo_, 1e-6), 0, 1))
                   [:, :, :3] * 255).astype(np.uint8)
            Image.fromarray(rgb).save(
                os.path.join(frames_dir, f"{view.view_id}_rgb.jpg"), quality=95)
            Image.fromarray(disp).save(
                os.path.join(frames_dir, f"{view.view_id}_disp.jpg"), quality=95)
            np.save(os.path.join(frames_dir, f"{view.view_id}_depth.npy"), d)

            vid = str(view.view_id)
            rgb_l = Image.fromarray(rgb)
            ImageDraw.Draw(rgb_l).text((6, 4), vid, fill=(255, 255, 0))
            disp_l = Image.fromarray(disp)
            ImageDraw.Draw(disp_l).text(
                (6, 4), f"{vid} depth", fill=(255, 255, 0))
            sep = np.ones((4, W, 3), np.uint8) * 200
            frames.append(np.concatenate(
                [np.asarray(rgb_l), sep, np.asarray(disp_l)], axis=0))
        accelerator.print(f"chunk {ti // model.num_target_views} done", flush=True)

    try:
        from tools.make_traj_videos import encode
        fp = os.path.join(output_dir, f"{shot}_{os.path.basename(args.run_dir)}_step{step}.mp4")
        encode(frames, args.fps, fp)
        accelerator.print(f"wrote {fp} ({len(frames)} frames)")
    except Exception as exc:
        accelerator.print(f"video encode skipped: {exc}")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(centers[:, 0], centers[:, 1], "o-", label="src path")
    bev_xy = np.array([p[:3, 3][:2] for p in bevs])
    ax.plot(bev_xy[:, 0], bev_xy[:, 1], "s--", label="bev waypoints")
    for i, p in enumerate(bevs):
        ax.annotate(str(i), bev_xy[i], fontsize=8)
    ax.set_title(f"BEV flyover {args.scene_id} @step{step}")
    ax.legend()
    ax.axis("equal")
    fig.savefig(os.path.join(output_dir, "bev_path_topdown.png"), dpi=130,
                bbox_inches="tight")
    accelerator.print("all done")

def score_outputs(lateral,trajectory):
    import json,cv2
    from pathlib import Path
    lateral=Path(lateral);trajectory=Path(trajectory)
    rows=[json.loads(x) for x in (trajectory/'per_frame_metrics.jsonl').read_text().splitlines()]
    assert len(rows)==15
    real_scores={k:float(np.mean([x[k] for x in rows])) for k in ['tgt_rgb_psnr','tgt_rgb_lpips','tgt_depth_abs_rel_aligned','tgt_depth_delta1','nearest_src_psnr']}
    pose=json.loads((lateral/'poses.json').read_text());source=np.array(Image.open(lateral/'anchor_rgb.png'));h,w=source.shape[:2];K=np.array(pose['K']);values=[]
    for i,spec in enumerate(pose['specs'][:9]):
        rad=np.deg2rad(spec['degrees']);R=np.array([[np.cos(rad),0,np.sin(rad)],[0,1,0],[-np.sin(rad),0,np.cos(rad)]]);H=K@R.T@np.linalg.inv(K)
        gt=cv2.warpPerspective(source,H,(w,h));mask=cv2.erode(cv2.warpPerspective(np.ones((h,w),np.uint8),H,(w,h)),np.ones((5,5),np.uint8)).astype(bool)
        pred=np.array(Image.open(lateral/'frames'/f'bev{i:02d}_rgb.jpg'));mse=np.mean((pred[mask].astype(float)-gt[mask])**2);values.append(float(10*np.log10(255**2/max(mse,1e-10))))
    videos={}
    for name,directory in [('lateral',lateral),('trajectory',trajectory)]:
        candidates=list(directory.glob('*.mp4'));assert len(candidates)==1,candidates
        cap=cv2.VideoCapture(str(candidates[0]));ok,_=cap.read();count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release();assert ok and count==15,(name,count)
        videos[name]=str(candidates[0])
    return {'trajectory':real_scores,'yaw':{'yaw0_psnr':values[4],'nonzero_psnr':float(np.mean(values[:4]+values[5:])),'large15_20_psnr':float(np.mean([values[i] for i in [0,1,7,8]])),'per_yaw':values},'videos':videos,'outputs':{'trajectory':str(trajectory),'lateral':str(lateral)}}


if __name__=="__main__":main()
