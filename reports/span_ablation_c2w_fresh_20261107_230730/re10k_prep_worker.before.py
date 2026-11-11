"""re10k -> WAI-style layout + DA3 depth, one worker per NPU.

Per scene (stride-subsampled frames):
  <scene>/<frame>.jpg   raw jpeg bytes copied from the .torch shard
  <scene>/<frame>.npz   depth(fp16, native res) + R/t cam2world + intrinsics(px)
  <scene>/poses.json    per-frame K / c2w lists for the manifest builder

Cameras come from DA3's own multi-view prediction (c2w, self-consistent with
its depth); re10k SfM cameras are not used. Scenes with an existing poses.json
are skipped, so the job is resumable.

Usage:
  ASCEND_RT_VISIBLE_DEVICES=<i> python re10k_prep_worker.py --shard-dir .../train \
      --out-root .../re10k/extracted --worker 0 --num-workers 6
"""

import argparse
import faulthandler
import io
import json
import os
import sys
import time
import types

faulthandler.enable(all_threads=True)

# moviepy / plyfile / pycolmap / trimesh / evo / matplotlib 等只被 DA3 的导出与
# 可视化工具引用，深度推理不会触达；迭代式惰性补桩，避免改动共享环境。
from unittest.mock import MagicMock  # noqa: E402


def _stub_module(name):
    if name in sys.modules:
        return
    m = types.ModuleType(name)
    m.__getattr__ = lambda attr: MagicMock() if not attr.startswith("__") else (
        (_ for _ in ()).throw(AttributeError(attr))
    )
    sys.modules[name] = m
    if "." in name:
        parent = name.rsplit(".", 1)[0]
        _stub_module(parent)
        setattr(sys.modules[parent], name.rsplit(".", 1)[1], m)


sys.path.insert(0, "/mnt/sfs_turbo/t00845730/zhangyiheng/Depth-Anything-3/src")
for _ in range(40):
    try:
        import depth_anything_3.api  # noqa: E402
        break
    except ModuleNotFoundError as e:
        _stub_module(e.name)

DA3_SRC = "/mnt/sfs_turbo/t00845730/zhangyiheng/Depth-Anything-3/src"
DA3_CKPT = "/mnt/sfs_turbo/t00845730/zhangyiheng/data/UAV-MOT-training-assets-20261009/checkpoints/DA3-GIANT"
sys.path.insert(0, DA3_SRC)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.npu  # noqa: E402

FRAME_STRIDE = 5
MIN_VIEWS = 12  # keep only scenes with at least this many usable frames
PROCESS_RES = 504


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_model(device):
    # 注意：不要把 torch.cuda.is_bf16_supported() 补丁成 True —— NPU 上 bf16
    # autocast 会触发 CANN 内核崩溃；保持 False（fp16 autocast）已验证可用。
    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(DA3_CKPT)
    model = model.to(device)
    model.eval()
    return model


def save_scene(model, device, scene, out_dir, jpg_bytes_list, frame_ids, native_size):
    from PIL import Image

    W, H = native_size
    imgs = [Image.open(io.BytesIO(b)).convert("RGB") for b in jpg_bytes_list]
    for b, fid in zip(jpg_bytes_list, frame_ids):
        with open(os.path.join(out_dir, f"{fid}.jpg"), "wb") as fh:
            fh.write(b)

    with torch.no_grad():
        pred = model.inference(imgs, process_res=PROCESS_RES)

    depth = np.asarray(pred.depth, dtype=np.float32)  # (N, Hp, Wp)
    extr = np.asarray(pred.extrinsics, dtype=np.float64)  # (N, 3, 4) c2w
    intr = np.asarray(pred.intrinsics, dtype=np.float64)  # (N, 3, 3) px @ (Wp,Hp)
    if extr.shape[-2:] == (3, 4):  # 补齐成齐次 4x4
        extr = np.concatenate([extr, np.tile(np.eye(4)[3:], (extr.shape[0], 1, 1))], axis=1)
    N, Hp, Wp = depth.shape

    sx, sy = W / Wp, H / Hp
    frames = []
    for i, fid in enumerate(frame_ids):
        d = depth[i]
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        d = np.clip(d, 0.0, 1e4)
        # bilinear resize (Hp,Wp) -> (H,W)
        yi = np.clip((np.arange(H) + 0.5) / sy - 0.5, 0, Hp - 1)
        xi = np.clip((np.arange(W) + 0.5) / sx - 0.5, 0, Wp - 1)
        y0 = np.floor(yi).astype(np.int64)
        x0 = np.floor(xi).astype(np.int64)
        y1 = np.minimum(y0 + 1, Hp - 1)
        x1 = np.minimum(x0 + 1, Wp - 1)
        wy = (yi - y0)[:, None]
        wx = (xi - x0)[None, :]
        d = (d[np.ix_(y0, x0)] * (1 - wy) * (1 - wx)
             + d[np.ix_(y0, x1)] * (1 - wy) * wx
             + d[np.ix_(y1, x0)] * wy * (1 - wx)
             + d[np.ix_(y1, x1)] * wy * wx)

        K = intr[i].copy()
        K[0, :] *= sx
        K[1, :] *= sy
        R = extr[i][:3, :3]
        t = extr[i][:3, 3]
        np.savez_compressed(
            os.path.join(out_dir, f"{fid}.npz"),
            depth=d.astype(np.float16),
            R_cam2world=R.astype(np.float64),
            t_cam2world=t.astype(np.float64),
            intrinsics=K.astype(np.float64),
        )
        frames.append({
            "view_id": fid,
            "K": K.tolist(),
            "c2w": extr[i].tolist(),
        })

    poses = {"scene_id": scene["key"], "native_size": [W, H], "frames": frames}
    tmp = os.path.join(out_dir, "poses.json.tmp")
    with open(tmp, "w") as fh:
        json.dump(poses, fh)
    os.replace(tmp, os.path.join(out_dir, "poses.json"))
    return len(frame_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", default="/mnt/sfs_turbo/t00845730/re10k/train")
    ap.add_argument("--out-root", default="/mnt/sfs_turbo/t00845730/re10k/extracted")
    ap.add_argument("--worker", type=int, required=True)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--limit-shards", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    import torch_npu  # noqa: F401

    shards = sorted(
        os.path.join(args.shard_dir, f) for f in os.listdir(args.shard_dir) if f.endswith(".torch")
    )
    if args.limit_shards:
        shards = shards[: args.limit_shards]
    mine = shards[args.worker :: args.num_workers]
    log(f"worker {args.worker}: {len(mine)} shards")

    device = "npu"
    # ASCEND_RT_VISIBLE_DEVICES 已把本进程限制到唯一物理卡（逻辑编号 0）。
    torch.npu.set_device(0)
    t0 = time.time()
    model = load_model(device)
    torch.npu.synchronize()
    log(f"model ready in {time.time() - t0:.0f}s")

    from PIL import Image

    n_scenes = n_frames = n_skip = 0
    for si, shard_path in enumerate(mine):
        shard_name = os.path.basename(shard_path)
        try:
            data = torch.load(shard_path, map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001
            log(f"shard {shard_name}: LOAD FAIL {exc}")
            continue
        for scene in data:
            out_dir = os.path.join(args.out_root, scene["key"])
            if os.path.exists(os.path.join(out_dir, "poses.json")):
                n_skip += 1
                continue
            os.makedirs(out_dir, exist_ok=True)
            total = len(scene["images"])
            frame_ids = [f"{k:08d}" for k in range(0, total, FRAME_STRIDE)]
            if len(frame_ids) < MIN_VIEWS:
                n_skip += 1
                continue
            try:
                jpgs = [scene["images"][int(fid)].numpy().tobytes() for fid in frame_ids]
                size = Image.open(io.BytesIO(jpgs[0])).size
                t0 = time.time()
                nf = save_scene(model, device, scene, out_dir, jpgs, frame_ids, size)
                torch.npu.synchronize()
                n_scenes += 1
                n_frames += nf
                if n_scenes % 20 == 0:
                    rate = (time.time() - t0) / max(nf, 1)
                    log(
                        f"shard {si + 1}/{len(mine)} scenes={n_scenes} frames={n_frames} "
                        f"last={time.time() - t0:.1f}s/scene (~{rate:.2f}s/frame)"
                    )
            except Exception as exc:  # noqa: BLE001
                log(f"scene {scene.get('key')} FAIL: {type(exc).__name__} {exc}")
                continue
    log(f"DONE worker {args.worker}: scenes={n_scenes} frames={n_frames} skipped={n_skip}")


if __name__ == "__main__":
    main()
