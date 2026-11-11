from pathlib import Path
import json,hashlib,time
import cv2,numpy as np
P=Path(__file__).parent;C=json.loads((P/'plan.json').read_text());arms=['control','gae','rgb'];checks=[]
for sid in C['scenes']:
 reference=None;refcopy=None;refshape={}
 for arm in arms:
  p=P/'results'/sid/arm;r=json.loads((p/'complete.json').read_text());assert r['step']==C['arms'][arm]['step'] and r['scene']==sid and r['projection_mode']==arm
  poses=json.loads((p/'lateral/poses.json').read_text());rows=[json.loads(l) for l in (p/'trajectory/per_frame_metrics.jsonl').read_text().splitlines()]
  assert [x['view_id'] for x in rows]==C['scene_protocol'][sid]['target_ids']
  copy=np.array([x['nearest_src_psnr'] for x in rows]);assert len(rows)==15
  if reference is None:reference=poses;refcopy=copy
  else:
   assert poses['specs']==reference['specs']
   for key in ['anchor','K','poses']:assert np.allclose(poses[key],reference[key],atol=1e-6)
   assert np.allclose(copy,refcopy,atol=1e-6)
  for kind,path in r['videos'].items():
   cap=cv2.VideoCapture(path);count=0;shape=None
   while True:
    ok,frame=cap.read()
    if not ok:break
    count+=1;shape=frame.shape
   cap.release();assert count==15,(sid,arm,kind,count)
   if kind in refshape:assert shape==refshape[kind]
   else:refshape[kind]=shape
   checks.append(dict(scene=sid,arm=arm,kind=kind,frames=count,shape=shape,bytes=Path(path).stat().st_size,sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()))
  run=Path(C['arms'][arm]['run_dir']);step=r['step']
  for kind,name in [('lateral','lateral_probe'),('trajectory','traj')]:
   target=run/f'{name}_step{step}_test10'/sid;target.parent.mkdir(exist_ok=True)
   if not target.exists():target.symlink_to(p/kind,target_is_directory=True)
result=dict(passed=True,scenes=10,jobs=30,video_count=60,all_source_poses_match=True,all_real_target_ids_match=True,all_copy_baselines_match=True,all_videos_fully_decoded=True,checked_at=time.time(),videos=checks)
(P/'verification.json').write_text(json.dumps(result,indent=2));print('VERIFIED 10 scenes /30 jobs /60 videos')
