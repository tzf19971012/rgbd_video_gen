"""Pose-only DL3DV audit; compare full-view and 16-view linspace random 6+3 draws."""
import json,random
from pathlib import Path
from PIL import Image
import numpy as np
r=Path('/mnt/sfs_turbo/t00845730/data/DL3DV-ALL-960P/10K_extracted');out=Path('/mnt/sfs_turbo/t00845730/rgbd_video_gen_gae/reports/dl3dv_camera_span');out.mkdir(exist_ok=True)
rng=random.Random(42);vals={};rows=[];bad=[];dimensions={};lens={};image_total=0;missing_total=0

def add(k,x):vals.setdefault(k,[]).extend(np.asarray(x).reshape(-1).tolist())
def angles(rot):return np.degrees(np.arccos(np.clip((rot.reshape(len(rot),9)@rot.reshape(len(rot),9).T-1)/2,-1,1)))
for folder in sorted(r.iterdir()):
 if not folder.is_dir():continue
 f=folder/'transforms.json'
 if not f.exists():bad.append({'scene':folder.name,'reason':'missing transforms.json'});continue
 try:
  data=json.load(open(f));frames=data['frames'];poses=np.array([v['transform_matrix'] for v in frames],dtype=float)
  if len(poses)<9 or not np.isfinite(poses).all():raise ValueError('invalid or too few poses')
 except Exception as e:bad.append({'scene':folder.name,'reason':str(e)});continue
 files={p.stem:p for p in (folder/'images_4').iterdir() if p.suffix.lower() in ['.png','.jpg','.jpeg']}
 present=[i for i,v in enumerate(frames) if Path(v['file_path']).stem in files];missing=len(frames)-len(present);image_total+=len(present);missing_total+=missing
 if len(present)<9:bad.append({'scene':folder.name,'reason':'too few images'});continue
 with Image.open(files[Path(frames[present[0]]['file_path']).stem]) as im:dim=im.size
 key=f'{dim[0]}x{dim[1]}';dimensions[key]=dimensions.get(key,0)+1
 model=data.get('camera_model','unknown');lens[model]=lens.get(model,0)+1
 poses=poses[present];rot=poses[:,:3,:3];pos=poses[:,:3,3];allang=angles(rot);dist=np.linalg.norm(pos[:,None]-pos[None],axis=-1);diam=dist.max();orth=np.max(np.abs(rot.transpose(0,2,1)@rot-np.eye(3)));det=np.linalg.det(rot)
 row={'scene_id':folder.name,'frames':len(poses),'missing_images':missing,'max_rotation_deg':float(allang.max()),'rotation_orthogonality_max_error':float(orth),'det_min':float(det.min()),'det_max':float(det.max()),'coordinate_translation_diameter':float(diam)}
 for mode in ['full','linspace16']:
  pool=list(range(len(poses))) if mode=='full' else np.linspace(0,len(poses)-1,min(16,len(poses)),dtype=int).tolist()
  elev_obs=[]
  for _ in range(32):
   tgt=rng.sample(pool,3);src=rng.sample([i for i in pool if i not in tgt],6);rng.shuffle(src)
   cross=allang[np.ix_(tgt,src)];add(mode+'_nearest_rotation_deg',cross.min(1));add(mode+'_all_pair_rotation_deg',cross)
   add(mode+'_nearest_translation_over_diameter',dist[np.ix_(tgt,src)].min(1)/max(diam,1e-9))
   # NeRF/OpenGL convention assumed: forward=-Z, up=+Y. Joint sign
   # reversal leaves these absolute elevation metrics invariant.
   up=rot[src,:,1].mean(0);up/=max(np.linalg.norm(up),1e-9);forward=-rot[:,:,2]
   elev=np.degrees(np.arcsin(np.clip(forward@up,-1,1)));gap=np.abs(elev[tgt,None]-elev[src][None]).min(1)
   add(mode+'_nearest_elevation_gap_deg',gap);add(mode+'_target_elevation_abs_deg',np.abs(elev[tgt]));elev_obs.extend(gap.tolist())
  row[mode+'_fraction_elevation_gap_ge10']=float((np.array(elev_obs)>=10).mean())
 rows.append(row)
for k in ['frames','max_rotation_deg']:vals[k]=[x[k] for x in rows]
summary={'scenes':len(rows),'invalid_directories':bad,'matched_frames':image_total,'missing_frame_files':missing_total,'image_dimensions_scene_counts':dimensions,'camera_models':lens,'sampling':'32 draws per scene per scheme, seed42, target-first random 3+6; equal scene weight; linspace16 approximates planned cache, not existing training','caveat':'Pose-only; no visibility/depth check; translation is SfM coordinate scale, normalized by trajectory diameter; elevation is relative to source mean up, not gravity. No distortion correction applied.'}
for k,v in vals.items():
 x=np.array(v);summary[k]={'count':len(x),'percentiles':dict(zip(['p10','p50','p75','p90','p95','p99','max'],map(float,np.percentile(x,[10,50,75,90,95,99,100])))),'fraction_ge':{str(t):float((x>=t).mean()) for t in ([.05,.1,.2,.5] if 'translation' in k else [5,10,15,20,30,45,60,80])}}
(out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'per_scene.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows));print(json.dumps(summary,indent=2))
