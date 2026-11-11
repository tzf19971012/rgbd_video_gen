"""Pose-only audit of cache-restricted 6-source/3-target random sampling."""
import json,random
from pathlib import Path
import numpy as np
R=Path('/mnt/sfs_turbo/t00845730');out=R/'rgbd_video_gen_gae/reports/re10k_camera_span';out.mkdir(exist_ok=True)
rng=random.Random(42);values={};scenes=[];dropped=0

def add(k,x):values.setdefault(k,[]).extend(np.asarray(x).reshape(-1).tolist())
def angles(rot):
 flat=rot.reshape(len(rot),9);cos=(flat@flat.T-1)/2
 return np.degrees(np.arccos(np.clip(cos,-1,1)))
for line in open(R/'re10k/manifest_train.jsonl'):
 rec=json.loads(line);sid=rec['scene_id'];meta=R/f'gae_latent_cache/re10k_train/{sid}.meta.json'
 if not meta.exists():dropped+=1;continue
 allowed={str(x) for x in json.load(open(meta))['views']};allviews=rec['views'];v=[x for x in allviews if str(x['view_id']) in allowed]
 if len(v)<9:dropped+=1;continue
 poses=np.array([x['c2w'] for x in v],dtype=float);rot=poses[:,:3,:3];pos=poses[:,:3,3];a=angles(rot);dist=np.linalg.norm(pos[:,None]-pos[None],axis=-1);diam=dist.max();forward=rot[:,:,2];fangle=np.degrees(np.arccos(np.clip(forward@forward.T,-1,1)))
 fullrot=np.array([x['c2w'] for x in allviews],dtype=float)[:,:3,:3]
 fullmax=float(angles(fullrot).max())
 # Same target-first random selection as dataset._select_views, 32 draws/scene.
 for _ in range(32):
  tgt=rng.sample(range(len(v)),3);src=rng.sample([i for i in range(len(v)) if i not in tgt],6);rng.shuffle(src)
  cross=a[np.ix_(tgt,src)];cd=dist[np.ix_(tgt,src)]
  add('target_to_source_rotation_deg_all_pairs',cross)
  add('target_nearest_source_rotation_deg',cross.min(1));add('target_farthest_source_rotation_deg',cross.max(1))
  add('target_nearest_source_translation_over_scene_diameter',cd.min(1)/max(diam,1e-9))
  up=-rot[src,:,1].mean(0);up/=max(np.linalg.norm(up),1e-9)
  elevations=np.degrees(np.arcsin(np.clip(forward@up,-1,1)))
  add('target_nearest_source_elevation_gap_deg',np.abs(elevations[tgt,None]-elevations[src][None]).min(1))
  add('target_elevation_relative_source_up_abs_deg',np.abs(elevations[tgt]))
 scenes.append(dict(scene_id=sid,manifest_views=len(allviews),cached_views=len(v),full_max_rotation_deg=fullmax,cache_max_rotation_deg=float(a.max()),cache_max_forward_angle_deg=float(fangle.max()),scene_translation_diameter=float(diam)))
summary={'scenes':len(scenes),'dropped':dropped,'draws_per_scene':32,'target_observations':len(scenes)*32*3,'seed':42,'note':'Pose-only Monte Carlo audit, not reconstruction of historical RNG. Translation normalized by cached trajectory diameter, not metric depth. Elevation uses sampled source mean up, not gravity.'}
for k in ['full_max_rotation_deg','cache_max_rotation_deg','cache_max_forward_angle_deg','cached_views']:values[k]=[s[k] for s in scenes]
for k,v in values.items():
 x=np.array(v);summary[k]={'count':len(x),'percentiles':dict(zip(['p10','p50','p75','p90','p95','p99','max'],map(float,np.percentile(x,[10,50,75,90,95,99,100])))),'fraction_ge':{str(t):float((x>=t).mean()) for t in ([.01,.05,.1,.2,.5] if 'translation' in k else [5,10,15,20,30,45,60,80])}}
(out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'per_scene.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in scenes));print(json.dumps(summary,indent=2))
