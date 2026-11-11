import os,sys,json,time,traceback
from pathlib import Path
r=Path(sys.argv[1]);arm=sys.argv[2];sys.path.insert(0,str(r));sys.path.insert(1,str(r.parent.parent))
import torch
import real_trajectory as real
p=json.loads((r/'protocol.json').read_text());a=next(a for a in p['arms'] if a['arm']==arm)
original=real.build_model;resident=[]
def build(*args,**kw):
 if not resident:
  model=original(*args,**kw);model.eval_num_inference_steps=25;resident.append(model)
 return resident[0]
real.build_model=build
try:
 for sid in p['scenes']:
  start=time.time();out=r/arm/sid
  if resident:resident[0]._latent_file_cache.clear()
  with torch.no_grad():real.run_scene(a['run'],a['checkpoint'],sid,6,15,str(out),shared_target_noise=True,manifest=p['manifest'],cache_name=p['cache'])
  real._synthesize_video(str(out),arm,a['step'],sid,fps=3)
  (out/'status.json').write_text(json.dumps(dict(returncode=0,seconds=time.time()-start)))
 (r/(arm+'.status.json')).write_text(json.dumps(dict(returncode=0)))
except Exception:
 (r/(arm+'.status.json')).write_text(json.dumps(dict(returncode=1,error=traceback.format_exc())));raise
