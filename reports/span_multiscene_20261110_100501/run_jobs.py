import os,sys,json,subprocess
from pathlib import Path
r=Path(sys.argv[1]);card=sys.argv[2];env=os.environ.copy();env.update(json.loads((r/'runtime_env.json').read_text()));env.update(ASCEND_RT_VISIBLE_DEVICES=card,OMP_NUM_THREADS='2');env['PYTHONPATH']=str(r.parent.parent)+':'+env.get('PYTHONPATH','')
for arm in sys.argv[3:]:
 with (r/(arm+'.log')).open('wb') as log:subprocess.run(['/data/l00580360/miniconda3/envs/wan14b/bin/python','-u',str(r/'evaluate.py'),str(r),arm],cwd=r.parent.parent,env=env,stdout=log,stderr=subprocess.STDOUT)
