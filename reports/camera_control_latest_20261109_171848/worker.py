import os,json,sys,time,subprocess
from pathlib import Path
r=Path(sys.argv[1]);card=sys.argv[2];env=os.environ.copy();env.update(json.loads((r/'runtime_env.json').read_text()));env.update(ASCEND_RT_VISIBLE_DEVICES=card,OMP_NUM_THREADS='2',PYTHONUNBUFFERED='1');env['PYTHONPATH']=str(r.parent.parent)+os.pathsep+env.get('PYTHONPATH','')
for k in ['RANK','LOCAL_RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT']:env.pop(k,None)
for a in json.loads((r/'protocol.json').read_text())['commands']:
 if a['arm'] not in sys.argv[3:]:continue
 t=time.time()
 with (r/(a['arm']+'.log')).open('wb') as log:p=subprocess.Popen(a['command'],cwd=r.parent.parent,env=env,stdout=log,stderr=subprocess.STDOUT);rc=p.wait()
 (r/(a['arm']+'.correctstats.status.json')).write_text(json.dumps(dict(returncode=rc,started=t,finished=time.time(),card=card)))
