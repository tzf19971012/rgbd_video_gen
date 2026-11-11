import json,subprocess,os,time,sys
from pathlib import Path
r=Path(sys.argv[1]); card=sys.argv[2]; arms=sys.argv[3:]
camp=r.parent;env=os.environ.copy();env.update(json.loads((camp/'runtime_env.json').read_text()));env['ASCEND_RT_VISIBLE_DEVICES']=card;env['OMP_NUM_THREADS']='8';env['PYTHONUNBUFFERED']='1'
for k in ['RANK','LOCAL_RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT']:env.pop(k,None)
for item in json.loads((r/'protocol.json').read_text())['commands']:
 if item['arm'] not in arms:continue
 start=time.time()
 with (r/(item['arm']+'.log')).open('wb') as log:
  p=subprocess.Popen(item['command'],cwd=camp.parent.parent,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  rc=p.wait()
 (r/(item['arm']+'.status.json')).write_text(json.dumps({'returncode':rc,'started':start,'finished':time.time(),'device':card}))
 if rc:sys.exit(rc)
