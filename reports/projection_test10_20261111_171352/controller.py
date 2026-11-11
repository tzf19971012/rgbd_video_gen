import os,json,time,subprocess,concurrent.futures,fcntl,threading,traceback
from pathlib import Path
P=Path(__file__).parent;C=json.loads((P/'plan.json').read_text());B=Path(C['base']);LOCK=threading.RLock();STOP=threading.Event();JOBS=[];CARD_LOCKS={a:threading.Lock() for a in [2,3,4]}
def atomic(path,obj):
 q=path.with_suffix(path.suffix+'.tmp');q.write_text(json.dumps(obj,indent=2));q.replace(path)
def refresh():
 with LOCK:
  data=dict(status='complete' if all(j['status']=='complete' for j in JOBS) else 'running',heartbeat=time.time(),jobs=JOBS,scenes=C['scenes'],arms=C['arms'],completed=sum(j['status']=='complete' for j in JOBS),total=len(JOBS),started=C['started'],report='report.html')
  atomic(P/'progress.json',data)
def work(job):
 with CARD_LOCKS[job.get("assigned_card",C["cards"][job["arm"]])]:
  return work_locked(job)
def work_locked(job):
 d=P/'results'/job['scene']/job['arm'];d.mkdir(parents=True,exist_ok=True)
 marker=d/'complete.json'
 if job.get('adopt_pid'):
  oldpid=job['adopt_pid']
  while Path('/proc/'+str(oldpid)).exists() and not marker.exists():
   with LOCK:job.update(status='running',pid=oldpid,adopted=True)
   refresh();time.sleep(5)
 if marker.exists():
  with LOCK:job.update(status='complete',result=json.loads(marker.read_text()))
  refresh();return
 a=C['arms'][job['arm']]
 env=os.environ.copy();env.update(json.loads((P/'runtime_env.json').read_text()));env.update(ASCEND_RT_VISIBLE_DEVICES=str(job.get('assigned_card',C['cards'][job['arm']])),OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1',PROJECTION_EVAL_CODE_ROOT=C['projection_code'])
 env['PYTHONPATH']=(C['projection_code']+os.pathsep if job['arm']!='control' else '')+str(B)+os.pathsep+env.get('PYTHONPATH','')
 for k in ['RANK','LOCAL_RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT']:env.pop(k,None)
 cmd=[C['python'],'-u',str(P/'probe.py'),'--arm',job['arm'],'--run_dir',a['run_dir'],'--checkpoint',a['checkpoint'],'--scene_id',job['scene'],'--job_dir',str(d),'--manifest',C['manifest'],'--cache_name','re10k_test','--num_inference_steps','25','--seed','1234']
 for attempt in range(1,3):
  with LOCK:job.update(status='running',started=time.time(),attempt=attempt)
  refresh()
  log=d/f'attempt{attempt}.log'
  with log.open('wb') as f:
   child=subprocess.Popen(cmd,cwd=B,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
   with LOCK:job.update(pid=child.pid,card=job.get('assigned_card',C['cards'][job['arm']]))
   refresh();rc=child.wait()
  if rc==0 and marker.exists():
   with LOCK:job.update(status='complete',finished=time.time(),result=json.loads(marker.read_text()))
   print('COMPLETE',job['scene'],job['arm'],flush=True);break
  tail=log.read_text(errors='replace')[-4000:]
  if 'out of memory' in tail.lower() and attempt==1:
   with LOCK:job.update(status='retry_wait',error=tail)
   refresh();time.sleep(45);continue
  with LOCK:job.update(status='failed',finished=time.time(),returncode=rc,error=tail)
  print('FAILED',job['scene'],job['arm'],tail[-800:],flush=True);break
 refresh()
 subprocess.run([C['python'],str(P/'report.py')],cwd=P,check=False)
def ticker():
 while not STOP.wait(10):refresh()
def main():
 own=(P/'controller.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
 (P/'controller.pid').write_text(str(os.getpid()))
 # The regular1000-step monitor's future probes take the same exclusive gate.
 # Its checkpoint scanner keeps running and pins new weights while waiting.
 gate=Path(C['card_gate']).open('a');fcntl.flock(gate,fcntl.LOCK_EX)
 previous=json.loads((P/'progress.json').read_text()) if (P/'progress.json').exists() else {'jobs':[]}
 oldjobs={(j['scene'],j['arm']):j for j in previous['jobs']}
 for scene in C['scenes']:
  for arm in ['control','gae','rgb']:
   j=dict(scene=scene,arm=arm,status='pending',card=C['cards'][arm]);oldj=oldjobs.get((scene,arm),{})
   if oldj.get('status')=='running' and oldj.get('pid') and Path('/proc/'+str(oldj['pid'])).exists():j.update(adopt_pid=oldj['pid'],pid=oldj['pid'],assigned_card=oldj.get('card',C['cards'][arm]),card=oldj.get('card',C['cards'][arm]))
   JOBS.append(j)
 refresh();threading.Thread(target=ticker,daemon=True).start()
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  pending=[j for j in JOBS if not j.get("adopt_pid")]
  def run_card(card):
   for job in JOBS:
    if job.get("adopt_pid") and job.get("assigned_card")==card:work(job)
   while True:
    with LOCK:
     if not pending:return
     job=pending.pop(0);job["assigned_card"]=card;job["card"]=card
    work(job)
  futures=[pool.submit(run_card,card) for card in [4,2,3]]
  for f in concurrent.futures.as_completed(futures):
   try:f.result()
   except Exception:print(traceback.format_exc(),flush=True)
 STOP.set();refresh();subprocess.run([C['python'],str(P/'report.py')],cwd=P,check=True)
 fcntl.flock(gate,fcntl.LOCK_UN)
 print('FINISHED',sum(j['status']=='complete' for j in JOBS),'/',len(JOBS),flush=True)
if __name__=='__main__':main()
