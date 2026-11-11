import argparse,csv,fcntl,hashlib,json,os,re,shutil,struct,subprocess,threading,time,traceback,zipfile
from pathlib import Path
R=Path(__file__).resolve().parent;CFG=json.loads((R/'config.json').read_text());BASE=R.parent.parent
LOCK=threading.RLock();STOP=threading.Event();JOBS={};ACTIVE={'status':'idle'}
for d in ['states','pins','jobs','logs']:(R/d).mkdir(exist_ok=True)

def atomic(path,value):
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2));os.replace(tmp,path)

def log(message):print(time.strftime('[%F %T]'),message,flush=True)

def ready(checkpoint):
    try:
        p=checkpoint/'model.safetensors';st=p.stat()
        if time.time()-st.st_mtime<10:return None
        with p.open('rb') as fh:
            n=struct.unpack('<Q',fh.read(8))[0]
            if n>100_000_000:return None
            raw=fh.read(n);header=json.loads(raw)
        end=max(v['data_offsets'][1] for k,v in header.items() if k!='__metadata__')
        if st.st_size!=8+n+end:return None
        if not all(zipfile.is_zipfile(checkpoint/name) for name in ['optimizer.bin','scheduler.bin']):return None
        if not all(zipfile.is_zipfile(checkpoint/f'random_states_{i}.pkl') for i in range(3)):return None
        return {'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'header_sha256':hashlib.sha256(raw).hexdigest()}
    except (OSError,ValueError,KeyError,struct.error):return None

def save(job):atomic(R/'states'/(job['id']+'.json'),job)

def free_gib(card):
    text=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=15).stdout;current=None
    for line in text.splitlines():
        m=re.match(r'\|\s*(\d+)\s+910',line)
        if m:current=int(m[1]);continue
        if current==card:
            pairs=re.findall(r'(\d+)\s*/\s*(\d+)',line)
            if pairs and int(pairs[-1][1])>=32000:
                used,total=map(int,pairs[-1]);return (total-used)/1024
    return 0.

def reconcile(job):
    p=R/'jobs'/job['id']/'complete.json'
    if p.exists():
        result=json.loads(p.read_text())
        if result['step']==job['step'] and result['projection_mode']==job['arm']:
            if job['status']!='complete':job.update(status='complete',result=result,finished=time.time());save(job)
            return True
    return False

def scan():
    with LOCK:
        for job in JOBS.values():reconcile(job)
    for run in CFG['runs']:
        for checkpoint in sorted(Path(run['run_dir']).glob('checkpoint-step-*')):
            try:step=int(checkpoint.name.rsplit('-',1)[1])
            except ValueError:continue
            if step<CFG['first_step'] or step%CFG['interval'] or step>run['max_step']:continue
            key=run['arm']+'_step'+str(step)
            with LOCK:
                if key in JOBS:continue
            meta=ready(checkpoint)
            if meta is None:continue
            pin=R/'pins'/run['arm']/checkpoint.name;pin.mkdir(parents=True,exist_ok=True)
            try:
                if not (pin/'model.safetensors').exists():os.link(checkpoint/'model.safetensors',pin/'model.safetensors')
            except OSError:continue
            jobdir=R/'jobs'/key;jobdir.mkdir(exist_ok=True)
            shutil.copy2(Path(run['run_dir'])/'args.yaml',jobdir/'training_args.yaml')
            job={'id':key,'arm':run['arm'],'run_dir':run['run_dir'],'step':step,'checkpoint':str(pin),'source_checkpoint':str(checkpoint),'weights':meta,'status':'pending','attempts':0,'queued_at':time.time()}
            with LOCK:JOBS[key]=job;save(job)
            log('queued '+key)

def summaries():
    fields=['arm','step','scene','trajectory_psnr','trajectory_lpips','depth_absrel_aligned','depth_delta1','yaw0_psnr','yaw_large15_20_psnr','trajectory_video','lateral_video']
    with LOCK:done=sorted([dict(x) for x in JOBS.values() if x['status']=='complete'],key=lambda x:(x['step'],x['arm']))
    tmp=R/'summary.csv.tmp'
    with tmp.open('w',newline='') as fh:
        writer=csv.DictWriter(fh,fieldnames=fields);writer.writeheader()
        for job in done:
            s=job['result'];t=s['trajectory'];y=s['yaw'];writer.writerow(dict(zip(fields,[job['arm'],job['step'],s['scene'],t['tgt_rgb_psnr'],t['tgt_rgb_lpips'],t['tgt_depth_abs_rel_aligned'],t['tgt_depth_delta1'],y['yaw0_psnr'],y['large15_20_psnr'],s['videos']['trajectory'],s['videos']['lateral']])))
    os.replace(tmp,R/'summary.csv')
    # Keep only the newest successful pin per arm, plus every pending/running pin.
    for arm in [r['arm'] for r in CFG['runs']]:
        old=[j for j in done if j['arm']==arm]
        for job in old[:-1]:
            p=Path(job['checkpoint'])
            if p.is_dir() and p.parent==R/'pins'/arm:shutil.rmtree(p)

def process_alive(job):
    try:return str(R/'probe.py') in (Path('/proc')/str(job.get('pid',0))/'cmdline').read_bytes().decode()
    except OSError:return False

def worker():
    while not STOP.is_set():
        with LOCK:
            candidates=sorted([j for j in JOBS.values() if j['status'] in ['pending','retry_wait','running'] and j.get('retry_after',0)<=time.time()],key=lambda x:(x['step'],x['arm']))
            job=candidates[0] if candidates else None
        if not job:STOP.wait(3);continue
        if job['status']=='running' and process_alive(job):
            ACTIVE.update(status='adopting_existing',job=job['id']);STOP.wait(3);continue
        with LOCK:
            if reconcile(job):summaries();continue
        try:
            free=free_gib(CFG['npu'])
            if free<CFG['minimum_free_gib']:
                ACTIVE.update(status='waiting_for_memory',free_gib=free,job=job['id']);STOP.wait(10);continue
            for name,digest in CFG['source_hashes'].items():
                assert hashlib.sha256((Path(CFG['code_root'])/name).read_bytes()).hexdigest()==digest,'Training implementation changed: '+name
            env=os.environ.copy();env.update(json.loads((R/'runtime_env.json').read_text()));env.update(ASCEND_RT_VISIBLE_DEVICES=str(CFG['npu']),OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1',PROJECTION_EVAL_CODE_ROOT=CFG['code_root'])
            env['PYTHONPATH']=CFG['code_root']+os.pathsep+str(BASE)+os.pathsep+env.get('PYTHONPATH','')
            for k in ['RANK','LOCAL_RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT']:env.pop(k,None)
            with LOCK:job['attempts']+=1;job.update(status='running',started=time.time());save(job)
            logfile=R/'logs'/f"{job['id']}.attempt{job['attempts']}.log"
            command=['/data/l00580360/miniconda3/envs/wan14b/bin/python','-u',str(R/'probe.py'),'--run_dir',job['run_dir'],'--checkpoint',job['checkpoint'],'--scene_id',CFG['scene'],'--manifest',CFG['manifest'],'--cache_name',CFG['cache_name'],'--seed',str(CFG['seed']),'--num_inference_steps',str(CFG['sampling_steps']),'--job_dir',str(R/'jobs'/job['id'])]
            ACTIVE.update(status='running',job=job['id'],card=CFG['npu']);log('running '+job['id'])
            with logfile.open('wb') as logfh:
                child=subprocess.Popen(command,cwd=BASE,env=env,stdin=subprocess.DEVNULL,stdout=logfh,stderr=subprocess.STDOUT)
                with LOCK:job.update(pid=child.pid,log=str(logfile),command=command);save(job)
                rc=child.wait()
            with LOCK:
                if rc==0 and reconcile(job):log('complete '+job['id'])
                else:
                    text=logfile.read_text(errors='replace');oom='out of memory' in text.lower();retry=oom and job['attempts']<3
                    job.update(status='retry_wait' if retry else 'failed',returncode=rc,error_tail=text[-2000:],retry_after=time.time()+300 if retry else 0);save(job);log('failed '+job['id']+' rc='+str(rc))
            summaries()
        except Exception:
            with LOCK:job.update(status='failed',error_tail=traceback.format_exc());save(job)
            log('worker exception '+job['id']);traceback.print_exc()
        ACTIVE.update(status='idle')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scan-once',action='store_true');parser.add_argument('--stop',action='store_true');args=parser.parse_args()
    if args.stop:(R/'STOP').touch();print('Stop requested; the current evaluation may finish first.');return
    lock=(R/'monitor.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:print('Monitor already running');return
    for p in (R/'states').glob('*.json'):
        job=json.loads(p.read_text());JOBS[job['id']]=job
        if job['status']=='running' and not process_alive(job):job['status']='pending';save(job)
    scan();summaries()
    if args.scan_once:print(json.dumps({k:v['status'] for k,v in JOBS.items()}));return
    (R/'STOP').unlink(missing_ok=True)
    thread=threading.Thread(target=worker,daemon=True);thread.start()
    phase='monitoring'
    try:
        while not (R/'STOP').exists():
            scan()
            with LOCK:atomic(R/'status.json',{'status':phase,'heartbeat':time.time(),'worker':dict(ACTIVE),'jobs':list(JOBS.values()),'pid':os.getpid()})
            if all(any(j['arm']==run['arm'] and j['step']==run['max_step'] and j['status']=='complete' for j in JOBS.values()) for run in CFG['runs']):phase='complete';break
            STOP.wait(CFG['poll_seconds'])
    finally:
        STOP.set();thread.join()
        atomic(R/'status.json',{'status':'complete' if phase=='complete' else 'stopped','heartbeat':time.time(),'worker':dict(ACTIVE),'jobs':list(JOBS.values()),'pid':os.getpid()})

if __name__=='__main__':main()
