"""Persistent four-arm OOM supervisor. Only owns jobs listed by this campaign."""
import argparse,fcntl,json,os,re,shutil,signal,struct,subprocess,time,zipfile
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parent
OOM=re.compile(r'out of memory|acl_error_rt_memory_allocation|memory allocation (?:failed|error)|alloc(?:ate)? (?:device )?memory failed',re.I)
LEVELS=[12,6,4,2,1]

def write_json(path,value):
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2));os.replace(tmp,path)

def alive(pid):
    p=Path('/proc')/str(pid)
    if not p.exists():return False
    try:return p.joinpath('stat').read_text().split(') ',1)[1].split()[0]!='Z'
    except OSError:return False

def complete(path):
    try:
        model=path/'model.safetensors'
        with model.open('rb') as fh:
            size=struct.unpack('<Q',fh.read(8))[0]
            if size>100000000:return False
            header=json.loads(fh.read(size))
        end=max(v['data_offsets'][1] for k,v in header.items() if k!='__metadata__')
        if model.stat().st_size!=8+size+end:return False
        return all(zipfile.is_zipfile(path/name) for name in ['optimizer.bin','scheduler.bin','random_states_0.pkl'])
    except (OSError,ValueError,KeyError,struct.error):return False

def pin_common(jobs):
    sets=[{int(p.name.rsplit('-',1)[1]) for p in Path(a['output']).glob('checkpoint-step-*') if p.is_dir()} for a in jobs]
    steps=set.intersection(*sets)
    recovery=ROOT/'oom_recovery';recovery.mkdir(exist_ok=True)
    record=recovery/'current.json';old=json.loads(record.read_text()) if record.exists() else None
    for step in sorted(steps,reverse=True):
        if old and step<=old['step']:return old
        sources={a['arm']:Path(a['output'])/f'checkpoint-step-{step}' for a in jobs}
        if not all(complete(p) for p in sources.values()):continue
        target=recovery/f'checkpoint-step-{step}';tmp=recovery/f'.pin-{step}'
        try:
            if tmp.exists():shutil.rmtree(tmp)
            tmp.mkdir()
            for arm,path in sources.items():shutil.copytree(path,tmp/arm,copy_function=os.link)
            os.replace(tmp,target)
        except OSError:
            if tmp.exists():shutil.rmtree(tmp)
            continue
        value={'step':step,'paths':{arm:str(target/arm) for arm in sources},'pinned_at':time.time()}
        write_json(record,value)
        # Only remove older snapshots owned by this supervisor; training assets are untouched.
        for p in recovery.glob('checkpoint-step-*'):
            if p!=target and p.is_dir():shutil.rmtree(p)
        return value
    return old

def stop_owned(jobs):
    for a in jobs:
        pid=a['pid']
        if not alive(pid):continue
        command=(Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        if a['config'] not in command or str(ROOT/'train_bs12.py') not in command or os.getpgid(pid)!=pid:
            raise RuntimeError(f'Refusing to stop unexpected process {pid}')
        os.killpg(pid,signal.SIGTERM)
    for _ in range(20):
        if not any(alive(a['pid']) for a in jobs):return
        time.sleep(1)
    for a in jobs:
        if alive(a['pid']):os.killpg(a['pid'],signal.SIGKILL)
    time.sleep(2)

def restart(jobs,recovery,batch,attempt):
    base=ROOT.parent.parent
    for a in jobs:
        cfg=yaml.safe_load(Path(a['config']).read_text())
        cfg.update(batch_size=batch,gradient_accumulation_steps=12//batch,
                   resume=bool(recovery),resume_weights_only=False,
                   training_state_dir=recovery['paths'][a['arm']] if recovery else None,
                   global_step=recovery['step'] if recovery else 0,
                   eval_batch_size=min(int(cfg.get('eval_batch_size',2)),batch))
        # Separate config per attempt retains the exact original launch settings.
        config=ROOT/f"{a['arm']}.bs{batch}.attempt{attempt}.yaml"
        config.write_text(yaml.safe_dump(cfg,sort_keys=False))
        env=os.environ.copy();env.update(json.loads((ROOT/'runtime_env.json').read_text()))
        env.update(ASCEND_RT_VISIBLE_DEVICES=str(a['npu']),OMP_NUM_THREADS='8',PYTHONUNBUFFERED='1')
        env['PYTHONPATH']=str(base)+os.pathsep+env.get('PYTHONPATH','')
        for key in ['RANK','LOCAL_RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT']:env.pop(key,None)
        cmd=[a['command'][0],'-u',str(ROOT/'train_bs12.py'),'--config',str(config)]
        logpath=ROOT/f"{a['arm']}.bs{batch}.attempt{attempt}.log"
        with logpath.open('wb') as log:
            process=subprocess.Popen(cmd,cwd=base,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        a.update(previous_pid=a['pid'],pid=process.pid,config=str(config),command=cmd,log=str(logpath),launched_at=time.time())
    write_json(ROOT/'launches.json',jobs)

def inspect(jobs):
    result=[]
    for a in jobs:
        p=Path(a['log']);text=p.read_text(errors='replace')[-100000:] if p.exists() else ''
        steps=[int(x) for x in re.findall(r'step (\d+): loss=',text)]
        healthy=re.findall(r'\[batch12-health\] (\{[^\n]+\})',text)
        health=json.loads(healthy[-1]) if healthy else None
        step=max(steps+([health['step']] if health else [])+[0])
        result.append({'arm':a['arm'],'pid':a['pid'],'alive':alive(a['pid']),'oom':bool(OOM.search(text)),'step_observed':step,'health':health})
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--once',action='store_true');args=parser.parse_args()
    lock=(ROOT/'oom_watchdog.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    statepath=ROOT/'oom_watchdog_state.json';state=json.loads(statepath.read_text()) if statepath.exists() else {'attempt':0,'events':[]}
    while True:
        jobs=json.loads((ROOT/'launches.json').read_text());recovery=pin_common(jobs);checks=inspect(jobs)
        state.update(checked_at=time.time(),runs=checks,recovery=recovery,status='monitoring')
        failed=[x for x in checks if x['oom'] and (not x['alive'] or 'Traceback' in Path(next(a['log'] for a in jobs if a['arm']==x['arm'])).read_text(errors='replace')[-100000:])]
        if failed and not args.once:
            batch=int(yaml.safe_load(Path(jobs[0]['config']).read_text())['batch_size']);next_batch=next((v for v in LEVELS if v<batch),None)
            if next_batch is None:
                stop_owned(jobs);state.update(status='stopped_oom_at_bs1');write_json(statepath,state);return
            stop_owned(jobs);recovery=pin_common(jobs)
            if not recovery and any(list(Path(a['output']).glob('checkpoint-step-*')) for a in jobs):
                state.update(status='stopped_no_verified_common_checkpoint');write_json(statepath,state);return
            state['attempt']+=1
            event={'time':time.time(),'trigger':[x['arm'] for x in failed],'from_batch':batch,'to_batch':next_batch,'effective_batch':12,'resume_step':recovery['step'] if recovery else 0}
            state['events'].append(event);write_json(statepath,state)
            restart(jobs,recovery,next_batch,state['attempt']);state['status']='restarted_after_oom'
            print(json.dumps(event),flush=True)
        elif any(not x['alive'] for x in checks):
            state['status']='process_exited_check_logs'
            if all(not x['alive'] and x['step_observed']>=10000 for x in checks):state['status']='completed'
        write_json(statepath,state)
        if args.once or state['status']=='completed':return
        time.sleep(30)

if __name__=='__main__':main()
