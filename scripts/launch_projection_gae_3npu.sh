#!/usr/bin/env bash
# Resume the GAE projection experiment with its frozen training implementation.
set -euo pipefail
PROJ_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PROJ_CAMPAIGN="$PROJ_REPO/reports/projection_ablation_20261109_113836/resume_3card_20261110_110818"
PROJ_CONFIG="$PROJ_CAMPAIGN/gae.yaml"
PROJ_DEVICES="${PROJECTION_DEVICES:-0,3,6}"
PROJ_PORT="${PROJECTION_PORT:-29642}"
PROJ_RESUME="latest"
PROJ_DRY_RUN=0
while (($#)); do
  case "$1" in
    --config) PROJ_CONFIG="${2:?missing config path}"; shift 2 ;;
    --devices) PROJ_DEVICES="${2:?missing devices}"; shift 2 ;;
    --port) PROJ_PORT="${2:?missing port}"; shift 2 ;;
    --resume) PROJ_RESUME="${2:?use latest or configured}"; shift 2 ;;
    --dry-run) PROJ_DRY_RUN=1; shift ;;
    -h|--help)
      cat <<'HELP'
Usage: bash scripts/launch_projection_gae_3npu.sh [options]
  --devices 0,3,6       Three physical NPUs (default:0,3,6)
  --port 29642          torchrun master port
  --config PATH        Campaign GAE YAML
  --resume latest      Resume latest complete checkpoint; fall back to YAML
  --resume configured  Use the YAML's original checkpoint (initially step1600)
  --dry-run            Print resolved command/config; do not launch training
The script refuses a duplicate active experiment. Run in the foreground, or use
nohup and redirect stdout/stderr to a log. Existing training is never stopped.
HELP
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
cd "$PROJ_REPO"
exec /data/l00580360/miniconda3/envs/wan14b/bin/python - \
  "$PROJ_REPO" "$PROJ_CAMPAIGN" "$PROJ_CONFIG" "$PROJ_DEVICES" \
  "$PROJ_PORT" "$PROJ_RESUME" "$PROJ_DRY_RUN" <<'PY'
import os,sys,json,shlex,struct,zipfile,time
from pathlib import Path
from omegaconf import OmegaConf
root,campaign,config,devices,port,resume,dry=sys.argv[1:]
root=Path(root);campaign=Path(campaign);config=Path(config).resolve();dry=dry=='1'
train=campaign/'code/train.py'
assert train.is_file() and config.is_file(), 'Missing frozen training source or config'
assert resume in ('latest','configured'), '--resume must be latest or configured'
assert len(devices.split(','))==3 and len(set(devices.split(',')))==3, 'This campaign requires3 distinct NPUs'
assert all(x.strip().isdigit() for x in devices.split(',')), 'Invalid NPU IDs'
assert port.isdigit() and 1024<=int(port)<=65535, 'Invalid port'
cfg=OmegaConf.load(config);assert cfg.projection_condition=='gae'
active=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  argv=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
  if str(train) not in argv or '--config' not in argv:continue
  other=Path(argv[argv.index('--config')+1])
  if other==config or str(OmegaConf.load(other).output_path)==str(cfg.output_path):active.append(int(proc.name))
 except (OSError,ValueError,IndexError,KeyError):continue
print('[projection-gae] active matching PIDs:',active,flush=True)
if active and not dry:raise SystemExit('Experiment already running; duplicate launch refused.')
selected=config
if resume=='latest':
 checkpoints=sorted(Path(cfg.output_path).glob('checkpoint-step-*'),key=lambda p:int(p.name.rsplit('-',1)[1]),reverse=True)
 for p in checkpoints:
  required=['model.safetensors','optimizer.bin','scheduler.bin']+[f'random_states_{i}.pkl' for i in range(3)]
  if not all((p/f).is_file() for f in required):continue
  with (p/'model.safetensors').open('rb') as f:n=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(n))
  expected=max(v['data_offsets'][1] for k,v in header.items() if k!='__metadata__')+8+n
  if expected!=(p/'model.safetensors').stat().st_size or not all(zipfile.is_zipfile(p/f) for f in required[1:]):continue
  cfg.training_state_dir=str(p);cfg.global_step=int(p.name.rsplit('-',1)[1]);cfg.resume=True;cfg.resume_weights_only=False
  selected=campaign/('gae_resume_latest_'+time.strftime('%Y%m%d_%H%M%S')+'.yaml')
  if not dry:OmegaConf.save(cfg,selected)
  print('[projection-gae] resume full state:',p,flush=True);break
runtime=root/'reports/span_ablation_20261106_164801/runtime_env.json'
env=os.environ.copy()
if runtime.is_file():env.update(json.loads(runtime.read_text()))
env['ASCEND_RT_VISIBLE_DEVICES']=devices;env['PYTHONUNBUFFERED']='1';env.setdefault('OMP_NUM_THREADS','4')
env['PYTHONPATH']=str(campaign/'code')+os.pathsep+str(root)+os.pathsep+env.get('PYTHONPATH','')
for key in ['RANK','LOCAL_RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT']:env.pop(key,None)
cmd=[sys.executable,'-u','-m','torch.distributed.run','--nproc_per_node=3','--master_port='+port,str(train),'--config',str(selected)]
print('[projection-gae] devices:',devices,'effective batch:',int(cfg.batch_size)*int(cfg.gradient_accumulation_steps)*3,flush=True)
print('[projection-gae] PYTHONPATH:',str(campaign/'code')+os.pathsep+str(root),flush=True)
print('[projection-gae] command:',shlex.join(cmd),flush=True)
if dry:print('[projection-gae] dry run; training was not launched.');raise SystemExit(0)
os.chdir(root);os.execvpe(cmd[0],cmd,env)
PY
