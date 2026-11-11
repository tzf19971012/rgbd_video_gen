import os,json,time,subprocess,importlib.util
from pathlib import Path
r=Path(__file__).parent;camp=r.parent;b=camp.parent.parent;job=json.loads((camp/"launch.json").read_text());checkpoint=Path(job["output"])/"checkpoint-step-4000"
span=Path((b/"reports/latest_span_c2w_fresh.txt").read_text());spec=importlib.util.spec_from_file_location("guard",span/"oom_watchdog.py");guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
while True:
    proc=Path("/proc")/str(job["pid"])
    active=proc.exists() and job["config"] in (proc/"cmdline").read_bytes().decode(errors="replace")
    ready=guard.complete(checkpoint)
    (r/"status.json").write_text(json.dumps({"status":"waiting_for_training_finish" if ready else "waiting_for_step4000","checked_at":time.time()}))
    if ready and not active:break
    if not active and not ready:
        (r/"status.json").write_text(json.dumps({"status":"training_exited_without_complete_step4000"}));raise SystemExit(2)
    time.sleep(30)
pin=r/"checkpoint-step-4000";pin.mkdir(exist_ok=True)
if not (pin/"model.safetensors").exists():os.link(checkpoint/"model.safetensors",pin/"model.safetensors")
env=os.environ.copy();env.update(json.loads((camp/"runtime_env.json").read_text()));env.update(ASCEND_RT_VISIBLE_DEVICES="7",OMP_NUM_THREADS="4",PYTHONUNBUFFERED="1");env["PYTHONPATH"]=str(b)+os.pathsep+env.get("PYTHONPATH","")
for key in ["RANK","LOCAL_RANK","WORLD_SIZE","MASTER_ADDR","MASTER_PORT"]:env.pop(key,None)
cmd=[job["command"][0],"-u",str(r/"probe.py"),"--run_dir",job["output"],"--checkpoint",str(pin),"--seed","1234","--num_inference_steps","25"]
(r/"status.json").write_text(json.dumps({"status":"running","started":time.time()}))
with (r/"inference.log").open("wb") as log:result=subprocess.run(cmd,cwd=b,env=env,stdout=log,stderr=subprocess.STDOUT)
(r/"status.json").write_text(json.dumps({"status":"complete" if result.returncode==0 else "failed","returncode":result.returncode,"finished":time.time()}))
