import json,time,subprocess,os
from pathlib import Path
r=Path(__file__).parent;arms=[a["arm"] for a in json.loads((r/"protocol.json").read_text())["commands"]]
while True:
 paths=[r/(a+".correctstats.status.json") for a in arms]
 if all(p.exists() for p in paths):break
 time.sleep(20)
if any(json.loads(p.read_text())["returncode"] for p in paths):
 (r/"report.status.json").write_text(json.dumps({"status":"inference_failure_check_logs"}));raise SystemExit(2)
env=os.environ.copy();env.update(json.loads((r/"runtime_env.json").read_text()));env["TORCH_DEVICE_BACKEND_AUTOLOAD"]="0";env["PYTHONPATH"]=str(r.parent.parent)+os.pathsep+env.get("PYTHONPATH","")
with (r/"summarize.log").open("wb") as log:
 result=subprocess.run(["/data/l00580360/miniconda3/envs/wan14b/bin/python",str(r/"summarize.py")],env=env,cwd=r.parent.parent,stdout=log,stderr=subprocess.STDOUT)
(r/"report.status.json").write_text(json.dumps({"returncode":result.returncode,"finished":time.time()}))
