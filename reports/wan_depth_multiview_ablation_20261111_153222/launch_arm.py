import os,sys,subprocess,json,time,traceback
from pathlib import Path
p=Path(__file__).parent;arm=sys.argv[1];out=p/arm;out.mkdir(exist_ok=True)
state=dict(pid=os.getpid(),arm=arm,status="starting");(out/"launch.json").write_text(json.dumps(state))
try:
 if arm=="multiview_dit":
  state["status"]="extracting_features";(out/"launch.json").write_text(json.dumps(state))
  with (p/"extract.log").open("ab") as f:subprocess.run([sys.executable,"-u",str(p/"extract.py")],cwd=p,stdout=f,stderr=subprocess.STDOUT,check=True)
 state["status"]="training";(out/"launch.json").write_text(json.dumps(state))
 with (out/"run.log").open("ab") as f:subprocess.run([sys.executable,"-u",str(p/"train.py"),"--arm",arm],cwd=p,stdout=f,stderr=subprocess.STDOUT,check=True)
 state["status"]="complete"
except Exception:state.update(status="failed",error=traceback.format_exc())
(out/"launch.json").write_text(json.dumps(state,indent=2))
