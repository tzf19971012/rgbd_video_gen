from pathlib import Path
import json,os,subprocess,time
r=Path(__file__).parent;a=json.loads((r/"protocol.json").read_text())["commands"][1];t=time.time()
with (r/"control8200.log").open("wb") as log:p=subprocess.run(a["command"],cwd=r.parent.parent,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
(r/"control8200.correctstats.status.json").write_text(json.dumps(dict(returncode=p.returncode,started=t,finished=time.time())))
