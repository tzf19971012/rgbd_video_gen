import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
r=Path(__file__).parent;data=json.loads((r/"training_analysis.json").read_text());metrics=data["metrics"]
fig,axes=plt.subplots(2,2,figsize=(11,7))
for ax,key in zip(axes.ravel(),["tgt_rgb_psnr","tgt_rgb_lpips","tgt_depth_abs_rel_aligned","tgt_depth_delta1"]):
 for arm,rows in metrics.items():ax.plot([x["step"] for x in rows],[x[key] for x in rows],marker="o",label=arm)
 ax.set_title(key);ax.set_xlabel("Optimizer step");ax.grid(alpha=.25)
axes[0,0].legend();fig.tight_layout();fig.savefig(r/"metric_trends.png",dpi=160)
jobs=json.loads((r.parent/"launches.json").read_text());canvas=Image.new("RGB",(1008,954),(22,22,22));draw=ImageDraw.Draw(canvas)
for idx,a in enumerate(jobs):
 im=Image.open(Path(a["output"])/"eval_step_001400/b00_pred_rgb.jpg");w=im.width//9;h=im.height
 if idx==0:
  for j in range(6):canvas.paste(im.crop((j*w,0,(j+1)*w,h)).resize((168,94)),(j*168,0))
 y=94+idx*215;draw.text((6,y+5),a["arm"]+" step1400: generated targets",fill="white");canvas.paste(im.crop((6*w,0,9*w,h)).resize((1008,189)),(0,y+25))
canvas.save(r/"sample0_targets.jpg",quality=93)
