import json,time,os
from pathlib import Path
import numpy as np
B=Path('/mnt/sfs_turbo/t00845730/rgbd_video_gen_gae');r=Path((B/'reports/latest_span_multiscene.txt').read_text().strip());p=json.loads((r/'protocol.json').read_text());arms=[a['arm'] for a in p['arms']]
while not all((r/(a+'.status.json')).exists() for a in arms):time.sleep(20)
for a in arms:
 status=json.loads((r/(a+'.status.json')).read_text())
 if status['returncode']:(r/'report.status.json').write_text(json.dumps(dict(returncode=1,arm=a,error=status)));raise SystemExit(1)
keys=['tgt_rgb_psnr','tgt_rgb_lpips','tgt_depth_abs_rel_aligned','tgt_depth_delta1','nearest_src_psnr'];scores={};per={}
for arm in arms:
 per[arm]={}
 for sid in p['scenes']:
  root=r/arm/sid;rows=[json.loads(x) for x in (root/'per_frame_metrics.jsonl').read_text().splitlines()]
  ref=r/arms[0]/sid;rr=[json.loads(x) for x in (ref/'per_frame_metrics.jsonl').read_text().splitlines()]
  assert [x['view_id'] for x in rows]==[x['view_id'] for x in rr]
  assert json.loads((root/'sources.json').read_text())==json.loads((ref/'sources.json').read_text())
  per[arm][sid]={k:float(np.mean([x[k] for x in rows])) for k in keys}
 scores[arm]={k:float(np.mean([per[arm][s][k] for s in p['scenes']])) for k in keys}
paired={};rng=np.random.default_rng(1234)
for a in arms[1:]:
 paired[a]={}
 for k in keys[:-1]:
  d=np.array([per[a][s][k]-per['control'][s][k] for s in p['scenes']]);boot=d[rng.integers(0,len(d),(10000,len(d)))].mean(1)
  paired[a][k]=dict(mean_delta=float(d.mean()),scene_wins=int((d>0).sum() if k in ['tgt_rgb_psnr','tgt_depth_delta1'] else (d<0).sum()),bootstrap95=np.percentile(boot,[2.5,97.5]).tolist())
report=dict(protocol=p,scene_mean=scores,per_scene=per,paired_vs_control=paired,paired_inputs_verified=True,note='8 held-out scenes; estimated depth reference; bootstrap intervals are exploratory')
(r/'summary.json').write_text(json.dumps(report,indent=2))
html=['<meta charset=utf-8><h1>Control / span：8个未见场景对比</h1><p>同场景配对，6源视角、最多15目标视角、25步采样、seed1234。深度参考为已有估计。均按场景等权平均。</p><table border=1><tr><th>实验</th><th>step</th>'+''.join('<th>'+k+'</th>' for k in keys)+'</tr>']
for a in p['arms']:html.append('<tr><td>'+a['arm']+'</td><td>'+str(a['step'])+'</td>'+''.join('<td>%.4f</td>'%scores[a['arm']][k] for k in keys)+'</tr>')
html+=['</table><h2>逐场景视频</h2>']
for sid in p['scenes']:
 html.append('<h3>'+sid+'</h3>')
 for a in arms:
  for v in (r/a/sid).glob('*.mp4'):html.append('<p>'+a+'</p><video controls width=640 src="'+str(v.relative_to(r))+'"></video>')
(r/'comparison.html').write_text('\n'.join(html));(r/'report.status.json').write_text(json.dumps(dict(returncode=0,paired_inputs_verified=True)));print(json.dumps(scores,indent=2))
