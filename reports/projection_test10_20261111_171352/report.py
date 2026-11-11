from pathlib import Path
import json,csv,html,statistics,time
import numpy as np
from PIL import Image,ImageDraw
P=Path(__file__).parent;C=json.loads((P/'plan.json').read_text());arms=['control','gae','rgb'];data={}
for sid in C['scenes']:
 data[sid]={}
 for a in arms:
  f=P/'results'/sid/a/'complete.json'
  if f.exists():data[sid][a]=json.loads(f.read_text())
paired=[sid for sid in C['scenes'] if len(data[sid])==3]
keys=[('trajectory','tgt_rgb_psnr','真实轨迹 PSNR ↑'),('trajectory','tgt_rgb_lpips','LPIPS ↓'),('trajectory','tgt_depth_abs_rel_aligned','深度 AbsRel ↓'),('trajectory','tgt_depth_delta1','深度 δ1 ↑'),('yaw','large15_20_psnr','±15°/±20°重叠区域 PSNR ↑'),('yaw','yaw0_psnr','0°重建 PSNR ↑')]
summary={a:{} for a in arms}
if paired:
 for a in arms:
  for section,key,label in keys:
   vals=[data[sid][a][section][key] for sid in paired]
   summary[a][key]=dict(mean=float(np.mean(vals)),std=float(np.std(vals)),median=float(np.median(vals)))
rows=[]
for sid in C['scenes']:
 for a,r in data[sid].items():
  row=dict(scene=sid,arm=a,step=r['step'])
  row.update({k:r[section][k] for section,k,_ in keys});row['nearest_src_psnr']=r['trajectory']['nearest_src_psnr'];rows.append(row)
with (P/'metrics.csv').open('w') as f:
 fields=['scene','arm','step']+[k for _,k,_ in keys]+['nearest_src_psnr'];w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
result=dict(complete=len(paired)==10,paired_scenes=paired,completed_jobs=len(rows),total_jobs=30,summary=summary,per_scene=data,protocol=C)
q=P/'report.json.tmp';q.write_text(json.dumps(result,indent=2));q.replace(P/'report.json')
esc=html.escape
parts=['<!doctype html><meta charset="utf-8"><title>Projection 10 场景实验报告</title><style>body{font:16px/1.6 system-ui;background:#10151c;color:#e9eef5;max-width:1400px;margin:36px auto;padding:0 24px}table{border-collapse:collapse;width:100%;margin:15px 0}td,th{padding:9px;border:1px solid #354252;text-align:left}a{color:#8dcaff}video{width:100%}.videos{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}img{max-width:100%}.muted{color:#a9b8c9}</style><h1>Projection 与 Control：10 个测试场景对比</h1>']
parts += [f'<p>完成 {len(rows)}/30 组推理，三组齐全的场景 {len(paired)}/10。只有三组齐全的场景参与下方配对统计。</p>','<p><a href="index.html">交互视频对比</a> · <a href="metrics.csv">逐场景指标 CSV</a> · <a href="report.json">完整结果 JSON</a> · <a href="plan.json">实验协议与源/目标帧清单</a></p>','<h2>目的和设置</h2><p>比较现有 checkpoint 的实际 NVS、旋转与横移质量。Control 8200 步，GAE projection 6000 步，RGB projection 6000 步。固定未见测试场景、6 源视角、每批3目标、25步采样、seed1234、共享目标初始噪声。每场景生成15张真实轨迹目标，以及9个旋转位置（−20°至+20°）和6个横移位置。</p>','<p>场景从满足帧数和缓存要求的测试集中按排序等间隔抽取，排除之前反复查看的 fe2fadf89a84e92a，不按生成质量筛选。所有方法使用相同输入视角和目标位姿。</p>','<h2>结果（场景宏平均 ± 场景间标准差）</h2><table><tr><th>指标</th><th>Control8200</th><th>GAE projection6000</th><th>RGB projection6000</th></tr>']
for _,key,label in keys:
 cells=[f'{summary[a][key]["mean"]:.4f} ± {summary[a][key]["std"]:.4f}' if paired else '等待' for a in arms];parts.append('<tr><td>'+label+'</td>'+''.join('<td>'+s+'</td>' for s in cells)+'</tr>')
parts.append('</table>')
if paired:
 parts.append('<h2>配对观察</h2><ul>')
 for a in ['gae','rgb']:
  rgbwins=sum(data[s][a]['trajectory']['tgt_rgb_psnr']>data[s]['control']['trajectory']['tgt_rgb_psnr'] for s in paired)
  yawwins=sum(data[s][a]['yaw']['large15_20_psnr']>data[s]['control']['yaw']['large15_20_psnr'] for s in paired)
  depthwins=sum(data[s][a]['trajectory']['tgt_depth_abs_rel_aligned']<data[s]['control']['trajectory']['tgt_depth_abs_rel_aligned'] for s in paired)
  parts.append(f'<li>{a.upper()} projection：真实轨迹 PSNR 在 {rgbwins}/{len(paired)} 个场景高于 control；大角度旋转在 {yawwins}/{len(paired)} 个场景更高；对齐深度 AbsRel 在 {depthwins}/{len(paired)} 个场景更低。</li>')
 parts.append('</ul>')
qual=P/'qualitative_analysis.json'
if qual.exists():
 q=json.loads(qual.read_text());parts.append('<h2>画面分析与结论</h2>')
 for item in q['sections']:
  parts.append('<h3>'+esc(item['title'])+'</h3>')
  for paragraph in item['paragraphs']:parts.append('<p>'+esc(paragraph)+'</p>')
parts.append('<h2>评估边界</h2><ul><li>深度参考是已有 DA3 估计值，并非传感器真值；深度指标逐图做中值尺度对齐，不能证明绝对尺度准确或超过 DA3。</li><li>旋转 PSNR 只计算纯旋转单应变换可验证的重叠区域；横移探针没有真实目标图，只做视觉诊断。</li><li>训练步数、历史、有效 batch 和相机跨度采样配置不同。结果比较当前模型表现，不能把差异全部归因于投影模块。</li><li>探针深度画面逐帧独立归一化，颜色不能用于比较跨帧绝对深度或尺度。</li><li>10个场景是小样本验证，不代表整个测试集；报告保留全部场景及失败情况。</li></ul>')
parts.append('<h2>逐场景结果与视频</h2>')
for sid in C['scenes']:
 parts.append(f'<h3>{sid}</h3><table><tr><th>方法</th>'+''.join('<th>'+label+'</th>' for _,_,label in keys)+'</tr>')
 for a in arms:
  r=data[sid].get(a);vals=[f'{r[sec][k]:.4f}' if r else '等待' for sec,k,_ in keys];parts.append('<tr><td>'+a+'</td>'+''.join('<td>'+v+'</td>' for v in vals)+'</tr>')
 parts.append('</table><div class="videos">')
 for a in arms:
  r=data[sid].get(a);parts.append('<div><b>'+a+'</b>')
  if r:
   rel=str(Path(r['videos']['lateral']).relative_to(P));parts.append('<video controls preload="none" src="'+esc(rel)+'"></video>')
   real=str(Path(r['videos']['trajectory']).relative_to(P));parts.append('<p><a href="'+esc(real)+'">真实轨迹：GT/预测 RGB 与深度</a></p>')
  else:parts.append('<p>等待推理</p>')
  parts.append('</div>')
 parts.append('</div>')
 if len(data[sid])==3:
  dest=P/'contacts'/f'{sid}.jpg';dest.parent.mkdir(exist_ok=True)
  if not dest.exists():
   ww,hh=336,189;ids=[0,4,8,9,14];img=Image.new('RGB',(ww*3,(hh*2+28)*len(ids)),'#171d26');draw=ImageDraw.Draw(img)
   for row,i in enumerate(ids):
    for col,a in enumerate(arms):
     base=Path(data[sid][a]['outputs']['lateral']);spec=json.loads((base/'poses.json').read_text())['specs'][i]
     label=(str(spec['degrees'])+' deg') if spec['kind']=='yaw' else ('shift '+str(spec['depth_fraction'])+' depth')
     draw.text((col*ww+4,row*(hh*2+28)+4),a+' '+label,fill='white')
     for j,suf in enumerate(['rgb','disp']):img.paste(Image.open(base/'frames'/f'bev{i:02d}_{suf}.jpg').resize((ww,hh)),(col*ww,row*(hh*2+28)+28+j*hh))
   img.save(dest,quality=90)
  parts.append('<p><a href="contacts/'+sid+'.jpg">打开旋转、横移 RGB/深度对比图</a></p>')
parts.append('<p class="muted">生成时间：'+time.strftime('%Y-%m-%d %H:%M:%S %z')+'</p>')
q=P/'report.html.tmp';q.write_text(''.join(parts));q.replace(P/'report.html')
