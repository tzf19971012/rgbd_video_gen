from pathlib import Path
import json, sys, numpy as np
from PIL import Image, ImageDraw

r=Path(sys.argv[1]); protocol=json.loads((r/'protocol.json').read_text())
arms=protocol['commands']; data={}; dirs={}
for a in arms:
    dirs[a['arm']]=Path(a['output'])
    f=dirs[a['arm']]/'per_frame_metrics.jsonl'
    data[a['arm']]=[json.loads(s) for s in f.read_text().splitlines() if s.strip()]
ids=[x['view_id'] for x in data[arms[0]['arm']]]
assert all([x['view_id'] for x in rows]==ids for rows in data.values())
metrics=['tgt_rgb_psnr','tgt_rgb_lpips','tgt_depth_delta1','tgt_depth_abs_rel_aligned']
groups={'all':lambda v:True,'source_range_le95':lambda v:int(v)<=95,'extrap_gt95':lambda v:int(v)>95,'tail_ge120':lambda v:int(v)>=120}
summary={}
for name,rows in data.items():
    summary[name]={}
    for group,select in groups.items():
        chosen=[x for x in rows if select(x['view_id'])]
        summary[name][group]={'n':len(chosen),**{k:float(np.mean([x[k] for x in chosen])) for k in metrics}}
(r/'summary.json').write_text(json.dumps(summary,indent=2))
images=[]
names=[a['arm'] for a in arms]
for vid in ids:
    tiles=[]
    for name in names:
        im=Image.open(dirs[name]/'frames'/(vid+'_side.jpg')).convert('RGB'); w=im.width//4
        tiles.append([im.crop((j*w,0,(j+1)*w,im.height)).resize((336,189)) for j in range(4)])
    canvas=Image.new('RGB',(1680,436),'white');draw=ImageDraw.Draw(canvas)
    draw.text((8,4),'STEP 9400 | view '+vid+' | columns: GT, control, 5-20, 5-35, 10-35',fill='black')
    for j,name in enumerate(['GT']+names):
        draw.text((j*336+8,23),name,fill='black')
        rgb=tiles[0][0] if j==0 else tiles[j-1][1]
        dep=tiles[0][2] if j==0 else tiles[j-1][3]
        canvas.paste(rgb,(j*336,54));canvas.paste(dep,(j*336,247))
    canvas.save(r/('compare_'+vid+'.jpg'),quality=92);images.append(np.asarray(canvas))
tail=Image.new('RGB',(1680,436*3),'white')
for i,vid in enumerate(['00000120','00000125','00000130']):tail.paste(Image.open(r/('compare_'+vid+'.jpg')),(0,436*i))
tail.save(r/'tail_comparison.jpg',quality=93)
sys.path.insert(0,'/mnt/sfs_turbo/t00845730/rgbd_video_gen_gae')
try:
    from tools.make_traj_videos import encode
    encode(images,3,str(r/'trajectory_comparison_step9400.mp4'))
except Exception as exc:print('video encoding:',repr(exc))
html=['<!doctype html><meta charset="utf-8"><title>Span trajectory step9400</title><style>body{font:16px/1.7 system-ui;max-width:1700px;margin:32px auto;padding:20px;color:#203047}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:10px}img,video{width:100%}h2{margin-top:36px}</style><h1>四组 span：step 9400 轨迹对比</h1><p>场景 00028da87cc5a4c4，训练集场景；固定六张源图 0/20/35/55/75/95；25 步采样，种子1234；目标每三帧独立采样，无自回归反馈。深度指标逐帧中位数尺度对齐，不能代表度量尺度精度。</p>']
for group in groups:
    html+=['<h2>'+group+'</h2><table><tr><th>Arm</th><th>N</th><th>PSNR ↑</th><th>LPIPS ↓</th><th>δ1 ↑</th><th>aligned AbsRel ↓</th></tr>']
    for name in names:
        d=summary[name][group];html+=['<tr><td>'+name+'</td><td>'+str(d['n'])+'</td>'+''.join('<td>'+format(d[k],'.4f')+'</td>' for k in metrics)+'</tr>']
    html+=['</table>']
html+=['<h2>尾帧 120 / 125 / 130</h2><img src="tail_comparison.jpg"><h2>全轨迹</h2><video controls src="trajectory_comparison_step9400.mp4"></video>']
(r/'comparison.html').write_text('\n'.join(html))
print(json.dumps(summary,indent=2))
