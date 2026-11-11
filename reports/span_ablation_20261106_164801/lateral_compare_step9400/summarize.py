from pathlib import Path
import json,sys,numpy as np,cv2
from PIL import Image,ImageDraw
r=Path(sys.argv[1]);protocol=json.loads((r/'protocol.json').read_text());arms=protocol['commands'];ref=json.loads((r/'reference_poses.json').read_text());names=[a['arm'] for a in arms];stats={};dirs={};frames=[]
for a in arms:
 p=Path(a['output']);dirs[a['arm']]=p;d=json.loads((p/'poses.json').read_text())
 for k in ['anchor','K','poses','median_depth']:assert np.allclose(d[k],ref[k]),(a['arm'],k)
 src=cv2.imread(str(p/'anchor_rgb.png'));h,w=src.shape[:2];K=np.array(d['K']);anchor=np.array(d['anchor']);rows=[]
 for i,(spec,pose) in enumerate(zip(d['specs'],d['poses'])):
  row=dict(index=i,**spec)
  if spec['kind']=='yaw':
   pose=np.array(pose);assert np.allclose(pose[:3,3],anchor[:3,3]);H=K@(pose[:3,:3].T@anchor[:3,:3])@np.linalg.inv(K);target=cv2.warpPerspective(src,H,(w,h));mask=cv2.erode(cv2.warpPerspective(np.ones((h,w),np.uint8),H,(w,h),flags=cv2.INTER_NEAREST),np.ones((5,5),np.uint8)).astype(bool);pred=cv2.imread(str(p/f'frames/bev{i:02d}_rgb.jpg'));mse=np.mean(((pred.astype(float)-target)/255.)[mask]**2);row.update(overlap_fraction=float(mask.mean()),overlap_psnr=float(-10*np.log10(max(mse,1e-12))))
  rows.append(row)
 stats[a['arm']]={'per_pose':rows,'yaw0_psnr':rows[4]['overlap_psnr'],'nonzero_yaw_mean':float(np.mean([x['overlap_psnr'] for x in rows if x['kind']=='yaw' and x['degrees']!=0])),'large_yaw_15_20_mean':float(np.mean([x['overlap_psnr'] for x in rows if x['kind']=='yaw' and abs(x['degrees'])>=15]))}
 (p/'yaw_metrics_step9400.json').write_text(json.dumps(rows,indent=2))
(r/'summary.json').write_text(json.dumps(stats,indent=2))
for i,spec in enumerate(ref['specs']):
 label=('yaw '+str(spec['degrees'])+' deg') if spec['kind']=='yaw' else 'lateral '+str(spec['depth_fraction'])+' x median depth'
 canvas=Image.new('RGB',(1680,438),'white');draw=ImageDraw.Draw(canvas);draw.text((8,5),'step 9400 | '+label+' | top RGB / bottom depth',fill='black')
 for j,name in enumerate(['SOURCE']+names):
  draw.text((336*j+8,28),name,fill='black')
  if j==0:rgb=Image.open(dirs[names[0]]/'anchor_rgb.png');dep=None
  else:
   rgb=Image.open(dirs[name]/f'frames/bev{i:02d}_rgb.jpg');dep=Image.open(dirs[name]/f'frames/bev{i:02d}_disp.jpg')
  canvas.paste(rgb.resize((336,189)),(336*j,55))
  if dep is not None:canvas.paste(dep.resize((336,189)),(336*j,249))
 canvas.save(r/f'pose_{i:02d}.jpg',quality=91);frames.append(np.asarray(canvas))
selected=[0,4,8,9,14];contact=Image.new('RGB',(1680,438*len(selected)),'white')
for j,i in enumerate(selected):contact.paste(Image.fromarray(frames[i]),(0,438*j))
contact.save(r/'selected_comparison.jpg',quality=93)
sys.path.insert(0,'/mnt/sfs_turbo/t00845730/rgbd_video_gen_gae')
try:
 from tools.make_traj_videos import encode
 encode(frames,1,str(r/'lateral_comparison_step9400.mp4'))
except Exception as exc:print('video',repr(exc))
html=['<!doctype html><meta charset="utf-8"><title>Span step9400 lateral probe</title><style>body{font:16px/1.8 system-ui;color:#203047;max-width:1700px;margin:30px auto;padding:20px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:10px}img,video{width:100%}</style><h1>四组 span：step 9400 相机控制探针</h1><p>场景 fe2fadf89a84e92a；精确复用旧 lateral_probe_step8000 的内参、锚点、15个目标位姿与尺度。25步采样，seed1234，shared_target_noise=True。六张源图按相同缓存规则选择。纯旋转用源图单应性变换构造重叠区参考；平移没有目标真值，暂仅作视觉比较。每帧深度可视化单独按百分位归一化，颜色不可用于比较绝对尺度。</p><table><tr><th>组别</th><th>0° PSNR ↑</th><th>非零 yaw 平均 ↑</th><th>|yaw|15/20°平均 ↑</th></tr>']
for n in names:
 d=stats[n];html+=['<tr><td>'+n+'</td>'+''.join('<td>'+format(d[k],'.3f')+'</td>' for k in ['yaw0_psnr','nonzero_yaw_mean','large_yaw_15_20_mean'])+'</tr>']
html+=['</table><h2>逐角度 PSNR</h2><table><tr><th>yaw</th>'+''.join('<th>'+n+'</th>' for n in names)+'</tr>']
for i in range(9):html+=['<tr><td>'+str(ref['specs'][i]['degrees'])+'</td>'+''.join('<td>'+format(stats[n]['per_pose'][i]['overlap_psnr'],'.3f')+'</td>' for n in names)+'</tr>']
html+=['</table><h2>并排视频</h2><video controls src="lateral_comparison_step9400.mp4"></video><h2>±20°、0°、±20%侧移</h2><img src="selected_comparison.jpg"><p>当前仅为单场景、单种子、step9400早期比较；不能据此判断跨场景泛化或所有大视角能力。源图重叠区域指标不评价新显露区域。</p>']
(r/'comparison.html').write_text('\n'.join(html));print(json.dumps(stats,indent=2))
