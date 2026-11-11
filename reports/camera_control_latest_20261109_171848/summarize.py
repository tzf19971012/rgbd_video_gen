import json,os,sys,math
from pathlib import Path
import numpy as np,cv2
from PIL import Image,ImageDraw
from tools.make_traj_videos import encode
r=Path(__file__).parent;p=json.loads((r/'protocol.json').read_text());items=p['commands'];arms=[a['arm'] for a in items];summary={};rgbvideos=[];yawvideos=[]
for a in items:
 status=json.loads((r/(a['arm']+'.correctstats.status.json')).read_text());assert status['returncode']==0,(a['arm'],status)
 root=Path(a['output']);rows=[json.loads(x) for x in (root/'trajectory/per_frame_metrics.jsonl').read_text().splitlines()];assert len(rows)==15
 real={k:float(np.mean([x[k] for x in rows])) for k in ['tgt_rgb_psnr','tgt_rgb_lpips','tgt_depth_abs_rel_aligned','tgt_depth_delta1','nearest_src_psnr']}
 beyond=[x for x in rows if int(x['view_id'])>75];real['after_last_source_psnr']=float(np.mean([x['tgt_rgb_psnr'] for x in beyond]));real['after_last_source_lpips']=float(np.mean([x['tgt_rgb_lpips'] for x in beyond]))
 poses=json.loads((root/'lateral/poses.json').read_text());anchor=np.array(Image.open(root/'lateral/anchor_rgb.png').convert('RGB'));h,w=anchor.shape[:2];K=np.array(poses['K']);scores=[];reverse=[]
 for i,spec in enumerate(poses['specs'][:9]):
  pred=np.array(Image.open(root/'lateral/frames'/f'bev{i:02d}_rgb.jpg').convert('RGB'))
  for sign,arr in [(1,scores),(-1,reverse)]:
   rad=np.deg2rad(spec['degrees']*sign);R=np.array([[np.cos(rad),0,np.sin(rad)],[0,1,0],[-np.sin(rad),0,np.cos(rad)]]);H=K@R.T@np.linalg.inv(K);gt=cv2.warpPerspective(anchor,H,(w,h));mask=cv2.warpPerspective(np.ones((h,w),np.uint8),H,(w,h));mask=cv2.erode(mask,np.ones((5,5),np.uint8)).astype(bool);mse=np.mean((pred[mask].astype(float)-gt[mask])**2);arr.append(float(10*np.log10(255**2/max(mse,1e-10))))
 nonzero=[0,1,2,3,5,6,7,8];yaw={'yaw0_psnr':scores[4],'nonzero_psnr':float(np.mean([scores[i] for i in nonzero])),'large15_20_psnr':float(np.mean([scores[i] for i in [0,1,7,8]])),'reverse_diagnostic_psnr':float(np.mean([reverse[i] for i in nonzero])),'per_yaw':scores}
 summary[a['arm']]={'step':a['step'],'trajectory':real,'yaw':yaw,'per_frame':rows}
# Check exact same anchors, poses, intrinsics, sources and targets across models.
base=Path(items[0]['output']);ref=json.loads((base/'lateral/poses.json').read_text());refsrc=json.loads((base/'trajectory/sources.json').read_text());refids=[x['view_id'] for x in summary[arms[0]]['per_frame']]
for a in items[1:]:
 root=Path(a['output']);test=json.loads((root/'lateral/poses.json').read_text());assert np.allclose(test['poses'],ref['poses']) and np.allclose(test['K'],ref['K']);assert np.array_equal(np.array(Image.open(root/'lateral/anchor_rgb.png')),np.array(Image.open(base/'lateral/anchor_rgb.png')));assert json.loads((root/'trajectory/sources.json').read_text())==refsrc;assert [x['view_id'] for x in summary[a['arm']]['per_frame']]==refids
W,H=336,189;labels=['GT/reference']+[a['arm']+' @'+str(a['step']) for a in items]
def sheet(rgb,depth,caption):
 img=Image.new('RGB',(W*6,30+H*2),(24,24,24));draw=ImageDraw.Draw(img)
 for c,(lab,rgbim,depim) in enumerate(zip(labels,rgb,depth)):
  draw.text((c*W+4,4),lab,fill='white');img.paste(rgbim.resize((W,H)),(c*W,30));img.paste(depim.resize((W,H)),(c*W,30+H))
 draw.text((6,30+H-16),caption,fill='yellow');return img
for i,vid in enumerate(refids):
 rgb=[];depth=[]
 for idx,a in enumerate(items):
  side=Image.open(Path(a['output'])/'trajectory/frames'/(vid+'_side.jpg'));w=side.width//4;h=side.height
  if idx==0:rgb.append(side.crop((0,0,w,h)));depth.append(side.crop((2*w,0,3*w,h)))
  rgb.append(side.crop((w,0,2*w,h)));depth.append(side.crop((3*w,0,4*w,h)))
 im=sheet(rgb,depth,'Real trajectory target '+vid);im.save(r/f'trajectory_{i:02d}.jpg',quality=90);rgbvideos.append(np.asarray(im))
encode(rgbvideos,2,str(r/'trajectory_comparison.mp4'))
for i,spec in enumerate(ref['specs']):
 rgb=[Image.open(base/'lateral/anchor_rgb.png')];depth=[Image.new('RGB',(W,H),(0,0,0))]
 if spec['kind']=='yaw':
  angle=np.deg2rad(spec['degrees']);rot=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]]);kk=np.array(ref['K']);source=np.array(rgb[0]);hh,ww=source.shape[:2]
  rgb[0]=Image.fromarray(cv2.warpPerspective(source,kk@rot.T@np.linalg.inv(kk),(ww,hh)))

 for a in items:
  f=Path(a['output'])/'lateral/frames';rgb.append(Image.open(f/f'bev{i:02d}_rgb.jpg'));depth.append(Image.open(f/f'bev{i:02d}_disp.jpg'))
 im=sheet(rgb,depth,str(spec));im.save(r/f'lateral_{i:02d}.jpg',quality=90);yawvideos.append(np.asarray(im))
encode(yawvideos,2,str(r/'lateral_comparison.mp4'))
(r/'summary.json').write_text(json.dumps({'protocol':p,'summary':summary,'paired_inputs_verified':True},indent=2))
html=['<meta charset="utf-8"><title>相机轨迹对比</title><style>body{max-width:1500px;margin:24px auto;font-family:system-ui}table{border-collapse:collapse}th,td{border:1px solid #aaa;padding:8px}video,img{max-width:100%}</style><h1>修正 c2w 后的相机轨迹对比</h1><p>测试场景 fe2fadf89a84e92a。GAE 四组为本次启动时最新 checkpoint、7884训练场景；GeoColor step3400、8训练场景。输入已核对一致。这是单场景诊断，不能据此做严格架构排名。</p><p>六个源帧：0、15、30、45、60、75；25步采样、seed1234、共享目标噪声。真实轨迹15帧；合成探针9个yaw和6个横移。</p><h2>真实轨迹：有真实目标图</h2><table><tr><th>模型</th><th>step</th><th>PSNR↑</th><th>LPIPS↓</th><th>深度对齐AbsRel↓</th><th>深度δ1↑</th><th>源帧范围之后PSNR↑</th></tr>']
for a,s in summary.items():
 d=s['trajectory'];html.append('<tr><td>'+a+'</td><td>'+str(s['step'])+'</td>'+''.join(f'<td>{d[k]:.3f}</td>' for k in ['tgt_rgb_psnr','tgt_rgb_lpips','tgt_depth_abs_rel_aligned','tgt_depth_delta1','after_last_source_psnr'])+'</tr>')
html+=['</table><video controls loop src="trajectory_comparison.mp4"></video><h2>旋转和平移探针</h2><p>纯旋转第一列为源图按旋转关系得到的参考图，黑色区域是未观测区域；横移第一列仅为原始源图。纯旋转分数仅覆盖与源图重叠的有效区域；新显露区域未计分。横移无真实目标图，因此不给准确率。反向yaw仅用于检查方向，不替代正常方向结果。深度伪彩色每帧单独归一化，不能直接比较绝对尺度；深度参考来自DA3预测。</p><table><tr><th>模型</th><th>yaw0 PSNR↑</th><th>非零yaw PSNR↑</th><th>±15/20° PSNR↑</th><th>反向yaw诊断</th></tr>']
for a,s in summary.items():
 d=s['yaw'];html.append('<tr><td>'+a+'</td>'+''.join(f'<td>{d[k]:.3f}</td>' for k in ['yaw0_psnr','nonzero_psnr','large15_20_psnr','reverse_diagnostic_psnr'])+'</tr>')
html+=['</table><video controls loop src="lateral_comparison.mp4"></video><h2>关键帧</h2>']
for i in [0,4,8,9,14]:html.append(f'<p>{ref["specs"][i]}</p><a href="lateral_{i:02d}.jpg"><img src="lateral_{i:02d}.jpg"></a>')
for i in [0,7,14]:html.append(f'<p>真实轨迹 {refids[i]}</p><a href="trajectory_{i:02d}.jpg"><img src="trajectory_{i:02d}.jpg"></a>')
(r/'comparison.html').write_text('\n'.join(html));print(json.dumps({a:{k:v for k,v in s.items() if k!='per_frame'} for a,s in summary.items()},indent=2))
