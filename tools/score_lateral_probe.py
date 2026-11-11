"""Score pure yaw against a source-image homography on the valid overlap only."""
import json,sys,cv2,numpy as np
from pathlib import Path
from PIL import Image,ImageDraw
r=Path(sys.argv[1]);data=json.load(open(r/'poses.json'));K=np.array(data['K']);anchor=np.array(data['anchor']);src=cv2.imread(str(r/'anchor_rgb.png'));h,w=src.shape[:2];rows=[]
canvas=Image.new('RGB',(3*448,5*285),'white');draw=ImageDraw.Draw(canvas)
for i,(spec,p) in enumerate(zip(data['specs'],data['poses'])):
 pred=cv2.imread(str(r/f'frames/bev{i:02d}_rgb.jpg'));assert pred is not None
 row=dict(index=i,**spec)
 if spec['kind']=='yaw':
  pose=np.array(p);assert np.allclose(pose[:3,3],anchor[:3,3]);rot=pose[:3,:3].T@anchor[:3,:3];H=K@rot@np.linalg.inv(K)
  ref=cv2.warpPerspective(src,H,(w,h));mask=cv2.warpPerspective(np.ones((h,w),np.uint8),H,(w,h),flags=cv2.INTER_NEAREST);mask=cv2.erode(mask,np.ones((5,5),np.uint8)).astype(bool)
  mse=np.mean(((pred.astype(float)-ref.astype(float))/255.)[mask]**2);row.update(overlap_fraction=float(mask.mean()),overlap_psnr=float(-10*np.log10(max(mse,1e-12))))
  cv2.imwrite(str(r/f'frames/bev{i:02d}_yaw_reference.jpg'),ref)
 im=Image.fromarray(cv2.cvtColor(pred,cv2.COLOR_BGR2RGB)).resize((448,252));x=i%3*448;y=i//3*285;canvas.paste(im,(x,y+30));label=f"yaw {spec['degrees']:+d} deg" if spec['kind']=='yaw' else f"shift {spec['depth_fraction']:+.2f} depth";draw.text((x+8,y+8),label,fill='black');rows.append(row)
canvas.save(r/'overview.jpg');(r/'yaw_metrics.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
