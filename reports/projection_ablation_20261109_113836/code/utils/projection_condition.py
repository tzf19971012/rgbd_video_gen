"""Source-only patch projection for RGB-vs-GAE conditioning ablation."""
import numpy as np
import torch
import torch.nn.functional as F

def route_numpy(features,depth,K,c2w,source_count,output_hw):
    # features [B,S,C,h,w]; depth [B,S,H,W]. Target RGB/depth/features never read.
    B,S,C,h,w=features.shape;H,W=depth.shape[-2:];V=K.shape[1]
    if (h,w)!=tuple(output_hw):raise ValueError('Feature grid mismatch')
    output=np.zeros((B,C+2,V,h,w),np.float32)
    yy,xx=np.meshgrid(np.arange(h),np.arange(w),indexing='ij')
    u=(xx.ravel()+.5)*W/w-.5;v=(yy.ravel()+.5)*H/h-.5
    pix=np.stack([u,v,np.ones_like(u)])
    for b in range(B):
        world=[];feat=[]
        for s in range(S):
            d=depth[b,s,np.rint(v).astype(int),np.rint(u).astype(int)]
            xyz=(np.linalg.inv(K[b,s].astype(np.float64))@pix)*d
            xyz=c2w[b,s,:3,:3]@xyz+c2w[b,s,:3,3:4]
            valid=np.isfinite(d)&(d>1e-6)&np.isfinite(xyz).all(0)
            world.append(xyz[:,valid]);feat.append(features[b,s].reshape(C,-1)[:,valid])
        world=np.concatenate(world,1);feat=np.concatenate(feat,1)
        for t in range(source_count,V):
            w2c=np.linalg.inv(c2w[b,t].astype(np.float64));cam=w2c[:3,:3]@world+w2c[:3,3:4];z=cam[2];p=K[b,t]@cam
            with np.errstate(divide='ignore',invalid='ignore'):
                x=(p[0]/z+.5)*w/W-.5;y=(p[1]/z+.5)*h/H-.5
            ok=np.isfinite(x)&np.isfinite(y)&np.isfinite(z)&(z>1e-6)
            x=np.rint(np.where(ok,x,-999)).astype(int);y=np.rint(np.where(ok,y,-999)).astype(int)
            ok&=(x>=0)&(x<w)&(y>=0)&(y<h)
            ids=(y*w+x)[ok];zz=z[ok];ff=feat[:,ok]
            near=np.full(h*w,np.inf);np.minimum.at(near,ids,zz)
            keep=zz<=near[ids]+np.maximum(1e-4,.01*near[ids]);ids=ids[keep];ff=ff[:,keep]
            count=np.bincount(ids,minlength=h*w).astype(np.float32)
            fused=np.zeros((h*w,C),np.float32);np.add.at(fused,ids,ff.T)
            fused/=np.maximum(count,1)[:,None]
            output[b,:C,t]=fused.T.reshape(C,h,w)
            output[b,C,t]=(count>0).reshape(h,w)
            output[b,C+1,t]=(np.minimum(count,source_count)/source_count).reshape(h,w)
    return output

def build_condition(data,z,mode,source_count):
    B,_,V,h,w=z.shape
    depth=data['depth_metric'][:,:source_count,0].detach().float().cpu().numpy()
    if mode=='gae':
        features=z[:,:,:source_count].detach().float().cpu().permute(0,2,1,3,4).numpy()
    elif mode=='rgb':
        rgb=data['images'][:,:source_count].detach().float().cpu()
        H,W=rgb.shape[-2:];ph,pw=H//h,W//w
        if H!=h*ph or W!=w*pw:raise ValueError('RGB image must tile latent grid exactly')
        # Pack all RGB patch pixels, no color averaging or pretrained RGB encoder.
        features=(rgb*2-1).reshape(B,source_count,3,h,ph,w,pw).permute(0,1,2,4,6,3,5).reshape(B,source_count,3*ph*pw,h,w).numpy()
    else:raise ValueError(mode)
    return torch.from_numpy(route_numpy(features,depth,data['K'].detach().float().cpu().numpy(),data['c2w'].detach().float().cpu().numpy(),source_count,(h,w))).to(z.device,z.dtype)
