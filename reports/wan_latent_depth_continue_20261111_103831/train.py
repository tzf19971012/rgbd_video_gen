import os,sys,json,time,random,math
from pathlib import Path
OLD='/mnt/sfs_turbo/t00845730/rgbd_video_gen_gae/reports/wan_latent_depth_probe_20261110_221707'
ROOT=Path('/mnt/sfs_turbo/t00845730/rgbd_video_gen_gae');sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from examples.dataset import Kubric4DRGBDNVSDataset
from diffsynth.pipelines.wan_video_new_determine import ModelConfig,WanVideoPipeline
OUT=Path(__file__).parent;H,W=384,672
random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(2)
device='npu:0'
def dataset(manifest):
 return Kubric4DRGBDNVSDataset(kubric4d_root='/mnt/sfs_turbo/t00845730/re10k/extracted',manifest_path=manifest,resolution=(H,W),num_source_views=6,num_target_views=3,random_permute_source_views=False,view_order='random',disparity_norm_mode='metric',samples_per_scene=1)
def uniform(items,n):return [items[i] for i in np.linspace(0,len(items)-1,min(n,len(items))).round().astype(int)]
train=dataset('/mnt/sfs_turbo/t00845730/re10k/manifest_train_c2w_v1.jsonl');test=dataset('/mnt/sfs_turbo/t00845730/re10k/manifest_test_c2w_v1.jsonl')
records=sorted(train.records,key=lambda r:r['scene_id']);selected=uniform(records,272);splits={'train':selected[:256],'val':selected[256:],'test':uniform(sorted(test.records,key=lambda r:r['scene_id']),32)}
sets={k:{r['scene_id'] for r in rs} for k,rs in splits.items()}
assert not sets['train']&sets['val'] and not (sets['train']|sets['val'])&sets['test']
(OUT/'protocol.json').write_text(json.dumps(dict(resolution=[H,W],split_scenes={k:sorted(s) for k,s in sets.items()},views_per_scene={'train':8,'val':4,'test':4},encoder='frozen Wan2.1 VAE; independent single frame; deterministic encode',state='16-channel native normalized Wan latent',target='existing estimated depth, not sensor ground truth',decoder='Conv + GroupNorm + bilinear upsampling; single view; no cameras/RGB input',steps=20000,batch_size=8,seed=42,selection='validation aligned AbsRel only; test evaluated once after selection'),indent=2))
index=json.loads(Path('/mnt/sfs_turbo/t00845730/rgbd_video_gen_gae/reports/wan_latent_depth_probe_20261110_221707/index.json').read_text())
class Cached(torch.utils.data.Dataset):
 def __init__(self,files):self.files=files
 def __len__(self):return len(self.files)
 def __getitem__(self,i):
  with np.load(self.files[i]) as f:return torch.from_numpy(f['z'].astype('float32')),torch.from_numpy(f['depth'].copy())
class Decoder(nn.Module):
 def __init__(self):
  super().__init__();self.start=nn.Conv2d(16,128,3,padding=1);self.blocks=nn.ModuleList([nn.Sequential(nn.GroupNorm(8,a),nn.SiLU(),nn.Conv2d(a,b,3,padding=1),nn.GroupNorm(8,b),nn.SiLU(),nn.Conv2d(b,b,3,padding=1)) for a,b in [(128,96),(96,64),(64,32)]]);self.head=nn.Conv2d(32,1,3,padding=1)
 def forward(self,z):
  x=self.start(z)
  for block in self.blocks:x=block(F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False))
  return self.head(x)[:,0].clamp(-12,12)
model=Decoder().to(device);opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.01)
loaders={s:torch.utils.data.DataLoader(Cached(fs),batch_size=8 if s=='train' else 2,shuffle=s=='train',num_workers=2,drop_last=s=='train') for s,fs in index.items()}
state=torch.load(Path(OLD)/'last.pt',map_location='cpu',weights_only=False)
model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
mean=state['mean'].to(device);std=state['std'].to(device)
resume_step=int(state['step'])
for values in opt.state.values():
 for k,v in values.items():
  if torch.is_tensor(v) and k!='step':values[k]=v.to(device)
loaders['train_probe']=torch.utils.data.DataLoader(Cached(uniform(index['train'],128)),batch_size=2,shuffle=False,num_workers=2)
index['train_probe']=uniform(index['train'],128)
print('RESUMED model + optimizer at step',resume_step,flush=True)
@torch.no_grad()
def evaluate(split,step):
 model.eval();rows=[]
 for batch,(z,d) in enumerate(loaders[split]):
  pred=model((z.to(device)-mean)/std).float().cpu().exp()
  for i in range(len(d)):
   mask=torch.isfinite(d[i])&(d[i]>1e-4);p=pred[i][mask];g=d[i][mask]
   if not len(g):continue
   scale=g.median()/p.median().clamp_min(1e-6);pa=p*scale
   rows.append(dict(constant_median_absrel=float(((g.median()-g).abs()/g).mean()),absrel=float(((p-g).abs()/g).mean()),aligned_absrel=float(((pa-g).abs()/g).mean()),delta1=float((torch.maximum(pa/g,g/pa)<1.25).float().mean()),log_rmse=float(((pa.log()-g.log()).square().mean()).sqrt())))
   if batch<4:
    fp=index[split][batch*2+i]
    with np.load(fp) as f:rgb=f['rgb']
    gt=d[i].numpy();pr=pred[i].numpy()*float(scale);valid=gt>1e-4;lo,hi=np.percentile(gt[valid],[2,98]);maps=[]
    for v in [gt,pr]:
     q=np.clip((v-lo)/max(hi-lo,1e-6),0,1);maps.append(np.repeat((q[:,:,None]*255).astype('uint8'),3,axis=2))
    folder=OUT/'visuals'/split;folder.mkdir(parents=True,exist_ok=True);Image.fromarray(np.concatenate([rgb,*maps],axis=1)).save(folder/('sample_%03d_step%d.jpg'%(batch*2+i,step)))
 metrics={k:float(np.mean([r[k] for r in rows])) for k in rows[0]};metrics.update(split=split,step=step,samples=len(rows));print('EVAL',json.dumps(metrics),flush=True);(OUT/(split+'_step%d.json'%step)).write_text(json.dumps(metrics,indent=2));model.train();return metrics
best=0.3883505701087415;step=resume_step;started=time.time();evaluate('train_probe',step);evaluate('val',step)
while step<20000:
 for z,d in loaders['train']:
  z=z.to(device);d=d.to(device);valid=torch.isfinite(d)&(d>1e-4)
  logp=model((z-mean)/std);logg=d.clamp_min(1e-4).log();e=torch.where(valid,logp-logg,0);count=valid.sum((1,2)).clamp_min(1);bias=e.sum((1,2))/count
  centered=e-bias[:,None,None];loss=(torch.where(valid,centered.square(),0).sum((1,2))/count+.1*bias.square()).mean()
  opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();step+=1
  if step%20==0:
   entry=dict(step=step,loss=float(loss.detach().cpu()),seconds=time.time()-started);print(json.dumps(entry),flush=True)
   with (OUT/'train.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
  if step%1000==0:
   evaluate('train_probe',step);m=evaluate('val',step);torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step,mean=mean.cpu(),std=std.cpu()),OUT/'last.pt')
   if m['aligned_absrel']<best:best=m['aligned_absrel'];shutil_unused=None;torch.save(dict(model=model.state_dict(),step=step,mean=mean.cpu(),std=std.cpu()),OUT/'best.pt')
  if step>=20000:break
state=torch.load(OUT/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(state['model']);result=evaluate('test',state['step']);(OUT/'status.json').write_text(json.dumps(dict(complete=True,best_validation_aligned_absrel=best,test=result),indent=2))
