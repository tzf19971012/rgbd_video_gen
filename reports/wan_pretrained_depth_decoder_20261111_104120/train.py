import os,sys,json,time,random,math
from pathlib import Path
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
(OUT/'protocol.json').write_text(json.dumps(dict(resolution=[H,W],split_scenes={k:sorted(s) for k,s in sets.items()},views_per_scene={'train':8,'val':4,'test':4},encoder='frozen Wan2.1 VAE; independent single frame; deterministic encode',state='16-channel native normalized Wan latent',target='existing estimated depth, not sensor ground truth',decoder='pretrained Wan VAE Decoder3d + conv2; RGB output head averaged to single log-depth channel; encoder frozen; native latent scale inversion; single-view T=1',steps=10000,batch_size=1,gradient_accumulation_steps=8,learning_rate=1e-5,seed=42,selection='validation aligned AbsRel only; test evaluated once after selection'),indent=2))
index=json.loads(Path('/mnt/sfs_turbo/t00845730/rgbd_video_gen_gae/reports/wan_latent_depth_probe_20261110_221707/index.json').read_text())
class Cached(torch.utils.data.Dataset):
 def __init__(self,files):self.files=files
 def __len__(self):return len(self.files)
 def __getitem__(self,i):
  with np.load(self.files[i]) as f:return torch.from_numpy(f['z'].astype('float32')),torch.from_numpy(f['depth'].copy())
class Decoder(nn.Module):
 def __init__(self):
  super().__init__()
  from diffsynth.models.wan_video_vae import CausalConv3d
  pipe=WanVideoPipeline.from_pretrained(torch_dtype=torch.float32,device='cpu',model_configs=[ModelConfig(model_id='Wan-AI/Wan2.1-T2V-1.3B',origin_file_pattern='Wan2.1_VAE.pth')],local_model_path='/mnt/sfs_turbo/t00845730/wan21',skip_download=True)
  self.core=pipe.vae.model;self.core.requires_grad_(False)
  layer=self.core.decoder.head[-1]
  assert layer.out_channels==3
  head=CausalConv3d(layer.in_channels,1,layer.kernel_size,padding=(layer._padding[4]//2,layer._padding[2],layer._padding[0]),stride=layer.stride,dilation=layer.dilation)
  with torch.no_grad():
   head.weight.copy_(layer.weight.mean(0,keepdim=True));head.bias.copy_(layer.bias.mean().reshape(1))
  self.core.decoder.head[-1]=head
  self.core.decoder.requires_grad_(True);self.core.conv2.requires_grad_(True)
  self.register_buffer('wan_mean',pipe.vae.scale[0].float().clone());self.register_buffer('wan_invstd',pipe.vae.scale[1].float().clone())
  self.core.encoder.eval()
  print('Pretrained Wan decoder + conv2 trainable params',sum(p.numel() for p in self.parameters() if p.requires_grad),flush=True)
 def train(self,mode=True):
  super().train(mode);self.core.encoder.eval();return self
 def forward(self,z):
  # Native un-clamped model.decode: public wrapper clamps RGB and uses inference helpers.
  # T=1 independent frame. Native scale inversion + causal cache behavior preserved.
  out=self.core.decode(z[:, :, None],[self.wan_mean,self.wan_invstd])
  assert out.shape[1:]==(1,1,H,W),out.shape
  return out[:,0,0].float().clamp(-12,12)
model=Decoder().to(device);opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5,weight_decay=.01)
loaders={s:torch.utils.data.DataLoader(Cached(fs),batch_size=1,shuffle=s=='train',num_workers=2,drop_last=s=='train') for s,fs in index.items()}
loaders['train_probe']=torch.utils.data.DataLoader(Cached(uniform(index['train'],64)),batch_size=1,shuffle=False,num_workers=2)
index['train_probe']=uniform(index['train'],64)
@torch.no_grad()
def evaluate(split,step):
 model.eval();rows=[]
 for batch,(z,d) in enumerate(loaders[split]):
  
  with torch.autocast(device_type='npu',dtype=torch.bfloat16):pred=model(z.to(device)).float().cpu().exp()
  for i in range(len(d)):
   mask=torch.isfinite(d[i])&(d[i]>1e-4);p=pred[i][mask];g=d[i][mask]
   if not len(g):continue
   scale=g.median()/p.median().clamp_min(1e-6);pa=p*scale
   rows.append(dict(constant_median_absrel=float(((g.median()-g).abs()/g).mean()),absrel=float(((p-g).abs()/g).mean()),aligned_absrel=float(((pa-g).abs()/g).mean()),delta1=float((torch.maximum(pa/g,g/pa)<1.25).float().mean()),log_rmse=float(((pa.log()-g.log()).square().mean()).sqrt())))
   if batch<4:
    fp=index[split][batch+i]
    with np.load(fp) as f:rgb=f['rgb']
    gt=d[i].numpy();pr=pred[i].numpy()*float(scale);valid=gt>1e-4;lo,hi=np.percentile(gt[valid],[2,98]);maps=[]
    for v in [gt,pr]:
     q=np.clip((v-lo)/max(hi-lo,1e-6),0,1);maps.append(np.repeat((q[:,:,None]*255).astype('uint8'),3,axis=2))
    folder=OUT/'visuals'/split;folder.mkdir(parents=True,exist_ok=True);Image.fromarray(np.concatenate([rgb,*maps],axis=1)).save(folder/('sample_%03d_step%d.jpg'%(batch+i,step)))
 metrics={k:float(np.mean([r[k] for r in rows])) for k in rows[0]};metrics.update(split=split,step=step,samples=len(rows));print('EVAL',json.dumps(metrics),flush=True);(OUT/(split+'_step%d.json'%step)).write_text(json.dumps(metrics,indent=2));model.train();return metrics
sample_z,sample_d=next(iter(loaders['train']))
model.train()
with torch.autocast(device_type='npu',dtype=torch.bfloat16):sample_out=model(sample_z.to(device))
assert sample_out.shape==(1,H,W)
mask=sample_d.to(device)>1e-4
smoke_loss=(sample_out[mask]-sample_d.to(device)[mask].log()).square().mean()
smoke_loss.backward()
assert model.core.decoder.head[-1].weight.grad is not None
assert model.core.conv2.weight.grad is not None
assert all(p.grad is None for p in model.core.encoder.parameters())
assert torch.isfinite(model.core.decoder.head[-1].weight.grad).all()
print('SMOKE PASS: output shape, decoder gradients, frozen encoder, finite gradients',flush=True)
opt.zero_grad();del sample_out,smoke_loss
best=float('inf');step=0;micro=0;started=time.time();opt.zero_grad();evaluate('train_probe',0);evaluate('val',0)
while step<10000:
 for z,d in loaders['train']:
  z=z.to(device);d=d.to(device);valid=torch.isfinite(d)&(d>1e-4)
  
  with torch.autocast(device_type='npu',dtype=torch.bfloat16):logp=model(z)
  logg=d.clamp_min(1e-4).log();e=torch.where(valid,logp-logg,0);count=valid.sum((1,2)).clamp_min(1);bias=e.sum((1,2))/count
  centered=e-bias[:,None,None];loss=(torch.where(valid,centered.square(),0).sum((1,2))/count+.1*bias.square()).mean()
  (loss/8).backward();micro+=1
  if micro%8:continue
  torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1);opt.step();opt.zero_grad();step+=1
  if step%20==0:
   entry=dict(step=step,loss=float(loss.detach().cpu()),seconds=time.time()-started);print(json.dumps(entry),flush=True)
   with (OUT/'train.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
  if step%500==0:
   evaluate('train_probe',step);m=evaluate('val',step);torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step),OUT/'last.pt')
   if m['aligned_absrel']<best:best=m['aligned_absrel'];shutil_unused=None;torch.save(dict(model=model.state_dict(),step=step),OUT/'best.pt')
  if step>=10000:break
state=torch.load(OUT/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(state['model']);result=evaluate('test',state['step']);(OUT/'status.json').write_text(json.dumps(dict(complete=True,best_validation_aligned_absrel=best,test=result),indent=2))
