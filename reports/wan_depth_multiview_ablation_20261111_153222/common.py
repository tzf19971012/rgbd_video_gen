import os,sys,json,math,random,time
from pathlib import Path
ROOT=Path('/mnt/sfs_turbo/t00845730/rgbd_video_gen_gae')
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from torch import nn
from torch.nn import functional as F
H,W=384,672
BASE=ROOT/'reports/wan_pretrained_depth_decoder_20261111_104120'
CACHE=ROOT/'reports/wan_latent_depth_probe_20261110_221707'
HERE=Path(__file__).parent

def init(seed=42):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(2)
 torch.npu.set_compile_mode(jit_compile=False)
 torch.npu.config.allow_internal_format=False
 torch.npu.set_per_process_memory_fraction(.32)

def atomic(p,data):
 p=Path(p);q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(data,indent=2));q.replace(p)

def groups():return json.loads((HERE/'groups.json').read_text())

def load_group(g):
 zs=[];ds=[];rgbs=[]
 for fp in g['files']:
  with np.load(fp) as f:
   zs.append(f['z'].astype('float32'));ds.append(f['depth'].copy());rgbs.append(f['rgb'].copy())
 return torch.from_numpy(np.stack(zs)),torch.from_numpy(np.stack(ds)),np.stack(rgbs)

# Exact pretrained decoder and scale inversion from the running single-view baseline.
src=(HERE/'baseline_snapshot.py').read_text()
part=src[src.index('class Decoder(nn.Module):'):src.index('model=Decoder().to(device)')]
namespace=dict(nn=nn,torch=torch,H=H,W=W)
from diffsynth.pipelines.wan_video_new_determine import ModelConfig,WanVideoPipeline
namespace.update(ModelConfig=ModelConfig,WanVideoPipeline=WanVideoPipeline)
exec(part,namespace)
SingleDecoder=namespace['Decoder']

class AttentionBlock(nn.Module):
 def __init__(self,c=96):
  super().__init__();self.norm=nn.LayerNorm(c);self.qkv=nn.Linear(c,3*c);self.out=nn.Linear(c,c);self.norm2=nn.LayerNorm(c);self.ff=nn.Sequential(nn.Linear(c,c*3),nn.GELU(),nn.Linear(c*3,c))
 def forward(self,x):
  b,n,c=x.shape;q,k,v=self.qkv(self.norm(x)).reshape(b,n,3,4,c//4).permute(2,0,3,1,4).unbind(0)
  att=(q.float()@k.float().transpose(-1,-2))*(c//4)**-.5
  y=(att.softmax(-1).to(v.dtype)@v).transpose(1,2).reshape(b,n,c)
  x=x+self.out(y);return x+self.ff(self.norm2(x))

class MultiDecoder(nn.Module):
 def __init__(self,dit_features=False):
  super().__init__();self.decoder=SingleDecoder();self.start=nn.Conv2d(16,96,1)
  self.feature=nn.Sequential(nn.GroupNorm(3,4608),nn.Conv2d(4608,96,1)) if dit_features else None
  self.pose=nn.Sequential(nn.Linear(16,96),nn.SiLU(),nn.Linear(96,96))
  self.xy=nn.Linear(2,96);self.blocks=nn.ModuleList([AttentionBlock(),AttentionBlock()]);self.to_latent=nn.Conv2d(96,16,1)
  nn.init.zeros_(self.to_latent.weight);nn.init.zeros_(self.to_latent.bias)
 def forward(self,z,camera,target,feat=None):
  # z [V,16,48,84], only RGB/cameras enter. No depth input.
  x=F.adaptive_avg_pool2d(self.start(z),(12,21))
  if self.feature is not None:
   assert feat is not None
   x=x+self.feature(feat)
  v,c,h,w=x.shape
  yy,xx=torch.meshgrid(torch.linspace(-1,1,h,device=x.device),torch.linspace(-1,1,w,device=x.device),indexing='ij')
  xy=self.xy(torch.stack([xx,yy],-1).to(x.dtype)).reshape(1,h*w,c)
  x=x.flatten(2).transpose(1,2)+self.pose(camera.to(x.dtype))[:,None,:]+xy
  x=x.reshape(1,v*h*w,c)
  for block in self.blocks:x=block(x)
  selected=x.reshape(v,h,w,c)[target].permute(2,0,1)[None]
  correction=F.interpolate(self.to_latent(selected),size=z.shape[-2:],mode='bilinear',align_corners=False)
  return self.decoder(z[target:target+1]+correction)

def loss_fn(logp,depth):
 valid=torch.isfinite(depth)&(depth>1e-4);logg=depth.clamp_min(1e-4).log();e=torch.where(valid,logp-logg,0);count=valid.sum((1,2)).clamp_min(1);bias=e.sum((1,2))/count
 return (torch.where(valid,(e-bias[:,None,None]).square(),0).sum((1,2))/count+.1*bias.square()).mean()

def metrics(pred,gt):
 m=torch.isfinite(gt)&(gt>1e-4);g=gt[m];p=pred[m]
 assert len(g) and torch.isfinite(p).all()
 pa=p*(g.median()/p.median().clamp_min(1e-6))
 return dict(absrel=float(((p-g).abs()/g).mean()),aligned_absrel=float(((pa-g).abs()/g).mean()),delta1=float((torch.maximum(pa/g,g/pa)<1.25).float().mean()),log_rmse=float((pa.log()-g.log()).square().mean().sqrt()))
