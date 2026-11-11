from common import *
import traceback
from diffsynth.pipelines.wan_video_new_determine import model_fn_wan_video
init();out=HERE/'features';out.mkdir(exist_ok=True)
try:
 atomic(out/'status.json',dict(status='loading',pid=os.getpid()))
 pipe=WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16,device='cpu',model_configs=[ModelConfig(model_id='Wan-AI/Wan2.1-T2V-1.3B',origin_file_pattern='diffusion_pytorch_model.safetensors')],local_model_path='/mnt/sfs_turbo/t00845730/wan21',skip_download=True)
 dit=pipe.dit.eval().requires_grad_(False).to('npu:0',torch.bfloat16)
 assert len(dit.blocks)==30
 ctx=torch.load(HERE/'empty_text_embed_wan.pt',map_location='cpu',weights_only=True)
 if isinstance(ctx,dict):ctx=ctx.get('context',ctx.get('prompt_embeds',next(iter(ctx.values()))))
 if ctx.ndim==2:ctx=ctx[None]
 ctx=ctx.to('npu:0',torch.bfloat16)
 captured={};handles=[]
 def hook(i):
  def save(module,inputs,output):
   assert output.shape==(1,4*24*42,1536),output.shape
   f=output.detach().reshape(4,24,42,1536).permute(0,3,1,2).contiguous()
   captured[i]=F.avg_pool2d(f.float(),2).to('cpu',torch.float16)
  return save
 for i in [9,19,29]:handles.append(dit.blocks[i].register_forward_hook(hook(i)))
 jobs=[(s,g) for s,ls in groups().items() for g in ls];started=time.time()
 for n,(split,g) in enumerate(jobs):
  dest=out/(g['id']+'.pt')
  if dest.exists():continue
  z,_,_=load_group(g);z=z.permute(1,0,2,3)[None].to('npu:0',torch.bfloat16)
  captured.clear()
  with torch.inference_mode():
   model_fn_wan_video(dit=dit,latents=z,timestep=torch.zeros(1,device='npu:0',dtype=torch.bfloat16),context=ctx,use_gradient_checkpointing=False)
  assert set(captured)=={9,19,29}
  feat=torch.cat([captured[i] for i in [9,19,29]],1)
  assert feat.shape==(4,4608,12,21) and torch.isfinite(feat).all()
  payload=dict(features=feat,group_id=g['id'],view_ids=g['view_ids'],layers=[9,19,29],timestep=0,source='frozen original Wan2.1 T2V1.3B; clean independent-frame RGB latents; one joint forward; not a denoising rollout')
  torch.save(payload,dest.with_suffix('.tmp'));dest.with_suffix('.tmp').replace(dest)
  row=dict(status='extracting',completed=n+1,total=len(jobs),last_group=g['id'],seconds=time.time()-started,peak_allocated_gib=torch.npu.max_memory_allocated()/2**30)
  atomic(out/'status.json',row)
  if n%8==0:print(json.dumps(row),flush=True)
 for h in handles:h.remove()
 atomic(out/'COMPLETE.json',dict(complete=True,groups=len(jobs),layers=[9,19,29],timestep=0,channels=4608,grid=[12,21],seconds=time.time()-started))
 print('EXTRACTION COMPLETE',flush=True)
except Exception:
 atomic(out/'FAILED.json',dict(error=traceback.format_exc()));raise
