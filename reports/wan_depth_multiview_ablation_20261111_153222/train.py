from common import *
import argparse
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--arm',choices=['multiview_latent','multiview_dit'],required=True);p.add_argument('--steps',type=int,default=10000);args=p.parse_args()
init();out=HERE/args.arm;out.mkdir(exist_ok=True);is_dit=args.arm=='multiview_dit';device='npu:0'
gs=groups();train=gs['train'];val=gs['val'];test=gs['test']
assert not {g['scene'] for g in train}&{g['scene'] for g in val}
assert not ({g['scene'] for g in train}|{g['scene'] for g in val})&{g['scene'] for g in test}
protocol=json.loads((BASE/'protocol.json').read_text());protocol.update(arm=args.arm,steps=args.steps,context_views=4,batch_size=1,gradient_accumulation_steps=8,learning_rate=1e-5,input='4 known RGB latents + relative cameras'+(' + frozen pretrained Wan DiT blocks 9/19/29 at t=0' if is_dit else ''),dit_frozen=True,decoder='same pretrained Wan depth decoder + two-block multiview latent fusion; target is one observed view per microbatch',comparison_note='Same split, target loss, pretrained decoder init and effective target batch as baseline; pilot adds 4-view context. Not a newly generated-view experiment.',feature_cache='3 layers, spatial average pooling from 24x42 to12x21; full1536 channels/layer; no random channel projection' if is_dit else None)
atomic(out/'protocol.json',protocol)
if is_dit:
 atomic(out/'status.json',dict(status='waiting_for_features',pid=os.getpid()))
 while not (HERE/'features/COMPLETE.json').exists():
  if (HERE/'features/FAILED.json').exists():raise RuntimeError((HERE/'features/FAILED.json').read_text())
  time.sleep(10)
model=MultiDecoder(is_dit).to(device);opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5,weight_decay=.01)
features=HERE/'features'
def inputs(g):
 z,d,rgb=load_group(g);cam=torch.tensor(g['camera'],dtype=torch.float32,device=device)
 feat=None
 if is_dit:
  f=torch.load(features/(g['id']+'.pt'),map_location='cpu',weights_only=True)
  assert f['group_id']==g['id'] and f['view_ids']==g['view_ids']
  feat=f['features'].to(device,torch.float32)
 return z.to(device),d,cam,feat,rgb
@torch.no_grad()
def evaluate(split,step):
 model.eval();rows=[]
 # Baseline uses 64 fixed train probes and all64 validation views; same val set.
 selected=gs[split] if split!='train' else train[::max(1,len(train)//16)][:16]
 for gi,g in enumerate(selected):
  z,d,cam,feat,rgb=inputs(g)
  for target in range(len(g['files'])):
   with torch.autocast(device_type='npu',dtype=torch.bfloat16):pred=model(z,cam,target,feat).float().cpu().exp()[0]
   rows.append(metrics(pred,d[target]))
   if gi==0:
    gt=d[target].numpy();pr=pred.numpy();mask=np.isfinite(gt)&(gt>1e-4);pr*=np.median(gt[mask])/max(np.median(pr[mask]),1e-6);lo,hi=np.percentile(gt[mask],[2,98]);maps=[np.repeat((np.clip((v-lo)/max(hi-lo,1e-6),0,1)[:,:,None]*255).astype('uint8'),3,axis=2) for v in [gt,pr]]
    folder=out/'visuals'/split;folder.mkdir(parents=True,exist_ok=True);Image.fromarray(np.concatenate([rgb[target],*maps],axis=1)).save(folder/f'view{target}_step{step}.jpg')
 result={k:float(np.mean([r[k] for r in rows])) for k in rows[0]};result.update(step=step,samples=len(rows),split=split)
 atomic(out/f'{split}_step{step}.json',result);print('EVAL',json.dumps(result),flush=True);model.train();return result
step=0;best=float('inf');last=out/'last.pt'
if last.exists():
 state=torch.load(last,map_location='cpu',weights_only=False);model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);step=state['step'];best=state['best'];random.setstate(state['python_rng']);torch.set_rng_state(state['torch_rng']);print('RESUME',step,flush=True)
# Verify both depth head and fusion participate; only the decoder/fusion has gradients.
z,d,cam,feat,_=inputs(train[0]);opt.zero_grad()
with torch.autocast(device_type='npu',dtype=torch.bfloat16):pred=model(z,cam,0,feat);loss=loss_fn(pred,d[:1].to(device))
loss.backward();assert torch.isfinite(loss);assert model.decoder.core.decoder.head[-1].weight.grad is not None;assert model.to_latent.weight.grad is not None;assert all(p.grad is None for p in model.decoder.core.encoder.parameters())
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
atomic(out/'smoke.json',dict(passed=True,output_shape=list(pred.shape),encoder_frozen=True,dit_frozen=True,depth_head_gradient=True,fusion_gradient=True,peak_allocated_gib=torch.npu.max_memory_allocated()/2**30))
print('SMOKE PASS',flush=True);opt.zero_grad();del pred,loss,z,d,cam,feat
if not step:evaluate('val',0)
started=time.time();base_step=step
while step<args.steps:
 opt.zero_grad();losses=[]
 for micro in range(8):
  g=random.choice(train);z,d,cam,feat,_=inputs(g);target=random.randrange(4)
  with torch.autocast(device_type='npu',dtype=torch.bfloat16):pred=model(z,cam,target,feat);loss=loss_fn(pred,d[target:target+1].to(device))
  assert torch.isfinite(loss), 'Non-finite training loss'
  (loss/8).backward();losses.append(float(loss.detach().cpu()))
 torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1);opt.step();step+=1
 if step%20==0:
  row=dict(step=step,loss=float(np.mean(losses)),seconds=time.time()-started,steps_this_session=step-base_step,peak_allocated_gib=torch.npu.max_memory_allocated()/2**30,peak_reserved_gib=torch.npu.max_memory_reserved()/2**30)
  with (out/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  atomic(out/'status.json',dict(status='training',pid=os.getpid(),**row));print(json.dumps(row),flush=True)
 if step%500==0:
  evaluate('train',step);m=evaluate('val',step)
  if m['aligned_absrel']<best:
   best=m['aligned_absrel'];torch.save(dict(model=model.state_dict(),step=step),out/'best.tmp');(out/'best.tmp').replace(out/'best.pt')
  torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step,best=best,python_rng=random.getstate(),torch_rng=torch.get_rng_state()),out/'last.tmp');(out/'last.tmp').replace(last)
state=torch.load(out/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(state['model']);result=evaluate('test',state['step']);atomic(out/'status.json',dict(status='complete',complete=True,best_validation_aligned_absrel=best,test=result))
