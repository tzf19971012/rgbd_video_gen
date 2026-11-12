"""Validate the16-rank GAE projection continuation without loading a model."""
import argparse,json,struct,zipfile,sys
from pathlib import Path
from omegaconf import OmegaConf
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--world-size',type=int,required=True);p.add_argument('--code-root',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--check-hardware',action='store_true');p.add_argument('--local-processes',type=int,default=8);a,extra=p.parse_known_args()
assert all('=' in x for x in extra), 'Extra arguments must be key=value'
cfg=OmegaConf.merge(OmegaConf.load(a.config),OmegaConf.from_dotlist(extra))
assert a.world_size==16 and cfg.projection_condition=='gae'
assert cfg.resume and not cfg.resume_weights_only
assert cfg.max_steps==30000 and cfg.global_step==8600
assert cfg.num_train_samples is None and cfg.scene_filter is None
assert cfg.num_source_views==6 and cfg.num_target_views==3
assert cfg.batch_size>=1 and cfg.gradient_accumulation_steps>=1
assert str(Path(a.output_dir).resolve())!=str(Path('/mnt/sfs_turbo/t00845730/rgbd_video_gen_gae/logs/projection_gae_3npu_resume_3card_20261110_110818').resolve()), 'Use a new cloud output directory'
ck=Path(a.checkpoint);assert ck.name=='checkpoint-step-8600'
for n in ['model.safetensors','optimizer.bin','scheduler.bin','random_states_0.pkl','random_states_1.pkl','random_states_2.pkl']:assert (ck/n).is_file(),n
for n in ['optimizer.bin','scheduler.bin','random_states_0.pkl','random_states_1.pkl','random_states_2.pkl']:assert zipfile.is_zipfile(ck/n),n
with (ck/'model.safetensors').open('rb') as f:
 n=struct.unpack('<Q',f.read(8))[0];head=json.loads(f.read(n))
assert max(v['data_offsets'][1] for k,v in head.items() if k!='__metadata__')+8+n==(ck/'model.safetensors').stat().st_size
adapter=[v['shape'] for k,v in head.items() if 'projection_adapter.0.weight' in k];assert adapter and adapter[0][1]==66,adapter
code=Path(a.code_root)
for rel in ['train.py','utils/projection_condition.py','examples/wanvideo/model_training/WanGAENVSTrainingModule.py']:assert (code/rel).is_file(),rel
assert 'accelerator.process_index' in (code/'train.py').read_text()
manifest=Path(str(cfg.manifest_path[0]));assert manifest.name=='manifest_train_c2w_v1.jsonl'
scenes={json.loads(line)['scene_id'] for line in manifest.open()};assert len(scenes)==7884,len(scenes)
cache=Path(cfg.gae_cache_root)/cfg.gae_dataset_name;assert cfg.gae_dataset_name=='re10k_train'
assert (cache/'latent_stats.pt').is_file()
missing=[sid for sid in scenes if not (cache/(sid+'.meta.json')).is_file()];assert not missing,('Missing scene caches',len(missing))
if a.check_hardware:
 import torch,torch_npu
 assert torch.npu.is_available() and torch.npu.device_count()>=a.local_processes
print(json.dumps(dict(preflight_passed=True,world_size=a.world_size,checkpoint=str(ck),projection='gae',train_scenes=len(scenes),test_data_used=False,start_step=8600,max_steps=30000,additional_steps=21400,batch_size=cfg.batch_size,gradient_accumulation_steps=cfg.gradient_accumulation_steps,effective_batch=cfg.batch_size*cfg.gradient_accumulation_steps*a.world_size,learning_rate=cfg.learning_rate,fresh_layer_learning_rate=cfg.learning_rate*cfg.fresh_layer_lr_mult,output=a.output_dir,training_entry=str(code/'train.py'),optimizer_resume_mode="full state at training startup",checkpoint_files_validated=True,rng='reseed per rank after world-size change'),indent=2))
