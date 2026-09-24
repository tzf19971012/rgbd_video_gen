import argparse
import gc
import os
import random
from datetime import timedelta
from itertools import cycle

import torch
from accelerate import Accelerator, accelerator
from accelerate.utils import InitProcessGroupKwargs, set_seed
from omegaconf import OmegaConf
from safetensors.torch import load_file
from tqdm import tqdm

from examples.dataset import (KITTI_VID_Dataset, ManifestVIDDataset,
                              NYUv2Dataset, Scannet_VID_Dataset)
# Import modules
from examples.wanvideo.model_training.DiffusionTrainingModule import \
    DiffusionTrainingModule
from examples.wanvideo.model_training.WanTrainingModule import (
    Validation, WanTrainingModule)

from .training_loss import GradientLoss3DSeparate

process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=30))


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def custom_collate_fn(batch):
    collated = {}
    for key in batch[0].keys():
        values = [d[key] for d in batch]
        if isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values)
        elif isinstance(values[0], str):
            collated[key] = values
        elif values[0] is None:
            collated[key] = None
        else:
            raise TypeError(
                f"Unsupported type for key '{key}': {type(values[0])}")
    return collated


def get_data(data, args):
    # print(f"data {data if isinstance(data,str) else type(data)}")
    input_data = {
        "images": data["images"],
        "disparity": data["disparity"],
        # Extra images
        "extra_images": data.get("extra_images", None),
        "extra_image_frame_index": data.get("extra_image_frame_index", None),
        # Shape
        "batch_size": data["images"].shape[0],
        "num_frames": data["images"].shape[1],
        "height": data["images"].shape[-2],
        "width": data["images"].shape[-1],
    }
    return input_data


def build_train_dataloaders(args, accelerator):
    """Build one train dataloader per entry in `args.train_datasets`.

    Each entry is a manifest.jsonl-backed video dataset, e.g.:
        - name: tartanair
          data_root: /path/to/TartanAir
          manifest: manifest.jsonl
          prob: 1
          scene_filter: null          # optional substring filter on scene_id
          min_num_frame: 45           # optional per-dataset overrides
          enlarge_factor: 100
    """
    dataloader_list, dataloader_prob = [], []
    for ds_cfg in args.train_datasets:
        dataset = ManifestVIDDataset(
            data_root=ds_cfg.data_root,
            manifest_path=ds_cfg.get("manifest", "manifest.jsonl"),
            name=ds_cfg.get("name", None),
            resolution=tuple(ds_cfg.get("resolution", [480, 640])),
            random_flip=args.random_flip,
            norm_type=args.norm_type,
            truncnorm_min=args.truncnorm_min,
            max_num_frame=ds_cfg.get("max_num_frame", args.max_num_frame),
            min_num_frame=ds_cfg.get("min_num_frame", args.min_num_frame),
            max_sample_stride=ds_cfg.get(
                "max_sample_stride", args.max_sample_stride),
            min_sample_stride=ds_cfg.get(
                "min_sample_stride", args.min_sample_stride),
            scene_filter=ds_cfg.get("scene_filter", None),
            train_ratio=ds_cfg.get("train_ratio", args.train_ratio),
            enlarge_factor=ds_cfg.get("enlarge_factor", 1),
            d_max=ds_cfg.get("d_max", 50),
        )
        num_workers = args.dataloader_num_workers
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=True,
            batch_size=ds_cfg.get("batch_size", args.batch_size),
            num_workers=num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=True,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
            drop_last=True,
        )
        dataloader_list.append(dataloader)
        dataloader_prob.append(ds_cfg.get("prob", 1))
        accelerator.print(
            f"[TrainDataset:{dataset.name}] {len(dataset)} samples, "
            f"batch_size={ds_cfg.get('batch_size', args.batch_size)}, "
            f"prob={ds_cfg.get('prob', 1)}"
        )
    return dataloader_list, dataloader_prob


def build_test_loader_dict(args, accelerator):
    """Build test dataloaders from `args.test_datasets`.

    Each test dataset has an `enable` flag so platforms without the
    test data can simply turn them off in the yaml.
    """
    test_loader_dict = {}
    dataset_range = {}

    def _make_loader(dataset):
        return torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            batch_size=args.test_batch_size,
            num_workers=2,
            collate_fn=custom_collate_fn,
            pin_memory=True,
            persistent_workers=True,
        )

    test_cfgs = args.get("test_datasets", None)
    if test_cfgs is None:
        return test_loader_dict, dataset_range

    if test_cfgs.get("kitti", {}).get("enable", False):
        dataset = KITTI_VID_Dataset(
            data_root=test_cfgs.kitti.data_root,
            max_num_frame=args.test_max_num_frame,
            min_num_frame=args.test_min_num_frame,
            max_sample_stride=args.test_max_sample_stride,
            min_sample_stride=args.test_min_sample_stride,
        )
        test_loader_dict["kitti"] = _make_loader(dataset)
        dataset_range["kitti"] = [1e-5, 80]
        accelerator.print(f"[TestDataset] kitti enabled: {test_cfgs.kitti.data_root}")

    if test_cfgs.get("scannet", {}).get("enable", False):
        dataset = Scannet_VID_Dataset(
            data_root=test_cfgs.scannet.data_root,
            split_ls=test_cfgs.scannet.get(
                "split_ls", "configs/scannetv1_test.txt"),
            test=False,
            max_num_frame=args.test_max_num_frame,
            min_num_frame=args.test_min_num_frame,
            max_sample_stride=args.test_max_sample_stride,
            min_sample_stride=args.test_min_sample_stride,
        )
        test_loader_dict["scannet"] = _make_loader(dataset)
        dataset_range["scannet"] = [1e-3, 10]
        accelerator.print(f"[TestDataset] scannet enabled: {test_cfgs.scannet.data_root}")

    if test_cfgs.get("nyuv2", {}).get("enable", False):
        dataset = NYUv2Dataset(
            data_root=test_cfgs.nyuv2.data_root,
            test=False,
        )
        test_loader_dict["nyuv2"] = _make_loader(dataset)
        dataset_range["nyuv2"] = [1e-3, 10]
        accelerator.print(f"[TestDataset] nyuv2 enabled: {test_cfgs.nyuv2.data_root}")

    return test_loader_dict, dataset_range


class ModelLogger:
    def __init__(
        self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x: x
    ):
        self.output_path = output_path
        import time
        os.makedirs(self.output_path, exist_ok=True)


def launch_training_task(
    accelerator,
    start_epoch,
    global_step,
    args,
    dataset_range,
    train_dataloader_list,
    test_loader_dict,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    num_epochs: int = 1,
    validate_step: int = 500,
    log_step: int = 10,
    dataloader_prob: list = None,
):
    validator = Validation()
    accelerator.print(
        f"Initial accelerator with gradient accumulation steps: {accelerator.gradient_accumulation_steps}"
    )
    accelerator.print(
        f"Using {accelerator.num_processes} processes for training.")

    accelerator.print(
        f"accelerator.state.deepspeed_plugin: {accelerator.state.deepspeed_plugin}")

    accelerator.print(f"Validate every {validate_step} steps.")

    if args.init_validate and len(test_loader_dict) > 0:
        accelerator.print(
            f"Starting validation with model at epoch {start_epoch}, global step {global_step}"
        )
        model.pipe.dit.eval()
        # if accelerator.is_main_process:
        validator.validate(
            accelerator=accelerator,
            dataset_range=dataset_range,
            pipe=model.pipe,
            global_step=global_step,
            args=args,
            test_loader_dict=test_loader_dict,
            output_path=model_logger.output_path,
        )

        model.pipe.scheduler.set_timesteps(
            training=True,
            denoise_step=args.denoise_step,
        )
        model.pipe.dit.train()
    accelerator.wait_for_everyone()

    optimizer.zero_grad()
    accumulate_depth_loss = 0.0
    accumulate_grad_loss = 0.0

    acm_cnt = 0
    rank = accelerator.process_index

    loader_iter_list = [iter(_train_dataloader)
                        for _train_dataloader in train_dataloader_list]
    if dataloader_prob is None:
        dataloader_prob = args.get(
            'prob', [1 for _ in range(len(train_dataloader_list))])
    assert len(dataloader_prob) == len(train_dataloader_list), (
        f"dataloader_prob {dataloader_prob} does not match "
        f"{len(train_dataloader_list)} train dataloaders"
    )
    prob = list(dataloader_prob)
    grad_loss = GradientLoss3DSeparate()

    pick_ranges = []
    start = 0
    for p in prob:
        end = start + p
        pick_ranges.append((start, end))
        start = end
    set_seed(42)

    print(f"{rank} Entering training loop...")

    for epoch_id in range(num_epochs):
        for small_batch_step in tqdm(range(100_000), desc=f"Epoch {epoch_id + 1}/{num_epochs}", disable=not accelerator.is_main_process):

            select_pos = random.choices(
                population=range(len(train_dataloader_list)),
                weights=prob,
                k=1
            )[0]

            data = None
            try:
                data = next(loader_iter_list[select_pos])
            except StopIteration:
                print(
                    f"GPU used up dataset {select_pos}, setting up new one...")
                loader_iter_list[select_pos] = iter(
                    train_dataloader_list[select_pos])
                data = next(loader_iter_list[select_pos])

            # Forward and backward pass
            with accelerator.accumulate(model):
                input_data = get_data(data, args=args)
                res_dict = model(input_data, args=args)
                depth_gt = res_dict['depth_gt']
                pred = res_dict['pred']

                # from torchvision.utils import save_image
                pred_rgb, pred_depth = None, None

                if isinstance(pred, tuple):
                    pred_depth, pred_rgb = pred
                else:
                    pred_depth = pred
                
                loss = torch.nn.functional.mse_loss(
                    depth_gt, pred_depth)
                
                
                accumulate_depth_loss += loss.item()

                if args.get('grad_loss', False):
                    _grad_loss = grad_loss(pred_depth, depth_gt)
                    _grad_t, _grad_h, _grad_w = _grad_loss
                    grad_co = args.get('grad_co', 1)
                    use_latent_flow = args.get('use_latent_flow', True)
                    if not use_latent_flow:
                        _grad_t = 0
                    loss += grad_co * (_grad_t+_grad_h+_grad_w)
                accumulate_grad_loss += loss.item()
                
                # accelerator.print(f"Microstep {small_batch_step} total loss: {loss.item()} pred_depth shape: {pred_depth.shape}, gt_depth shape: {depth_gt.shape}")
                accelerator.backward(loss)
                acm_cnt += 1

                # Update optimizer and scheduler
                if accelerator.sync_gradients:
                    
                    if args.get('clip_grad_norm', True):
                        accelerator.clip_grad_norm_(
                            model.trainable_modules(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    global_step += 1
                    # print(f"Step at {global_step} microstep {small_batch_step}")

                    # Calculate the average loss across all processes
                    if global_step % log_step == 0:
                        accumulate_depth_loss /= acm_cnt
                        accumulate_grad_loss /= acm_cnt

                        accumulate_grad_loss = accumulate_grad_loss - accumulate_depth_loss

                        print(
                            f"GPU {rank} step {global_step}: depth loss = {accumulate_depth_loss}, grad_loss = {accumulate_grad_loss}, learning rate : {scheduler.get_last_lr()[0]:.8f}"
                        )
                        accumulate_depth_loss = 0.0
                        accumulate_grad_loss = 0.0
                        acm_cnt = 0

                    if (global_step) % validate_step == 0:
                        model.pipe.dit.eval()
                        print(f"GPU {rank} saving training state...")
                        accelerator.save_state(
                            os.path.join(model_logger.output_path,
                                         f"checkpoint-step-{global_step}")
                        )
                        if accelerator.is_main_process:

                            torch.save(
                                {"global_step": global_step},
                                os.path.join(
                                    model_logger.output_path, "trainer_state.pt")
                            )
                            accelerator.print(
                                f"Checkpoint saved at step {global_step}")
                        validator.validate(
                            accelerator=accelerator,
                            pipe=model.pipe,
                            dataset_range=dataset_range,
                            global_step=global_step,
                            args=args,
                            test_loader_dict=test_loader_dict,
                            output_path=model_logger.output_path,
                        )

                        model.pipe.scheduler.set_timesteps(
                            training=True,
                            denoise_step=args.denoise_step,
                        )
                        model.pipe.dit.train()
                    accelerator.wait_for_everyone()

        accelerator.end_training()


if __name__ == "__main__":
    def get_config():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--config", type=str, default=None, help="Path to config yaml"
        )
        args = parser.parse_args()
        cfg = OmegaConf.load(args.config)
        return cfg

    cfg = get_config()
    args = cfg

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        kwargs_handlers=[process_group_kwargs],
    )
    accelerator.print(OmegaConf.to_yaml(cfg))

    # set_seed(42)

    # Save args
    os.makedirs(args.output_path, exist_ok=True)
    args_save_path = os.path.join(args.output_path, "args.yaml")

    if accelerator.is_main_process:
        accelerator.print(f"Saving args to {args_save_path}")
        with open(args_save_path, "w") as f:
            f.write(OmegaConf.to_yaml(args))
    accelerator.wait_for_everyone()

    # Load model
    model = WanTrainingModule(
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        lora_rank=args.lora_rank,
        lora_base_model=args.lora_base_model,
        local_model_path=args.wan_model_root,
        skip_download=args.skip_download,
        args=args,
        accelerator=accelerator,
    )

    model.set_training_param()

    model_logger = ModelLogger(
        args.output_path,
    )

    optimizer = torch.optim.AdamW(
        model.trainable_modules(), lr=args.learning_rate)
    world_size = accelerator.num_processes

    if args.warmup_steps > 0:
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.5, total_iters=args.warmup_steps*world_size
        )
    else:
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    # Training datasets: one manifest.jsonl-backed video dataset per entry
    # in `args.train_datasets` (see yaml).
    train_dataloader_list, dataloader_prob = build_train_dataloaders(
        args, accelerator)

    # Test datasets: each can be enabled/disabled in `args.test_datasets`.
    test_loader_dict, dataset_range = build_test_loader_dict(
        args, accelerator)
    accelerator.print(
        f"Test datasets enabled: {list(test_loader_dict.keys())}")

    start_epoch, global_step = 0, 0
    _num_train = len(train_dataloader_list)
    _num_test = len(test_loader_dict)
    prepared = accelerator.prepare(
        model, optimizer, scheduler,
        *train_dataloader_list, *test_loader_dict.values(),
    )
    model, optimizer, scheduler = prepared[0], prepared[1], prepared[2]
    train_dataloader_list = prepared[3:3 + _num_train]
    test_loader_dict = dict(
        zip(test_loader_dict.keys(), prepared[3 + _num_train:]))
    assert len(test_loader_dict) == _num_test

    if args.resume and args.training_state_dir is not None:

        _training_state_dir = args.training_state_dir
        if os.path.exists(_training_state_dir) and 'checkpoint' in _training_state_dir:

            accelerator.print(
                f"Resuming training state from {args.training_state_dir}...")
            # assign training state dir must come with a given global step
            global_step = args.global_step

        else:  # use the global step to find
            g_s_path = os.path.join(
                args.training_state_dir,  'trainer_state.pt')
            if os.path.exists(g_s_path):
                g_s_pt = torch.load(g_s_path)
                global_step = g_s_pt['global_step']

            args.training_state_dir = os.path.join(
                args.training_state_dir, 'checkpoint-step-{}'.format(global_step))

        if args.load_optimizer:
            accelerator.load_state(args.training_state_dir)
            for pg in optimizer.param_groups:
                pg['lr'] = args.learning_rate

        else:
            unwrapped_model = accelerator.unwrap_model(model)
            ckpt_path = os.path.join(
                args.training_state_dir, "model.safetensors")
            state_dict = load_file(ckpt_path, device="cpu")
            missing, unexpected = unwrapped_model.load_state_dict(
                state_dict, strict=False)
            assert len(unexpected) == 0

        if global_step > 0:
            accelerator.print(f"Resuming from global step {global_step}")
        else:
            accelerator.print(f"Training from scratch...")
        accelerator.print("Training state loaded.")
        accelerator.wait_for_everyone()

    if hasattr(model, 'module'):
        model = model.module

    launch_training_task(
        accelerator=accelerator,
        train_dataloader_list=train_dataloader_list,
        test_loader_dict=test_loader_dict,
        dataset_range=dataset_range,
        start_epoch=start_epoch,
        global_step=global_step,
        model=model,
        model_logger=model_logger,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        validate_step=args.validate_step,
        log_step=args.log_step,
        dataloader_prob=dataloader_prob,
        args=args,
    )
