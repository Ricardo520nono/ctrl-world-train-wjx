# from diffusers import StableVideoDiffusionPipeline
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.pipeline_stable_video_diffusion import StableVideoDiffusionPipeline
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from models.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from models.ctrl_world import CrtlWorld

import numpy as np
import torch
import torch.nn as nn
import einops
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin, InitProcessGroupKwargs
import datetime
import os
from accelerate.logging import get_logger
from tqdm.auto import tqdm
import json
from decord import VideoReader, cpu
import wandb
swanlab = None
import mediapy
from models.ctrl_world import CrtlWorld
from config import wm_args
import math
import time


def main(args):
    logger = get_logger(__name__, log_level="INFO")
    if swanlab is not None:
        try:
            swanlab.sync_wandb()
        except Exception as e:
            print(f"[WARN] swanlab.sync_wandb skipped: {e}")

    # Pre-initialize process group with extended timeout so the NCCL watchdog
    # uses 2-hour timeout instead of the default 10 minutes. Accelerate and
    # DeepSpeed will reuse this already-initialized group.
    if int(os.environ.get("LOCAL_RANK", -1)) != -1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(seconds=7200),
        )

    deepspeed_plugin = None
    if getattr(args, 'use_deepspeed', False):
        ds_config = {
            "zero_optimization": {
                "stage": 2,
                "overlap_comm": False,
                "contiguous_gradients": True,
                "allgather_bucket_size": 2e8,
                "reduce_bucket_size": 2e8,
            },
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "gradient_clipping": args.max_grad_norm,
            "train_micro_batch_size_per_gpu": args.train_batch_size,
            "fp16": {"enabled": args.mixed_precision == "fp16"},
            "bf16": {"enabled": args.mixed_precision == "bf16"},
        }
        deepspeed_plugin = DeepSpeedPlugin(
            hf_ds_config=ds_config,
            zero_stage=2,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_clipping=args.max_grad_norm,
        )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with='wandb',
        project_dir=args.output_dir,
        deepspeed_plugin=deepspeed_plugin,
        kwargs_handlers=[InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=7200))],
    )

    # model and optimizer
    model = CrtlWorld(args)
    if args.ckpt_path is not None:
        print(f"Loading checkpoint from {args.ckpt_path}!")
        state_dict = torch.load(args.ckpt_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
    model.to(accelerator.device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # logs
    if accelerator.is_main_process:
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        tag = args.tag
        run_name = f"train_{now}_{tag}"
        accelerator.init_trackers(args.wandb_project_name,config={}, init_kwargs={"wandb":{"name":run_name}})
        os.makedirs(args.output_dir, exist_ok=True)
        # count parameters num in each part
        num_params = sum(p.numel() for p in model.unet.parameters())
        print(f"Number of parameters in the unet: {num_params/1000000:.2f}M")
        num_params = sum(p.numel() for p in model.vae.parameters())
        print(f"Number of parameters in the vae: {num_params/1000000:.2f}M")
        num_params = sum(p.numel() for p in model.image_encoder.parameters())
        print(f"Number of parameters in the image_encoder: {num_params/1000000:.2f}M")
        num_params = sum(p.numel() for p in model.text_encoder.parameters())
        print(f"Number of parameters in the text_encoder: {num_params/1000000:.2f}M")
        num_params = sum(p.numel() for p in model.action_encoder.parameters())
        print(f"Number of parameters in the action_encoder: {num_params/1000000:.2f}M")

    # train and val datasets — ActionFollowingBench delta-ee
    if getattr(args, 'use_family_balanced_sampler', False):
        from dataset.dataset_delta_ee_family import DeltaEEFamilyBalancedDataset
        train_dataset = DeltaEEFamilyBalancedDataset(args, mode='train')
    else:
        from dataset.dataset_delta_ee import DeltaEEDataset
        train_dataset = DeltaEEDataset(args, mode='train')
    # val uses a separate episode split (test episodes) if val_episode_split is set
    if getattr(args, 'val_episode_split', None):
        import copy
        val_args = copy.copy(args)
        val_args.episode_split = args.val_episode_split
        from dataset.dataset_delta_ee import DeltaEEDataset
        val_dataset = DeltaEEDataset(val_args, mode='val')
    else:
        from dataset.dataset_delta_ee import DeltaEEDataset
        val_dataset = DeltaEEDataset(args, mode='val')
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.train_batch_size,
        shuffle=args.shuffle
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.train_batch_size,
        shuffle=args.shuffle
    )

    # Prepare everything with our accelerator
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )
   
    ############################ training ##############################
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_train_epochs = math.ceil(args.max_train_steps * args.gradient_accumulation_steps*total_batch_size / len(train_dataloader))
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  checkpointing_steps = {args.checkpointing_steps}")
    logger.info(f"  validation_steps = {args.validation_steps}")
    global_step = 0
    forward_step=0
    train_loss = 0.0
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    step_timer = time.time()
    data_timer = time.time()

    for epoch in range(num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            data_time = time.time() - data_timer
            is_last_accum = (forward_step + 1) % args.gradient_accumulation_steps == 0
            with accelerator.autocast():
                loss_gen, _ = model(batch)
            avg_loss = accelerator.gather(loss_gen.repeat(args.train_batch_size)).mean()
            train_loss += avg_loss.item() / args.gradient_accumulation_steps
            accelerator.backward(loss_gen)
            forward_step += 1
            grad_norm = None
            if is_last_accum:
                params_to_clip = model.parameters()
                grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            iter_time = time.time() - step_timer
            step_timer = time.time()
            data_timer = time.time()

            if is_last_accum:
                progress_bar.update(1)
                global_step += 1

                current_lr = optimizer.param_groups[0]["lr"]
                throughput = total_batch_size / max(iter_time, 1e-6)
                gpu_mem_gb = 0.0
                if torch.cuda.is_available():
                    gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

                log_payload = {
                    "train_loss": float(avg_loss.item()),
                    "learning_rate": float(current_lr),
                    "grad_norm": float(grad_norm.item()) if grad_norm is not None and torch.is_tensor(grad_norm) else (float(grad_norm) if grad_norm is not None else 0.0),
                    "global_step": int(global_step),
                    "epoch": int(epoch),
                    "throughput_samples_per_s": float(throughput),
                    "data_time_s": float(data_time),
                    "iter_time_s": float(iter_time),
                    "gpu_mem_gb": float(gpu_mem_gb),
                }
                accelerator.log(log_payload, step=global_step)

                if global_step % 20 == 0:
                    progress_bar.set_postfix({"loss": float(avg_loss.item()), "lr": current_lr})

                # save ckpt every checkpointing_steps (disabled when 0/None)
                if args.checkpointing_steps and global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    current_epoch = global_step * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes / len(train_dataset)
                    save_path = os.path.join(args.output_dir, f"checkpoint-step{global_step}-epoch{current_epoch:.2f}.pt")
                    torch.save(accelerator.unwrap_model(model).state_dict(), save_path)
                    logger.info(f"Saved checkpoint to {save_path}")
                # generate video every validation_steps
                if global_step % args.validation_steps == 5:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        model.eval()
                        with accelerator.autocast():
                            for id in range(args.video_num):
                                validate_video_generation(model, val_dataset, args,global_step, args.output_dir, id, accelerator)
                        model.train()
                    accelerator.wait_for_everyone()

                if global_step >= args.max_train_steps:
                    break

        # epoch-boundary checkpoint: save every N completed epochs (only when the
        # epoch fully finished, i.e. we did not break out early at max_train_steps)
        if getattr(args, 'checkpointing_epochs', None) and global_step < args.max_train_steps \
                and (epoch + 1) % args.checkpointing_epochs == 0 and accelerator.is_main_process:
            save_path = os.path.join(args.output_dir, f"checkpoint-epoch{epoch+1}-step{global_step}.pt")
            torch.save(accelerator.unwrap_model(model).state_dict(), save_path)
            logger.info(f"[epoch-ckpt] Saved checkpoint to {save_path}")

        if global_step >= args.max_train_steps:
            break

    # always save a final checkpoint at the end of training
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, f"checkpoint-final-step{global_step}.pt")
        torch.save(accelerator.unwrap_model(model).state_dict(), save_path)
        logger.info(f"[final-ckpt] Saved checkpoint to {save_path}")



def main_val(args):
    accelerator = Accelerator()
    model = CrtlWorld(args)
    # load form val_model_path
    print("load from val_model_path",args.val_model_path)
    model.load_state_dict(torch.load(args.val_model_path))
    model.to(accelerator.device)
    model.eval()
    validate_video_generation(model, None, args, 0, 'output', 0, accelerator, load_from_dataset=False)
    
            

def validate_video_generation(model, val_dataset, args, train_steps, videos_dir, id, accelerator, load_from_dataset=True):
    device = accelerator.device
    pipeline = model.module.pipeline if accelerator.num_processes > 1 else model.pipeline
    videos_row = args.video_num if not args.debug else 1
    videos_col = 2

    # sample from val dataset
    batch_id = list(range(0,len(val_dataset),max(1, int(len(val_dataset)/videos_row/videos_col))))
    batch_id = batch_id[int(id*(videos_col)):int((id+1)*(videos_col))]
    batch_list = [val_dataset.__getitem__(id) for id in batch_id]
    video_gt = torch.cat([t['latent'].unsqueeze(0) for i,t in enumerate(batch_list)],dim=0).to(device, non_blocking=True)
    text = [t['text'] for i,t in enumerate(batch_list)]
    actions = torch.cat([t['action'].unsqueeze(0) for i,t in enumerate(batch_list)],dim=0).to(device, non_blocking=True)
    his_latent_gt, future_latent_ft = video_gt[:,:args.num_history], video_gt[:,args.num_history:]
    current_latent = future_latent_ft[:,0]
    print("image",current_latent.shape, 'action', actions.shape)
    assert current_latent.shape[1:] == (4, 90, 40)
    assert actions.shape[1:] == (int(args.num_frames+args.num_history), args.action_dim)

    # start generate
    with torch.no_grad():
        bsz = actions.shape[0]
        action_latent = model.module.action_encoder(actions, text, model.module.tokenizer, model.module.text_encoder, args.frame_level_cond) if accelerator.num_processes > 1 else model.action_encoder(actions, text, model.tokenizer, model.text_encoder,args.frame_level_cond) # (8, 1, 1024)
        print("action_latent",action_latent.shape)

        _, pred_latents = CtrlWorldDiffusionPipeline.__call__(
            pipeline,
            image=current_latent,
            text=action_latent,
            width=args.width,
            height=int(3*args.height),
            num_frames=args.num_frames,
            history=his_latent_gt,
            num_inference_steps=args.num_inference_steps,
            decode_chunk_size=args.decode_chunk_size,
            max_guidance_scale=args.guidance_scale,
            fps=args.fps,
            motion_bucket_id=args.motion_bucket_id,
            mask=None,
            output_type='latent',
            return_dict=False,
            frame_level_cond=args.frame_level_cond,
            his_cond_zero=args.his_cond_zero,
        )
    
    pred_latents = einops.rearrange(pred_latents, 'b f c (m h) (n w) -> (b m n) f c h w', m=3,n=1) # (B, 8, 4, 32,32)
    video_gt = torch.cat([his_latent_gt, future_latent_ft], dim=1) # (B, 8, 4, 32,32)
    video_gt = einops.rearrange(video_gt, 'b f c (m h) (n w) -> (b m n) f c h w', m=3,n=1) # (B, 8, 4, 32,32)
    
    # decode latent
    vae_dtype = pipeline.vae.dtype
    if video_gt.shape[2] != 3:
        decoded_video = []
        bsz,frame_num = video_gt.shape[:2]
        video_gt = video_gt.flatten(0,1)
        decode_kwargs = {}
        for i in range(0,video_gt.shape[0],args.decode_chunk_size):
            chunk = (video_gt[i:i+args.decode_chunk_size]/pipeline.vae.config.scaling_factor).to(vae_dtype)
            decode_kwargs["num_frames"] = chunk.shape[0]
            decoded_video.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
        video_gt = torch.cat(decoded_video,dim=0)
        video_gt = video_gt.reshape(bsz,frame_num,*video_gt.shape[1:])

        decoded_video = []
        bsz,frame_num = pred_latents.shape[:2]
        pred_latents = pred_latents.flatten(0,1)
        decode_kwargs = {}
        for i in range(0,pred_latents.shape[0],args.decode_chunk_size):
            chunk = (pred_latents[i:i+args.decode_chunk_size]/pipeline.vae.config.scaling_factor).to(vae_dtype)
            decode_kwargs["num_frames"] = chunk.shape[0]
            decoded_video.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
        videos = torch.cat(decoded_video,dim=0)
        videos = videos.reshape(bsz,frame_num,*videos.shape[1:])

    video_gt = ((video_gt / 2.0 + 0.5).clamp(0, 1)*255)
    video_gt = video_gt.float().detach().cpu().numpy().transpose(0,1,3,4,2).astype(np.uint8)
    videos = ((videos / 2.0 + 0.5).clamp(0, 1)*255)
    videos = videos.float().detach().cpu().numpy().transpose(0,1,3,4,2).astype(np.uint8) #(2,16,256,256,3)
    videos = np.concatenate([video_gt[:, :args.num_history],videos],axis=1) #(2,16,512,256,3)
    videos = np.concatenate([video_gt,videos],axis=-3) #(2,16,512,256,3)
    videos = np.concatenate([video for video in videos],axis=-2).astype(np.uint8) # (16,512,256*batch,3)
    
    os.makedirs(f"{videos_dir}/samples", exist_ok=True)
    filename = f"{videos_dir}/samples/train_steps_{train_steps}_{id}.mp4"
    try:
        mediapy.write_video(filename, videos, fps=2)
    except Exception as e:
        print(f"[WARN] write_video failed (ffmpeg missing?): {e}")
    return



if __name__ == "__main__":
    # reset parameters with command line
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--svd_model_path', type=str, default=None)
    parser.add_argument('--clip_model_path', type=str, default=None)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--dataset_root_path', type=str, default=None)
    parser.add_argument('--dataset_meta_info_path', type=str, default=None)
    parser.add_argument('--dataset_cfgs', type=str, default=None)
    # dataset_names
    parser.add_argument('--dataset_names', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--wandb_project_name', type=str, default=None)
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--max_train_steps', type=int, default=None)
    parser.add_argument('--train_batch_size', type=int, default=None)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=None)
    parser.add_argument('--num_history', type=int, default=None)
    parser.add_argument('--num_frames', type=int, default=None)
    parser.add_argument('--action_dim', type=int, default=None)
    parser.add_argument('--height', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--checkpointing_steps', type=int, default=None,
                        help="Save a checkpoint every N optimizer steps. Set 0/empty to disable step-based saving.")
    parser.add_argument('--checkpointing_epochs', type=int, default=None,
                        help="If set, save a checkpoint at the end of every N completed epochs.")
    parser.add_argument('--validation_steps', type=int, default=None)
    parser.add_argument('--mixed_precision', type=str, default=None)
    parser.add_argument('--episode_split', type=str, default=None,
                        help="Episode index range for train split, e.g. '0-39'")
    parser.add_argument('--val_episode_split', type=str, default=None,
                        help="Episode index range for val split, e.g. '40-49'")
    parser.add_argument('--dataset_root_sampling', type=str, default=None,
                        help="Per-root sampling strategy, e.g. 'sliding+single'")
    parser.add_argument('--use_family_balanced_sampler', action='store_true')
    parser.add_argument('--family_root_paths', type=str, default=None,
                        help="Semicolon-separated roots, e.g. expert=/a;pca=/b;raw=/c;rf_uniform=/d;rf_weighted=/e")
    parser.add_argument('--family_sampling', type=str, default=None,
                        help="Comma-separated probabilities, e.g. expert=0.5,pca=0.166667,raw=0.166667,random_feasible=0.166667")
    parser.add_argument('--family_sampling_seed', type=int, default=None)
    parser.add_argument('--family_dataset_length', type=int, default=None,
                        help="Virtual dataset length for family-balanced sampling. Defaults to all available windows.")
    parser.add_argument('--use_abs_joint_action', action='store_true')
    parser.add_argument('--use_deepspeed', action='store_true')
    args_new = parser.parse_args()
    args = wm_args()

    def merge_args(args, new_args):
        for k, v in new_args.__dict__.items():
            if v is not None:
                args.__dict__[k] = v
        return args
    
    args = merge_args(args, args_new)

    if str(getattr(args, 'ckpt_path', '')).strip() in {'', 'none', 'None', 'null', 'NULL'}:
        args.ckpt_path = None

    if not hasattr(args, 'use_abs_joint_action'):
        args.use_abs_joint_action = False

    main(args)

    # CUDA_VISIBLE_DEVICES=0,1 WANDB_MODE=offline accelerate launch --main_process_port 29501 train_wm.py --dataset_root_path dataset_example --dataset_meta_info_path dataset_meta_info
    # CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 29506 unit_test2.py

    # args = Args()
    # from video_dataset.dataset_droid_exp33 import Dataset_mix
    # dataset = Dataset_mix(args,mode='val')
    # from torch.utils.data import DataLoader
    # dataloader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=2)
    # model = CrtlWorld(args).to('cuda')
    # # print model parameter num
    # num_params = sum(p.numel() for p in model.parameters())
    # print(f"Number of parameters in the model: {num_params/1000000:.2f}M")
    # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-6)
    # total_elements = sum(p.numel() for group in optimizer.param_groups for p in group['params'])
    # print(f"Total number of learnable parameters: {total_elements}")
    # model.train()
    

    # for batch in dataloader:
    #     print(batch['latent'].shape)
    #     print(batch['text'])
    #     print(batch['action'].shape)

    #     loss,_ = model(batch)
    #     loss.backward()
    #     optimizer.step()
    #     optimizer.zero_grad()
    #     print(loss.item())





    # device = 'cuda'
    # video_encoder = VideoEncoder(hidden_size=1024).to(device)
    # # count the parameters of the model
    # num_params = sum(p.numel() for p in video_encoder.parameters())
    # print(f"Number of parameters in the model: {num_params/1000000:.2f}M")
    # vae_latent = torch.randn(8, 1, 4, 32, 32).to(device)
    # clip_latent = torch.randn(8, 20, 512).to(device)
    # current_img = video_encoder(vae_latent, clip_latent)
    # print(current_img.shape)  # (8, 1, 4, 32, 32)


    # pos_emb = get_2d_sincos_pos_embed(1024, 16)
    # print(pos_emb.shape)  # (256, 1024)
    # clip_emb = get_1d_sincos_pos_embed_from_grid(1024, np.arange(20))
    # print(clip_emb.shape)  # (20, 512)
