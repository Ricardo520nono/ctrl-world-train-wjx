#!/usr/bin/env python3
"""Run one Ctrl-World ActionFollowingData checkpoint inference.

Example:
  python scripts/infer_action_following_ckpt.py --preset clean --sample_index 0

The output video places GT on the left and prediction on the right.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import einops
import mediapy
import numpy as np
import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from config import wm_args  # noqa: E402
from dataset.dataset_action_following import ActionFollowingCtrlWorldDataset  # noqa: E402
from models.ctrl_world import CrtlWorld  # noqa: E402
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline  # noqa: E402


ASSET_ROOT = Path("/mnt/dataset/public_data/cscsx_projects/ctrl_world_train/assets/models")
LATENT_ROOT = Path("/mnt/dataset/public_data/cscsx_projects/ctrl_world_train/latents/action_following_chunk32_clean_enhanced_test4v2_20260701")
OUTPUT_ROOT = Path("/tmp/ctrlworld_action_following_infer")

RUNS = {
    "clean": {
        "ckpt": "/mnt/dataset/public_data/cscsx_projects/ctrl_world_train/outputs/ACWM_ctrlworld_action_following_clean_test4v2_chunk32_8gpu_ckpt2500_20260702/checkpoint-step2500-epoch0.21.pt",
        "manifest": "clean_train.jsonl",
        "stat": "/mnt/dataset/public_data/cscsx_projects/ctrl_world_train/dataset_meta_info/action_following_chunk32_clean_only_test4v2_20260701/stat.json",
        "protocol": "clean_only",
    },
    "mix3": {
        "ckpt": "/mnt/dataset/public_data/cscsx_projects/ctrl_world_train/outputs/ACWM_ctrlworld_action_following_mix3_test4v2_chunk32_8gpu_ckpt2500_20260702/checkpoint-step2500-epoch0.21.pt",
        "manifest": "train.jsonl",
        "stat": "/mnt/dataset/public_data/cscsx_projects/ctrl_world_train/dataset_meta_info/action_following_chunk32_clean_enhanced_test4v2_20260701/stat.json",
        "protocol": "mix_3to1to1to1",
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=RUNS.keys(), default="clean")
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--mode", choices=["train", "test_quick"], default="train")
    parser.add_argument("--steps", type=int, default=8, help="diffusion inference steps")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def make_args(preset: str, out_dir: Path, mode: str):
    run = RUNS[preset]
    args = wm_args()
    args.svd_model_path = str(ASSET_ROOT / "stable-video-diffusion-img2vid")
    args.clip_model_path = str(ASSET_ROOT / "clip-vit-base-patch32")
    args.dataset_type = "action_following"
    args.dataset_root_path = str(LATENT_ROOT)
    args.dataset_meta_info_path = str(Path(run["stat"]).parent.parent)
    args.dataset_cfgs = Path(run["stat"]).parent.name
    args.dataset_names = "action_following_test4v2"
    args.output_dir = str(out_dir)
    args.action_following_latent_root = str(LATENT_ROOT)
    args.action_following_train_manifest = str(LATENT_ROOT / "manifests" / run["manifest"])
    args.action_following_val_manifest = str(LATENT_ROOT / "manifests" / "test_quick.jsonl")
    args.action_following_stat_path = run["stat"]
    args.action_following_sampling_protocol = run["protocol"]
    args.action_following_chunk_size = 32
    args.action_following_sampling_seed = 20260703
    args.num_history = 6
    args.num_frames = 26
    args.action_dim = 14
    args.height = 240
    args.width = 320
    args.train_batch_size = 1
    args.shuffle = False
    args.use_ee_head = False
    args.mode = mode
    return args


def load_model(args, ckpt: Path, device: torch.device):
    model = CrtlWorld(args)
    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.unet.to(dtype=torch.bfloat16)
    model.vae.to(dtype=torch.bfloat16)
    model.image_encoder.to(dtype=torch.bfloat16)
    model.text_encoder.to(dtype=torch.bfloat16)
    model.action_encoder.to(dtype=torch.bfloat16)
    model.eval()
    return model


def decode_latents(pipeline, latents, chunk=7):
    bsz, frames = latents.shape[:2]
    flat = latents.flatten(0, 1).to(pipeline.vae.device)
    decoded = []
    for start in range(0, flat.shape[0], chunk):
        z = flat[start : start + chunk] / pipeline.vae.config.scaling_factor
        decoded.append(pipeline.vae.decode(z.to(pipeline.vae.dtype), num_frames=z.shape[0]).sample)
    video = torch.cat(decoded).reshape(bsz, frames, *decoded[0].shape[1:])
    video = ((video / 2 + 0.5).clamp(0, 1) * 255)
    return video.float().cpu().numpy().transpose(0, 1, 3, 4, 2).astype(np.uint8)


def main():
    cli = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = RUNS[cli.preset]
    ckpt = cli.ckpt or Path(run["ckpt"])
    out = cli.out or OUTPUT_ROOT / f"{cli.preset}_{cli.mode}_sample{cli.sample_index}_steps{cli.steps}"
    out.mkdir(parents=True, exist_ok=True)

    args = make_args(cli.preset, out, cli.mode)
    model = load_model(args, ckpt, device)
    dataset = ActionFollowingCtrlWorldDataset(args, mode=cli.mode)
    sample = dataset[cli.sample_index]

    latent = sample["latent"].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    action = sample["action"].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    text = [sample["text"]]

    history = latent[:, : args.num_history]
    future_gt = latent[:, args.num_history :]
    current = future_gt[:, 0]

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        action_latent = model.action_encoder(action, text, model.tokenizer, model.text_encoder, args.frame_level_cond)
        _, future_pred = CtrlWorldDiffusionPipeline.__call__(
            model.pipeline,
            image=current,
            text=action_latent,
            width=args.width,
            height=3 * args.height,
            num_frames=args.num_frames,
            history=history,
            num_inference_steps=cli.steps,
            decode_chunk_size=7,
            max_guidance_scale=1.0,
            fps=args.fps,
            motion_bucket_id=args.motion_bucket_id,
            generator=torch.Generator(device=device).manual_seed(20260703),
            output_type="latent",
            return_dict=False,
            frame_level_cond=args.frame_level_cond,
            his_cond_zero=args.his_cond_zero,
        )

    pred = torch.cat([history, future_pred.to(history.dtype)], dim=1)
    mse = torch.mean((future_pred.float() - future_gt.float()) ** 2).item()

    torch.save({"pred_latents": pred.cpu(), "gt_latents": latent.cpu(), "action": action.cpu(), "text": text}, out / "prediction_latents.pt")
    (out / "metadata.json").write_text(json.dumps({
        "preset": cli.preset,
        "ckpt": str(ckpt),
        "mode": cli.mode,
        "sample_index": cli.sample_index,
        "task": sample.get("task", ""),
        "family": sample.get("family", ""),
        "sample_id": sample.get("sample_id", ""),
        "chunk_start": int(sample.get("chunk_start", 0)),
        "text": text[0],
        "latent_mse_future": mse,
    }, indent=2) + "\n")

    gt_views = einops.rearrange(latent, "b f c (view h) w -> (b view) f c h w", view=3)
    pred_views = einops.rearrange(pred, "b f c (view h) w -> (b view) f c h w", view=3)
    gt_video = decode_latents(model.pipeline, gt_views)
    pred_video = decode_latents(model.pipeline, pred_views)
    rows = [np.concatenate([gt_video[i], pred_video[i]], axis=2) for i in range(3)]
    mediapy.write_video(str(out / "gt_left_pred_right.mp4"), np.concatenate(rows, axis=1), fps=2)

    print(f"video: {out / 'gt_left_pred_right.mp4'}")
    print(f"latents: {out / 'prediction_latents.pt'}")
    print(f"metadata: {out / 'metadata.json'}")
    print(f"latent_mse_future: {mse:.6f}")


if __name__ == "__main__":
    main()
