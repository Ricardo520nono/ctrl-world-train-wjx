#!/usr/bin/env python3
"""Autoregressively replay one clean expert trajectory through Ctrl-World.

The script reads one precomputed ActionFollowing clean expert latent trajectory,
feeds actions to the world model in 32-step chunks, and uses each chunk's final
predicted frame as the next chunk's current frame.
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
from models.ctrl_world import CrtlWorld  # noqa: E402
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline  # noqa: E402


ASSET_ROOT = Path("/mnt/dataset/public_data/cscsx_projects/ctrl_world_train/assets/models")
LATENT_ROOT = Path(
    "/mnt/dataset/public_data/cscsx_projects/ctrl_world_train/latents/"
    "action_following_current1_future32_clean_only_test4v2_fulldesc_rot6d20_retry1_20260707"
)
STAT_PATH = Path(
    "/mnt/dataset/public_data/cscsx_projects/ctrl_world_train/dataset_meta_info/"
    "action_following_current1_future32_clean_only_test4v2_fulldesc_rot6d20_retry1_20260707/stat.json"
)
DEFAULT_CKPT = Path(
    "/mnt/dataset/csx_workspace/Ideas/data_AF3/ctrl-world/outputs/"
    "ACWM_ctrlworld_action_following_clean_test4v2_chunk32_8gpu_current1_future32_fulldesc_rot6d20_retry1_20260707/"
    "checkpoint-step10000-epoch5.39.pt"
)
OUTPUT_ROOT = Path("/tmp/ctrlworld_clean_expert_autoreg")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--latent_root", type=Path, default=LATENT_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--stat", type=Path, default=STAT_PATH)
    parser.add_argument("--task", default=None, help="optional task name; sample_index is applied inside this task")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=8, help="diffusion steps; raise for higher-quality videos")
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--max_actions", type=int, default=0, help="0 means replay the full trajectory")
    parser.add_argument("--chunk_size", type=int, default=32, help="backward-compatible action chunk size")
    parser.add_argument("--num_history", type=int, default=1)
    parser.add_argument("--num_frames", type=int, default=None, help="future frames per model call; defaults to chunk_size")
    parser.add_argument("--action_chunk_size", type=int, default=None, help="action tokens per model call; defaults to chunk_size")
    parser.add_argument("--action_dim", type=int, default=20)
    parser.add_argument("--decode_chunk", type=int, default=7)
    parser.add_argument("--fps", type=int, default=2)
    return parser.parse_args()


def read_jsonl(path: Path):
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def resolve_under(root: Path, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


def make_args(cli, out_dir: Path):
    args = wm_args()
    num_frames = int(cli.num_frames or cli.chunk_size)
    action_chunk_size = int(cli.action_chunk_size or cli.chunk_size)
    args.svd_model_path = str(ASSET_ROOT / "stable-video-diffusion-img2vid")
    args.clip_model_path = str(ASSET_ROOT / "clip-vit-base-patch32")
    args.dataset_type = "action_following"
    args.dataset_root_path = str(cli.latent_root)
    args.dataset_meta_info_path = str(cli.stat.parent.parent)
    args.dataset_cfgs = cli.stat.parent.name
    args.output_dir = str(out_dir)
    args.action_following_latent_root = str(cli.latent_root)
    args.action_following_stat_path = str(cli.stat)
    args.action_following_chunk_size = action_chunk_size
    args.action_following_action_chunk_size = action_chunk_size
    args.num_history = int(cli.num_history)
    args.num_frames = num_frames
    args.action_dim = int(cli.action_dim)
    args.height = 240
    args.width = 320
    args.train_batch_size = 1
    args.shuffle = False
    args.use_ee_head = False
    args.text_cond = True
    args.frame_level_cond = True
    args.his_cond_zero = False
    args.mixed_precision = "bf16"
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
    model.pipeline.set_progress_bar_config(disable=False)
    return model


def load_record(cli):
    manifest = cli.manifest or (cli.latent_root / "manifests" / "clean_train.jsonl")
    records = read_jsonl(manifest)
    if cli.task:
        records = [rec for rec in records if rec.get("task") == cli.task]
    if not records:
        task_msg = f" task={cli.task}" if cli.task else ""
        raise RuntimeError(f"empty manifest selection:{task_msg} manifest={manifest}")
    rec = records[int(cli.sample_index) % len(records)]
    data_path = resolve_under(cli.latent_root, rec["file"])
    data = torch.load(str(data_path), map_location="cpu", weights_only=False)

    action_path = rec.get("action_file")
    if action_path:
        actions = np.load(resolve_under(cli.latent_root, action_path)).astype(np.float32)
    else:
        actions = np.asarray(data["action_pos"], dtype=np.float32)

    text = rec.get("text") or data.get("text") or str(rec.get("task", "")).replace("_", " ")
    latent = data["latent"].float()
    if latent.ndim != 4:
        raise RuntimeError(f"expected latent [T,C,H,W], got {tuple(latent.shape)} from {data_path}")
    if actions.ndim != 2 or actions.shape[1] < cli.action_dim:
        raise RuntimeError(f"expected actions [T,{cli.action_dim}+], got {actions.shape} from {data_path}")

    action_limit = min(actions.shape[0], latent.shape[0])
    total_future = max(0, latent.shape[0] - int(cli.num_history))
    if cli.max_actions > 0:
        total_future = min(total_future, int(cli.max_actions))
    if total_future <= 0:
        raise RuntimeError(f"record has no usable action/latent pairs: {data_path}")

    return rec, data_path, latent[: int(cli.num_history) + total_future], actions[:action_limit, : cli.action_dim], text


def normalize_actions(actions: np.ndarray, stat_path: Path, action_dim: int) -> np.ndarray:
    stat = json.loads(stat_path.read_text())
    p01 = np.asarray(stat["state_01"], dtype=np.float32)[:action_dim]
    p99 = np.asarray(stat["state_99"], dtype=np.float32)[:action_dim]
    return np.clip(2 * (actions - p01) / (p99 - p01 + 1e-8) - 1, -1, 1).astype(np.float32)


def encode_action_condition(model, args, action, text):
    action_latent = model.action_encoder(action, [text], model.tokenizer, model.text_encoder, args.frame_level_cond)
    expected = int(args.num_history + args.num_frames)
    if args.frame_level_cond and action_latent.shape[1] == args.num_frames and args.num_history > 0:
        history_latent = torch.zeros(
            action_latent.shape[0],
            args.num_history,
            action_latent.shape[2],
            device=action_latent.device,
            dtype=action_latent.dtype,
        )
        action_latent = torch.cat([history_latent, action_latent], dim=1)
    if args.frame_level_cond and action_latent.shape[1] != expected:
        raise RuntimeError(f"action condition has {action_latent.shape[1]} frames, expected {expected}")
    return action_latent


def pad_action_chunk(chunk: np.ndarray, action_chunk_size: int) -> np.ndarray:
    valid = int(chunk.shape[0])
    if valid == action_chunk_size:
        return chunk
    if valid <= 0:
        raise RuntimeError("empty action chunk")
    pad = np.repeat(chunk[-1:], action_chunk_size - valid, axis=0)
    return np.concatenate([chunk, pad], axis=0)


@torch.no_grad()
def autoregressive_replay(model, args, actions_norm: np.ndarray, init_latent: torch.Tensor, text: str, cli):
    device = next(model.unet.parameters()).device
    dtype = torch.bfloat16
    num_history = int(args.num_history)
    num_frames = int(args.num_frames)
    action_chunk_size = int(cli.action_chunk_size or cli.chunk_size)
    total_future = int(getattr(cli, "_total_future", actions_norm.shape[0]))
    current = init_latent[None].to(device=device, dtype=dtype)
    if current.ndim != 5 or current.shape[1] != num_history:
        raise RuntimeError(f"expected initial history [H,C,h,w] with H={num_history}, got {tuple(init_latent.shape)}")
    pred_chunks = []

    for chunk_id, start in enumerate(range(0, total_future, num_frames)):
        valid = min(num_frames, total_future - start)
        chunk = pad_action_chunk(actions_norm[start : start + action_chunk_size], action_chunk_size)
        action = torch.tensor(chunk, device=device, dtype=dtype).unsqueeze(0)
        action_latent = encode_action_condition(model, args, action, text)
        generator = torch.Generator(device=device).manual_seed(int(cli.seed) + chunk_id)

        _, future = CtrlWorldDiffusionPipeline.__call__(
            model.pipeline,
            image=current[:, -1],
            text=action_latent,
            width=args.width,
            height=3 * args.height,
            num_frames=num_frames,
            history=current,
            num_inference_steps=cli.steps,
            decode_chunk_size=cli.decode_chunk,
            max_guidance_scale=1.0,
            fps=args.fps,
            motion_bucket_id=args.motion_bucket_id,
            generator=generator,
            output_type="latent",
            return_dict=False,
            frame_level_cond=args.frame_level_cond,
            his_cond_zero=args.his_cond_zero,
        )

        keep = future[:, :valid].to(dtype)
        pred_chunks.append(keep.cpu())
        current = torch.cat([current, keep], dim=1)[:, -num_history:].to(device=device, dtype=dtype)
        print(f"[autoregressive] chunk={chunk_id} action_range=[{start},{start + valid})", flush=True)

    return torch.cat([init_latent[None].cpu(), torch.cat(pred_chunks, dim=1)], dim=1)


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


def decode_three_views(pipeline, latent_seq: torch.Tensor, decode_chunk: int):
    views = einops.rearrange(latent_seq, "b f c (view h) w -> (b view) f c h w", view=3)
    return decode_latents(pipeline, views, chunk=decode_chunk)


def stack_views(video_views: np.ndarray) -> np.ndarray:
    return np.concatenate([video_views[i] for i in range(video_views.shape[0])], axis=1)


def main():
    cli = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this SVD-scale autoregressive replay script.")

    out = cli.out or OUTPUT_ROOT / f"sample{cli.sample_index}_steps{cli.steps}"
    out.mkdir(parents=True, exist_ok=True)
    args = make_args(cli, out)

    rec, data_path, gt_latent, actions, text = load_record(cli)
    cli._total_future = int(gt_latent.shape[0] - int(cli.num_history))
    actions_norm = normalize_actions(actions, cli.stat, cli.action_dim)
    model = load_model(args, cli.ckpt, device)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_latent = autoregressive_replay(model, args, actions_norm, gt_latent[: int(cli.num_history)], text, cli)

    gt_latent = gt_latent[None, : pred_latent.shape[1]].cpu()
    mse = torch.mean((pred_latent[:, 1:].float() - gt_latent[:, 1:].float()) ** 2).item()

    torch.save(
        {
            "pred_latents": pred_latent,
            "gt_latents": gt_latent,
            "actions_norm": torch.tensor(actions_norm),
            "text": text,
            "record": rec,
        },
        out / "autoregressive_replay_latents.pt",
    )

    pred_views = decode_three_views(model.pipeline, pred_latent.to(device=device, dtype=torch.bfloat16), cli.decode_chunk)
    gt_views = decode_three_views(model.pipeline, gt_latent.to(device=device, dtype=torch.bfloat16), cli.decode_chunk)
    pred_video = stack_views(pred_views)
    side_by_side = np.concatenate(
        [np.concatenate([gt_views[i], pred_views[i]], axis=2) for i in range(3)],
        axis=1,
    )
    mediapy.write_video(str(out / "autoregressive_pred.mp4"), pred_video, fps=cli.fps)
    mediapy.write_video(str(out / "gt_left_autoreg_right.mp4"), side_by_side, fps=cli.fps)

    metadata = {
        "ckpt": str(cli.ckpt),
        "latent_root": str(cli.latent_root),
        "manifest": str(cli.manifest or (cli.latent_root / "manifests" / "clean_train.jsonl")),
        "stat": str(cli.stat),
        "sample_index": int(cli.sample_index),
        "task_filter": cli.task,
        "task": rec.get("task", ""),
        "sample_id": rec.get("sample_id", ""),
        "data_path": str(data_path),
        "text": text,
        "action_dim": int(cli.action_dim),
        "chunk_size": int(cli.chunk_size),
        "num_history": int(cli.num_history),
        "num_frames": int(cli.num_frames or cli.chunk_size),
        "action_chunk_size": int(cli.action_chunk_size or cli.chunk_size),
        "num_actions": int(actions.shape[0]),
        "num_video_frames": int(pred_latent.shape[1]),
        "num_inference_steps": int(cli.steps),
        "latent_mse_without_initial_frame": mse,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")

    print(f"pred video: {out / 'autoregressive_pred.mp4'}")
    print(f"compare video: {out / 'gt_left_autoreg_right.mp4'}")
    print(f"latents: {out / 'autoregressive_replay_latents.pt'}")
    print(f"metadata: {out / 'metadata.json'}")
    print(f"latent_mse_without_initial_frame: {mse:.6f}")


if __name__ == "__main__":
    main()
