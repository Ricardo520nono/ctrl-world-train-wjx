"""
Pre-encode RGB frames from EnhancedData (perturbed_pca_gaussian) into SVD-VAE latent tensors.

Directory structure:
  {data_root}/{task}/{episode_N_start_S_seed_K}/data.hdf5

Takes the first --max_episodes episodes per task (sorted by episode/start/seed numerically).

Usage:
    python precompute_latents_enhanced.py \
        --data_root /mnt/public_ckp/.../c_8_sigma_0p05 \
        --out_root  /mnt/gyc_ckp/wjx/ctrlworld/precomputed_latents_enhanced_14d \
        --svd_path  /mnt/gyc/Ctrl-World/assets/models/stable-video-diffusion-img2vid \
        --max_episodes 100
"""

import argparse
import io
import json
import os
import re

import h5py
import numpy as np
import torch
from PIL import Image

CAMERA_KEYS = ["front_camera", "head_camera", "left_camera"]
IMG_SIZE = (320, 240)  # width x height -> VAE latent: 40 x 30


def sort_key(d):
    m = re.match(r'episode_(\d+)_start_(\d+)_seed_(\d+)', d)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return (999999, 999999, 999999)


def load_vae(svd_path: str, device: str):
    from diffusers import AutoencoderKLTemporalDecoder
    vae = AutoencoderKLTemporalDecoder.from_pretrained(svd_path, subfolder="vae")
    return vae.to(device).eval()


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    img = img.resize(IMG_SIZE, Image.BILINEAR).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr.transpose(2, 0, 1))


def encode_frames(vae, frames_rgb: list, device: str, batch_size: int = 16) -> torch.Tensor:
    results = []
    for i in range(0, len(frames_rgb), batch_size):
        batch = frames_rgb[i: i + batch_size]
        tensors = torch.stack([pil_to_tensor(img) for img in batch]).to(device)
        with torch.no_grad():
            z = vae.encode(tensors).latent_dist.sample() * vae.config.scaling_factor
        results.append(z.cpu())
        del tensors, z
        torch.cuda.empty_cache()
    return torch.cat(results, dim=0)


def process_task(task_name: str, task_src_dir: str, task_out_dir: str,
                 vae, device: str, max_episodes: int):
    os.makedirs(task_out_dir, exist_ok=True)

    all_dirs = [d for d in os.listdir(task_src_dir) if not d.startswith('.')]
    all_dirs = sorted(all_dirs, key=sort_key)
    ep_dirs = all_dirs[:max_episodes]
    print(f"[{task_name}] {len(ep_dirs)} episodes to encode (from {len(all_dirs)} total)")

    meta = []
    for ep_idx, ep_dir in enumerate(ep_dirs):
        hdf5_path = os.path.join(task_src_dir, ep_dir, "data.hdf5")
        if not os.path.exists(hdf5_path):
            print(f"  [{task_name}] WARNING: {hdf5_path} not found, skipping.")
            continue

        print(f"  [{task_name}] Episode {ep_idx+1}/{len(ep_dirs)}: {ep_dir}", flush=True)

        with h5py.File(hdf5_path, "r") as f:
            l_pose = np.array(f["delta_ee_action/left_delta_pose"],  dtype=np.float32)  # (T, 6)
            l_grip = np.array(f["delta_ee_action/left_gripper"],     dtype=np.float32)[:, None]  # (T, 1)
            r_pose = np.array(f["delta_ee_action/right_delta_pose"], dtype=np.float32)  # (T, 6)
            r_grip = np.array(f["delta_ee_action/right_gripper"],    dtype=np.float32)[:, None]  # (T, 1)
            action_pos = np.concatenate([l_pose, l_grip, r_pose, r_grip], axis=1)  # (T, 14)
            T = action_pos.shape[0]

            cam_latents = {}
            for cam_key in CAMERA_KEYS:
                rgb_bytes = f[f"observation/{cam_key}/rgb"][:]
                frames = [Image.open(io.BytesIO(b)) for b in rgb_bytes]
                cam_latents[cam_key] = encode_frames(vae, frames, device)  # (T, 4, 30, 40)

        # stack 3 cameras vertically: (T, 4, 90, 40)
        latent = torch.cat([cam_latents[k] for k in CAMERA_KEYS], dim=2)

        out_file = os.path.join(task_out_dir, f"ep_{ep_idx:04d}_{ep_dir}.pt")
        torch.save({"latent": latent, "action_pos": action_pos}, out_file)
        meta.append({"file": out_file, "T": T, "ep_dir": ep_dir})

    with open(os.path.join(task_out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[{task_name}] Done. {len(meta)} episodes saved.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--out_root",  type=str, required=True)
    parser.add_argument("--svd_path",  type=str, required=True)
    parser.add_argument("--max_episodes", type=int, default=100)
    parser.add_argument("--task_name", type=str, default=None,
                        help="Process a single task only (for parallelism). If not set, process all.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading VAE on {device}...")
    vae = load_vae(args.svd_path, device)

    all_tasks = sorted([t for t in os.listdir(args.data_root)
                        if os.path.isdir(os.path.join(args.data_root, t)) and t != "meta"])
    if args.task_name is not None:
        all_tasks = [args.task_name]

    for task in all_tasks:
        task_src = os.path.join(args.data_root, task)
        task_out = os.path.join(args.out_root, task)
        if os.path.exists(os.path.join(task_out, "meta.json")):
            print(f"[INFO] Latents already exist for {task}, skipping.")
            continue
        process_task(task, task_src, task_out, vae, device, args.max_episodes)

    print("[INFO] All tasks done.")


if __name__ == "__main__":
    main()
