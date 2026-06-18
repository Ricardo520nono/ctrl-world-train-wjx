"""
Pre-encode RGB frames from ActionFollowingBench delta-ee HDF5 into SVD-VAE latent tensors.
Reads HDF5 directly (no zip). Action from delta_ee_action/left_delta_pose (already aligned).

Usage:
    python precompute_latents_delta_ee.py \
        --data_dir /path/to/task/data \
        --out_dir  /path/to/output_latents \
        --svd_path /path/to/stable-video-diffusion-img2vid \
        --task_name place_container_plate
"""

import argparse
import io
import json
import os

import h5py
import numpy as np
import torch
from PIL import Image

# Default camera set for current mainline headwrist training. Override with --cameras.
# Must be exactly 3 cameras: they are stacked vertically into (T, 4, 90, 40).
CAMERA_KEYS = ["head_camera", "left_camera", "right_camera"]
IMG_SIZE = (320, 240)  # width x height -> VAE latent: 40 x 30


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


def process_data_dir(data_dir: str, out_dir: str, svd_path: str, task_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{task_name}] Loading VAE on {device}...")
    vae = load_vae(svd_path, device)

    os.makedirs(out_dir, exist_ok=True)
    meta = []

    hdf5_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".hdf5")])
    for ep_idx, fname in enumerate(hdf5_files):
        print(f"  [{task_name}] Episode {ep_idx+1}/{len(hdf5_files)} ...", flush=True)

        with h5py.File(os.path.join(data_dir, fname), "r") as f:
            # action: left (delta_pose 6D + gripper 1D) + right (delta_pose 6D + gripper 1D) = 14D
            l_pose = np.array(f["delta_ee_action/left_delta_pose"],  dtype=np.float32)  # (T, 6)
            l_grip = np.array(f["delta_ee_action/left_gripper"],     dtype=np.float32)[:, None]  # (T, 1)
            r_pose = np.array(f["delta_ee_action/right_delta_pose"], dtype=np.float32)  # (T, 6)
            r_grip = np.array(f["delta_ee_action/right_gripper"],    dtype=np.float32)[:, None]  # (T, 1)
            action_pos = np.concatenate([l_pose, l_grip, r_pose, r_grip], axis=1)  # (T, 14)
            T = action_pos.shape[0]

            # encode 3 cameras
            cam_latents = {}
            for cam_key in CAMERA_KEYS:
                rgb_bytes = f[f"observation/{cam_key}/rgb"][:]
                frames = [Image.open(io.BytesIO(b)) for b in rgb_bytes]
                cam_latents[cam_key] = encode_frames(vae, frames, device)  # (T, 4, 30, 40)

        # stack 3 cameras vertically: (T, 4, 90, 40)
        stacked = torch.zeros(T, 4, 90, 40)
        for i, cam_key in enumerate(CAMERA_KEYS):
            stacked[:, :, i*30:(i+1)*30, :] = cam_latents[cam_key][:, :, :30, :40]

        ep_out = os.path.join(out_dir, f"episode_{ep_idx:04d}.pt")
        torch.save({"latent": stacked, "action_pos": action_pos, "task": task_name}, ep_out)
        meta.append({"episode": ep_idx, "file": ep_out, "T": T})

    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=2)
    print(f"[{task_name}] Done. {len(meta)} episodes saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",  required=True)
    parser.add_argument("--out_dir",   required=True)
    parser.add_argument("--svd_path",  required=True)
    parser.add_argument("--task_name", required=True)
    parser.add_argument("--cameras", type=str, default=",".join(CAMERA_KEYS),
                        help="Comma-separated camera keys to stack vertically (must be exactly 3).")
    args = parser.parse_args()

    cams = [c.strip() for c in args.cameras.split(",") if c.strip()]
    assert len(cams) == 3, f"--cameras must list exactly 3 cameras, got {cams}"
    CAMERA_KEYS = cams
    print(f"[{args.task_name}] Using cameras (top->bottom): {CAMERA_KEYS}")

    process_data_dir(args.data_dir, args.out_dir, args.svd_path, args.task_name)
