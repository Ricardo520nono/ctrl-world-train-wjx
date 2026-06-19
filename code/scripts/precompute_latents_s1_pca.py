"""
Pre-encode RGB frames from EnhancedData (perturbed_pca_gaussian) into SVD-VAE latent tensors.
Filters episodes by source episode index range (e.g. 0-39 for train, 40-49 for test).

Directory structure:
  {data_root}/{task}/{episode_N_start_S_seed_K}/data.hdf5

Usage:
    python precompute_latents_s1_pca.py \
        --data_root /mnt/public_ckp/.../c_8_sigma_0p05 \
        --out_root  /mnt/public_ckp/cscsx_projects/ctrl_world_train/latents/s1B_latents_pca_train_headwrist \
        --svd_path  /mnt/public_ckp/cscsx_projects/ctrl_world_train/assets/models/stable-video-diffusion-img2vid \
        --tasks     click_alarmclock click_bell place_object_basket open_laptop stack_blocks_two \
        --episode_min 0 --episode_max 39
"""

import argparse
import io
import json
import os
import re
import sys

import h5py
import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.ee_targets import ee_target_from_hdf5

# Default camera set for current mainline headwrist training. Override with --cameras.
# Must be exactly 3 cameras: they are stacked vertically into (T, 4, 90, 40).
CAMERA_KEYS = ["head_camera", "left_camera", "right_camera"]
IMG_SIZE = (320, 240)


def episode_sort_key(d):
    m = re.match(r'episode_(\d+)_start_(\d+)_seed_(\d+)', d)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return (999999, 999999, 999999)


def load_vae(svd_path, device):
    from diffusers import AutoencoderKLTemporalDecoder
    vae = AutoencoderKLTemporalDecoder.from_pretrained(svd_path, subfolder="vae")
    return vae.to(device).eval()


def pil_to_tensor(img):
    img = img.resize(IMG_SIZE, Image.BILINEAR).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr.transpose(2, 0, 1))


def encode_frames(vae, frames_rgb, device, batch_size=16):
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


def process_task(task_name, task_src_dir, task_out_dir, vae, device, episode_min, episode_max):
    os.makedirs(task_out_dir, exist_ok=True)

    all_dirs = [d for d in os.listdir(task_src_dir) if not d.startswith('.')]
    all_dirs = sorted(all_dirs, key=episode_sort_key)

    # filter by episode index
    filtered = []
    for d in all_dirs:
        m = re.match(r'episode_(\d+)_', d)
        if m:
            ep_idx = int(m.group(1))
            if episode_min <= ep_idx <= episode_max:
                filtered.append((ep_idx, d))

    print(f"[{task_name}] {len(filtered)} dirs after episode filter [{episode_min},{episode_max}] "
          f"(from {len(all_dirs)} total)")

    meta = []
    for i, (ep_idx, ep_dir) in enumerate(filtered):
        hdf5_path = os.path.join(task_src_dir, ep_dir, "data.hdf5")
        if not os.path.exists(hdf5_path):
            print(f"  [{task_name}] WARNING: {hdf5_path} not found, skipping.")
            continue

        print(f"  [{task_name}] {i+1}/{len(filtered)}: {ep_dir}", flush=True)

        with h5py.File(hdf5_path, "r") as f:
            l_pose = np.array(f["delta_ee_action/left_delta_pose"],  dtype=np.float32)
            l_grip = np.array(f["delta_ee_action/left_gripper"],     dtype=np.float32)[:, None]
            r_pose = np.array(f["delta_ee_action/right_delta_pose"], dtype=np.float32)
            r_grip = np.array(f["delta_ee_action/right_gripper"],    dtype=np.float32)[:, None]
            action_pos = np.concatenate([l_pose, l_grip, r_pose, r_grip], axis=1)  # (T, 14)
            ee_target = ee_target_from_hdf5(f)  # (T, 20): left/right xyz + rot6d + gripper
            T = action_pos.shape[0]
            if ee_target.shape[0] != T:
                raise RuntimeError(f"EE target length mismatch: action={T}, ee_target={ee_target.shape[0]}")

            cam_latents = {}
            for cam_key in CAMERA_KEYS:
                rgb_bytes = f[f"observation/{cam_key}/rgb"][:]
                frames = [Image.open(io.BytesIO(b)) for b in rgb_bytes]
                cam_latents[cam_key] = encode_frames(vae, frames, device)

        latent = torch.cat([cam_latents[k] for k in CAMERA_KEYS], dim=2)  # (T, 4, 90, 40)

        out_file = os.path.join(task_out_dir, f"ep{ep_idx:04d}_{ep_dir}.pt")
        torch.save({"latent": latent, "action_pos": action_pos, "ee_target": ee_target}, out_file)
        meta.append({"episode": ep_idx, "file": out_file, "T": T, "ep_dir": ep_dir})

    with open(os.path.join(task_out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[{task_name}] Done. {len(meta)} episodes saved.")


def main():
    global CAMERA_KEYS
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",    type=str, required=True)
    parser.add_argument("--out_root",     type=str, required=True)
    parser.add_argument("--svd_path",     type=str, required=True)
    parser.add_argument("--tasks",        type=str, nargs="+", required=True)
    parser.add_argument("--episode_min",  type=int, default=0)
    parser.add_argument("--episode_max",  type=int, default=39)
    parser.add_argument("--task_name",    type=str, default=None,
                        help="Process a single task (for parallelism).")
    parser.add_argument("--cameras",      type=str, default=",".join(CAMERA_KEYS),
                        help="Comma-separated camera keys to stack vertically (must be exactly 3).")
    args = parser.parse_args()

    cams = [c.strip() for c in args.cameras.split(",") if c.strip()]
    assert len(cams) == 3, f"--cameras must list exactly 3 cameras, got {cams}"
    CAMERA_KEYS = cams
    print(f"[INFO] Using cameras (top->bottom): {CAMERA_KEYS}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading VAE on {device}...")
    vae = load_vae(args.svd_path, device)

    tasks = [args.task_name] if args.task_name else args.tasks
    for task in tasks:
        task_out = os.path.join(args.out_root, task)
        if os.path.exists(os.path.join(task_out, "meta.json")):
            print(f"[INFO] Latents already exist for {task}, skipping.")
            continue
        process_task(task, os.path.join(args.data_root, task),
                     task_out, vae, device, args.episode_min, args.episode_max)

    print("[INFO] All tasks done.")


if __name__ == "__main__":
    main()
