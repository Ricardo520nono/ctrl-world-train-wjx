"""
Compute action normalization stat (p01/p99) from two data sources combined:
  1. Original expert latents (precomputed, action stored in .pt files)
  2. Enhanced perturbed latents (precomputed, action stored in .pt files)

Saves stat.json to --out_dir compatible with existing DeltaEEDataset format.

Usage:
    python compute_stat_enhanced.py \
        --latent_root_orig /mnt/gyc_ckp/wjx/ctrlworld/precomputed_latents_delta_ee_all50_14d \
        --latent_root_enhanced /mnt/gyc_ckp/wjx/ctrlworld/precomputed_latents_enhanced_14d \
        --out_dir /mnt/gyc/Ctrl-World/dataset_meta_info/delta_ee_all50_enhanced_14d \
        --action_dim 14
"""

import argparse
import json
import os

import numpy as np
import torch


def collect_actions_from_root(latent_root: str, action_dim: int) -> np.ndarray:
    all_actions = []
    tasks = sorted([t for t in os.listdir(latent_root)
                    if os.path.isdir(os.path.join(latent_root, t))])
    for task in tasks:
        meta_path = os.path.join(latent_root, task, "meta.json")
        if not os.path.exists(meta_path):
            print(f"  [WARN] no meta.json in {task}, skipping")
            continue
        meta = json.load(open(meta_path))
        for item in meta:
            data = torch.load(item["file"], map_location="cpu", weights_only=False)
            action = np.array(data["action_pos"], dtype=np.float32)[:, :action_dim]
            all_actions.append(action)
        print(f"  {task}: {len(meta)} episodes loaded")
    return np.concatenate(all_actions, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_root_orig",     type=str, required=True)
    parser.add_argument("--latent_root_enhanced", type=str, required=True)
    parser.add_argument("--out_dir",  type=str, required=True)
    parser.add_argument("--action_dim", type=int, default=14)
    args = parser.parse_args()

    print("[INFO] Collecting actions from original latents...")
    actions_orig = collect_actions_from_root(args.latent_root_orig, args.action_dim)
    print(f"  orig total frames: {len(actions_orig)}")

    print("[INFO] Collecting actions from enhanced latents...")
    actions_enh = collect_actions_from_root(args.latent_root_enhanced, args.action_dim)
    print(f"  enhanced total frames: {len(actions_enh)}")

    all_actions = np.concatenate([actions_orig, actions_enh], axis=0)
    print(f"[INFO] Combined total frames: {len(all_actions)}")

    p01 = np.percentile(all_actions, 1, axis=0).tolist()
    p99 = np.percentile(all_actions, 99, axis=0).tolist()

    print("[INFO] p01:", [round(v, 6) for v in p01])
    print("[INFO] p99:", [round(v, 6) for v in p99])

    os.makedirs(args.out_dir, exist_ok=True)
    stat = {"state_01": p01, "state_99": p99}
    out_path = os.path.join(args.out_dir, "stat.json")
    with open(out_path, "w") as f:
        json.dump(stat, f, indent=2)
    print(f"[INFO] Saved stat to {out_path}")


if __name__ == "__main__":
    main()
