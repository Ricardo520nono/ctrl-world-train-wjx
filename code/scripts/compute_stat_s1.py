"""
Compute action normalization stat (p01/p99) for S1 experiments.

Two modes:
  --group A: only expert latents, episodes filtered by episode_split
  --group B: expert latents + pca off-expert latents, same episode filter

Usage:
    # A group
    python compute_stat_s1.py \
        --group A \
        --expert_root /mnt/public_ckp/cscsx_projects/ctrl_world_train/latents/precomputed_latents_s1A_5tasks_14d_headwrist \
        --tasks click_alarmclock click_bell place_object_basket open_laptop stack_blocks_two \
        --episode_split 0-39 \
        --out_dir /mnt/public_ckp/cscsx_projects/ctrl_world_train/code/dataset_meta_info/s1_A_expert_only

    # B group
    python compute_stat_s1.py \
        --group B \
        --expert_root /mnt/public_ckp/cscsx_projects/ctrl_world_train/latents/precomputed_latents_s1B_5tasks_14d_headwrist \
        --pca_root    /mnt/public_ckp/cscsx_projects/ctrl_world_train/latents/s1B_latents_pca_train_headwrist \
        --tasks click_alarmclock click_bell place_object_basket open_laptop stack_blocks_two \
        --episode_split 0-39 \
        --out_dir /mnt/public_ckp/cscsx_projects/ctrl_world_train/code/dataset_meta_info/s1_B_expert_plus_pca
"""

import argparse
import json
import os

import numpy as np
import torch


def parse_episode_split(split_str):
    if split_str is None:
        return None
    lo, hi = split_str.split("-")
    return set(range(int(lo), int(hi) + 1))


def collect_actions(latent_root, tasks, action_dim, allowed_episodes=None):
    all_actions = []
    for task in tasks:
        meta_path = os.path.join(latent_root, task, "meta.json")
        if not os.path.exists(meta_path):
            print(f"  [WARN] no meta.json for {task} in {latent_root}, skipping")
            continue
        meta = json.load(open(meta_path))
        count = 0
        for item in meta:
            if allowed_episodes is not None:
                ep_idx = item.get("episode", None)
                if ep_idx is None or ep_idx not in allowed_episodes:
                    continue
            data = torch.load(item["file"], map_location="cpu", weights_only=False)
            action = np.array(data["action_pos"], dtype=np.float32)[:, :action_dim]
            all_actions.append(action)
            count += 1
        print(f"  {task}: {count} episodes loaded from {latent_root}")
    return all_actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group",         type=str, required=True, choices=["A", "B"])
    parser.add_argument("--expert_root",   type=str, required=True)
    parser.add_argument("--pca_root",      type=str, default=None)
    parser.add_argument("--tasks",         type=str, nargs="+", required=True)
    parser.add_argument("--episode_split", type=str, default="0-39")
    parser.add_argument("--out_dir",       type=str, required=True)
    parser.add_argument("--action_dim",    type=int, default=14)
    args = parser.parse_args()

    allowed_episodes = parse_episode_split(args.episode_split)

    print(f"[INFO] Group {args.group}, episode_split={args.episode_split}")
    print("[INFO] Collecting expert actions...")
    all_actions = collect_actions(args.expert_root, args.tasks, args.action_dim, allowed_episodes)

    if args.group == "B":
        assert args.pca_root is not None, "--pca_root required for group B"
        print("[INFO] Collecting pca off-expert actions...")
        # pca latents are already filtered to train episodes, no need to filter again
        all_actions += collect_actions(args.pca_root, args.tasks, args.action_dim, allowed_episodes=None)

    all_actions = np.concatenate(all_actions, axis=0)
    print(f"[INFO] Total frames: {len(all_actions)}")

    p01 = np.percentile(all_actions, 1,  axis=0).tolist()
    p99 = np.percentile(all_actions, 99, axis=0).tolist()

    print("[INFO] p01:", [round(v, 6) for v in p01])
    print("[INFO] p99:", [round(v, 6) for v in p99])

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "stat.json")
    with open(out_path, "w") as f:
        json.dump({"state_01": p01, "state_99": p99}, f, indent=2)
    print(f"[INFO] Saved to {out_path}")


if __name__ == "__main__":
    main()
