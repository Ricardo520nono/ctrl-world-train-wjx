"""
Compute p01/p99 action stats from multiple precomputed latent roots.

Each root should contain {task}/meta.json entries pointing to .pt files with
action_pos. Episode split is applied when meta entries carry an "episode" key.
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


def collect_actions(latent_root, tasks, action_dim, allowed_episodes):
    all_actions = []
    for task in tasks:
        meta_path = os.path.join(latent_root, task, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"meta.json not found: {meta_path}")
        meta = json.load(open(meta_path))
        count = 0
        frames = 0
        for item in meta:
            if allowed_episodes is not None:
                ep_idx = item.get("episode", None)
                if ep_idx is None or ep_idx not in allowed_episodes:
                    continue
            data = torch.load(item["file"], map_location="cpu", weights_only=False)
            action = np.array(data["action_pos"], dtype=np.float32)[:, :action_dim]
            all_actions.append(action)
            count += 1
            frames += len(action)
        print(f"  {latent_root} :: {task}: {count} trajectories, {frames} frames")
    return all_actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_roots", type=str, nargs="+", required=True)
    parser.add_argument("--tasks", type=str, nargs="+", required=True)
    parser.add_argument("--episode_split", type=str, default="0-39")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--action_dim", type=int, default=14)
    args = parser.parse_args()

    allowed_episodes = parse_episode_split(args.episode_split)
    all_actions = []
    for root in args.latent_roots:
        all_actions.extend(collect_actions(root, args.tasks, args.action_dim, allowed_episodes))

    all_actions = np.concatenate(all_actions, axis=0)
    print(f"[INFO] total frames: {len(all_actions)}")

    p01 = np.percentile(all_actions, 1, axis=0).tolist()
    p99 = np.percentile(all_actions, 99, axis=0).tolist()
    print("[INFO] p01:", [round(v, 6) for v in p01])
    print("[INFO] p99:", [round(v, 6) for v in p99])

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "stat.json")
    with open(out_path, "w") as f:
        json.dump({"state_01": p01, "state_99": p99}, f, indent=2)
    print(f"[INFO] saved: {out_path}")


if __name__ == "__main__":
    main()
