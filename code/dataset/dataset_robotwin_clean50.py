"""
Dataset for RobotWin2.0 clean50 (pcp + pob) using pre-computed SVD latents.
Reads:
  - latent tensor (.pt)     from precomputed latent dir
  - action (abs joint 6D)   from same .pt (action_pos field)
"""
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset


class RobotwinClean50Dataset(Dataset):
    def __init__(self, args, mode: str = "train"):
        self.args = args
        self.mode = mode
        self.num_history = args.num_history
        self.num_frames = args.num_frames
        self.T = self.num_history + self.num_frames
        self.action_dim = int(getattr(args, "action_dim", 6))

        # normalisation stats
        stat_path = os.path.join(args.dataset_meta_info_path, args.dataset_cfgs.split("+")[0], "stat.json")
        with open(stat_path) as f:
            stat = json.load(f)
        self.p01 = np.array(stat["state_01"], dtype=np.float32)[: self.action_dim]
        self.p99 = np.array(stat["state_99"], dtype=np.float32)[: self.action_dim]

        # collect episodes from all task latent dirs
        latent_root = args.dataset_root_path  # e.g. /mnt/gyc_wjx/Ctrl-World/model_ckpt/precomputed_latents
        task_names = args.dataset_names.split("+")
        self.episodes = []
        for task in task_names:
            task_dir = os.path.join(latent_root, task)
            meta_file = os.path.join(task_dir, "meta.json")
            if not os.path.exists(meta_file):
                raise FileNotFoundError(f"meta.json not found in {task_dir}. Run precompute_latents_clean50.py first.")
            meta = json.load(open(meta_file))
            # use all episodes for training, no val split
            for item in meta:
                self.episodes.append({"file": item["file"], "T": item["T"], "task": task})

        if len(self.episodes) == 0:
            raise RuntimeError("No episodes found. Check dataset_root_path and dataset_names.")
        print(f"[RobotwinClean50Dataset] mode={mode}, {len(self.episodes)} episodes, tasks={task_names}")

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        ep = self.episodes[idx % len(self.episodes)]
        data = torch.load(ep["file"], map_location="cpu", weights_only=False)

        latent_full = data["latent"]       # (T_full, 4, 72, 40)
        action_pos  = data["action_pos"]   # (T_full, 6)

        # guard: use min length in case latent and action_pos are mismatched
        T_full = min(latent_full.shape[0], action_pos.shape[0])
        if T_full == 0:
            return self.__getitem__((idx + 1) % len(self.episodes))
        latent_full = latent_full[:T_full]
        action_pos  = action_pos[:T_full]

        if T_full < self.T:
            # pad if episode too short
            pad = self.T - T_full
            latent_full = torch.cat([latent_full, latent_full[-1:].repeat(pad, 1, 1, 1)], dim=0)
            action_pos  = np.concatenate([action_pos, action_pos[-1:].repeat(pad, axis=0)], axis=0)
            T_full = self.T

        # random start
        max_start = T_full - self.T
        start = random.randint(0, max_start) if max_start > 0 else 0
        latent = latent_full[start: start + self.T]   # (T, 4, 72, 40)
        action = action_pos[start: start + self.T, : self.action_dim]

        # normalize action
        action = np.clip(2 * (action - self.p01) / (self.p99 - self.p01 + 1e-8) - 1, -1, 1)

        task_text = ep["task"].replace("_", " ")
        return {
            "latent": latent.float(),
            "text": task_text,
            "action": torch.tensor(action, dtype=torch.float32),
        }
