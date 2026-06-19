"""
Dataset for ActionFollowingBench delta-ee using pre-computed SVD latents.
Latent shape: (T, 4, 90, 40)  — 3 cameras × 30 height × 40 width
Action: delta_ee 14D (left 6D pose + 1D gripper + right 6D pose + 1D gripper)

dataset_root_path supports multiple roots separated by '+', e.g.:
    /path/to/orig+/path/to/enhanced

dataset_root_sampling: per-root sampling strategy, separated by '+', e.g.:
    sliding+single
    - sliding: expand each episode into (T_full - T + 1) windows (default)
    - single:  one sample per episode (first window only)

episode_split: if set, only load episodes whose index falls in the given range.
    Format: "0-39" for train, "40-49" for test.
    If not set, load all episodes (original behavior).
"""
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


def _parse_episode_split(split_str):
    """Parse '0-39' -> set(range(0, 40))"""
    if split_str is None:
        return None
    lo, hi = split_str.split("-")
    return set(range(int(lo), int(hi) + 1))


class DeltaEEDataset(Dataset):
    def __init__(self, args, mode: str = "train"):
        self.args = args
        self.mode = mode
        self.num_history = args.num_history
        self.num_frames = args.num_frames
        self.T = self.num_history + self.num_frames
        self.action_dim = int(getattr(args, "action_dim", 6))
        self.use_ee_head = args.use_ee_head

        stat_path = os.path.join(args.dataset_meta_info_path, args.dataset_cfgs.split("+")[0], "stat.json")
        with open(stat_path) as f:
            stat = json.load(f)
        self.p01 = np.array(stat["state_01"], dtype=np.float32)[: self.action_dim]
        self.p99 = np.array(stat["state_99"], dtype=np.float32)[: self.action_dim]

        # episode split filter
        episode_split = getattr(args, "episode_split", None)
        allowed_episodes = _parse_episode_split(episode_split)

        # per-root sampling strategy: "sliding" or "single"
        latent_roots = args.dataset_root_path.split("+")
        sampling_str = getattr(args, "dataset_root_sampling", None)
        if sampling_str:
            sampling_modes = sampling_str.split("+")
        else:
            sampling_modes = ["sliding"] * len(latent_roots)
        # pad if fewer sampling modes than roots
        while len(sampling_modes) < len(latent_roots):
            sampling_modes.append("sliding")

        task_names = args.dataset_names.split("+")

        self.windows = []
        num_episodes = 0
        for latent_root, sampling in zip(latent_roots, sampling_modes):
            for task in task_names:
                task_dir = os.path.join(latent_root, task)
                meta_file = os.path.join(task_dir, "meta.json")
                if not os.path.exists(meta_file):
                    raise FileNotFoundError(f"meta.json not found in {task_dir}. Run precompute script first.")
                for item in json.load(open(meta_file)):
                    if allowed_episodes is not None:
                        ep_idx = item.get("episode", None)
                        if ep_idx is None or ep_idx not in allowed_episodes:
                            continue
                    T_full = item["T"]
                    if sampling == "single":
                        # only first window
                        self.windows.append((item["file"], 0, T_full, task))
                    else:
                        # sliding window
                        num_windows = max(1, T_full - self.T + 1)
                        for start in range(num_windows):
                            self.windows.append((item["file"], start, T_full, task))
                    num_episodes += 1

        if len(self.windows) == 0:
            raise RuntimeError("No windows found.")
        print(f"[DeltaEEDataset] mode={mode}, split={episode_split}, sampling={sampling_modes}, "
              f"{num_episodes} episodes, {len(self.windows)} windows, "
              f"tasks={task_names}, roots={latent_roots}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        file, start, T_full, task = self.windows[idx]
        data = torch.load(file, map_location="cpu", weights_only=False)

        latent_full = data["latent"]     # (T_full, 4, 90, 40)
        action_pos  = data["action_pos"] # (T_full, 14)
        if self.use_ee_head:
            ee_target_full = data["ee_target"] # (T_full, 20)

        T_actual = min(latent_full.shape[0], action_pos.shape[0])
        if self.use_ee_head:
            T_actual = min(T_actual, ee_target_full.shape[0])
        latent_full = latent_full[:T_actual]
        action_pos  = action_pos[:T_actual]
        if self.use_ee_head:
            ee_target_full = ee_target_full[:T_actual]

        # pad if shorter than T (edge case)
        if T_actual < self.T:
            pad = self.T - T_actual
            latent_full = torch.cat([latent_full, latent_full[-1:].repeat(pad, 1, 1, 1)], dim=0)
            action_pos  = np.concatenate([action_pos, action_pos[-1:].repeat(pad, axis=0)], axis=0)
            if self.use_ee_head:
                ee_target_full = np.concatenate([ee_target_full, ee_target_full[-1:].repeat(pad, axis=0)], axis=0)
            start = 0

        latent = latent_full[start: start + self.T]
        action = action_pos[start: start + self.T, : self.action_dim]

        action = np.clip(2 * (action - self.p01) / (self.p99 - self.p01 + 1e-8) - 1, -1, 1)

        task_text = task.replace("_", " ")
        sample = {
            "latent": latent.float(),
            "text":   task_text,
            "action": torch.tensor(action, dtype=torch.float32),
        }
        if self.use_ee_head:
            ee_target = ee_target_full[start: start + self.T]
            sample["ee_target"] = torch.tensor(ee_target, dtype=torch.float32)
        return sample
