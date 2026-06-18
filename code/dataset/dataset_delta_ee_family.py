"""
Family-balanced Delta-EE dataset for S1 mixed training.

Sampling hierarchy:
  family -> task -> trajectory/sample -> window

The random_feasible family may contain two variants (uniform/weighted). When
both are present, variants are sampled uniformly inside the family.
"""
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


def _parse_episode_split(split_str):
    if split_str is None:
        return None
    lo, hi = split_str.split("-")
    return set(range(int(lo), int(hi) + 1))


def _parse_family_roots(spec):
    """Parse 'expert=/a;pca=/b;raw=/c;rf_uniform=/d;rf_weighted=/e'."""
    roots = {}
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid family root spec: {part}")
        key, value = part.split("=", 1)
        roots[key.strip()] = value.strip()
    return roots


def _parse_sampling(spec):
    """Parse 'expert=0.5,pca=0.166667,raw=0.166667,random_feasible=0.166667'."""
    probs = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid family sampling spec: {part}")
        key, value = part.split("=", 1)
        probs[key.strip()] = float(value)
    total = sum(probs.values())
    if total <= 0:
        raise ValueError("family sampling probabilities sum to zero")
    return {k: v / total for k, v in probs.items()}


class DeltaEEFamilyBalancedDataset(Dataset):
    def __init__(self, args, mode: str = "train"):
        self.args = args
        self.mode = mode
        self.num_history = args.num_history
        self.num_frames = args.num_frames
        self.T = self.num_history + self.num_frames
        self.action_dim = int(getattr(args, "action_dim", 14))
        self.seed = int(getattr(args, "family_sampling_seed", 20260610))

        stat_path = os.path.join(args.dataset_meta_info_path, args.dataset_cfgs.split("+")[0], "stat.json")
        with open(stat_path) as f:
            stat = json.load(f)
        self.p01 = np.array(stat["state_01"], dtype=np.float32)[: self.action_dim]
        self.p99 = np.array(stat["state_99"], dtype=np.float32)[: self.action_dim]

        task_names = args.dataset_names.split("+")
        allowed_episodes = _parse_episode_split(getattr(args, "episode_split", None))
        roots = _parse_family_roots(args.family_root_paths)
        probs = _parse_sampling(args.family_sampling)

        self.family_probs = probs
        self.family_names = list(probs.keys())
        self.family_p = np.array([probs[k] for k in self.family_names], dtype=np.float64)

        self.data = {}
        for family in ["expert", "pca", "raw"]:
            if family in roots:
                self.data[family] = self._load_root(roots[family], task_names, allowed_episodes)

        rf_variants = {}
        if "rf_uniform" in roots:
            rf_variants["uniform"] = self._load_root(roots["rf_uniform"], task_names, allowed_episodes)
        if "rf_weighted" in roots:
            rf_variants["weighted"] = self._load_root(roots["rf_weighted"], task_names, allowed_episodes)
        if rf_variants:
            self.data["random_feasible"] = rf_variants

        for family in self.family_names:
            if family not in self.data:
                raise RuntimeError(f"family '{family}' requested but no data loaded")

        default_len = 0
        for family in self.family_names:
            default_len += self._count_windows_for_family(family)
        self.virtual_length = int(getattr(args, "family_dataset_length", 0) or default_len)
        if self.virtual_length <= 0:
            raise RuntimeError("No windows found for family-balanced dataset.")

        self._print_summary(task_names, roots)

    def _load_root(self, root, task_names, allowed_episodes):
        by_task = {}
        for task in task_names:
            meta_path = os.path.join(root, task, "meta.json")
            if not os.path.exists(meta_path):
                raise FileNotFoundError(f"meta.json not found for task={task} root={root}")
            items = []
            for item in json.load(open(meta_path)):
                if allowed_episodes is not None:
                    ep_idx = item.get("episode", None)
                    if ep_idx is None or ep_idx not in allowed_episodes:
                        continue
                T_full = int(item["T"])
                nwin = max(1, T_full - self.T + 1)
                items.append({"file": item["file"], "T": T_full, "nwin": nwin})
            if not items:
                raise RuntimeError(f"No items for task={task} root={root}")
            by_task[task] = items
        return by_task

    def _count_windows_for_task_data(self, task_data):
        return sum(item["nwin"] for items in task_data.values() for item in items)

    def _count_windows_for_family(self, family):
        family_data = self.data[family]
        if family != "random_feasible":
            return self._count_windows_for_task_data(family_data)
        return sum(self._count_windows_for_task_data(v) for v in family_data.values())

    def _print_summary(self, task_names, roots):
        print("[DeltaEEFamilyBalancedDataset]")
        print(f"  mode={self.mode}, T={self.T}, virtual_length={self.virtual_length}")
        print(f"  families={self.family_probs}")
        print(f"  tasks={task_names}")
        print(f"  roots={roots}")
        for family in self.family_names:
            print(f"  {family}: {self._count_windows_for_family(family)} windows")
            if family == "random_feasible":
                for variant, data in self.data[family].items():
                    print(f"    {variant}: {self._count_windows_for_task_data(data)} windows")

    def __len__(self):
        return self.virtual_length

    def _choose_task_item_window(self, rng, task_data):
        tasks = list(task_data.keys())
        task = tasks[int(rng.integers(0, len(tasks)))]
        items = task_data[task]
        item = items[int(rng.integers(0, len(items)))]
        start = int(rng.integers(0, item["nwin"]))
        return task, item["file"], start

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.seed + int(idx))
        family = self.family_names[int(rng.choice(len(self.family_names), p=self.family_p))]

        if family == "random_feasible":
            variants = list(self.data[family].keys())
            variant = variants[int(rng.integers(0, len(variants)))]
            task, file, start = self._choose_task_item_window(rng, self.data[family][variant])
        else:
            task, file, start = self._choose_task_item_window(rng, self.data[family])

        data = torch.load(file, map_location="cpu", weights_only=False)
        latent_full = data["latent"]
        action_pos = data["action_pos"]

        T_actual = min(latent_full.shape[0], action_pos.shape[0])
        latent_full = latent_full[:T_actual]
        action_pos = action_pos[:T_actual]

        if T_actual < self.T:
            pad = self.T - T_actual
            latent_full = torch.cat([latent_full, latent_full[-1:].repeat(pad, 1, 1, 1)], dim=0)
            action_pos = np.concatenate([action_pos, action_pos[-1:].repeat(pad, axis=0)], axis=0)
            start = 0

        latent = latent_full[start:start + self.T]
        action = action_pos[start:start + self.T, : self.action_dim]
        action = np.clip(2 * (action - self.p01) / (self.p99 - self.p01 + 1e-8) - 1, -1, 1)

        return {
            "latent": latent.float(),
            "text": task.replace("_", " "),
            "action": torch.tensor(action, dtype=torch.float32),
        }
