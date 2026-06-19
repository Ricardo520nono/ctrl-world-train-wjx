"""
Local smoke checks for Ctrl-World EE trajectory auxiliary head.

This script avoids loading SVD/CLIP weights. It validates:
  1. EE target construction from RoboTwin HDF5 endpose fields.
  2. DeltaEEDataset no-head path still works without ee_target.
  3. DeltaEEDataset and DeltaEEFamilyBalancedDataset return aligned ee_target
     when --use_ee_head is enabled.
  4. The EE head and loss compute on a tiny batch.
"""
import json
import os
import shutil
import sys
import tempfile
from types import SimpleNamespace

import h5py
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.dataset_delta_ee import DeltaEEDataset
from dataset.dataset_delta_ee_family import DeltaEEFamilyBalancedDataset
from dataset.ee_targets import EE_TARGET_DIM, ee_target_from_hdf5
from models.ee_head import EETrajectoryHead, compute_ee_losses
from scripts.backfill_ee_targets import _hdf5_path


TASKS = [
    "click_alarmclock",
    "click_bell",
    "place_object_basket",
    "open_laptop",
    "stack_blocks_two",
]


def _write_toy_root(root, tasks, with_ee_target):
    for task_i, task in enumerate(tasks):
        task_dir = os.path.join(root, task)
        os.makedirs(task_dir, exist_ok=True)
        meta = []
        for ep in range(2):
            length = 28 + task_i + ep
            latent = torch.randn(length, 4, 90, 40)
            action = np.random.default_rng(task_i * 100 + ep).normal(size=(length, 14)).astype(np.float32)
            sample = {"latent": latent, "action_pos": action}
            if with_ee_target:
                ee_target = np.random.default_rng(task_i * 200 + ep).normal(size=(length, EE_TARGET_DIM)).astype(np.float32)
                ee_target[:, 9] = (ee_target[:, 9] > 0).astype(np.float32)
                ee_target[:, 19] = (ee_target[:, 19] > 0).astype(np.float32)
                sample["ee_target"] = ee_target
            path = os.path.join(task_dir, f"{task}_episode_{ep:04d}.pt")
            torch.save(sample, path)
            meta.append({"episode": ep, "file": path, "T": length})
        with open(os.path.join(task_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)


def _write_stat(meta_root, cfg):
    stat_dir = os.path.join(meta_root, cfg)
    os.makedirs(stat_dir, exist_ok=True)
    with open(os.path.join(stat_dir, "stat.json"), "w") as f:
        json.dump({"state_01": [-2.0] * 14, "state_99": [2.0] * 14}, f)


def _base_args(tmp, dataset_root, use_ee_head):
    return SimpleNamespace(
        num_history=6,
        num_frames=16,
        action_dim=14,
        use_ee_head=use_ee_head,
        dataset_root_path=dataset_root,
        dataset_meta_info_path=os.path.join(tmp, "meta"),
        dataset_cfgs="toy",
        episode_split="0-1",
        dataset_root_sampling=None,
        dataset_names="+".join(TASKS),
        family_sampling_seed=123,
        family_dataset_length=64,
    )


def check_hdf5_target_shape():
    path = "/mnt/dataset/public_data/cscsx_projects/data/ActionFollowingBench/data_delta_ee/demo_clean_zed2i_visible/click_alarmclock/data/episode0.hdf5"
    if not os.path.exists(path):
        print(f"[skip] HDF5 sample not found: {path}")
        return
    with h5py.File(path, "r") as f:
        ee_target = ee_target_from_hdf5(f)
    if ee_target.ndim != 2 or ee_target.shape[1] != EE_TARGET_DIM:
        raise AssertionError(f"bad ee_target shape: {ee_target.shape}")
    print("  ok hdf5 ee_target:", ee_target.shape)


def check_backfill_path_mapping():
    clean_files = [
        "/data/click_alarmclock/data/episode0.hdf5",
        "/data/click_alarmclock/data/episode1.hdf5",
    ]
    clean_path = _hdf5_path(
        "/data",
        "click_alarmclock",
        {"episode": 1},
        "clean",
        clean_files,
    )
    enhanced_path = _hdf5_path(
        "/enhanced",
        "click_alarmclock",
        {"ep_dir": "episode_1_start_0_seed_0"},
        "enhanced",
        None,
    )
    if clean_path != clean_files[1]:
        raise AssertionError(clean_path)
    expected = "/enhanced/click_alarmclock/episode_1_start_0_seed_0/data.hdf5"
    if enhanced_path != expected:
        raise AssertionError(enhanced_path)
    print("  ok backfill path mapping")


def check_datasets_and_loss():
    tmp = tempfile.mkdtemp(prefix="ctrlworld_ee_head_smoke_")
    try:
        _write_stat(os.path.join(tmp, "meta"), "toy")
        no_head_root = os.path.join(tmp, "no_head")
        ee_root = os.path.join(tmp, "ee")
        _write_toy_root(no_head_root, TASKS, with_ee_target=False)
        _write_toy_root(ee_root, TASKS, with_ee_target=True)

        no_head_ds = DeltaEEDataset(_base_args(tmp, no_head_root, use_ee_head=False), mode="train")
        no_head_sample = no_head_ds[0]
        if "ee_target" in no_head_sample:
            raise AssertionError("no-head sample unexpectedly contains ee_target")
        print("  ok no-head sample:", no_head_sample["latent"].shape, no_head_sample["action"].shape)

        ee_args = _base_args(tmp, ee_root, use_ee_head=True)
        ee_ds = DeltaEEDataset(ee_args, mode="train")
        ee_sample = ee_ds[0]
        assert ee_sample["latent"].shape == (22, 4, 90, 40)
        assert ee_sample["action"].shape == (22, 14)
        assert ee_sample["ee_target"].shape == (22, EE_TARGET_DIM)
        print("  ok ee sample:", ee_sample["latent"].shape, ee_sample["ee_target"].shape)

        family_roots = {}
        for family in ["expert", "pca", "raw", "rf_uniform", "rf_weighted"]:
            family_roots[family] = os.path.join(tmp, family)
            _write_toy_root(family_roots[family], TASKS, with_ee_target=True)
        family_args = _base_args(tmp, family_roots["expert"], use_ee_head=True)
        family_args.family_root_paths = (
            f"expert={family_roots['expert']};"
            f"pca={family_roots['pca']};"
            f"raw={family_roots['raw']};"
            f"rf_uniform={family_roots['rf_uniform']};"
            f"rf_weighted={family_roots['rf_weighted']}"
        )
        family_args.family_sampling = "expert=0.5,pca=0.166667,raw=0.166667,random_feasible=0.166667"
        family_ds = DeltaEEFamilyBalancedDataset(family_args, mode="train")
        family_sample = family_ds[0]
        assert family_sample["ee_target"].shape == (22, EE_TARGET_DIM)
        print("  ok family ee sample:", family_sample["latent"].shape, family_sample["ee_target"].shape)

        head = EETrajectoryHead(hidden_dim=32)
        latent = torch.stack([ee_sample["latent"][6:]], dim=0)
        target = torch.stack([ee_sample["ee_target"][6:]], dim=0)
        pred = head(latent)
        loss, loss_dict = compute_ee_losses(pred, target)
        if pred.shape != target.shape:
            raise AssertionError(f"pred/target mismatch: {pred.shape} vs {target.shape}")
        if not torch.isfinite(loss):
            raise AssertionError(f"non-finite loss: {loss}")
        loss.backward()
        print("  ok ee head loss:", pred.shape, float(loss.item()), sorted(loss_dict.keys()))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    print("[1/3] Checking HDF5 EE target construction...")
    check_hdf5_target_shape()
    print("[2/3] Checking backfill path mapping...")
    check_backfill_path_mapping()
    print("[3/3] Checking dataset shapes and EE loss...")
    check_datasets_and_loss()
    print("[OK] EE head smoke validation passed.")


if __name__ == "__main__":
    main()
