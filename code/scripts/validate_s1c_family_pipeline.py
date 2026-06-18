"""
Local validation for the S1-C family-balanced training pipeline.

This script does not run VAE precompute or distributed training. It checks:
  1. Real input paths and HDF5 schemas for expert/PCA/raw/RF data.
  2. Existing expert/PCA headwrist latent meta files.
  3. The family-balanced dataset on a tiny synthetic latent fixture.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile

import h5py
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TRAIN_PACKAGE_ROOT = os.environ.get("TRAIN_PACKAGE_ROOT", "/mnt/public_ckp/cscsx_projects/ctrl_world_train")
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(TRAIN_PACKAGE_ROOT, "code"))
CACHE_ROOT = os.environ.get("CACHE_ROOT", os.path.join(TRAIN_PACKAGE_ROOT, "latents"))

TASKS = [
    "click_alarmclock",
    "click_bell",
    "place_object_basket",
    "open_laptop",
    "stack_blocks_two",
]

EXPERT_LATENT_ROOT = os.environ.get(
    "EXPERT_LATENT_ROOT",
    os.path.join(CACHE_ROOT, "precomputed_latents_s1A_5tasks_14d_headwrist"),
)
PCA_LATENT_ROOT = os.environ.get(
    "PCA_LATENT_ROOT",
    os.path.join(CACHE_ROOT, "s1B_latents_pca_train_headwrist"),
)
DATA_ROOT_RAW = "/mnt/public_ckp/cscsx_projects/data/ActionFollowingBench/EnhancedData/perturbed_raw_gaussian/sigma_0p0025"
RF_BASE = "/mnt/public_ckp/cscsx_projects/data/ActionFollowingBench/EnhancedData/random_feasible_300step_5task_2ep5start_formal_v1/random_feasible_random_walk"
DATA_ROOT_RF_UNIFORM = f"{RF_BASE}/rf_5task_300step_2ep5start_formal_uniform_10seed_v1"
DATA_ROOT_RF_WEIGHTED = f"{RF_BASE}/rf_5task_300step_2ep5start_formal_weighted_10seed_v1"


def require(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)


def first_hdf5(task_root):
    for dirpath, _, filenames in os.walk(task_root):
        for name in filenames:
            if name.endswith(".hdf5"):
                return os.path.join(dirpath, name)
    raise FileNotFoundError(f"No .hdf5 under {task_root}")


def check_hdf5_schema(path):
    with h5py.File(path, "r") as f:
        for key in [
            "delta_ee_action/left_delta_pose",
            "delta_ee_action/left_gripper",
            "delta_ee_action/right_delta_pose",
            "delta_ee_action/right_gripper",
            "observation/head_camera/rgb",
            "observation/left_camera/rgb",
            "observation/right_camera/rgb",
        ]:
            if key not in f:
                raise KeyError(f"{key} missing in {path}")
        t_action = f["delta_ee_action/left_delta_pose"].shape[0]
        t_head = f["observation/head_camera/rgb"].shape[0]
        if t_action != t_head:
            raise RuntimeError(f"T mismatch in {path}: action={t_action}, head={t_head}")
        return t_action


def check_real_inputs():
    print("[1/3] Checking real paths and source schemas...")
    for root in [EXPERT_LATENT_ROOT, PCA_LATENT_ROOT]:
        for task in TASKS:
            meta_path = os.path.join(root, task, "meta.json")
            require(meta_path)
            meta = json.load(open(meta_path))
            if not meta:
                raise RuntimeError(f"empty meta: {meta_path}")
        print(f"  ok latent root: {root}")

    for name, root in [
        ("raw", DATA_ROOT_RAW),
        ("rf_uniform", DATA_ROOT_RF_UNIFORM),
        ("rf_weighted", DATA_ROOT_RF_WEIGHTED),
    ]:
        for task in TASKS:
            task_root = os.path.join(root, task)
            require(task_root)
            sample = first_hdf5(task_root)
            t = check_hdf5_schema(sample)
            print(f"  ok {name}/{task}: sample_T={t}, sample={sample}")


def write_toy_root(root, family_name, tasks):
    for task_i, task in enumerate(tasks):
        task_dir = os.path.join(root, task)
        os.makedirs(task_dir, exist_ok=True)
        meta = []
        for ep in range(2):
            T = 28 + task_i + ep
            latent = torch.randn(T, 4, 90, 40)
            action = np.random.default_rng(task_i * 100 + ep).normal(size=(T, 14)).astype(np.float32)
            path = os.path.join(task_dir, f"{family_name}_{task}_episode_{ep:04d}.pt")
            torch.save({"latent": latent, "action_pos": action}, path)
            meta.append({"episode": ep, "file": path, "T": T, "family": family_name})
        with open(os.path.join(task_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)


def check_toy_family_dataset():
    print("[2/3] Checking family-balanced dataset on toy latents...")
    from dataset.dataset_delta_ee_family import DeltaEEFamilyBalancedDataset

    tmp = tempfile.mkdtemp(prefix="ctrlworld_s1c_validate_")
    try:
        roots = {}
        for family in ["expert", "pca", "raw", "rf_uniform", "rf_weighted"]:
            roots[family] = os.path.join(tmp, family)
            write_toy_root(roots[family], family, TASKS)

        meta_dir = os.path.join(tmp, "meta", "toy")
        os.makedirs(meta_dir, exist_ok=True)
        with open(os.path.join(meta_dir, "stat.json"), "w") as f:
            json.dump({"state_01": [-2.0] * 14, "state_99": [2.0] * 14}, f)

        class Args:
            num_history = 6
            num_frames = 16
            action_dim = 14
            dataset_meta_info_path = os.path.join(tmp, "meta")
            dataset_cfgs = "toy"
            episode_split = "0-1"
            dataset_names = "+".join(TASKS)
            family_root_paths = (
                f"expert={roots['expert']};"
                f"pca={roots['pca']};"
                f"raw={roots['raw']};"
                f"rf_uniform={roots['rf_uniform']};"
                f"rf_weighted={roots['rf_weighted']}"
            )
            family_sampling = "expert=0.5,pca=0.166667,raw=0.166667,random_feasible=0.166667"
            family_sampling_seed = 123
            family_dataset_length = 128

        ds = DeltaEEFamilyBalancedDataset(Args(), mode="train")
        if len(ds) != 128:
            raise AssertionError(f"unexpected dataset len: {len(ds)}")
        sample = ds[0]
        assert sample["latent"].shape == (22, 4, 90, 40)
        assert sample["action"].shape == (22, 14)
        assert isinstance(sample["text"], str)
        print("  ok toy sample:", sample["latent"].shape, sample["action"].shape, sample["text"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_training_script_args():
    print("[3/3] Checking training script argparse additions...")
    train_script = os.path.join(PROJECT_ROOT, "scripts", "train_delta_ee.py")
    text = open(train_script).read()
    for needle in [
        "--use_family_balanced_sampler",
        "--family_root_paths",
        "--family_sampling",
        "DeltaEEFamilyBalancedDataset",
    ]:
        if needle not in text:
            raise RuntimeError(f"missing {needle} in train_delta_ee.py")
    print("  ok train_delta_ee.py contains family-balanced wiring")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-real-inputs", action="store_true")
    args = parser.parse_args()

    if not args.skip_real_inputs:
        check_real_inputs()
    check_toy_family_dataset()
    check_training_script_args()
    print("[OK] S1-C family-balanced pipeline validation passed.")


if __name__ == "__main__":
    main()
