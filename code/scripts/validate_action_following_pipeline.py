"""Lightweight validation for ActionFollowing Ctrl-World dataset wiring."""
import json
import os
import shutil
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.dataset_action_following import ActionFollowingCtrlWorldDataset


FAMILIES = [
    ("clean", "clean", 80),
    ("perturbed", "pca", 33),
    ("perturbed", "raw", 33),
    ("random_feasible", "uniform", 64),
    ("random_feasible", "weighted", 64),
    ("counterfactual_replay", None, 72),
    ("exploration", "policy_rollout", 96),
]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def make_fixture(root):
    rows = []
    quick_rows = []
    rng = np.random.default_rng(7)
    for family, subtype, length in FAMILIES:
        for task in ["place_can_basket", "turn_switch"]:
            sample_id = f"{family}_{subtype or 'default'}_{task}"
            out_dir = os.path.join(root, "samples", family, subtype or "default", task)
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"{sample_id}.pt")
            torch.save(
                {
                    "latent": torch.randn(length, 4, 90, 40),
                    "action_pos": rng.normal(size=(length, 14)).astype(np.float32),
                    "ee_target": rng.normal(size=(length, 20)).astype(np.float32),
                    "text": task.replace("_", " "),
                },
                path,
            )
            rec = {
                "file": os.path.relpath(path, root),
                "family": family,
                "subtype": subtype,
                "task": task,
                "sample_id": sample_id,
                "length": length,
                "unit_level": "chunk" if family == "perturbed" else "trajectory",
            }
            rows.append(rec)
            if family != "clean":
                q = dict(rec)
                q["chunk_start"] = 1 if length > 33 else 0
                q["fixed_start"] = q["chunk_start"]
                quick_rows.append(q)
    write_jsonl(os.path.join(root, "manifests", "train.jsonl"), rows)
    write_jsonl(os.path.join(root, "manifests", "test_quick.jsonl"), quick_rows)
    stat_dir = os.path.join(root, "meta")
    os.makedirs(stat_dir, exist_ok=True)
    with open(os.path.join(stat_dir, "stat.json"), "w") as f:
        json.dump({"state_01": [-2.0] * 14, "state_99": [2.0] * 14}, f)
    return os.path.join(stat_dir, "stat.json")


def make_args(root, protocol, stat_path):
    return SimpleNamespace(
        num_history=6,
        num_frames=26,
        action_dim=14,
        use_ee_head=False,
        dataset_meta_info_path=os.path.join(root, "meta_root"),
        dataset_cfgs="unused",
        action_following_latent_root=root,
        action_following_train_manifest=os.path.join(root, "manifests", "train.jsonl"),
        action_following_val_manifest=os.path.join(root, "manifests", "test_quick.jsonl"),
        action_following_stat_path=stat_path,
        action_following_sampling_protocol=protocol,
        action_following_chunk_size=32,
        action_following_dataset_length=64,
        action_following_sampling_seed=123,
    )


def check(protocol):
    tmp = tempfile.mkdtemp(prefix="ctrlworld_af_validate_")
    try:
        stat_path = make_fixture(tmp)
        ds = ActionFollowingCtrlWorldDataset(make_args(tmp, protocol, stat_path), mode="train")
        sample = ds[0]
        assert sample["latent"].shape == (32, 4, 90, 40)
        assert sample["action"].shape == (32, 14)
        assert len(ds) == 64
        val = ActionFollowingCtrlWorldDataset(make_args(tmp, protocol, stat_path), mode="test_quick")
        assert len(val) > 0
        val_sample = val[0]
        assert val_sample["latent"].shape == (32, 4, 90, 40)
        assert val_sample["family"] != "clean"
        print(f"ok protocol={protocol}: train_len={len(ds)}, quick_len={len(val)}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    check("mix_4to1to1to1to1")
    check("mix_1to1to1to1to1")
    check("enhanced_1to1to1to1")
    check("clean_only")
    print("[OK] ActionFollowing Ctrl-World dataset validation passed.")


if __name__ == "__main__":
    main()
