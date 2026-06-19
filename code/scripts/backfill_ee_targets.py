import argparse
import json
import os
import sys

import h5py
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.ee_targets import EE_TARGET_DIM, ee_target_from_hdf5


def _clean_hdf5_files(data_root, task):
    data_dir = os.path.join(data_root, task, "data")
    return sorted(
        os.path.join(data_dir, name)
        for name in os.listdir(data_dir)
        if name.endswith(".hdf5")
    )


def _hdf5_path(data_root, task, item, layout, clean_files):
    if layout == "clean":
        return clean_files[int(item["episode"])]
    if layout == "enhanced":
        return os.path.join(data_root, task, item["ep_dir"], "data.hdf5")
    raise ValueError(f"Unknown layout: {layout}")


def _load_ee_target(path):
    with h5py.File(path, "r") as f:
        return ee_target_from_hdf5(f)


def _check_existing(data, latent_file):
    ee_target = data["ee_target"]
    if ee_target.shape[1] != EE_TARGET_DIM:
        raise RuntimeError(f"Bad ee_target dim in {latent_file}: {ee_target.shape}")
    if ee_target.shape[0] != data["latent"].shape[0]:
        raise RuntimeError(
            f"Bad ee_target length in {latent_file}: ee={ee_target.shape[0]}, latent={data['latent'].shape[0]}"
        )


def backfill_task(latent_root, data_root, task, layout, dry_run):
    meta_path = os.path.join(latent_root, task, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    clean_files = _clean_hdf5_files(data_root, task) if layout == "clean" else None

    updated = 0
    skipped = 0
    for item in meta:
        latent_file = item["file"]
        data = torch.load(latent_file, map_location="cpu", weights_only=False)
        if "ee_target" in data:
            _check_existing(data, latent_file)
            skipped += 1
            continue

        source_hdf5 = _hdf5_path(data_root, task, item, layout, clean_files)
        ee_target = _load_ee_target(source_hdf5)
        if ee_target.shape[0] != data["latent"].shape[0]:
            raise RuntimeError(
                f"Length mismatch for {latent_file}: ee={ee_target.shape[0]}, latent={data['latent'].shape[0]}, source={source_hdf5}"
            )
        if ee_target.shape[1] != EE_TARGET_DIM:
            raise RuntimeError(f"Bad ee_target dim from {source_hdf5}: {ee_target.shape}")
        data["ee_target"] = ee_target
        updated += 1
        if not dry_run:
            tmp_path = f"{latent_file}.ee_target_tmp"
            torch.save(data, tmp_path)
            os.replace(tmp_path, latent_file)
    print(f"[backfill-ee] task={task} layout={layout} updated={updated} skipped={skipped}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_root", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--layout", choices=["clean", "enhanced"], required=True)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    for task in args.tasks:
        backfill_task(args.latent_root, args.data_root, task, args.layout, args.dry_run)


if __name__ == "__main__":
    main()
