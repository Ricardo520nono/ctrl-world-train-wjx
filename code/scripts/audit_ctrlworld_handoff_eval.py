#!/usr/bin/env python3
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--stat", type=Path, required=True)
    parser.add_argument("--policy_ckpt", type=Path, required=True)
    parser.add_argument("--wm_steps", type=int, default=20)
    parser.add_argument("--policy_ddim_steps", type=int, default=10)
    parser.add_argument("--action_dim", type=int, default=20)
    parser.add_argument("--summary", type=Path, default=None)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def same_file(left, right):
    return Path(left).resolve(strict=True) == Path(right).resolve(strict=True)


def probe_video(path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    info = {
        "path": str(path),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frames": int(stream["nb_read_frames"]),
    }
    if min(info["width"], info["height"], info["frames"]) <= 0:
        raise RuntimeError(f"invalid video: {info}")
    return info


def require(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def audit_task(root, task, args, stat_sha):
    task_root = root / task
    open_root = task_root / "open_loop_expert_actions"
    closed_root = task_root / "closed_loop_qwenoft_step90000"
    require(task_root / ".complete")
    contract = require(task_root / "eval_contract.txt").read_text()
    if "POLICY_VIEW_ORDER=cam_high,cam_left_wrist,cam_right_wrist" not in contract:
        raise RuntimeError(f"{task} policy input view order missing from eval contract")

    open_meta = json.loads(require(open_root / "metadata.json").read_text())
    closed_meta = json.loads(require(closed_root / "metadata.json").read_text())
    for label, metadata in (("open", open_meta), ("closed", closed_meta)):
        if metadata.get("task") != task or metadata.get("task_filter") != task:
            raise RuntimeError(f"{task} {label} task mismatch: {metadata.get('task')}")
        if int(metadata.get("action_dim", -1)) != args.action_dim:
            raise RuntimeError(f"{task} {label} action_dim mismatch")
        if int(metadata.get("chunk_size", -1)) != 32:
            raise RuntimeError(f"{task} {label} chunk_size mismatch")
        if int(metadata.get("num_history", -1)) != 1 or int(metadata.get("num_frames", -1)) != 32:
            raise RuntimeError(f"{task} {label} temporal contract mismatch")
        if int(metadata.get("num_inference_steps", -1)) != args.wm_steps:
            raise RuntimeError(f"{task} {label} WM denoise steps mismatch")

    if not same_file(open_meta["ckpt"], args.ckpt) or not same_file(closed_meta["wm_ckpt"], args.ckpt):
        raise RuntimeError(f"{task} checkpoint provenance mismatch")
    if not same_file(open_meta["stat"], args.stat) or not same_file(closed_meta["wm_stat"], args.stat):
        raise RuntimeError(f"{task} stat provenance mismatch")
    if not same_file(closed_meta["policy_ckpt"], args.policy_ckpt):
        raise RuntimeError(f"{task} policy checkpoint mismatch")
    if int(closed_meta.get("policy_num_ddim_steps", -1)) != args.policy_ddim_steps:
        raise RuntimeError(f"{task} policy DDIM steps mismatch")
    if closed_meta.get("policy_view_order_canonical") != [
        "head_camera",
        "left_camera",
        "right_camera",
    ]:
        raise RuntimeError(f"{task} canonical policy view order mismatch")
    if not open_meta.get("text") or open_meta.get("text") != closed_meta.get("text"):
        raise RuntimeError(f"{task} open/closed text mismatch")

    bridge = closed_meta.get("action_normalization_bridge", {})
    if bridge.get("schema") != "policy_normalized_to_physical_rot6d20_to_ctrlworld_p01p99_normalized":
        raise RuntimeError(f"{task} action bridge schema mismatch")
    if bridge.get("wm_stat_sha256") != stat_sha:
        raise RuntimeError(f"{task} action bridge stat hash mismatch")

    arrays = {}
    expected_closed_chunks = (int(closed_meta["num_video_frames"]) - 1 + 31) // 32
    for name in (
        "policy_action_chunks_norm.npy",
        "policy_action_chunks_physical.npy",
        "wm_action_chunks_norm.npy",
    ):
        array = np.load(require(closed_root / name))
        if array.ndim != 3 or array.shape[1:] != (32, args.action_dim) or not np.isfinite(array).all():
            raise RuntimeError(f"{task} invalid action array {name}: {array.shape}")
        if array.shape[0] != expected_closed_chunks:
            raise RuntimeError(f"{task} action chunk count mismatch for {name}: {array.shape}")
        arrays[name] = list(array.shape)

    videos = {
        "open_prediction": probe_video(require(open_root / "autoregressive_pred.mp4")),
        "open_comparison": probe_video(require(open_root / "gt_left_autoreg_right.mp4")),
        "closed_prediction": probe_video(require(closed_root / "policy_server_autoreg_pred.mp4")),
        "closed_comparison": probe_video(require(closed_root / "gt_left_policy_autoreg_right.mp4")),
    }
    expected_frames = int(open_meta["num_video_frames"])
    if any(videos[name]["frames"] != expected_frames for name in ("open_prediction", "open_comparison")):
        raise RuntimeError(f"{task} open video frame mismatch")
    expected_closed_frames = int(closed_meta["num_video_frames"])
    if any(
        videos[name]["frames"] != expected_closed_frames
        for name in ("closed_prediction", "closed_comparison")
    ):
        raise RuntimeError(f"{task} closed video frame mismatch")

    return {
        "task": task,
        "sample_id": open_meta.get("sample_id"),
        "text": open_meta.get("text"),
        "open_latent_mse": open_meta.get("latent_mse_without_initial_frame"),
        "closed_latent_mse": closed_meta.get("latent_mse_without_initial_history"),
        "action_arrays": arrays,
        "videos": videos,
        "status": "pass",
    }


def main():
    args = parse_args()
    root = args.root.resolve(strict=True)
    ckpt = args.ckpt.resolve(strict=True)
    stat = args.stat.resolve(strict=True)
    policy_ckpt = args.policy_ckpt.resolve(strict=True)
    summary_path = args.summary or (root / "audit_summary.json")
    stat_sha = sha256(stat)
    summary = {
        "status": "pass",
        "root": str(root),
        "ckpt": str(ckpt),
        "stat": str(stat),
        "stat_sha256": stat_sha,
        "policy_ckpt": str(policy_ckpt),
        "wm_steps": args.wm_steps,
        "policy_ddim_steps": args.policy_ddim_steps,
        "tasks": [audit_task(root, task, args, stat_sha) for task in args.tasks],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (root / ".complete").touch()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
