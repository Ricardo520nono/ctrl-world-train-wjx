#!/usr/bin/env python3
"""Autoregressively roll out Ctrl-World using actions from a StarVLA policy server."""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import mediapy
import numpy as np
import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline  # noqa: E402
from replay_clean_expert_autoreg import (  # noqa: E402
    decode_three_views,
    encode_action_condition,
    load_model,
    make_args,
    read_jsonl,
    resolve_under,
    stack_views,
)


DEFAULT_WM_RUN_DIR = Path(
    "/mnt/dataset/public_data/cscsx_projects/AF3/starVLA_dev/legacy_flat_results_20260706/"
    "ctrl-world/outputs/"
    "ACWM_ctrlworld_action_following_mix4_test4v2_chunk32_8gpu_rgbfix_explore_resume15000_20260706"
)
DEFAULT_LATENT_ROOT = Path(
    "/mnt/dataset/public_data/cscsx_projects/ctrl_world_train/latents/"
    "action_following_chunk32_clean_enhanced_explore_test4v2_20260705"
)
DEFAULT_STAT = Path(
    "/mnt/dataset/public_data/cscsx_projects/ctrl_world_train/dataset_meta_info/"
    "action_following_chunk32_clean_enhanced_explore_test4v2_20260705/stat.json"
)
DEFAULT_STARVLA_ROOT = Path("/mnt/dataset/csx_workspace/Ideas/AF3/code/starVLA_dev")


VIEW_ALIASES = {
    "cam_high": "head_camera",
    "head": "head_camera",
    "head_camera": "head_camera",
    "image_0": "head_camera",
    "cam_left_wrist": "left_camera",
    "left_camera": "left_camera",
    "left_wrist": "left_camera",
    "image_1": "left_camera",
    "cam_right_wrist": "right_camera",
    "right_camera": "right_camera",
    "right_wrist": "right_camera",
    "image_2": "right_camera",
}

CANONICAL_VIEW_INDEX = {
    "head_camera": 0,
    "left_camera": 1,
    "right_camera": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_WM_RUN_DIR / "checkpoint-step25000-epoch2.06.pt")
    parser.add_argument("--latent_root", type=Path, default=DEFAULT_LATENT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_LATENT_ROOT / "manifests" / "clean_train.jsonl")
    parser.add_argument("--stat", type=Path, default=DEFAULT_STAT)
    parser.add_argument("--task", default="rotate_qrcode")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=20, help="Ctrl-World diffusion denoising steps")
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--max_actions", type=int, default=0, help="0 means match selected GT trajectory length")
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--num_history", type=int, default=6)
    parser.add_argument("--num_frames", type=int, default=26)
    parser.add_argument("--action_chunk_size", type=int, default=32)
    parser.add_argument("--action_dim", type=int, default=14)
    parser.add_argument("--decode_chunk", type=int, default=7)
    parser.add_argument("--fps", type=int, default=2)

    parser.add_argument("--policy_host", default="127.0.0.1")
    parser.add_argument("--policy_port", type=int, default=5694)
    parser.add_argument("--policy_mode", default="vla")
    parser.add_argument("--policy_num_ddim_steps", type=int, default=10)
    parser.add_argument("--policy_ckpt", type=Path, default=None, help="policy checkpoint provenance for metadata")
    parser.add_argument("--policy_bridge_python", type=Path, default=DEFAULT_STARVLA_ROOT / ".venv" / "bin" / "python")
    parser.add_argument("--starvla_root", type=Path, default=DEFAULT_STARVLA_ROOT)
    parser.add_argument("--policy_view_order", default="server", help="'server' or comma-separated view names")
    parser.add_argument(
        "--legacy_history_action_mode",
        choices=("previous", "repeat_first", "zero", "policy_prefix"),
        default="previous",
        help="How to fill the first num_history action tokens when action_chunk_size=num_history+num_frames.",
    )
    parser.add_argument("--allow_action_truncate", action="store_true")
    return parser.parse_args()


def load_initial_record(cli: argparse.Namespace):
    records = read_jsonl(cli.manifest)
    if cli.task:
        records = [rec for rec in records if rec.get("task") == cli.task]
    if not records:
        raise RuntimeError(f"empty manifest selection: task={cli.task} manifest={cli.manifest}")
    rec = records[int(cli.sample_index) % len(records)]
    data_path = resolve_under(cli.latent_root, rec["file"])
    data = torch.load(str(data_path), map_location="cpu", weights_only=False)
    latent = data["latent"].float()
    if latent.ndim != 4:
        raise RuntimeError(f"expected latent [T,C,H,W], got {tuple(latent.shape)} from {data_path}")
    text = rec.get("text") or data.get("text") or str(rec.get("task", "")).replace("_", " ")
    total_future = max(0, latent.shape[0] - int(cli.num_history))
    if cli.max_actions > 0:
        total_future = min(total_future, int(cli.max_actions))
    if total_future <= 0:
        raise RuntimeError(f"record has no future frames after num_history={cli.num_history}: {data_path}")
    return rec, data_path, latent[: int(cli.num_history) + total_future], text, total_future


def encode_image_payload(image: np.ndarray) -> dict[str, Any]:
    image = np.ascontiguousarray(image.astype(np.uint8, copy=False))
    return {
        "shape": list(image.shape),
        "dtype": str(image.dtype),
        "data": base64.b64encode(image.tobytes()).decode("ascii"),
    }


class StarVLAPolicyBridge:
    def __init__(self, cli: argparse.Namespace):
        if not cli.policy_bridge_python.exists():
            raise FileNotFoundError(f"policy_bridge_python not found: {cli.policy_bridge_python}")
        if not cli.starvla_root.exists():
            raise FileNotFoundError(f"starvla_root not found: {cli.starvla_root}")
        bridge_script = SCRIPT_DIR / "starvla_policy_bridge.py"
        self.proc = subprocess.Popen(
            [
                str(cli.policy_bridge_python),
                str(bridge_script),
                "--starvla_root",
                str(cli.starvla_root),
                "--host",
                cli.policy_host,
                "--port",
                str(cli.policy_port),
                "--mode",
                cli.policy_mode,
                "--num_ddim_steps",
                str(cli.policy_num_ddim_steps),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        ready = self._read_json_line()
        if ready.get("type") != "ready":
            raise RuntimeError(f"policy bridge did not report ready: {ready}")
        self.metadata = ready.get("metadata", {})

    def _read_json_line(self) -> dict[str, Any]:
        if self.proc.stdout is None:
            raise RuntimeError("policy bridge stdout is closed")
        line = self.proc.stdout.readline()
        if not line:
            stderr = ""
            if self.proc.stderr is not None:
                stderr = self.proc.stderr.read()
            raise RuntimeError(f"policy bridge exited early with code {self.proc.poll()}; stderr:\n{stderr}")
        return json.loads(line)

    def predict(self, *, request_id: str, lang: str, images: list[np.ndarray], cli: argparse.Namespace, seed: int) -> np.ndarray:
        if self.proc.stdin is None:
            raise RuntimeError("policy bridge stdin is closed")
        payload = {
            "request_id": request_id,
            "lang": lang,
            "images": [encode_image_payload(image) for image in images],
            "mode": cli.policy_mode,
            "num_ddim_steps": int(cli.policy_num_ddim_steps),
            "num_steps": int(cli.policy_num_ddim_steps),
            "seed": int(seed),
        }
        self.proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        response = self._read_json_line()
        if not response.get("ok"):
            raise RuntimeError(f"policy bridge inference failed: {response}")
        actions = np.asarray(response["normalized_actions"], dtype=np.float32)
        if actions.ndim == 3:
            actions = actions[0]
        if actions.ndim != 2:
            raise RuntimeError(f"expected policy actions [T,D] or [B,T,D], got {actions.shape}")
        return actions

    def close(self) -> None:
        if self.proc.poll() is not None:
            return
        try:
            if self.proc.stdin is not None:
                self.proc.stdin.write(json.dumps({"type": "close"}) + "\n")
                self.proc.stdin.flush()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def view_order_from_metadata(metadata: dict[str, Any], cli: argparse.Namespace) -> list[str]:
    if cli.policy_view_order != "server":
        raw_views = [item.strip() for item in cli.policy_view_order.split(",") if item.strip()]
    else:
        raw = metadata.get("input_views") or metadata.get("view_order") or []
        raw_views = list(raw) if isinstance(raw, (list, tuple)) else []
    if not raw_views:
        raw_views = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    canonical = []
    for name in raw_views:
        key = VIEW_ALIASES.get(str(name))
        if key is None:
            raise KeyError(f"Unsupported policy view name {name!r}; pass --policy_view_order explicitly.")
        canonical.append(key)
    return canonical


def decode_current_policy_images(model, current_history: torch.Tensor, decode_chunk: int, canonical_view_order: list[str]) -> list[np.ndarray]:
    current_latent = current_history[:, -1:].to(device=next(model.unet.parameters()).device, dtype=torch.bfloat16)
    views = decode_three_views(model.pipeline, current_latent, decode_chunk)
    per_view = [views[i, 0] for i in range(3)]
    return [per_view[CANONICAL_VIEW_INDEX[name]] for name in canonical_view_order]


def coerce_policy_actions(policy_actions: np.ndarray, cli: argparse.Namespace) -> np.ndarray:
    if policy_actions.shape[1] < cli.action_dim:
        raise RuntimeError(f"policy action dim {policy_actions.shape[1]} < required WM action_dim {cli.action_dim}")
    if policy_actions.shape[1] > cli.action_dim:
        if not cli.allow_action_truncate:
            raise RuntimeError(
                f"policy action dim {policy_actions.shape[1]} > WM action_dim {cli.action_dim}; "
                "pass --allow_action_truncate only if this is intentional."
            )
        policy_actions = policy_actions[:, : cli.action_dim]
    return policy_actions.astype(np.float32, copy=False)


def build_wm_action_condition(
    policy_actions: np.ndarray,
    *,
    previous_future_actions: np.ndarray | None,
    cli: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    policy_actions = coerce_policy_actions(policy_actions, cli)
    if policy_actions.shape[0] < cli.num_frames:
        pad = np.repeat(policy_actions[-1:], cli.num_frames - policy_actions.shape[0], axis=0)
        policy_actions = np.concatenate([policy_actions, pad], axis=0)

    if cli.action_chunk_size == cli.num_frames:
        future = policy_actions[: cli.num_frames]
        return future, future

    if cli.action_chunk_size != cli.num_history + cli.num_frames:
        raise RuntimeError(
            "Only current-frame action_chunk_size=num_frames or legacy "
            "action_chunk_size=num_history+num_frames layouts are supported."
        )

    if cli.legacy_history_action_mode == "policy_prefix":
        if policy_actions.shape[0] < cli.action_chunk_size:
            pad = np.repeat(policy_actions[-1:], cli.action_chunk_size - policy_actions.shape[0], axis=0)
            policy_actions = np.concatenate([policy_actions, pad], axis=0)
        return policy_actions[: cli.action_chunk_size], policy_actions[cli.num_history : cli.action_chunk_size]

    future = policy_actions[: cli.num_frames]
    if cli.legacy_history_action_mode == "zero":
        history = np.zeros((cli.num_history, cli.action_dim), dtype=np.float32)
    elif previous_future_actions is not None and previous_future_actions.shape[0] >= cli.num_history:
        history = previous_future_actions[-cli.num_history :]
    else:
        history = np.repeat(future[:1], cli.num_history, axis=0)
    return np.concatenate([history, future], axis=0), future


@torch.no_grad()
def autoregressive_policy_rollout(model, args, cli, gt_latent: torch.Tensor, text: str, bridge: StarVLAPolicyBridge):
    device = next(model.unet.parameters()).device
    dtype = torch.bfloat16
    current = gt_latent[: cli.num_history][None].to(device=device, dtype=dtype)
    pred_chunks = []
    policy_action_chunks = []
    wm_action_chunks = []
    previous_future_actions = None
    total_future = int(cli._total_future)
    view_order = view_order_from_metadata(bridge.metadata, cli)

    for chunk_id, start in enumerate(range(0, total_future, cli.num_frames)):
        valid = min(cli.num_frames, total_future - start)
        images = decode_current_policy_images(model, current, cli.decode_chunk, view_order)
        policy_actions = bridge.predict(
            request_id=f"ctrlworld-{chunk_id:04d}",
            lang=text,
            images=images,
            cli=cli,
            seed=cli.seed + chunk_id,
        )
        wm_action, previous_future_actions = build_wm_action_condition(
            policy_actions,
            previous_future_actions=previous_future_actions,
            cli=cli,
        )
        action = torch.tensor(wm_action, device=device, dtype=dtype).unsqueeze(0)
        action_latent = encode_action_condition(model, args, action, text)
        generator = torch.Generator(device=device).manual_seed(int(cli.seed) + chunk_id)

        _, future = CtrlWorldDiffusionPipeline.__call__(
            model.pipeline,
            image=current[:, -1],
            text=action_latent,
            width=args.width,
            height=3 * args.height,
            num_frames=args.num_frames,
            history=current,
            num_inference_steps=cli.steps,
            decode_chunk_size=cli.decode_chunk,
            max_guidance_scale=1.0,
            fps=args.fps,
            motion_bucket_id=args.motion_bucket_id,
            generator=generator,
            output_type="latent",
            return_dict=False,
            frame_level_cond=args.frame_level_cond,
            his_cond_zero=args.his_cond_zero,
        )

        keep = future[:, :valid].to(dtype)
        pred_chunks.append(keep.cpu())
        policy_action_chunks.append(policy_actions)
        wm_action_chunks.append(wm_action)
        current = torch.cat([current, keep], dim=1)[:, -cli.num_history :].to(device=device, dtype=dtype)
        print(
            f"[policy_autoreg] chunk={chunk_id} action_range=[{start},{start + valid}) "
            f"policy_actions={tuple(policy_actions.shape)} wm_action={tuple(wm_action.shape)}",
            flush=True,
        )

    pred_latent = torch.cat([gt_latent[: cli.num_history][None].cpu(), torch.cat(pred_chunks, dim=1)], dim=1)
    return pred_latent, policy_action_chunks, wm_action_chunks, view_order


def main() -> None:
    cli = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Ctrl-World policy-server autoregressive rollout.")

    out = cli.out or DEFAULT_WM_RUN_DIR / "rollout_policy_server_rotate_qrcode" / "checkpoint-step25000-epoch2.06_sample0_steps20"
    out.mkdir(parents=True, exist_ok=True)
    args = make_args(cli, out)

    rec, data_path, gt_latent, text, total_future = load_initial_record(cli)
    cli._total_future = total_future
    model = load_model(args, cli.ckpt, torch.device("cuda"))

    bridge = StarVLAPolicyBridge(cli)
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_latent, policy_action_chunks, wm_action_chunks, view_order = autoregressive_policy_rollout(
                model, args, cli, gt_latent, text, bridge
            )
    finally:
        bridge.close()

    gt_latent = gt_latent[None, : pred_latent.shape[1]].cpu()
    mse = torch.mean((pred_latent[:, cli.num_history :].float() - gt_latent[:, cli.num_history :].float()) ** 2).item()

    torch.save(
        {
            "pred_latents": pred_latent,
            "gt_latents": gt_latent,
            "text": text,
            "record": rec,
            "policy_action_chunks_norm": policy_action_chunks,
            "wm_action_chunks_norm": wm_action_chunks,
        },
        out / "policy_server_autoreg_latents.pt",
    )
    np.save(out / "policy_action_chunks_norm.npy", np.asarray(policy_action_chunks, dtype=np.float32))
    np.save(out / "wm_action_chunks_norm.npy", np.asarray(wm_action_chunks, dtype=np.float32))

    pred_views = decode_three_views(model.pipeline, pred_latent.to(device="cuda", dtype=torch.bfloat16), cli.decode_chunk)
    gt_views = decode_three_views(model.pipeline, gt_latent.to(device="cuda", dtype=torch.bfloat16), cli.decode_chunk)
    pred_video = stack_views(pred_views)
    side_by_side = np.concatenate(
        [np.concatenate([gt_views[i], pred_views[i]], axis=2) for i in range(3)],
        axis=1,
    )
    mediapy.write_video(str(out / "policy_server_autoreg_pred.mp4"), pred_video, fps=cli.fps)
    mediapy.write_video(str(out / "gt_left_policy_autoreg_right.mp4"), side_by_side, fps=cli.fps)

    metadata = {
        "wm_ckpt": str(cli.ckpt),
        "wm_stat": str(cli.stat),
        "latent_root": str(cli.latent_root),
        "manifest": str(cli.manifest),
        "sample_index": int(cli.sample_index),
        "task_filter": cli.task,
        "task": rec.get("task", ""),
        "sample_id": rec.get("sample_id", ""),
        "data_path": str(data_path),
        "text": text,
        "action_dim": int(cli.action_dim),
        "chunk_size": int(cli.chunk_size),
        "num_history": int(cli.num_history),
        "num_frames": int(cli.num_frames),
        "action_chunk_size": int(cli.action_chunk_size),
        "legacy_history_action_mode": cli.legacy_history_action_mode,
        "num_video_frames": int(pred_latent.shape[1]),
        "num_inference_steps": int(cli.steps),
        "policy_host": cli.policy_host,
        "policy_port": int(cli.policy_port),
        "policy_mode": cli.policy_mode,
        "policy_num_ddim_steps": int(cli.policy_num_ddim_steps),
        "policy_ckpt": str(cli.policy_ckpt) if cli.policy_ckpt else None,
        "policy_bridge_python": str(cli.policy_bridge_python),
        "starvla_root": str(cli.starvla_root),
        "policy_server_metadata": bridge.metadata,
        "policy_view_order_canonical": view_order,
        "latent_mse_without_initial_history": mse,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")

    print(f"pred video: {out / 'policy_server_autoreg_pred.mp4'}")
    print(f"compare video: {out / 'gt_left_policy_autoreg_right.mp4'}")
    print(f"latents: {out / 'policy_server_autoreg_latents.pt'}")
    print(f"metadata: {out / 'metadata.json'}")
    print(f"latent_mse_without_initial_history: {mse:.6f}")


if __name__ == "__main__":
    main()
