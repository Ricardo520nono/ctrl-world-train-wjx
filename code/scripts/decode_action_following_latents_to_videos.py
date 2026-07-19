#!/usr/bin/env python3
"""Decode precomputed ActionFollowing Ctrl-World latents into RGB videos.

The latent files produced by ActionFollowing precompute store three camera views
stacked along the latent height dimension: [T, 4, 90, 40] -> 3 x [T, 4, 30, 40].
This script decodes those latents with the SVD VAE and writes a side-by-side
three-view mp4 for each latent sample.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import einops
import mediapy
import numpy as np
import torch
from diffusers import AutoencoderKLTemporalDecoder


DEFAULT_SVD_PATH = Path(
    "/mnt/dataset/public_data/cscsx_projects/ctrl_world_train/assets/models/stable-video-diffusion-img2vid"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_root", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, default=None)
    parser.add_argument("--svd_path", type=Path, default=DEFAULT_SVD_PATH)
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        default=[],
        help="Optional manifest jsonl; can be repeated. Defaults to scanning samples/**/*.pt.",
    )
    parser.add_argument(
        "--input_subdir",
        type=Path,
        default=None,
        help="Optional relative subdirectory under latent_root to scan, e.g. samples/clean/clean/rotate_qrcode.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of samples to decode; 0 means all.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--decode_chunk", type=int, default=32)
    parser.add_argument("--view_count", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def read_manifest_records(paths: list[Path]) -> list[dict]:
    records: list[dict] = []
    seen: set[str] = set()
    for path in paths:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rel = rec.get("file") or rec.get("latent_file")
                if not rel or rel in seen:
                    continue
                seen.add(rel)
                records.append(rec)
    return records


def scan_records(latent_root: Path, input_subdir: Path | None = None) -> list[dict]:
    records = []
    scan_root = resolve_under(latent_root, input_subdir) if input_subdir else (latent_root / "samples")
    if not scan_root.exists():
        raise FileNotFoundError(f"scan root does not exist: {scan_root}")
    for path in sorted(scan_root.rglob("*.pt")):
        records.append({"file": str(path.relative_to(latent_root))})
    return records


def resolve_under(root: Path, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


def output_path_for(out_root: Path, rel_file: Path) -> Path:
    return (out_root / rel_file).with_suffix(".mp4")


def load_svd_vae(svd_path: Path, device: torch.device, dtype: torch.dtype):
    kwargs = {"subfolder": "vae", "torch_dtype": dtype}
    try:
        vae = AutoencoderKLTemporalDecoder.from_pretrained(str(svd_path), **kwargs)
    except OSError as exc:
        print(f"[WARN] Default VAE weights not found, retrying variant=fp16: {exc}", flush=True)
        vae = AutoencoderKLTemporalDecoder.from_pretrained(str(svd_path), variant="fp16", **kwargs)
    return vae.to(device=device, dtype=dtype).eval()


def load_latent(path: Path) -> torch.Tensor:
    data = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        latent = data.get("latent")
    else:
        latent = data
    if latent is None:
        raise KeyError(f"latent key not found in {path}")
    if not torch.is_tensor(latent):
        latent = torch.as_tensor(latent)
    latent = latent.float()
    if latent.ndim != 4:
        raise RuntimeError(f"expected latent [T,C,H,W], got {tuple(latent.shape)} from {path}")
    return latent


@torch.no_grad()
def decode_latents(vae, latents: torch.Tensor, decode_chunk: int) -> np.ndarray:
    decoded = []
    for start in range(0, latents.shape[0], decode_chunk):
        z = latents[start : start + decode_chunk] / vae.config.scaling_factor
        frame = vae.decode(z.to(device=vae.device, dtype=vae.dtype), num_frames=z.shape[0]).sample
        decoded.append(frame)
    video = torch.cat(decoded, dim=0)
    video = ((video / 2 + 0.5).clamp(0, 1) * 255)
    return video.float().cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8)


@torch.no_grad()
def decode_three_view_video(vae, latent: torch.Tensor, view_count: int, decode_chunk: int) -> np.ndarray:
    if latent.shape[2] % view_count != 0:
        raise RuntimeError(f"latent height {latent.shape[2]} is not divisible by view_count={view_count}")
    views = einops.rearrange(latent, "t c (v h) w -> (v t) c h w", v=view_count)
    decoded = decode_latents(vae, views, decode_chunk)
    frames = latent.shape[0]
    decoded = decoded.reshape(view_count, frames, *decoded.shape[1:])
    return np.concatenate([decoded[i] for i in range(view_count)], axis=2)


def main() -> None:
    args = parse_args()
    latent_root = args.latent_root.resolve()
    out_root = args.out_root or latent_root.with_name(f"{latent_root.name}_decoded_videos")
    out_root.mkdir(parents=True, exist_ok=True)

    records = read_manifest_records(args.manifest) if args.manifest else scan_records(latent_root, args.input_subdir)
    if args.shard_count <= 0:
        raise ValueError("--shard_count must be positive")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise ValueError("--shard_index must be in [0, shard_count)")
    if args.shard_count > 1:
        records = [rec for i, rec in enumerate(records) if i % args.shard_count == args.shard_index]
    records = records[int(args.offset) :]
    if args.limit > 0:
        records = records[: int(args.limit)]
    if not records:
        raise RuntimeError(f"no latent records found under {latent_root}")

    print(f"[decode] latent_root={latent_root}", flush=True)
    print(f"[decode] out_root={out_root}", flush=True)
    print(
        f"[decode] records={len(records)} shard={args.shard_index}/{args.shard_count} "
        f"dry_run={args.dry_run} force={args.force}",
        flush=True,
    )

    if args.dry_run:
        for rec in records[:20]:
            rel = Path(rec["file"])
            print(f"[dry-run] {resolve_under(latent_root, rel)} -> {output_path_for(out_root, rel)}", flush=True)
        return

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    vae = load_svd_vae(args.svd_path, device, dtype)

    summary = []
    for idx, rec in enumerate(records):
        rel = Path(rec["file"])
        src = resolve_under(latent_root, rel)
        dst = output_path_for(out_root, rel)
        if dst.exists() and not args.force:
            print(f"[skip] {idx + 1}/{len(records)} {dst}", flush=True)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        latent = load_latent(src)
        video = decode_three_view_video(vae, latent.to(device=device), args.view_count, args.decode_chunk)
        mediapy.write_video(str(dst), video, fps=args.fps)
        info = {
            "index": idx,
            "source": str(src),
            "output": str(dst),
            "frames": int(video.shape[0]),
            "height": int(video.shape[1]),
            "width": int(video.shape[2]),
            "fps": int(args.fps),
        }
        summary.append(info)
        print(f"[done] {idx + 1}/{len(records)} {dst} frames={video.shape[0]}", flush=True)

    if args.shard_count > 1:
        summary_path = out_root / f"decode_summary_shard{args.shard_index:05d}-of-{args.shard_count:05d}.json"
    else:
        summary_path = out_root / "decode_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"[decode] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
