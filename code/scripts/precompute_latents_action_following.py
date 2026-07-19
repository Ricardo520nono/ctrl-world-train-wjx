"""
Precompute Ctrl-World SVD latents for ActionFollowingData.

Inputs:
  - expert clean LeRobot root
  - enhanced_v1_split train/test_quick canonical root

Outputs:
  {out_root}/samples/<family>/<subtype>/<task>/<sample_id>.pt
  {out_root}/manifests/train.jsonl
  {out_root}/manifests/test_quick.jsonl

The output records are consumed by dataset.dataset_action_following.
"""
import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
from collections import defaultdict

import h5py
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.ee_targets import ee_target_from_hdf5


CAMERA_KEYS = ["head_camera", "left_camera", "right_camera"]
LEROBOT_VIDEO_KEYS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]
IMG_SIZE = (320, 240)
ASSET_FAMILY = {
    "perturbed_pca_c8_sigma0p05_v1": ("perturbed", "pca"),
    "perturbed_raw_sigma0p0025_v1": ("perturbed", "raw"),
    "random_feasible_300step_uniform_2ep5start_10seed_v1": ("random_feasible", "uniform"),
    "random_feasible_300step_weighted_2ep5start_10seed_v1": ("random_feasible", "weighted"),
    "counterfactual_replay_50task_20src5seed_v1": ("counterfactual_replay", None),
    "exploration_policy_rollout_starvla_official48_step150000_50task10traj_v1": ("exploration", "policy_rollout"),
}


def import_image_modules():
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Pillow is required in the training environment to precompute latents.") from exc
    return Image


def import_lerobot_modules():
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError("pandas/pyarrow are required to read LeRobot data.") from exc
    try:
        from decord import VideoReader, cpu
    except Exception:
        VideoReader = None
        cpu = None
    try:
        import imageio.v3 as iio
    except Exception:
        iio = None
    return pd, VideoReader, cpu, iio


def pil_to_tensor(img, Image):
    img = img.resize(IMG_SIZE, Image.BILINEAR).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr.transpose(2, 0, 1))


def load_vae(svd_path, device):
    from diffusers import AutoencoderKLTemporalDecoder
    try:
        vae = AutoencoderKLTemporalDecoder.from_pretrained(svd_path, subfolder="vae")
    except OSError as exc:
        print(f"[WARN] Default SVD VAE weights not found, retrying variant=fp16: {exc}", flush=True)
        vae = AutoencoderKLTemporalDecoder.from_pretrained(svd_path, subfolder="vae", variant="fp16")
    return vae.to(device).eval()


def encode_frames(vae, frames, device, batch_size, Image):
    encoded = []
    for start in range(0, len(frames), batch_size):
        batch = frames[start:start + batch_size]
        tensors = torch.stack([pil_to_tensor(img, Image) for img in batch]).to(device)
        with torch.no_grad():
            latent = vae.encode(tensors).latent_dist.sample() * vae.config.scaling_factor
        encoded.append(latent.cpu())
        del tensors, latent
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return torch.cat(encoded, dim=0)


def stack_camera_latents(cam_latents):
    return torch.cat([cam_latents[key] for key in CAMERA_KEYS], dim=2)


def safe_text_from_instruction(path, fallback):
    if not os.path.exists(path):
        return fallback
    try:
        data = json.load(open(path))
    except Exception:
        return fallback
    for key in ["seen", "unseen"]:
        values = data.get(key)
        if isinstance(values, list) and values:
            return str(values[0])
    value = data.get("instruction")
    if value:
        return str(value)
    return fallback


def task_full_description(task, instruction_root):
    fallback = task.replace("_", " ")
    if not instruction_root:
        return fallback
    path = os.path.join(instruction_root, f"{task}.json")
    if not os.path.exists(path):
        return fallback
    try:
        data = json.load(open(path))
    except Exception:
        return fallback
    value = data.get("full_description")
    if value:
        return str(value)
    return fallback


def enhanced_family_subtype(asset_id, family, subtype):
    if family:
        family = str(family)
    if family in {"perturbed", "random_feasible", "counterfactual_replay", "exploration"}:
        return family, subtype
    if asset_id in ASSET_FAMILY:
        return ASSET_FAMILY[asset_id]
    text = f"{asset_id} {family}".lower()
    if "counterfactual" in text:
        return "counterfactual_replay", None
    if "exploration" in text or "policy_rollout" in text:
        return "exploration", "policy_rollout"
    if "random_feasible" in text:
        return "random_feasible", "weighted" if "weighted" in text else "uniform"
    if "perturbed" in text or "pca" in text or "raw" in text:
        return "perturbed", "raw" if "raw" in text else "pca"
    raise ValueError(f"Cannot infer family/subtype for asset={asset_id}, family={family}, subtype={subtype}")


def load_enhanced_sample(sample_dir, vae, device, batch_size, Image):
    h5_path = os.path.join(sample_dir, "data.hdf5")
    with h5py.File(h5_path, "r") as f:
        action_npy = os.path.join(sample_dir, "action.npy")
        if os.path.exists(action_npy):
            action_pos = np.load(action_npy).astype(np.float32)
        elif "delta_ee_action/vector" in f:
            action_pos = np.array(f["delta_ee_action/vector"], dtype=np.float32)
        else:
            raise FileNotFoundError(f"missing action.npy and delta_ee_action/vector in {sample_dir}")
        ee_target = ee_target_from_hdf5(f)
        T = min(len(ee_target), len(action_pos) + 1)
        if T <= 0:
            raise RuntimeError(f"empty observation/action sequence in {sample_dir}")
        cam_latents = {}
        for cam_key in CAMERA_KEYS:
            rgb_bytes = f[f"observation/{cam_key}/rgb"][:T]
            frames = [Image.open(io.BytesIO(blob)).convert("RGB") for blob in rgb_bytes]
            cam_latents[cam_key] = encode_frames(vae, frames, device, batch_size, Image)
    latent = stack_camera_latents(cam_latents)
    return latent, action_pos[:T], ee_target[:T]


_VIDEO_FRAME_COUNT_CACHE = {}


def video_frame_count(video_path, VideoReader=None, cpu=None, iio=None):
    video_path = os.path.abspath(video_path)
    if video_path in _VIDEO_FRAME_COUNT_CACHE:
        return _VIDEO_FRAME_COUNT_CACHE[video_path]
    if VideoReader is not None:
        try:
            count = len(VideoReader(video_path, ctx=cpu(0)))
            _VIDEO_FRAME_COUNT_CACHE[video_path] = int(count)
            return int(count)
        except Exception:
            pass
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=nb_read_frames,nb_frames",
                "-of",
                "default=nw=1:nk=1",
                video_path,
            ],
            text=True,
            timeout=60,
        )
        for line in out.splitlines():
            line = line.strip()
            if line and line != "N/A":
                count = int(float(line))
                _VIDEO_FRAME_COUNT_CACHE[video_path] = count
                return count
    except Exception:
        pass
    if iio is not None:
        try:
            count = sum(1 for _ in iio.imiter(video_path))
            _VIDEO_FRAME_COUNT_CACHE[video_path] = int(count)
            return int(count)
        except Exception:
            pass
    raise RuntimeError(f"unable to determine frame count for {video_path}")


def discover_lerobot_video_chunks(video_dir, VideoReader=None, cpu=None, iio=None):
    if not os.path.isdir(video_dir):
        raise FileNotFoundError(f"missing LeRobot video directory: {video_dir}")
    video_paths = []
    for root, _, files in os.walk(video_dir):
        for name in files:
            if name.endswith(".mp4"):
                video_paths.append(os.path.join(root, name))
    video_paths = sorted(video_paths)
    if not video_paths:
        raise FileNotFoundError(f"no mp4 files under LeRobot video directory: {video_dir}")
    chunks = []
    offset = 0
    for path in video_paths:
        count = video_frame_count(path, VideoReader, cpu, iio)
        chunks.append((path, offset, offset + count))
        offset += count
    return chunks


def read_video_frames(video_path, frame_indices, Image, VideoReader=None, cpu=None, iio=None):
    if VideoReader is not None:
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            return [Image.fromarray(vr[int(idx)].asnumpy()).convert("RGB") for idx in frame_indices]
        except Exception as exc:
            if iio is None:
                raise
            print(f"[WARN] decord failed for {video_path}; falling back to imageio: {exc}", flush=True)
    if iio is not None:
        requested = [int(idx) for idx in frame_indices]
        frames = []
        failed_random_access = False
        for idx in requested:
            try:
                frame = iio.imread(video_path, index=int(idx))
            except Exception as exc:
                print(
                    f"[WARN] imageio random access failed for {video_path} frame={int(idx)}; "
                    f"falling back to one-pass sequential decode: {exc}",
                    flush=True,
                )
                failed_random_access = True
                break
            frames.append(Image.fromarray(frame).convert("RGB"))
        if failed_random_access:
            wanted = set(requested)
            found = {}
            max_wanted = max(wanted) if wanted else -1
            for seq_idx, seq_frame in enumerate(iio.imiter(video_path)):
                if seq_idx in wanted:
                    found[seq_idx] = Image.fromarray(seq_frame).convert("RGB")
                    if len(found) == len(wanted):
                        break
                if seq_idx > max_wanted:
                    break
            missing = [idx for idx in requested if idx not in found]
            if missing:
                raise RuntimeError(f"unable to decode frame {missing[0]} from {video_path}")
            frames = [found[idx] for idx in requested]
        return frames
    raise RuntimeError("Either decord or imageio is required to read LeRobot videos.")


def read_lerobot_video_frames(video_dir, frame_indices, Image, VideoReader=None, cpu=None, iio=None):
    chunks = discover_lerobot_video_chunks(video_dir, VideoReader, cpu, iio)
    total_frames = chunks[-1][2]
    requested = [int(idx) for idx in frame_indices]
    if requested and (min(requested) < 0 or max(requested) >= total_frames):
        raise RuntimeError(
            f"requested frame range [{min(requested)}, {max(requested)}] exceeds "
            f"{total_frames} frames under {video_dir}"
        )
    grouped = defaultdict(list)
    positions = defaultdict(list)
    for pos, idx in enumerate(requested):
        for path, start, end in chunks:
            if start <= idx < end:
                grouped[path].append(idx - start)
                positions[path].append(pos)
                break
        else:
            raise RuntimeError(f"unable to map global frame {idx} under {video_dir}")
    frames = [None] * len(requested)
    for path, local_indices in grouped.items():
        local_frames = read_video_frames(path, local_indices, Image, VideoReader, cpu, iio)
        for pos, frame in zip(positions[path], local_frames):
            frames[pos] = frame
    missing = [i for i, frame in enumerate(frames) if frame is None]
    if missing:
        raise RuntimeError(f"failed to read {len(missing)} frames under {video_dir}; first missing position={missing[0]}")
    return frames


def load_lerobot_episode(task_root, video_task_root, episode_idx, vae, device, batch_size, Image, pd, VideoReader, cpu, iio):
    parquet_path = os.path.join(task_root, "data", "chunk-000", "file-000.parquet")
    df = pd.read_parquet(parquet_path)
    ep_df = df[df["episode_index"] == episode_idx].sort_values("frame_index")
    if ep_df.empty:
        raise RuntimeError(f"episode {episode_idx} not found in {parquet_path}")
    action_all = np.stack(ep_df["action"].to_numpy()).astype(np.float32)
    if len(action_all) <= 1:
        raise RuntimeError(f"episode {episode_idx} is too short in {parquet_path}")
    action_pos = action_all[:-1]

    cam_latents = {}
    # LeRobot videos may be split across multiple mp4 chunk files. `frame_index`
    # resets inside each episode, while `index` is the global frame position in
    # the concatenated task video stream. Using `frame_index` here repeats the
    # beginning of file-000.mp4 for every episode; using `index` without chunk
    # mapping fails once a camera has file-001.mp4.
    video_frame_col = "index" if "index" in ep_df.columns else "frame_index"
    video_frame_indices = ep_df[video_frame_col].astype(int).to_numpy()
    for video_key, cam_key in zip(LEROBOT_VIDEO_KEYS, CAMERA_KEYS):
        video_dir = os.path.join(video_task_root, "videos", video_key)
        frames = read_lerobot_video_frames(video_dir, video_frame_indices, Image, VideoReader, cpu, iio)
        cam_latents[cam_key] = encode_frames(vae, frames, device, batch_size, Image)

    latent = stack_camera_latents(cam_latents)
    return latent, action_pos, None


def ensure_out_path(out_root, family, subtype, task, sample_id):
    parts = [out_root, "samples", family]
    if subtype:
        parts.append(subtype)
    parts.extend([task, f"{sample_id}.pt"])
    out_file = os.path.join(*parts)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    return out_file


def relpath(path, root):
    return os.path.relpath(path, root)


def write_record(out_file, latent, action_pos, ee_target, text, record):
    action_pos = np.asarray(action_pos, dtype=np.float32)
    payload = {
        "latent": latent,
        "action_pos": action_pos,
        "text": text,
        "record": record,
        "latent_length": int(latent.shape[0]),
        "action_length": int(action_pos.shape[0]),
    }
    if ee_target is not None:
        payload["ee_target"] = np.asarray(ee_target, dtype=np.float32)
    torch.save(payload, out_file)
    np.save(action_sidecar_path(out_file), action_pos)


def action_sidecar_path(out_file):
    return out_file + ".action.npy"


def ensure_action_sidecar(out_file):
    sidecar = action_sidecar_path(out_file)
    if os.path.exists(sidecar):
        return
    data = torch.load(out_file, map_location="cpu", weights_only=False)
    np.save(sidecar, np.asarray(data["action_pos"], dtype=np.float32))


def append_jsonl(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def already_done(out_file, overwrite):
    return os.path.exists(out_file) and not overwrite


def selected_tasks(all_tasks, tasks):
    if not tasks:
        return all_tasks
    wanted = set(tasks)
    return [task for task in all_tasks if task in wanted]


def load_manifest_records(path, tasks=None, limit_per_family_task=None):
    counts = defaultdict(int)
    out = []
    for rec in read_jsonl(path):
        task = rec.get("task")
        if tasks and task not in tasks:
            continue
        family, subtype = enhanced_family_subtype(rec.get("asset_id"), rec.get("family"), rec.get("subtype"))
        key = (family, subtype or "default", task)
        if limit_per_family_task is not None and counts[key] >= limit_per_family_task:
            continue
        rec = dict(rec)
        rec["_family"] = family
        rec["_subtype"] = subtype
        counts[key] += 1
        out.append(rec)
    return out


def stable_mod(text, num_shards):
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest, 16) % num_shards


def shard_records(records, num_shards, shard_index, key_fn):
    if num_shards <= 1:
        return records
    return [rec for rec in records if stable_mod(key_fn(rec), num_shards) == shard_index]


def manifest_with_suffix(name, suffix):
    if not suffix:
        return name
    stem, ext = os.path.splitext(name)
    return f"{stem}_{suffix}{ext}"


def read_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def process_enhanced(records, enhanced_split_root, out_root, manifest_name, vae, device, batch_size, Image, overwrite, task_instruction_root):
    manifest_path = os.path.join(out_root, "manifests", manifest_name)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w"):
        pass
    processed = 0
    skipped = 0
    for i, rec in enumerate(records, 1):
        family = rec["_family"]
        subtype = rec["_subtype"]
        task = rec["task"]
        sample_id = rec["sample_id"]
        sample_rel = rec.get("quick_backing_path") or rec.get("split_path")
        if not sample_rel:
            raise KeyError(f"missing split path in record: {rec}")
        if sample_rel.startswith("enhanced_v1_split/"):
            action_following_root = os.path.dirname(enhanced_split_root)
            sample_dir = os.path.join(action_following_root, sample_rel)
        else:
            sample_dir = os.path.join(enhanced_split_root, sample_rel)
        if not os.path.exists(sample_dir):
            split_name = "test_quick/backing_samples" if "quick_backing_path" in rec else "train"
            sample_dir = os.path.join(
                enhanced_split_root,
                split_name,
                family,
                *([subtype] if subtype else []),
                "tasks",
                task,
                sample_id,
            )
        text = task_full_description(task, task_instruction_root)
        out_file = ensure_out_path(out_root, family, subtype, task, sample_id)
        if already_done(out_file, overwrite):
            ensure_action_sidecar(out_file)
            data = torch.load(out_file, map_location="cpu", weights_only=False)
            action_length = int(len(np.asarray(data["action_pos"])))
            latent_length = int(data["latent"].shape[0])
            skipped += 1
        else:
            latent, action_pos, ee_target = load_enhanced_sample(sample_dir, vae, device, batch_size, Image)
            write_record(out_file, latent, action_pos, ee_target, text, rec)
            action_length = int(len(action_pos))
            latent_length = int(latent.shape[0])
            processed += 1

        fixed_start = rec.get("chunk_start")
        unit_level = "chunk" if family == "perturbed" else "trajectory"
        manifest_rec = {
            "file": relpath(out_file, out_root),
            "action_file": relpath(action_sidecar_path(out_file), out_root),
            "family": family,
            "subtype": subtype,
            "task": task,
            "sample_id": sample_id,
            "length": action_length,
            "action_length": action_length,
            "latent_length": latent_length,
            "unit_level": unit_level,
            "text": text,
        }
        if fixed_start is not None:
            manifest_rec["chunk_start"] = int(fixed_start)
            manifest_rec["fixed_start"] = int(fixed_start)
        append_jsonl(manifest_path, manifest_rec)
        if i % 100 == 0:
            print(f"[enhanced:{manifest_name}] {i}/{len(records)} processed={processed} skipped={skipped}", flush=True)
    return {"records": len(records), "processed": processed, "skipped": skipped, "manifest": manifest_path}


def load_clean_episode_infos(clean_root, tasks, limit_per_task, task_instruction_root):
    records = []
    for task in tasks:
        task_root = os.path.join(clean_root, task)
        info_path = os.path.join(task_root, "meta", "delta_ee_summary.json")
        if os.path.exists(info_path):
            rows = json.load(open(info_path))
        else:
            parquet_path = os.path.join(task_root, "data", "chunk-000", "file-000.parquet")
            if not os.path.exists(parquet_path):
                raise FileNotFoundError(f"missing clean summary and parquet: {info_path}, {parquet_path}")
            import pandas as pd
            df = pd.read_parquet(parquet_path, columns=["episode_index", "frame_index"])
            rows = []
            for ep, ep_df in df.groupby("episode_index", sort=True):
                rows.append({
                    "episode_index": int(ep),
                    "frames": int(len(ep_df)),
                    "actions": max(0, int(len(ep_df)) - 1),
                    "instruction": task_full_description(task, task_instruction_root),
                })
        for row in rows[:limit_per_task if limit_per_task is not None else len(rows)]:
            ep = int(row["episode_index"])
            length = int(row.get("actions") or max(0, int(row.get("frames", 0)) - 1))
            records.append({
                "family": "clean",
                "subtype": "clean",
                "task": task,
                "episode_index": ep,
                "sample_id": f"episode_{ep:04d}",
                "length": length,
                "text": task_full_description(task, task_instruction_root),
            })
    return records


def process_clean(records, clean_root, clean_video_root, out_root, manifest_name, vae, device, batch_size, Image, pd, VideoReader, cpu, iio, overwrite):
    manifest_path = os.path.join(out_root, "manifests", manifest_name)
    if os.path.exists(manifest_path):
        os.remove(manifest_path)
    processed = 0
    skipped = 0
    for i, rec in enumerate(records, 1):
        out_file = ensure_out_path(out_root, "clean", "clean", rec["task"], rec["sample_id"])
        if already_done(out_file, overwrite):
            ensure_action_sidecar(out_file)
            data = torch.load(out_file, map_location="cpu", weights_only=False)
            rec["length"] = int(len(np.asarray(data["action_pos"])))
            latent_length = int(data["latent"].shape[0])
            skipped += 1
        else:
            task_root = os.path.join(clean_root, rec["task"])
            video_task_root = os.path.join(clean_video_root, rec["task"])
            latent, action_pos, ee_target = load_lerobot_episode(
                task_root, video_task_root, rec["episode_index"], vae, device, batch_size, Image, pd, VideoReader, cpu, iio
            )
            write_record(out_file, latent, action_pos, ee_target, rec["text"], rec)
            rec["length"] = len(action_pos)
            latent_length = int(latent.shape[0])
            processed += 1
        manifest_rec = {
            "file": relpath(out_file, out_root),
            "action_file": relpath(action_sidecar_path(out_file), out_root),
            "family": "clean",
            "subtype": "clean",
            "task": rec["task"],
            "sample_id": rec["sample_id"],
            "episode_index": rec["episode_index"],
            "length": int(rec["length"]),
            "action_length": int(rec["length"]),
            "latent_length": latent_length,
            "unit_level": "trajectory",
            "text": rec["text"],
        }
        append_jsonl(manifest_path, manifest_rec)
        if i % 10 == 0:
            print(f"[clean:{manifest_name}] {i}/{len(records)} processed={processed} skipped={skipped}", flush=True)
    return {"records": len(records), "processed": processed, "skipped": skipped, "manifest": manifest_path}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--svd_path", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--enhanced_split_root", default="/mnt/dataset/csx_workspace/Ideas/data/ActionFollowingData/enhanced_v1_split")
    parser.add_argument("--enhanced_train_manifest_path", default=None)
    parser.add_argument("--enhanced_test_quick_manifest_path", default=None)
    parser.add_argument("--clean_lerobot_root", default="/mnt/dataset/csx_workspace/Ideas/data/ActionFollowingBench/data_lerobot/robotwin_delta_ee/demo_clean_zed2i_visible")
    parser.add_argument("--clean_lerobot_video_root", default=None)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--split", choices=["train", "test_quick", "both"], default="both")
    parser.add_argument("--include_clean", action="store_true")
    parser.add_argument("--include_enhanced", action="store_true")
    parser.add_argument("--limit_per_family_task", type=int, default=None)
    parser.add_argument("--clean_limit_per_task", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--task_instruction_root", default="/mnt/dataset/csx_workspace/Ideas/AF3/code/RoboTwin/description/task_instruction")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--manifest_suffix", default="")
    parser.add_argument("--skip_train_merge", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard_index must be in [0, num_shards)")

    include_clean = args.include_clean or not args.include_enhanced
    include_enhanced = args.include_enhanced or not args.include_clean
    clean_video_root = args.clean_lerobot_video_root or args.clean_lerobot_root
    os.makedirs(args.out_root, exist_ok=True)

    Image = import_image_modules()
    pd = VideoReader = cpu = iio = None
    if include_clean:
        pd, VideoReader, cpu, iio = import_lerobot_modules()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading SVD VAE on {device}")
    vae = load_vae(args.svd_path, device)

    summary = {}
    if include_clean and args.split in {"train", "both"}:
        clean_records = load_clean_episode_infos(
            args.clean_lerobot_root, args.tasks, args.clean_limit_per_task, args.task_instruction_root
        )
        clean_records = shard_records(
            clean_records,
            args.num_shards,
            args.shard_index,
            lambda rec: f"clean|{rec['task']}|{rec['episode_index']}",
        )
        summary["clean_train"] = process_clean(
            clean_records, args.clean_lerobot_root, clean_video_root, args.out_root,
            manifest_with_suffix("clean_train.jsonl", args.manifest_suffix),
            vae, device, args.batch_size, Image, pd, VideoReader, cpu, iio, args.overwrite
        )

    if include_enhanced and args.split in {"train", "both"}:
        train_manifest = args.enhanced_train_manifest_path or os.path.join(
            args.enhanced_split_root, "manifests", "train_samples.jsonl"
        )
        if not os.path.isfile(train_manifest):
            raise FileNotFoundError(f"enhanced train manifest does not exist: {train_manifest}")
        records = load_manifest_records(train_manifest, set(args.tasks), args.limit_per_family_task)
        records = shard_records(
            records,
            args.num_shards,
            args.shard_index,
            lambda rec: f"{rec['_family']}|{rec['_subtype']}|{rec['task']}|{rec['sample_id']}",
        )
        summary["enhanced_train"] = process_enhanced(
            records, args.enhanced_split_root, args.out_root,
            manifest_with_suffix("enhanced_train.jsonl", args.manifest_suffix),
            vae, device, args.batch_size, Image, args.overwrite, args.task_instruction_root
        )

    if include_enhanced and args.split in {"test_quick", "both"}:
        quick_manifest = args.enhanced_test_quick_manifest_path or os.path.join(
            args.enhanced_split_root, "manifests", "test_quick_chunks.jsonl"
        )
        if not os.path.isfile(quick_manifest):
            raise FileNotFoundError(f"enhanced test_quick manifest does not exist: {quick_manifest}")
        records = load_manifest_records(quick_manifest, set(args.tasks), args.limit_per_family_task)
        records = shard_records(
            records,
            args.num_shards,
            args.shard_index,
            lambda rec: f"{rec['_family']}|{rec['_subtype']}|{rec['task']}|{rec['sample_id']}",
        )
        summary["test_quick"] = process_enhanced(
            records, args.enhanced_split_root, args.out_root,
            manifest_with_suffix("test_quick.jsonl", args.manifest_suffix),
            vae, device, args.batch_size, Image, args.overwrite, args.task_instruction_root
        )

    if args.split in {"train", "both"} and not args.skip_train_merge:
        train_out = os.path.join(args.out_root, "manifests", "train.jsonl")
        if os.path.exists(train_out):
            os.remove(train_out)
        for name in [
            manifest_with_suffix("clean_train.jsonl", args.manifest_suffix),
            manifest_with_suffix("enhanced_train.jsonl", args.manifest_suffix),
        ]:
            path = os.path.join(args.out_root, "manifests", name)
            if os.path.exists(path):
                with open(train_out, "a") as out, open(path) as src:
                    for line in src:
                        out.write(line)

    summary_name = manifest_with_suffix("precompute_summary.json", args.manifest_suffix)
    with open(os.path.join(args.out_root, summary_name), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
