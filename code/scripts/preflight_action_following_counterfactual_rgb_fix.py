"""Fail-closed RGB repair checks for state-major Counterfactual Ctrl-World caches."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path

import h5py
from PIL import Image


EXPECTED_TRAIN = 50
EXPECTED_QUICK = 200
CAMERA_KEYS = ("head_camera", "left_camera", "right_camera")
DEFAULT_REPAIR_ROOT = Path(
    "/mnt/dataset/public_data/cscsx_projects/AF3/counterfactual_replay_state_major/"
    "rgb_fix_20260719"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("source", "cache"))
    parser.add_argument("--task", required=True)
    parser.add_argument("--repair_root", default=str(DEFAULT_REPAIR_ROOT))
    parser.add_argument("--out", required=True)
    parser.add_argument("--enhanced_split_root")
    parser.add_argument("--train_selection_manifest")
    parser.add_argument("--quick_selection_manifest")
    parser.add_argument("--latent_root")
    parser.add_argument("--expected_origin", default="state_major_rgbfix_20260719")
    return parser.parse_args()


def read_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def read_jsonl(path):
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_family(row):
    text = f"{row.get('family', '')} {row.get('asset_id', '')} {row.get('subtype', '')}".lower()
    if "counterfactual" in text:
        return "counterfactual_replay"
    return str(row.get("family", ""))


def require_global_evidence(repair_root):
    repair_root = Path(repair_root).resolve()
    repair_summary_path = repair_root / "hdf5_repair_inplace_logs_full_4w/repair_summary.json"
    hdf5_validation_summary_path = (
        repair_root / "hdf5_repair_inplace_logs_full_4w/head_first_frame_rgb_validation_summary.json"
    )
    hdf5_validation_path = (
        repair_root / "hdf5_repair_inplace_logs_full_4w/head_first_frame_rgb_validation.jsonl"
    )
    lerobot_summary_path = repair_root / "lerobot_wrist_rebuild_rot6d_full_32w/summary_s000_of_001.json"
    lerobot_validation_summary_path = (
        repair_root / "lerobot_wrist_rebuild_rot6d_full_32w/first_frame_color_validation_summary.json"
    )

    repair = read_json(repair_summary_path)
    hdf5_validation = read_json(hdf5_validation_summary_path)
    lerobot = read_json(lerobot_summary_path)
    lerobot_validation = read_json(lerobot_validation_summary_path)
    checks = (
        (repair.get("status") == "ok", "HDF5 repair status"),
        (repair.get("rows") == 12500, "HDF5 repair row count"),
        (not repair.get("errors"), "HDF5 repair errors"),
        (hdf5_validation.get("status") == "ok", "HDF5 validation status"),
        (hdf5_validation.get("processed") == 12500, "HDF5 validation count"),
        (hdf5_validation.get("counts", {}).get("ok") == 12500, "HDF5 validation ok count"),
        (not hdf5_validation.get("bad_examples"), "HDF5 validation bad examples"),
        (lerobot.get("status") == "ok", "LeRobot rebuild status"),
        (lerobot.get("processed") == 5000, "LeRobot rebuild count"),
        (lerobot.get("counts", {}).get("ok") == 5000, "LeRobot rebuild ok count"),
        (lerobot_validation.get("status") == "ok", "LeRobot validation status"),
        (lerobot_validation.get("processed") == 5000, "LeRobot validation count"),
        (lerobot_validation.get("counts", {}).get("ok") == 5000, "LeRobot validation ok count"),
        (not lerobot_validation.get("bad_examples"), "LeRobot validation bad examples"),
    )
    failed = [label for passed, label in checks if not passed]
    if failed:
        raise RuntimeError(f"global RGB repair evidence failed: {failed}")

    validation_index = {}
    for row in read_jsonl(hdf5_validation_path):
        validation_index[os.path.realpath(row["hdf5_path"])] = row
    if len(validation_index) != 12500:
        raise RuntimeError(f"expected 12500 unique HDF5 validation rows, got {len(validation_index)}")
    evidence = {
        "repair_summary": str(repair_summary_path),
        "repair_summary_sha256": sha256(repair_summary_path),
        "hdf5_validation_summary": str(hdf5_validation_summary_path),
        "hdf5_validation_summary_sha256": sha256(hdf5_validation_summary_path),
        "hdf5_validation_rows": str(hdf5_validation_path),
        "hdf5_validation_rows_sha256": sha256(hdf5_validation_path),
        "lerobot_summary": str(lerobot_summary_path),
        "lerobot_summary_sha256": sha256(lerobot_summary_path),
        "lerobot_validation_summary": str(lerobot_validation_summary_path),
        "lerobot_validation_summary_sha256": sha256(lerobot_validation_summary_path),
    }
    return evidence, validation_index, hdf5_validation_summary_path.stat().st_mtime_ns


def resolve_sample_dir(row, enhanced_split_root):
    enhanced_split_root = Path(enhanced_split_root).resolve()
    sample_rel = row.get("quick_backing_path") or row.get("split_path")
    if not sample_rel:
        raise KeyError(f"missing split path: {row}")
    if sample_rel.startswith("enhanced_v1_split/"):
        sample_dir = enhanced_split_root.parent / sample_rel
    else:
        sample_dir = enhanced_split_root / sample_rel
    if not sample_dir.exists():
        family = canonical_family(row)
        subtype = row.get("subtype")
        split_name = "test_quick/backing_samples" if "quick_backing_path" in row else "train"
        parts = [split_name, family]
        if subtype:
            parts.append(str(subtype))
        parts.extend(("tasks", row["task"], row["sample_id"]))
        sample_dir = enhanced_split_root.joinpath(*parts)
    return sample_dir.resolve()


def external_source_path(rot6d_hdf5):
    with h5py.File(rot6d_hdf5, "r") as handle:
        observation_link = handle.get("observation", getlink=True)
        third_view_link = handle.get("third_view_rgb", getlink=True)
        if not isinstance(observation_link, h5py.ExternalLink):
            raise RuntimeError(f"observation is not an external link: {rot6d_hdf5}")
        if not isinstance(third_view_link, h5py.ExternalLink):
            raise RuntimeError(f"third_view_rgb is not an external link: {rot6d_hdf5}")
        source_path = Path(observation_link.filename)
        if not source_path.is_absolute():
            source_path = Path(rot6d_hdf5).parent / source_path
        third_path = Path(third_view_link.filename)
        if not third_path.is_absolute():
            third_path = Path(rot6d_hdf5).parent / third_path
        if os.path.realpath(source_path) != os.path.realpath(third_path):
            raise RuntimeError(f"camera external links disagree: {rot6d_hdf5}")
        if handle["delta_ee_action/vector"].shape[-1] != 20:
            raise RuntimeError(f"expected Rot6D20 action in {rot6d_hdf5}")
        for camera in CAMERA_KEYS:
            blob = handle[f"observation/{camera}/rgb"][0]
            with Image.open(io.BytesIO(blob)) as image:
                image.convert("RGB").load()
    return Path(os.path.realpath(source_path))


def source_audit(args, evidence, validation_index, evidence_mtime_ns):
    required = (args.enhanced_split_root, args.train_selection_manifest, args.quick_selection_manifest)
    if any(value is None for value in required):
        raise ValueError("source mode requires enhanced root and both selection manifests")
    outputs = {}
    all_sources = set()
    min_rgb_margin = None
    for split, manifest, expected in (
        ("train", args.train_selection_manifest, EXPECTED_TRAIN),
        ("test_quick", args.quick_selection_manifest, EXPECTED_QUICK),
    ):
        rows = [row for row in read_jsonl(manifest) if canonical_family(row) == "counterfactual_replay"]
        if len(rows) != expected:
            raise RuntimeError(f"{split} counterfactual count: expected={expected}, actual={len(rows)}")
        sample_dirs = set()
        for row in rows:
            if row.get("task") != args.task:
                raise RuntimeError(f"unexpected task in {manifest}: {row.get('task')}")
            sample_dir = resolve_sample_dir(row, args.enhanced_split_root)
            sample_dirs.add(sample_dir)
        for sample_dir in sorted(sample_dirs):
            contract = read_json(sample_dir / "rgb_contract_20260717.json")
            if contract.get("contract") != "semantic_rgb" or contract.get("rgb_swap_required") is not False:
                raise RuntimeError(f"invalid RGB contract: {sample_dir}")
            source_path = external_source_path(sample_dir / "data.hdf5")
            validation = validation_index.get(os.path.realpath(source_path))
            if validation is None or validation.get("status") != "ok":
                raise RuntimeError(f"missing passing repair validation for {source_path}")
            if validation.get("task") != args.task:
                raise RuntimeError(f"repair validation task mismatch for {source_path}")
            if source_path.stat().st_mtime_ns > evidence_mtime_ns:
                raise RuntimeError(f"HDF5 changed after RGB validation evidence: {source_path}")
            margin = float(validation["mad_rb_swap"]) - float(validation["mad_rgb"])
            if margin <= 0:
                raise RuntimeError(f"semantic RGB validation margin is not positive: {source_path}")
            min_rgb_margin = margin if min_rgb_margin is None else min(min_rgb_margin, margin)
            all_sources.add(str(source_path))
        outputs[split] = {
            "manifest": str(Path(manifest).resolve()),
            "manifest_sha256": sha256(manifest),
            "counterfactual_rows": len(rows),
            "unique_samples": len(sample_dirs),
        }
    return {
        "mode": "source",
        "status": "pass",
        "task": args.task,
        "repair_evidence": evidence,
        "splits": outputs,
        "unique_external_hdf5_sources": len(all_sources),
        "minimum_rgb_over_rb_margin": min_rgb_margin,
    }


def cache_audit(args, evidence, evidence_mtime_ns):
    if args.latent_root is None:
        raise ValueError("cache mode requires --latent_root")
    latent_root = Path(args.latent_root).resolve()
    if not (latent_root / ".incremental_manifests_ready").is_file():
        raise RuntimeError(f"incremental cache manifests are incomplete: {latent_root}")
    outputs = {}
    checked_files = set()
    for split, manifest_name, expected in (
        ("train", "train.jsonl", EXPECTED_TRAIN),
        ("test_quick", "test_quick.jsonl", EXPECTED_QUICK),
    ):
        manifest = latent_root / "manifests" / manifest_name
        rows = [row for row in read_jsonl(manifest) if canonical_family(row) == "counterfactual_replay"]
        if len(rows) != expected:
            raise RuntimeError(f"{split} counterfactual count: expected={expected}, actual={len(rows)}")
        for row in rows:
            if row.get("task") != args.task:
                raise RuntimeError(f"unexpected task in {manifest}: {row.get('task')}")
            if row.get("cache_origin") != args.expected_origin:
                raise RuntimeError(f"unexpected cache origin in {manifest}: {row.get('cache_origin')}")
            for field in ("file", "action_file"):
                value = Path(row[field])
                if value.is_absolute():
                    raise RuntimeError(f"RGB-fix counterfactual path must be task-local: {value}")
                path = (latent_root / value).resolve()
                try:
                    path.relative_to(latent_root)
                except ValueError as exc:
                    raise RuntimeError(f"counterfactual cache escapes RGB-fix root: {path}") from exc
                if not path.is_file() or path.stat().st_size <= 0:
                    raise FileNotFoundError(path)
                if path.stat().st_mtime_ns <= evidence_mtime_ns:
                    raise RuntimeError(f"counterfactual cache predates RGB validation evidence: {path}")
                checked_files.add(path)
        outputs[split] = {
            "manifest": str(manifest),
            "manifest_sha256": sha256(manifest),
            "counterfactual_rows": len(rows),
        }
    return {
        "mode": "cache",
        "status": "pass",
        "task": args.task,
        "repair_evidence": evidence,
        "latent_root": str(latent_root),
        "expected_origin": args.expected_origin,
        "checked_unique_files": len(checked_files),
        "splits": outputs,
    }


def main():
    args = parse_args()
    evidence, validation_index, evidence_mtime_ns = require_global_evidence(args.repair_root)
    if args.mode == "source":
        result = source_audit(args, evidence, validation_index, evidence_mtime_ns)
    else:
        result = cache_audit(args, evidence, evidence_mtime_ns)
    write_json_atomic(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
