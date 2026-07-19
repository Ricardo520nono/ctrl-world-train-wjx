"""Plan and finalize a task-local cache that reuses an older Ctrl-World cache."""

import argparse
import json
from collections import Counter
from pathlib import Path


FAMILIES = {
    "perturbed",
    "random_feasible",
    "counterfactual_replay",
    "exploration",
}
EXPECTED_TRAIN = {
    "perturbed": 5000,
    "random_feasible": 100,
    "counterfactual_replay": 50,
    "exploration": 5,
}
EXPECTED_QUICK = {family: 200 for family in FAMILIES}
EXPECTED_CLEAN = 50
PROVENANCE_FIELDS = (
    "asset_id",
    "source_group_id",
    "state_group",
    "action_group",
    "split_unit",
    "split_version",
)


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "finalize"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--task", required=True)
        subparser.add_argument("--old_cache_root", required=True)
        subparser.add_argument("--enhanced_split_root", required=True)
        subparser.add_argument("--out_root", required=True)
        if name == "finalize":
            subparser.add_argument("--num_shards", type=int, default=8)
    return parser.parse_args()


def read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl_atomic(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    tmp.replace(path)


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def canonical_family(row):
    text = f"{row.get('family', '')} {row.get('asset_id', '')} {row.get('subtype', '')}".lower()
    if "counterfactual" in text:
        return "counterfactual_replay"
    if "exploration" in text or "policy_rollout" in text:
        return "exploration"
    if "random_feasible" in text:
        return "random_feasible"
    if "perturbed" in text or "pca" in text or "raw" in text:
        return "perturbed"
    if "clean" in text:
        return "clean"
    return str(row.get("family", ""))


def record_key(row, quick=False):
    key = (
        str(row.get("task", "")),
        canonical_family(row),
        str(row.get("subtype") or ""),
        str(row.get("sample_id", "")),
    )
    if quick:
        start = row.get("chunk_start", row.get("fixed_start"))
        if start is None:
            raise ValueError(f"test_quick row lacks chunk_start/fixed_start: {row}")
        key += (int(start),)
    return key


def unique_map(rows, quick=False, label="manifest"):
    result = {}
    for row in rows:
        key = record_key(row, quick=quick)
        if key in result:
            raise RuntimeError(f"duplicate key in {label}: {key}")
        result[key] = row
    return result


def filter_task(rows, task):
    return [row for row in rows if row.get("task") == task]


def family_counts(rows):
    return dict(sorted(Counter(canonical_family(row) for row in rows).items()))


def require_counts(rows, expected, label):
    actual = Counter(canonical_family(row) for row in rows)
    if dict(actual) != expected:
        raise RuntimeError(f"{label} family counts mismatch: expected={expected}, actual={dict(actual)}")


def require_state_major(rows, label):
    bad = []
    for row in rows:
        if canonical_family(row) != "counterfactual_replay":
            continue
        if "state_major" not in str(row.get("asset_id", "")) or not row.get("state_group"):
            bad.append(row.get("sample_id"))
    if bad:
        raise RuntimeError(f"{label} has non-state-major counterfactual rows: {bad[:5]}")


def current_rows(enhanced_root, task):
    manifest_root = Path(enhanced_root) / "manifests"
    train = filter_task(read_jsonl(manifest_root / "train_samples.jsonl"), task)
    quick = filter_task(read_jsonl(manifest_root / "test_quick_chunks.jsonl"), task)
    require_counts(train, EXPECTED_TRAIN, f"{task} current train")
    require_counts(quick, EXPECTED_QUICK, f"{task} current test_quick")
    require_state_major(train, f"{task} current train")
    require_state_major(quick, f"{task} current test_quick")
    return train, quick


def old_rows(old_root, task):
    manifest_root = Path(old_root) / "manifests"
    clean = filter_task(read_jsonl(manifest_root / "clean_train.jsonl"), task)
    enhanced = filter_task(read_jsonl(manifest_root / "enhanced_train.jsonl"), task)
    quick = filter_task(read_jsonl(manifest_root / "test_quick.jsonl"), task)
    if len(clean) != EXPECTED_CLEAN:
        raise RuntimeError(f"{task} old clean count mismatch: expected={EXPECTED_CLEAN}, actual={len(clean)}")
    return clean, enhanced, quick


def split_reuse_delta(current, old, quick=False):
    old_map = unique_map(old, quick=quick, label="old cache")
    reuse = []
    delta = []
    for row in current:
        key = record_key(row, quick=quick)
        if canonical_family(row) != "counterfactual_replay" and key in old_map:
            reuse.append(row)
        else:
            delta.append(row)
    return reuse, delta


def expected_delta_counts(task):
    quick = {"counterfactual_replay": 200}
    if task == "place_burger_fries":
        quick["random_feasible"] = 10
    return {"counterfactual_replay": 50}, quick


def plan(args):
    old_root = Path(args.old_cache_root).resolve()
    enhanced_root = Path(args.enhanced_split_root).resolve()
    out_root = Path(args.out_root).resolve()
    current_train, current_quick = current_rows(enhanced_root, args.task)
    old_clean, old_train, old_quick = old_rows(old_root, args.task)
    reuse_train, delta_train = split_reuse_delta(current_train, old_train)
    reuse_quick, delta_quick = split_reuse_delta(current_quick, old_quick, quick=True)

    expected_train_delta, expected_quick_delta = expected_delta_counts(args.task)
    if family_counts(delta_train) != expected_train_delta:
        raise RuntimeError(
            f"{args.task} unexpected train delta: expected={expected_train_delta}, "
            f"actual={family_counts(delta_train)}"
        )
    if family_counts(delta_quick) != expected_quick_delta:
        raise RuntimeError(
            f"{args.task} unexpected quick delta: expected={expected_quick_delta}, "
            f"actual={family_counts(delta_quick)}"
        )

    selection_root = out_root / "selection_manifests"
    train_selection = selection_root / "train_delta_source.jsonl"
    quick_selection = selection_root / "test_quick_delta_source.jsonl"
    write_jsonl_atomic(train_selection, delta_train)
    write_jsonl_atomic(quick_selection, delta_quick)

    unique_delta = {
        (
            canonical_family(row),
            str(row.get("subtype") or ""),
            str(row.get("sample_id", "")),
        )
        for row in delta_train + delta_quick
    }
    summary = {
        "status": "ready",
        "task": args.task,
        "old_cache_root": str(old_root),
        "enhanced_split_root": str(enhanced_root),
        "out_root": str(out_root),
        "old_clean_records": len(old_clean),
        "current_train_records": len(current_train),
        "current_test_quick_records": len(current_quick),
        "reuse_train_records": len(reuse_train),
        "reuse_test_quick_records": len(reuse_quick),
        "delta_train_records": len(delta_train),
        "delta_train_family_counts": family_counts(delta_train),
        "delta_test_quick_records": len(delta_quick),
        "delta_test_quick_family_counts": family_counts(delta_quick),
        "delta_unique_trajectories": len(unique_delta),
        "train_selection_manifest": str(train_selection),
        "test_quick_selection_manifest": str(quick_selection),
    }
    write_json_atomic(out_root / "incremental_plan.json", summary)
    (out_root / ".incremental_plan_ready").write_text("ready\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


def absolutize(row, root):
    result = dict(row)
    for field in ("file", "latent_file", "action_file"):
        value = result.get(field)
        if value and not Path(value).is_absolute():
            result[field] = str((root / value).resolve())
    return result


def attach_provenance(cache_row, source_row, origin):
    result = dict(cache_row)
    result["cache_origin"] = origin
    for field in PROVENANCE_FIELDS:
        if field in source_row:
            result[field] = source_row[field]
    return result


def ensure_files(rows, out_root):
    checked = set()
    for row in rows:
        for field in ("file", "action_file"):
            value = row.get(field)
            if not value:
                raise RuntimeError(f"manifest row missing {field}: {row}")
            path = Path(value) if Path(value).is_absolute() else out_root / value
            path = path.resolve()
            if path in checked:
                continue
            if not path.is_file() or path.stat().st_size <= 0:
                raise FileNotFoundError(path)
            checked.add(path)
    return len(checked)


def finalize(args):
    if args.num_shards < 1:
        raise ValueError("--num_shards must be positive")
    old_root = Path(args.old_cache_root).resolve()
    enhanced_root = Path(args.enhanced_split_root).resolve()
    out_root = Path(args.out_root).resolve()
    manifest_root = out_root / "manifests"
    current_train, current_quick = current_rows(enhanced_root, args.task)
    old_clean, old_train, old_quick = old_rows(old_root, args.task)
    old_train_map = unique_map(old_train, label="old enhanced_train")
    old_quick_map = unique_map(old_quick, quick=True, label="old test_quick")

    delta_train = []
    delta_quick = []
    for shard in range(args.num_shards):
        suffix = f"delta_shard{shard:02d}"
        delta_train.extend(read_jsonl(manifest_root / f"enhanced_train_{suffix}.jsonl"))
        delta_quick.extend(read_jsonl(manifest_root / f"test_quick_{suffix}.jsonl"))
    delta_train_map = unique_map(delta_train, label="delta enhanced_train")
    delta_quick_map = unique_map(delta_quick, quick=True, label="delta test_quick")

    final_train = []
    train_origins = Counter()
    for source_row in current_train:
        key = record_key(source_row)
        if canonical_family(source_row) != "counterfactual_replay" and key in old_train_map:
            cache_row = absolutize(old_train_map[key], old_root)
            origin = "old_50task_cache"
        else:
            if key not in delta_train_map:
                raise RuntimeError(f"missing delta train row: {key}")
            cache_row = delta_train_map[key]
            origin = "state_major_incremental"
        final_train.append(attach_provenance(cache_row, source_row, origin))
        train_origins[origin] += 1

    final_quick = []
    quick_origins = Counter()
    for source_row in current_quick:
        key = record_key(source_row, quick=True)
        if canonical_family(source_row) != "counterfactual_replay" and key in old_quick_map:
            cache_row = absolutize(old_quick_map[key], old_root)
            origin = "old_50task_cache"
        else:
            if key not in delta_quick_map:
                raise RuntimeError(f"missing delta test_quick row: {key}")
            cache_row = delta_quick_map[key]
            origin = "state_major_incremental"
        final_quick.append(attach_provenance(cache_row, source_row, origin))
        quick_origins[origin] += 1

    final_clean = []
    for row in old_clean:
        cached = absolutize(row, old_root)
        cached["cache_origin"] = "old_50task_cache"
        final_clean.append(cached)

    require_counts(final_train, EXPECTED_TRAIN, f"{args.task} final enhanced_train")
    require_counts(final_quick, EXPECTED_QUICK, f"{args.task} final test_quick")
    if len(final_clean) != EXPECTED_CLEAN:
        raise RuntimeError(f"{args.task} final clean count mismatch: {len(final_clean)}")
    all_train = final_clean + final_train
    checked_files = ensure_files(all_train + final_quick, out_root)

    write_jsonl_atomic(manifest_root / "clean_train.jsonl", final_clean)
    write_jsonl_atomic(manifest_root / "enhanced_train.jsonl", final_train)
    write_jsonl_atomic(manifest_root / "train.jsonl", all_train)
    write_jsonl_atomic(manifest_root / "test_quick.jsonl", final_quick)

    summary = {
        "status": "ready",
        "task": args.task,
        "old_cache_root": str(old_root),
        "out_root": str(out_root),
        "clean_train_records": len(final_clean),
        "enhanced_train_records": len(final_train),
        "train_records": len(all_train),
        "test_quick_records": len(final_quick),
        "enhanced_train_family_counts": family_counts(final_train),
        "test_quick_family_counts": family_counts(final_quick),
        "train_cache_origin_counts": dict(sorted(train_origins.items())),
        "test_quick_cache_origin_counts": dict(sorted(quick_origins.items())),
        "checked_unique_files": checked_files,
        "num_shards": args.num_shards,
    }
    write_json_atomic(out_root / "incremental_finalize_summary.json", summary)
    (out_root / ".incremental_manifests_ready").write_text("ready\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    args = parse_args()
    if args.command == "plan":
        plan(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
