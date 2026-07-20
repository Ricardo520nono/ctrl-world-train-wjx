"""Merge audited task-local ActionFollowing caches into one multi-task cache view."""

import argparse
import json
from collections import Counter
from pathlib import Path


FAMILIES = {"clean", "perturbed", "random_feasible", "counterfactual_replay", "exploration"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_cache", action="append", help="task=/absolute/cache/root")
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--audit_only", action="store_true")
    parser.add_argument("--stat_path")
    parser.add_argument("--audit_out")
    return parser.parse_args()


def read_jsonl(path):
    rows = []
    with Path(path).open() as handle:
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


def absolutize(row, root):
    result = dict(row)
    for field in ("file", "latent_file", "action_file"):
        value = result.get(field)
        if value and not Path(value).is_absolute():
            result[field] = str((root / value).resolve())
    return result


def require_source_cache(task, root):
    required = (
        root / ".incremental_complete",
        root / "rgb_preflight_source.json",
        root / "rgb_preflight_cache.json",
        root / "manifests/train.jsonl",
        root / "manifests/test_quick.jsonl",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"task cache is incomplete for {task}: {missing}")
    for audit_name in ("rgb_preflight_source.json", "rgb_preflight_cache.json"):
        audit = json.loads((root / audit_name).read_text())
        if audit.get("status") != "pass" or audit.get("task") != task:
            raise RuntimeError(f"invalid {audit_name} for {task}: {audit}")


def ensure_manifest_files(rows):
    checked = set()
    for row in rows:
        for field in ("file", "action_file"):
            path = Path(row[field])
            if not path.is_absolute():
                raise RuntimeError(f"merged manifest path is not absolute: {path}")
            path = path.resolve()
            if not path.is_file() or path.stat().st_size <= 0:
                raise FileNotFoundError(path)
            checked.add(path)
    return len(checked)


def audit_existing(out_root, stat_path=None):
    out_root = Path(out_root).resolve()
    if not (out_root / ".merged_manifests_ready").is_file():
        raise RuntimeError(f"merged manifests are incomplete: {out_root}")
    summary = json.loads((out_root / "merge_summary.json").read_text())
    tasks = summary.get("tasks", [])
    if len(tasks) < 2 or len(tasks) != len(set(tasks)):
        raise RuntimeError(f"invalid merged task set: {tasks}")
    for task, root in summary.get("task_caches", {}).items():
        require_source_cache(task, Path(root).resolve())

    train = read_jsonl(out_root / "manifests/train.jsonl")
    quick = read_jsonl(out_root / "manifests/test_quick.jsonl")
    expected_train = {
        "clean": 50,
        "perturbed": 5000,
        "random_feasible": 100,
        "counterfactual_replay": 50,
        "exploration": 5,
    }
    expected_quick = {
        "perturbed": 200,
        "random_feasible": 200,
        "counterfactual_replay": 200,
        "exploration": 200,
    }
    counts = {}
    for task in tasks:
        train_counts = Counter(canonical_family(row) for row in train if row.get("task") == task)
        quick_counts = Counter(canonical_family(row) for row in quick if row.get("task") == task)
        if dict(train_counts) != expected_train:
            raise RuntimeError(f"merged train counts mismatch for {task}: {dict(train_counts)}")
        if dict(quick_counts) != expected_quick:
            raise RuntimeError(f"merged quick counts mismatch for {task}: {dict(quick_counts)}")
        bad_origins = [
            row.get("sample_id") for row in train + quick
            if row.get("task") == task
            and canonical_family(row) == "counterfactual_replay"
            and row.get("cache_origin") != "state_major_rgbfix_20260719"
        ]
        if bad_origins:
            raise RuntimeError(f"non-RGB-fix counterfactual cache rows for {task}: {bad_origins[:5]}")
        counts[task] = {"train": dict(train_counts), "test_quick": dict(quick_counts)}

    result = {
        "status": "pass",
        "out_root": str(out_root),
        "tasks": tasks,
        "counts": counts,
        "checked_unique_files": ensure_manifest_files(train + quick),
    }
    if stat_path:
        stat_path = Path(stat_path).resolve()
        stat = json.loads(stat_path.read_text())
        if stat.get("action_dim") != 20 or stat.get("records") != len(train):
            raise RuntimeError(f"invalid merged stat: {stat_path}")
        result["stat_path"] = str(stat_path)
        result["stat_records"] = stat["records"]
        result["action_dim"] = stat["action_dim"]
    return result


def main():
    args = parse_args()
    if args.audit_only:
        result = audit_existing(args.out_root, args.stat_path)
        if not args.audit_out:
            raise ValueError("--audit_only requires --audit_out")
        write_json_atomic(args.audit_out, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if not args.task_cache:
        raise ValueError("at least one --task_cache is required when not using --audit_only")
    task_caches = []
    for value in args.task_cache:
        if "=" not in value:
            raise ValueError(f"invalid --task_cache: {value}")
        task, root = value.split("=", 1)
        root = Path(root).resolve()
        require_source_cache(task, root)
        task_caches.append((task, root))
    tasks = [task for task, _ in task_caches]
    if len(tasks) != len(set(tasks)) or len(tasks) < 2:
        raise RuntimeError(f"expected at least two unique tasks, got {tasks}")

    out_root = Path(args.out_root).resolve()
    manifests = {}
    summary = {"status": "ready", "tasks": tasks, "task_caches": {}, "manifests": {}}
    for manifest_name in ("clean_train.jsonl", "enhanced_train.jsonl", "train.jsonl", "test_quick.jsonl"):
        combined = []
        keys = set()
        per_task_family = Counter()
        for task, root in task_caches:
            rows = read_jsonl(root / "manifests" / manifest_name)
            for row in rows:
                if row.get("task") != task:
                    raise RuntimeError(f"unexpected task in {root}/{manifest_name}: {row.get('task')}")
                row = absolutize(row, root)
                key = (
                    row.get("task"), canonical_family(row), row.get("subtype"), row.get("sample_id"),
                    row.get("chunk_start", row.get("fixed_start")),
                )
                if key in keys:
                    raise RuntimeError(f"duplicate merged manifest key: {key}")
                keys.add(key)
                combined.append(row)
                per_task_family[(task, canonical_family(row))] += 1
        manifests[manifest_name] = combined
        summary["manifests"][manifest_name] = {
            "records": len(combined),
            "per_task_family": {
                f"{task}/{family}": count
                for (task, family), count in sorted(per_task_family.items())
            },
        }

    for task, root in task_caches:
        summary["task_caches"][task] = str(root)
    train_families_by_task = {
        task: {canonical_family(row) for row in manifests["train.jsonl"] if row.get("task") == task}
        for task in tasks
    }
    for task, families in train_families_by_task.items():
        if families != FAMILIES:
            raise RuntimeError(f"merged train families mismatch for {task}: {families}")

    for manifest_name, rows in manifests.items():
        write_jsonl_atomic(out_root / "manifests" / manifest_name, rows)
    write_json_atomic(out_root / "merge_summary.json", summary)
    (out_root / ".merged_manifests_ready").write_text("ready\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
