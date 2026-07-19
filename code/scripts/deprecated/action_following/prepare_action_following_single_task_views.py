"""Deprecated task-local views, superseded by the incremental cache workflow."""

import argparse
import json
from collections import Counter
from pathlib import Path


MANIFEST_NAMES = ("clean_train", "enhanced_train", "train", "test_quick")
TRAIN_FAMILIES = {
    "clean",
    "perturbed",
    "random_feasible",
    "counterfactual_replay",
    "exploration",
}
QUICK_FAMILIES = TRAIN_FAMILIES - {"clean"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_root", required=True)
    parser.add_argument("--view_root", required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    return parser.parse_args()


def canonical_family(record):
    text = f"{record.get('family', '')} {record.get('asset_id', '')} {record.get('subtype', '')}".lower()
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
    return str(record.get("family", ""))


def read_jsonl(path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def make_paths_absolute(record, source_root):
    record = dict(record)
    for key in ("file", "latent_file", "action_file"):
        value = record.get(key)
        if value and not Path(value).is_absolute():
            record[key] = str((source_root / value).resolve())
    return record


def write_jsonl_atomic(path, records):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    tmp.replace(path)


def main():
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    source_manifests = source_root / "manifests"
    view_root = Path(args.view_root).resolve()

    missing = [name for name in MANIFEST_NAMES if not (source_manifests / f"{name}.jsonl").is_file()]
    if missing:
        raise FileNotFoundError(f"source manifests missing under {source_manifests}: {missing}")

    tasks = list(dict.fromkeys(args.tasks))
    if len(tasks) != len(args.tasks):
        raise ValueError(f"duplicate task names are not allowed: {args.tasks}")

    source_rows = {
        name: list(read_jsonl(source_manifests / f"{name}.jsonl"))
        for name in MANIFEST_NAMES
    }
    summaries = {}
    for task in tasks:
        task_root = view_root / task
        manifest_root = task_root / "manifests"
        manifest_root.mkdir(parents=True, exist_ok=True)
        task_summary = {"task": task, "source_root": str(source_root), "view_root": str(task_root)}

        filtered_by_name = {}
        for name, rows in source_rows.items():
            filtered = [
                make_paths_absolute(row, source_root)
                for row in rows
                if row.get("task") == task
            ]
            if not filtered:
                raise RuntimeError(f"task {task!r} has no rows in {name}.jsonl")
            write_jsonl_atomic(manifest_root / f"{name}.jsonl", filtered)
            filtered_by_name[name] = filtered
            task_summary[f"{name}_records"] = len(filtered)
            task_summary[f"{name}_family_counts"] = dict(
                sorted(Counter(canonical_family(row) for row in filtered).items())
            )

        train_families = set(task_summary["train_family_counts"])
        quick_families = set(task_summary["test_quick_family_counts"])
        if train_families != TRAIN_FAMILIES:
            raise RuntimeError(
                f"task {task!r} train families mismatch: expected={sorted(TRAIN_FAMILIES)}, "
                f"actual={sorted(train_families)}"
            )
        if quick_families != QUICK_FAMILIES:
            raise RuntimeError(
                f"task {task!r} test_quick families mismatch: expected={sorted(QUICK_FAMILIES)}, "
                f"actual={sorted(quick_families)}"
            )

        expected_train = len(filtered_by_name["clean_train"]) + len(filtered_by_name["enhanced_train"])
        if len(filtered_by_name["train"]) != expected_train:
            raise RuntimeError(
                f"task {task!r} train count mismatch: train={len(filtered_by_name['train'])}, "
                f"clean+enhanced={expected_train}"
            )

        summary_path = task_root / "manifest_view_summary.json"
        summary_path.write_text(json.dumps(task_summary, indent=2, sort_keys=True) + "\n")
        summaries[task] = task_summary
        print(json.dumps(task_summary, sort_keys=True), flush=True)

    view_root.mkdir(parents=True, exist_ok=True)
    (view_root / "summary.json").write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
