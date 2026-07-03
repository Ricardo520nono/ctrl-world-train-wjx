"""Compute Ctrl-World action normalization stats from ActionFollowing latents."""
import argparse
import json
import os

import numpy as np


def read_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_root", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--action_dim", type=int, default=14)
    parser.add_argument("--max_records", type=int, default=0)
    args = parser.parse_args()

    manifest = args.manifest or os.path.join(args.latent_root, "manifests", "train.jsonl")
    actions = []
    records = 0
    frames = 0
    for rec in read_jsonl(manifest):
        if args.max_records and records >= args.max_records:
            break
        action_path = rec.get("action_file")
        if action_path:
            if not os.path.isabs(action_path):
                action_path = os.path.join(args.latent_root, action_path)
            action = np.load(action_path).astype(np.float32)[:, : args.action_dim]
        else:
            path = rec["file"]
            if not os.path.isabs(path):
                path = os.path.join(args.latent_root, path)
            import torch
            data = torch.load(path, map_location="cpu", weights_only=False)
            action = np.asarray(data["action_pos"], dtype=np.float32)[:, : args.action_dim]
        actions.append(action)
        records += 1
        frames += len(action)
        if records % 1000 == 0:
            print(f"[stat] records={records}, frames={frames}", flush=True)

    if not actions:
        raise RuntimeError(f"No actions found in manifest: {manifest}")
    all_actions = np.concatenate(actions, axis=0)
    p01 = np.percentile(all_actions, 1, axis=0).tolist()
    p99 = np.percentile(all_actions, 99, axis=0).tolist()

    os.makedirs(args.out_dir, exist_ok=True)
    out = {
        "state_01": p01,
        "state_99": p99,
        "records": records,
        "frames": frames,
        "manifest": manifest,
        "action_dim": args.action_dim,
    }
    out_path = os.path.join(args.out_dir, "stat.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[stat] saved {out_path}")
    print("[stat] p01:", [round(v, 6) for v in p01])
    print("[stat] p99:", [round(v, 6) for v in p99])


if __name__ == "__main__":
    main()
