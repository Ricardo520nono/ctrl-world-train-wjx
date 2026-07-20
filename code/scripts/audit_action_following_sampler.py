"""Preflight audit for ActionFollowingData chunk-level sampling."""
import argparse
import json
import os
import sys
from types import SimpleNamespace

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.dataset_action_following import ActionFollowingCtrlWorldDataset  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_root", required=True)
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--val_manifest", default=None)
    parser.add_argument("--stat_path", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--num_history", type=int, default=6)
    parser.add_argument("--num_frames", type=int, default=26)
    parser.add_argument("--action_dim", type=int, default=14)
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--task_balanced", action="store_true")
    return parser.parse_args()


def main():
    cli = parse_args()
    args = SimpleNamespace(
        num_history=cli.num_history,
        num_frames=cli.num_frames,
        action_dim=cli.action_dim,
        use_ee_head=False,
        dataset_meta_info_path=os.path.dirname(os.path.dirname(cli.stat_path)),
        dataset_cfgs=os.path.basename(os.path.dirname(cli.stat_path)),
        action_following_latent_root=cli.latent_root,
        action_following_train_manifest=cli.train_manifest,
        action_following_val_manifest=cli.val_manifest,
        action_following_stat_path=cli.stat_path,
        action_following_sampling_protocol=cli.protocol,
        action_following_chunk_size=cli.chunk_size,
        action_following_action_chunk_size=cli.chunk_size,
        action_following_dataset_length=0,
        action_following_sampling_seed=cli.seed,
        action_following_task_balanced=cli.task_balanced,
    )
    dataset = ActionFollowingCtrlWorldDataset(args, mode="train")
    report = dataset.audit_sampler(
        num_samples=cli.num_samples,
        seed=cli.seed,
        tolerance=cli.tolerance,
        raise_on_fail=True,
    )
    print("[ActionFollowing sampler audit]")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
