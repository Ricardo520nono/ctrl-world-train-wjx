"""
Single-GPU smoke for the Ctrl-World EE-head forward path.

This loads real SVD/CLIP weights and runs one synthetic batch through
CrtlWorld.forward. It is intended for local A800 debugging, not AIHC training.
"""
import argparse
import os
import sys
from types import SimpleNamespace

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ctrl_world import CrtlWorld


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--svd_model_path", required=True)
    parser.add_argument("--clip_model_path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ee_head_hidden_dim", type=int, default=64)
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    model_args = SimpleNamespace(
        svd_model_path=args.svd_model_path,
        clip_model_path=args.clip_model_path,
        action_dim=14,
        num_history=6,
        num_frames=16,
        text_cond=True,
        frame_level_cond=True,
        his_cond_zero=False,
        motion_bucket_id=127,
        fps=7,
        use_ee_head=True,
        ee_loss_weight=0.05,
        ee_head_hidden_dim=args.ee_head_hidden_dim,
    )
    print("[real-forward] loading model")
    model = CrtlWorld(model_args)
    model.to(device)
    model.train()

    torch.manual_seed(11)
    batch = {
        "latent": torch.randn(1, 22, 4, 90, 40, device=device),
        "action": torch.randn(1, 22, 14, device=device),
        "text": ["click alarmclock"],
        "ee_target": torch.randn(1, 22, 20, device=device),
    }
    batch["ee_target"][..., 9] = (batch["ee_target"][..., 9] > 0).float()
    batch["ee_target"][..., 19] = (batch["ee_target"][..., 19] > 0).float()

    print("[real-forward] running forward")
    with torch.no_grad():
        loss, loss_dict = model(batch)
    for key in ["video_loss", "ee_loss", "ee_position_loss", "ee_rotation_loss", "ee_gripper_loss"]:
        if key not in loss_dict:
            raise AssertionError(f"{key} missing from loss_dict")
    if not torch.isfinite(loss):
        raise AssertionError(f"non-finite loss: {loss}")
    print("[real-forward] loss", float(loss.detach().cpu()))
    print("[real-forward] keys", sorted(loss_dict.keys()))
    print("[OK] Real Ctrl-World EE-head forward validation passed.")


if __name__ == "__main__":
    main()
