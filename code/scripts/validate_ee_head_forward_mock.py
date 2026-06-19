"""
Mocked forward smoke for Ctrl-World EE head wiring.

It bypasses SVD/CLIP weight loading and injects tiny dummy modules into
CrtlWorld, then checks the no-head and EE-head forward paths.
"""
import os
import sys
import types
import importlib.machinery
from types import SimpleNamespace

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

decord_stub = types.ModuleType("decord")
decord_stub.__spec__ = importlib.machinery.ModuleSpec("decord", loader=None)
decord_stub.VideoReader = object
decord_stub.cpu = lambda *args, **kwargs: None
sys.modules.setdefault("decord", decord_stub)

from models.ctrl_world import CrtlWorld
from models.ee_head import EETrajectoryHead


class DummyVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(scaling_factor=1.0)


class DummyPipeline:
    def __init__(self):
        self.vae = DummyVAE()

    def _get_add_time_ids(self, fps, motion_bucket_id, noise_aug_strength, dtype, batch_size, num_videos_per_prompt, do_classifier_free_guidance):
        return torch.zeros(batch_size, 3, dtype=dtype)


class DummyActionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(14, 1024)

    def forward(self, action, texts=None, text_tokinizer=None, text_encoder=None, frame_level_cond=True):
        return self.proj(action.float()).to(action.dtype)


class DummyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))

    @property
    def dtype(self):
        return self.scale.dtype

    @property
    def device(self):
        return self.scale.device

    def forward(self, input_latents, c_noise, encoder_hidden_states=None, added_time_ids=None, frame_level_cond=True):
        return SimpleNamespace(sample=input_latents[:, :, :4] * self.scale)


def make_model(use_ee_head):
    model = CrtlWorld.__new__(CrtlWorld)
    nn.Module.__init__(model)
    model.args = SimpleNamespace(
        num_history=6,
        num_frames=16,
        action_dim=14,
        text_cond=True,
        frame_level_cond=True,
        his_cond_zero=False,
        motion_bucket_id=127,
        fps=7,
        use_ee_head=use_ee_head,
        ee_loss_weight=0.05,
        ee_head_hidden_dim=32,
    )
    model.pipeline = DummyPipeline()
    model.vae = model.pipeline.vae
    model.unet = DummyUNet()
    model.action_encoder = DummyActionEncoder()
    model.tokenizer = None
    model.text_encoder = None
    model.use_ee_head = use_ee_head
    if use_ee_head:
        model.ee_head = EETrajectoryHead(hidden_dim=32)
    return model


def make_batch():
    torch.manual_seed(7)
    batch = {
        "latent": torch.randn(2, 22, 4, 90, 40),
        "action": torch.randn(2, 22, 14),
        "text": ["click alarmclock", "place object basket"],
        "ee_target": torch.randn(2, 22, 20),
    }
    batch["ee_target"][..., 9] = (batch["ee_target"][..., 9] > 0).float()
    batch["ee_target"][..., 19] = (batch["ee_target"][..., 19] > 0).float()
    return batch


def check_forward(use_ee_head):
    model = make_model(use_ee_head)
    batch = make_batch()
    loss, loss_dict = model(batch)
    if not torch.isfinite(loss):
        raise AssertionError(f"non-finite loss for use_ee_head={use_ee_head}: {loss}")
    if "video_loss" not in loss_dict:
        raise AssertionError("video_loss missing")
    if use_ee_head:
        for key in ["ee_loss", "ee_position_loss", "ee_rotation_loss", "ee_gripper_loss"]:
            if key not in loss_dict:
                raise AssertionError(f"{key} missing")
        expected = loss_dict["video_loss"] + model.args.ee_loss_weight * loss_dict["ee_loss"]
        if not torch.allclose(loss.detach(), expected.detach(), rtol=1e-5, atol=1e-5):
            raise AssertionError("total loss does not include weighted ee_loss")
    else:
        if "ee_loss" in loss_dict:
            raise AssertionError("ee_loss present in no-head path")
    loss.backward()
    print(f"  ok forward use_ee_head={use_ee_head}: loss={float(loss.item()):.6f}, keys={sorted(loss_dict.keys())}")


def main():
    check_forward(use_ee_head=False)
    check_forward(use_ee_head=True)
    print("[OK] Mocked Ctrl-World forward validation passed.")


if __name__ == "__main__":
    main()
