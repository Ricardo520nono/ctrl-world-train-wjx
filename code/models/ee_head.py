import torch
import torch.nn as nn
import torch.nn.functional as F


EE_TARGET_DIM = 20
POSITION_DIMS = [0, 1, 2, 10, 11, 12]
ROTATION_6D_DIMS = [3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18]
GRIPPER_DIMS = [9, 19]


class EETrajectoryHead(nn.Module):
    def __init__(self, in_channels=4, hidden_dim=256, target_dim=EE_TARGET_DIM):
        super().__init__()
        self.target_dim = target_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, latent):
        if latent.ndim != 5:
            raise ValueError(f"EE head expects (B, F, C, H, W), got {tuple(latent.shape)}")
        bsz, frames, channels, height, width = latent.shape
        latent = latent.reshape(bsz * frames, channels, height, width)
        hidden = self.encoder(latent).flatten(1)
        pred = self.proj(hidden)
        return pred.reshape(bsz, frames, self.target_dim)


def compute_ee_losses(pred, target):
    if pred.shape != target.shape:
        raise ValueError(f"EE pred/target shape mismatch: pred={tuple(pred.shape)}, target={tuple(target.shape)}")
    if pred.shape[-1] != EE_TARGET_DIM:
        raise ValueError(f"EE target dim must be {EE_TARGET_DIM}, got {pred.shape[-1]}")

    pred_f = pred.float()
    target_f = target.float()
    position_loss = F.mse_loss(pred_f[..., POSITION_DIMS], target_f[..., POSITION_DIMS])
    rotation_loss = F.mse_loss(pred_f[..., ROTATION_6D_DIMS], target_f[..., ROTATION_6D_DIMS])
    gripper_loss = F.binary_cross_entropy_with_logits(pred_f[..., GRIPPER_DIMS], target_f[..., GRIPPER_DIMS])
    total_loss = position_loss + rotation_loss + gripper_loss
    return total_loss, {
        "ee_position_loss": position_loss,
        "ee_rotation_loss": rotation_loss,
        "ee_gripper_loss": gripper_loss,
    }
