import numpy as np


EE_TARGET_DIM = 20


def quat_wxyz_to_rot6d(quat):
    quat = np.asarray(quat, dtype=np.float32)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    quat = quat / np.clip(norm, 1e-8, None)

    w = quat[..., 0]
    x = quat[..., 1]
    y = quat[..., 2]
    z = quat[..., 3]

    rot = np.empty((*quat.shape[:-1], 3, 3), dtype=np.float32)
    rot[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rot[..., 0, 1] = 2.0 * (x * y - z * w)
    rot[..., 0, 2] = 2.0 * (x * z + y * w)
    rot[..., 1, 0] = 2.0 * (x * y + z * w)
    rot[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rot[..., 1, 2] = 2.0 * (y * z - x * w)
    rot[..., 2, 0] = 2.0 * (x * z - y * w)
    rot[..., 2, 1] = 2.0 * (y * z + x * w)
    rot[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)

    return rot[..., :, :2].reshape(*quat.shape[:-1], 6).astype(np.float32)


def build_ee_target(left_endpose, left_gripper, right_endpose, right_gripper):
    left_endpose = np.asarray(left_endpose, dtype=np.float32)
    right_endpose = np.asarray(right_endpose, dtype=np.float32)
    left_gripper = np.asarray(left_gripper, dtype=np.float32)[:, None]
    right_gripper = np.asarray(right_gripper, dtype=np.float32)[:, None]

    if left_endpose.shape[-1] != 7 or right_endpose.shape[-1] != 7:
        raise ValueError(f"endpose must be 7D, got left={left_endpose.shape}, right={right_endpose.shape}")

    left = np.concatenate(
        [left_endpose[:, :3], quat_wxyz_to_rot6d(left_endpose[:, 3:7]), left_gripper],
        axis=1,
    )
    right = np.concatenate(
        [right_endpose[:, :3], quat_wxyz_to_rot6d(right_endpose[:, 3:7]), right_gripper],
        axis=1,
    )
    ee_target = np.concatenate([left, right], axis=1).astype(np.float32)
    if ee_target.shape[1] != EE_TARGET_DIM:
        raise RuntimeError(f"ee_target dim mismatch: {ee_target.shape}")
    return ee_target


def ee_target_from_hdf5(h5_file):
    return build_ee_target(
        h5_file["endpose/left_endpose"][:],
        h5_file["endpose/left_gripper"][:],
        h5_file["endpose/right_endpose"][:],
        h5_file["endpose/right_gripper"][:],
    )
