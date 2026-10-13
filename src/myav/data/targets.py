"""Training targets computed from comma2k19's global pose."""

import numpy as np


def lateral_accel_from_pose(pose_path, smooth=5):
    """
    Returns per-frame times, speed (m/s) and signed lateral acceleration (m/s^2, left positive).
    lat_accel = ((v x a) . up) / |v|, which equals curvature * v^2
    """
    t = np.load(f"{pose_path}/frame_times")
    pos = np.load(f"{pose_path}/frame_positions")
    vel = np.load(f"{pose_path}/frame_velocities")

    acc = np.gradient(vel, t, axis=0)
    kernel = np.ones(smooth) / smooth
    acc = np.stack([np.convolve(acc[:, i], kernel, mode="same") for i in range(3)], axis=1)

    up = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    speed = np.linalg.norm(vel, axis=1)
    lat_accel = np.einsum("ij,ij->i", np.cross(vel, acc), up) / np.maximum(speed, 1.0)
    return t, speed.astype(np.float32), lat_accel.astype(np.float32)
