# op_dataset.py
# Iterable dataset producing openpilot style inputs and lateral acceleration targets from comma2k19

import os
import random
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from src.data.op_frames import read_warped_frames

MIN_ACTION_T = 0.1  # seconds, roughly the range openpilot passes as lat_action_t
MAX_ACTION_T = 0.6


def lateral_accel_from_pose(pose_path, smooth=5):
    """
    Computes speed and signed lateral acceleration (left positive) per frame from global_pose.
    lat_accel = ((v x a) . up) / |v|, which equals curvature * v^2
    """
    t = np.load(os.path.join(pose_path, "frame_times"))
    pos = np.load(os.path.join(pose_path, "frame_positions"))
    vel = np.load(os.path.join(pose_path, "frame_velocities"))

    acc = np.gradient(vel, t, axis=0)
    kernel = np.ones(smooth) / smooth
    acc = np.stack([np.convolve(acc[:, i], kernel, mode="same") for i in range(3)], axis=1)

    up = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    speed = np.linalg.norm(vel, axis=1)
    lat_accel = np.einsum("ij,ij->i", np.cross(vel, acc), up) / np.maximum(speed, 1.0)
    return t, speed.astype(np.float32), lat_accel.astype(np.float32)


class OpWindowDataset(IterableDataset):
    """
    Yields windows of consecutive frames from shuffled segments.
    Each item:
        imgs:     [T+1, 6, 128, 256] uint8 (frame 0 is only used as the previous frame)
        v_ego:    [T] float32 m/s
        action_t: [T] float32 seconds
        target:   [T] float32 lateral accel (m/s^2) at t + action_t
    """
    def __init__(self, base_dataset, indices, window_size=40, windows_per_segment=20, shuffle=True):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.window_size = window_size
        self.windows_per_segment = windows_per_segment
        self.shuffle = shuffle

    def _segments(self):
        indices = self.indices[:]
        info = get_worker_info()
        if info is not None:
            indices = indices[info.id::info.num_workers]
        if self.shuffle:
            random.shuffle(indices)
        return indices

    def __iter__(self):
        T = self.window_size
        for idx in self._segments():
            sample = self.base_dataset[idx]
            if sample["video_path"] is None or sample["pose_path"] is None:
                continue
            frames = read_warped_frames(sample["video_path"])
            t, speed, lat_accel = lateral_accel_from_pose(sample["pose_path"])
            n = min(len(frames), len(t))
            if n < T + 1:
                continue

            starts = list(range(0, n - T - 1))
            if self.shuffle:
                starts = random.sample(starts, min(self.windows_per_segment, len(starts)))
            else:
                starts = starts[::T][:self.windows_per_segment]

            for s in starts:
                frame_idx = np.arange(s + 1, s + T + 1)
                action_t = np.random.uniform(MIN_ACTION_T, MAX_ACTION_T, T).astype(np.float32)
                target = np.interp(t[frame_idx] + action_t, t[:n], lat_accel[:n]).astype(np.float32)
                yield {
                    "imgs": torch.from_numpy(frames[s:s + T + 1].copy()),
                    "v_ego": torch.from_numpy(speed[frame_idx]),
                    "action_t": torch.from_numpy(action_t),
                    "target": torch.from_numpy(target),
                }
