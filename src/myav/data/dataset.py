"""Streams training windows of openpilot-format frames with lateral acceleration targets."""

import warnings

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from myav.data.frames import read_warped_frames
from myav.data.targets import lateral_accel_from_pose

# range of action delays (seconds) to train on, roughly what openpilot passes as lat_action_t
MIN_ACTION_T = 0.1
MAX_ACTION_T = 0.6


class WindowDataset(IterableDataset):
    """
    Decodes a whole segment once, then yields windows of consecutive frames from it.

    Each item:
        imgs:     [T+1, 6, 128, 256] uint8 (frame 0 is only used as the previous frame of frame 1)
        v_ego:    [T] float32 m/s
        action_t: [T] float32 seconds
        target:   [T] float32 lateral accel (m/s^2) at t + action_t

    Training keeps `segments_in_flight` segments open per worker and draws windows from them at random,
    so a batch mixes several drives instead of being made of near-identical windows of one video.
    Validation is deterministic: fixed non-overlapping windows and a fixed action_t per segment.
    """

    def __init__(self, segments, window=40, windows_per_segment=20, segments_in_flight=4, train=True, seed=0):
        self.segments = list(segments)
        self.window = window
        self.windows_per_segment = windows_per_segment
        self.segments_in_flight = segments_in_flight if train else 1
        self.train = train
        self.seed = seed

    def __iter__(self):
        # DataLoader seeds torch differently for every worker and epoch, so this varies when it should
        base_seed = torch.randint(2**31, ()).item() if self.train else self.seed
        order = list(range(len(self.segments)))
        info = get_worker_info()
        if info is not None:
            order = order[info.id::info.num_workers]
        if self.train:
            np.random.default_rng(base_seed).shuffle(order)

        pending = iter(order)
        pool = []
        pool_rng = np.random.default_rng(base_seed)
        while True:
            while len(pool) < self.segments_in_flight:
                idx = next(pending, None)
                if idx is None:
                    break
                pool.append(self._windows(self.segments[idx], np.random.default_rng([base_seed, idx])))
            if not pool:
                return
            k = pool_rng.integers(len(pool))
            try:
                yield next(pool[k])
            except StopIteration:
                pool.pop(k)

    def _windows(self, segment, rng):
        T = self.window
        try:
            frames = read_warped_frames(str(segment.video))
        except RuntimeError as e:
            warnings.warn(f"skipping segment: {e}")
            return
        t, speed, lat_accel = lateral_accel_from_pose(segment.pose)
        n = min(len(frames), len(t))
        if n < T + 1:
            return

        if self.train:
            starts = rng.choice(n - T, size=min(self.windows_per_segment, n - T), replace=False)
        else:
            starts = range(0, n - T, T)[:self.windows_per_segment]

        for s in starts:
            frame_idx = np.arange(s + 1, s + T + 1)
            action_t = rng.uniform(MIN_ACTION_T, MAX_ACTION_T, T).astype(np.float32)
            target = np.interp(t[frame_idx] + action_t, t[:n], lat_accel[:n]).astype(np.float32)
            yield {
                "imgs": torch.from_numpy(frames[s:s + T + 1]),
                "v_ego": torch.from_numpy(speed[frame_idx]),
                "action_t": torch.from_numpy(action_t),
                "target": torch.from_numpy(target),
            }
