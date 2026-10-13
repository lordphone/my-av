"""Builds a tiny fake comma2k19 tree, with video decoding replaced by random frames, so tests need no data."""

import numpy as np
import pytest

import myav.data.dataset

N_FRAMES = 60


def write_pose(pose_dir, n=N_FRAMES, turn_rate=0.1):
    """Constant speed circle on a plane tangent to the earth at the north pole."""
    pose_dir.mkdir(parents=True)
    t = np.arange(n) * 0.05
    r, R = 20.0 / turn_rate, 6.4e6
    pos = np.stack([r * np.cos(turn_rate * t), r * np.sin(turn_rate * t), np.full_like(t, R)], axis=1)
    vel = np.stack([-20 * np.sin(turn_rate * t), 20 * np.cos(turn_rate * t), np.zeros_like(t)], axis=1)
    for name, arr in [("frame_times", t), ("frame_positions", pos), ("frame_velocities", vel)]:
        with open(pose_dir / name, "wb") as f:
            np.save(f, arr)


@pytest.fixture
def fake_dataset(tmp_path, monkeypatch):
    for route in ["routeA", "routeB", "routeC", "routeD"]:
        for seg in range(2):
            seg_dir = tmp_path / "Chunk_1" / route / str(seg)
            write_pose(seg_dir / "global_pose")
            (seg_dir / "video.hevc").write_bytes(b"")

    def fake_read(video_path):
        return np.random.default_rng(0).integers(0, 256, (N_FRAMES, 6, 128, 256), dtype=np.uint8)

    monkeypatch.setattr(myav.data.dataset, "read_warped_frames", fake_read)
    return tmp_path
