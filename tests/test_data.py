import numpy as np

from conftest import N_FRAMES, write_pose
from myav.data.comma2k19 import find_segments, split_by_route
from myav.data.dataset import WindowDataset
from myav.data.targets import lateral_accel_from_pose


def test_lateral_accel_left_turn_positive(tmp_path):
    write_pose(tmp_path / "pose", n=200, turn_rate=0.2)
    _, speed, lat = lateral_accel_from_pose(tmp_path / "pose")
    np.testing.assert_allclose(speed, 20.0, rtol=1e-6)
    np.testing.assert_allclose(lat[10:-10], 20.0 * 0.2, rtol=1e-2)


def test_find_segments_and_route_split(fake_dataset):
    segments = find_segments(fake_dataset)
    assert len(segments) == 8
    train, val = split_by_route(segments, val_fraction=0.25)
    assert len(val) == 2
    assert not {s.route for s in train} & {s.route for s in val}


def test_window_shapes_and_counts(fake_dataset):
    segments = find_segments(fake_dataset)
    items = list(WindowDataset(segments, window=10, windows_per_segment=3, train=True))
    assert len(items) == 8 * 3
    item = items[0]
    assert item["imgs"].shape == (11, 6, 128, 256)
    assert item["v_ego"].shape == item["action_t"].shape == item["target"].shape == (10,)


def test_validation_is_deterministic(fake_dataset):
    segments = find_segments(fake_dataset)
    ds = WindowDataset(segments, window=10, windows_per_segment=100, train=False)
    a, b = list(ds), list(ds)
    assert len(a) == 8 * ((N_FRAMES - 10 + 9) // 10)
    for x, y in zip(a, b):
        assert (x["action_t"] == y["action_t"]).all()
