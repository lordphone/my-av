"""Finds segments in the comma2k19 dataset and splits them into train/val by route.

Expected layout:  <root>/Chunk_<n>/<route_id>/<segment_number>/{video.hevc, global_pose/}
"""

import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Segment:
    route: str
    number: int
    path: Path

    @property
    def video(self) -> Path:
        return self.path / "video.hevc"

    @property
    def pose(self) -> Path:
        return self.path / "global_pose"


def find_segments(root) -> list[Segment]:
    """Returns every segment that has a video and a global pose, sorted by route then segment number."""
    segments = []
    for path in Path(root).glob("Chunk_*/*/*"):
        if not path.name.isdigit():
            continue
        seg = Segment(route=path.parent.name, number=int(path.name), path=path)
        if seg.video.is_file() and (seg.pose / "frame_times").is_file():
            segments.append(seg)
    return sorted(segments, key=lambda s: (s.route, s.number))


def split_by_route(segments, val_fraction=0.1, seed=0):
    """
    Splits whole routes (drives) into train and val. Neighbouring segments of the same drive look
    almost identical, so splitting by segment would leak training data into validation.
    """
    routes = sorted({s.route for s in segments})
    random.Random(seed).shuffle(routes)
    val_routes = set(routes[:max(1, round(len(routes) * val_fraction))])
    train = [s for s in segments if s.route not in val_routes]
    val = [s for s in segments if s.route in val_routes]
    return train, val
