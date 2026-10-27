"""RawSlam wearing the interface the video app (3d-slam.py) expects of its SLAM session.

The app only needs a few things from a session: `process(frame)`, the live tracks, the pose, the
trajectory and a map to draw. `RawSession` supplies those from `raw_slam.RawSlam` and nothing else:
no mesh, no clamping, no carried speed, no map thinning. The map is every landmark RawSlam has
triangulated, kept at the last position it had when its track ended.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from raw_slam import RawSlam

UNKNOWN_COLOR = 0.7  # grey, for a landmark whose pixel colour is not known


class RawMapStore:
    """The part of MapStore that MapRenderer uses: points and colours to draw."""

    def __init__(self) -> None:
        self.positions = np.empty((1024, 3))
        self.colors = np.empty((1024, 3))
        self.count = 0
        self.row: dict[int, int] = {}

    def __len__(self) -> int:
        return self.count

    @property
    def live_count(self) -> int:
        return self.count

    @property
    def archive_count(self) -> int:
        return 0

    def update(self, landmarks: dict[int, np.ndarray], colors: dict[int, np.ndarray]) -> None:
        """Move known landmarks to their current position and add the new ones (with their colour, if known)."""
        for track, entry in landmarks.items():
            position = entry[:3] / entry[3]
            row = self.row.get(track)
            if row is None:
                if self.count == len(self.positions):
                    self.positions = np.vstack([self.positions, np.empty_like(self.positions)])
                    self.colors = np.vstack([self.colors, np.empty_like(self.colors)])
                row = self.row[track] = self.count
                self.colors[row] = colors.get(track, np.full(3, UNKNOWN_COLOR))
                self.count += 1
            self.positions[row] = position

    def render_subset(self, eye: np.ndarray, target: np.ndarray, aspect: float, *, max_points: int = 150_000):
        step = max(1, self.count // max_points)
        return self.positions[: self.count : step].copy(), self.colors[: self.count : step].copy()

    def finalize(self) -> None:
        pass


class RawSession:
    def __init__(
        self,
        first_frame: np.ndarray,
        intrinsic: np.ndarray,
        exclude_mask: np.ndarray | None = None,
        fps: float = 30.0,
    ) -> None:
        self.slam = RawSlam(first_frame, intrinsic, exclude_mask)
        self.intrinsic = intrinsic
        self.exclude_mask = exclude_mask
        self.fps = fps
        self.height, self.width = first_frame.shape[:2]
        self.processed = 0
        self.trajectory = [self.slam.world_position.copy()]
        self.map_store = RawMapStore()
        # the video app reads these; the raw SLAM has none of them
        self.mesh_builder = None
        self.anchors = None
        self.saturation_map = None
        self.motion_sample = None
        self.stats: defaultdict[str, int] = defaultdict(int)  # "fixes" is kept current

    @property
    def track_points(self) -> np.ndarray:
        return self.slam.track_points

    @property
    def world_rotation(self) -> np.ndarray:
        return self.slam.world_rotation

    @property
    def world_position(self) -> np.ndarray:
        return self.slam.world_position

    def process(self, frame: np.ndarray) -> None:
        self.processed += 1
        if self.slam.process(frame):
            self.trajectory.append(self.slam.world_position.copy())
            self._update_map(frame)
        self.stats["fixes"] = self.slam.fixes

    def _update_map(self, frame: np.ndarray) -> None:
        """Colours come from the frame, at the track's pixel, the first time its landmark is stored."""
        new = [t for t in self.slam.landmarks if t not in self.map_store.row]
        colors: dict[int, np.ndarray] = {}
        if new:
            pixel = dict(zip(self.slam.track_ids.tolist(), self.slam.track_points))
            for track in new:
                if track in pixel:
                    x, y = np.round(pixel[track]).astype(int)
                    bgr = frame[min(max(y, 0), frame.shape[0] - 1), min(max(x, 0), frame.shape[1] - 1)]
                    colors[track] = bgr[::-1].astype(np.float64) / 255.0
        self.map_store.update(self.slam.landmarks, colors)

    def describe(self) -> str:
        return self.slam.describe()
