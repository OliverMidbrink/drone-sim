"""Adaptive map resolution: map a region until it is complete, then keep only anchors.

A region is "complete" when a patch of the map (sized relative to its distance
from the camera path, so near and far things are judged alike) holds enough
well-observed landmarks. A complete patch is thinned to its best few landmarks,
which become *settled anchors*: they stay in the map to localise against, and
any new observation that lands next to one is redundant and not added. The
tracker then spends its feature budget on what is still unmapped.
"""

from __future__ import annotations

import cv2
import numpy as np

CELL_REL = 0.15  # patch side, as a fraction of the point's distance from the path
MIN_POINTS = 12  # landmarks in a patch before it can be complete
MIN_MEAN_WEIGHT = 8.0  # mean observation weight (accumulated triangulation angle)
MIN_MEAN_AGE = 3.0  # seconds of video the patch has had to refine
KEEP_FRACTION = 0.25  # share of a completed patch kept as anchors
MIN_KEEP = 4
GATE_REL = 0.10  # a new point this close (x its distance) to a settled anchor is redundant

_COORD_BITS = 19
_COORD_BIAS = 1 << (_COORD_BITS - 1)


def voxel_keys(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Patch id per point: a cubic cell whose side scales (in octaves) with distance."""
    level = np.floor(np.log2(np.maximum(CELL_REL * distance, 1e-4)))
    size = 2.0 ** level
    cell = np.floor(points / size[:, None]).astype(np.int64) + _COORD_BIAS
    cell = np.clip(cell, 0, (1 << _COORD_BITS) - 1)
    tag = np.clip(level + 32, 0, 63).astype(np.int64)
    return (
        (tag << (3 * _COORD_BITS))
        | (cell[:, 0] << (2 * _COORD_BITS))
        | (cell[:, 1] << _COORD_BITS)
        | cell[:, 2]
    )


def distance_to_path(points: np.ndarray, trajectory: np.ndarray) -> np.ndarray:
    poses = np.ascontiguousarray(
        trajectory[:: max(1, len(trajectory) // 400)], dtype=np.float32
    )
    if len(poses) < 2:
        return np.linalg.norm(points - poses[0], axis=1)
    index = cv2.flann_Index(poses, dict(algorithm=1, trees=1))
    _, d2 = index.knnSearch(np.ascontiguousarray(points, dtype=np.float32), 1, params={})
    return np.sqrt(d2[:, 0])


class SaturationMap:
    def __init__(self) -> None:
        self.anchor_ids = np.empty(0, dtype=np.int64)  # sorted ids of settled anchors
        self._index = None
        self._dirty = True
        self.dropped_total = 0

    def __len__(self) -> int:
        return len(self.anchor_ids)

    def mark_dirty(self) -> None:
        """Anchor positions moved (loop-closure warp): rebuild the lookup lazily."""
        self._dirty = True

    def _rebuild(self, store) -> None:
        self._dirty = False
        self._index = None
        if len(self.anchor_ids) == 0:
            return
        chunks = []
        for ids, positions in (
            (store.live_ids, store.live_positions),
            (store.archive_ids, store.archive_positions),
        ):
            if len(ids):
                chunks.append(positions[np.isin(ids, self.anchor_ids)])
        positions = np.vstack(chunks) if chunks else np.empty((0, 3))
        if len(positions) >= 4:
            self._index = cv2.flann_Index(
                np.ascontiguousarray(positions, dtype=np.float32), dict(algorithm=1, trees=2)
            )

    def settled(self, store, points: np.ndarray, distance: np.ndarray) -> np.ndarray:
        """True where a point sits next to a settled anchor (its patch is complete)."""
        out = np.zeros(len(points), dtype=bool)
        if len(points) == 0 or len(self.anchor_ids) == 0:
            return out
        if self._dirty:
            self._rebuild(store)
        if self._index is None:
            return out
        _, d2 = self._index.knnSearch(
            np.ascontiguousarray(points, dtype=np.float32), 1, params={}
        )
        return np.sqrt(d2[:, 0]) <= GATE_REL * np.maximum(distance, 1e-6)

    def consolidate(self, store, now: float, trajectory: np.ndarray) -> tuple[int, int]:
        """Find complete patches, thin them to anchors: (anchors made, landmarks dropped)."""
        positions = np.vstack([store.live_positions, store.archive_positions])
        if len(positions) < MIN_POINTS or len(trajectory) == 0:
            return 0, 0
        ids = np.concatenate([store.live_ids, store.archive_ids])
        weights = np.concatenate([store.live_weights, store.archive_weights])
        born = np.concatenate([store.live_born, store.archive_born])

        work = np.flatnonzero(~np.isin(ids, self.anchor_ids))
        if len(work) < MIN_POINTS:
            return 0, 0
        distance = distance_to_path(positions[work], trajectory)
        keys = voxel_keys(positions[work], distance)
        _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
        mean_weight = np.bincount(inverse, weights=weights[work]) / counts
        mean_age = np.bincount(inverse, weights=now - born[work]) / counts
        complete = (
            (counts >= MIN_POINTS)
            & (mean_weight >= MIN_MEAN_WEIGHT)
            & (mean_age >= MIN_MEAN_AGE)
        )
        if not complete.any():
            return 0, 0

        in_complete = complete[inverse]
        rows = work[in_complete]
        group = inverse[in_complete]
        order = np.lexsort((-weights[rows], group))  # by patch, best weight first
        rows, group = rows[order], group[order]
        starts = np.r_[0, np.flatnonzero(np.diff(group)) + 1]
        sizes = np.diff(np.r_[starts, len(group)])
        rank = np.arange(len(group)) - np.repeat(starts, sizes)
        keep_n = np.maximum(MIN_KEEP, np.ceil(KEEP_FRACTION * np.repeat(sizes, sizes))).astype(int)
        keep = rank < keep_n

        self.anchor_ids = np.union1d(self.anchor_ids, ids[rows[keep]])
        doomed = np.zeros(len(positions), dtype=bool)
        doomed[rows[~keep]] = True
        dropped = store.remove_rows(doomed)
        self.dropped_total += dropped
        self._dirty = True
        return int(keep.sum()), dropped
