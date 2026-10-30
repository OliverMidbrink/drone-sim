"""Collision queries for a sphere against the terrain and the trees, cheap enough to run hundreds of times a second.

The terrain is a height lookup. Every tree trunk is a vertical stack of discs whose radius follows
the tree's real silhouette (see trees.Collider); trunks live in a grid of cells and only the 3 x 3
cells around the query point are tested, with that candidate set cached while the sphere stays in a cell.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from terrain import Terrain
from trees import PROFILE_BINS, TreeModel

CELL = 14.0  # metres; bigger than the widest tree plus a drone


@dataclass
class Contact:
    kind: str  # "ground" or "tree"
    normal: np.ndarray  # unit vector pointing out of the obstacle
    depth: float  # how far the sphere is inside, metres


class CollisionWorld:
    def __init__(self, terrain: Terrain, trees: list[np.ndarray], models: list[TreeModel]) -> None:
        self.terrain = terrain
        self.cells: dict[tuple[int, int], np.ndarray] = {}
        self._cache_key: tuple[int, int] | None = None
        self._candidates = np.empty(0, dtype=int)
        rows, profiles = [], []
        for instances, model in zip(trees, models):
            for collider in model.colliders:
                profiles.append(collider.profile)
                profile_index = len(profiles) - 1
                if len(instances) == 0:
                    continue
                x, y, z, yaw, height = (instances[:, i].astype(np.float64) for i in range(5))
                c, s = np.cos(yaw), np.sin(yaw)
                ox, oz = collider.offset
                rows.append(
                    np.stack(
                        [
                            x + (c * ox + s * oz) * height,  # same turn as the tree vertex shader
                            z + (-s * ox + c * oz) * height,
                            y,
                            height,
                            np.full(len(x), collider.top),
                            np.full(len(x), profile_index),
                        ],
                        axis=1,
                    )
                )
        self.profiles = np.array(profiles).reshape(-1, PROFILE_BINS)
        self.trunks = np.concatenate(rows) if rows else np.empty((0, 6))  # x, z, base y, height, top (unit), profile
        if len(self.trunks):
            keys = np.floor(self.trunks[:, :2] / CELL).astype(int)
            order = np.lexsort((keys[:, 1], keys[:, 0]))
            sorted_keys = keys[order]
            starts = np.flatnonzero(np.any(np.diff(sorted_keys, axis=0) != 0, axis=1)) + 1
            for group in np.split(order, starts):
                self.cells[(int(keys[group[0], 0]), int(keys[group[0], 1]))] = group

    def _nearby(self, x: float, z: float) -> np.ndarray:
        key = (int(np.floor(x / CELL)), int(np.floor(z / CELL)))
        if key != self._cache_key:
            found = [self.cells[(key[0] + i, key[1] + j)] for i in (-1, 0, 1) for j in (-1, 0, 1) if (key[0] + i, key[1] + j) in self.cells]
            self._candidates = np.concatenate(found) if found else np.empty(0, dtype=int)
            self._cache_key = key
        return self._candidates

    def resolve(self, position: np.ndarray, radius: float) -> list[Contact]:
        """Everything a sphere of this radius at `position` is currently inside of."""
        x, y, z = float(position[0]), float(position[1]), float(position[2])
        contacts: list[Contact] = []
        depth = float(self.terrain.height_at(x, z)) + radius - y
        if depth > 0.0:
            contacts.append(Contact("ground", self.terrain.normal_at(x, z), depth))

        near = self._nearby(x, z)
        if len(near):
            trunks = self.trunks[near]
            height = trunks[:, 3]
            relative = (y - trunks[:, 2]) / height  # 0 at the ground, 1 at the tree's nominal top
            alive = (relative > -radius / height) & (relative < trunks[:, 4] + radius / height)
            if alive.any():
                trunks, relative, height = trunks[alive], relative[alive], height[alive]
                position_in_bins = np.clip(relative * PROFILE_BINS - 0.5, 0.0, PROFILE_BINS - 1.0)
                low = np.floor(position_in_bins).astype(int)
                high = np.minimum(low + 1, PROFILE_BINS - 1)
                blend = position_in_bins - low
                profile = self.profiles[trunks[:, 5].astype(int)]
                rows = np.arange(len(trunks))
                reach = (profile[rows, low] * (1.0 - blend) + profile[rows, high] * blend) * height + radius
                dx, dz = x - trunks[:, 0], z - trunks[:, 1]
                distance = np.hypot(dx, dz)
                for hit in np.flatnonzero(distance < reach):
                    away = np.array([dx[hit], 0.0, dz[hit]]) / distance[hit] if distance[hit] > 1e-6 else np.array([1.0, 0.0, 0.0])
                    contacts.append(Contact("tree", away, float(reach[hit] - distance[hit])))
        return contacts
