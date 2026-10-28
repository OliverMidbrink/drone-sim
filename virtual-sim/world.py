"""A generated world: terrain plus scattered trees, and where to start flying in it."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from camera import Camera
from terrain import Terrain
from trees import TreeModel, place_trees

HEIGHTMAP_DIR = Path(__file__).resolve().parent / "assets" / "heightmaps"
SPAWN_HEIGHT = 30.0  # metres above the ground at the origin


@dataclass
class World:
    seed: int
    terrain: Terrain
    trees: list[np.ndarray]  # per tree model, (N, 6) instances (see trees.place_trees)

    @classmethod
    def generate(
        cls,
        seed: int,
        models: list[TreeModel],
        size: float = 2000.0,
        resolution: int = 801,
        tree_count: int = 6000,
        heightmap: Path | None = None,
        heightmap_max: float = 600.0,
        low_mountains: bool = False,
    ) -> "World":
        if heightmap is None and (HEIGHTMAP_DIR / "terrain.png").is_file():
            heightmap = HEIGHTMAP_DIR / "terrain.png"  # a terrain image dropped into assets/heightmaps/ wins
        start = time.perf_counter()
        terrain = Terrain.generate(seed, size, resolution, heightmap, heightmap_max, low_mountains)
        trees = place_trees(terrain, models, seed, tree_count)
        source = (f"heightmap {heightmap.name}" if heightmap else f"seed {seed}") + (", low mountains" if low_mountains else "")
        print(f"World ({source}): {size:.0f} m, {sum(len(t) for t in trees)} trees, built in {time.perf_counter() - start:.2f} s")
        return cls(seed, terrain, trees)

    def spawn_camera(self, fov_deg: float = 90.0) -> Camera:
        """Above the middle of the map, facing the highest ground."""
        terrain = self.terrain
        row, col = np.unravel_index(np.argmax(terrain.heights), terrain.heights.shape)
        scale = terrain.size / (terrain.resolution - 1)
        peak = np.array([col * scale - terrain.half, float(terrain.heights[row, col]), row * scale - terrain.half])
        position = np.array([0.0, float(terrain.height_at(0.0, 0.0)) + SPAWN_HEIGHT, 0.0])
        distance = max(float(np.hypot(peak[0], peak[2])), 1.0)
        pitch = np.arctan2(peak[1] - position[1], distance) * 0.6
        return Camera(position, yaw=float(np.arctan2(peak[0], -peak[2])), pitch=float(np.clip(pitch, -0.2, 0.3)), fov_deg=fov_deg)
