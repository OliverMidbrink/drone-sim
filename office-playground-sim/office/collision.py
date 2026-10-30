"""Collision boxes that follow a model's real shape: the mesh is voxelised and the voxels are merged into few boxes.

A chair becomes a base, a stem, a seat and a back with air between them instead of one block, a desk
becomes a top and legs. All boxes are axis-aligned in the world (the model is rotated first), made of
cells of `CELL` metres, and carry a bottom and a top height, so the walker can stand on a seat or the
robot can pass under a shelf. The boxes are as much as one cell fatter than the surface.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

CELL = 0.05  # m
FILL = 3  # air gaps of at most this many cells inside a column are closed: it is the thickness of a surface, not the space under a seat


def _triangles(primitives, matrix: np.ndarray) -> np.ndarray:
    """(T, 3, 3) world-space triangles of all the primitives."""
    parts = []
    for primitive in primitives:
        world = primitive.positions.astype(float) @ matrix[:3, :3].T + matrix[:3, 3]
        parts.append(world[primitive.indices.reshape(-1, 3)])
    return np.concatenate(parts)


def _sample(triangles: np.ndarray, size: float) -> np.ndarray:
    """Points spread over the triangles at most `size` apart: a barycentric lattice with as many steps as the longest edge needs."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    longest = np.maximum.reduce([np.linalg.norm(a - b, axis=1), np.linalg.norm(b - c, axis=1), np.linalg.norm(c - a, axis=1)])
    steps = np.maximum(np.ceil(longest / size).astype(int), 1)
    points = []
    for n in np.unique(steps):
        group = triangles[steps == n]
        i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        keep = i + j <= n
        u, v = (i[keep] / n)[None, :, None], (j[keep] / n)[None, :, None]
        ga, gb, gc = group[:, 0][:, None], group[:, 1][:, None], group[:, 2][:, None]
        points.append((ga + u * (gb - ga) + v * (gc - ga)).reshape(-1, 3))
    return np.concatenate(points)


def _runs(voxels: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Occupied (i, k, j0, j1) runs of every (x, z) column, with the thin gaps filled and anything near the floor joined to it. j is the height index."""
    order = np.lexsort((voxels[:, 1], voxels[:, 2], voxels[:, 0]))
    voxels = voxels[order]
    runs: list[tuple[int, int, int, int]] = []
    start = 0
    n = len(voxels)
    while start < n:
        i, k = voxels[start, 0], voxels[start, 2]
        end = start
        while end < n and voxels[end, 0] == i and voxels[end, 2] == k:
            end += 1
        heights = voxels[start:end, 1]
        j0 = prev = int(heights[0])
        for j in heights[1:]:
            j = int(j)
            if j - prev - 1 > FILL:
                runs.append((int(i), int(k), j0, prev))
                j0 = j
            prev = j
        runs.append((int(i), int(k), j0, prev))
        start = end
    return [(i, k, 0 if 0 < j0 <= FILL else j0, j1) for i, k, j0, j1 in runs]


def _merge(runs: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int, int, int]]:
    """Greedy merge of cell columns into boxes (i0, k0, i1, k1, j0, j1), first along z, then along x."""
    strips: dict[tuple[int, int, int], list[int]] = {}
    for i, k, j0, j1 in runs:
        strips.setdefault((i, j0, j1), []).append(k)
    rows: dict[tuple[int, int, int, int], list[int]] = {}
    for (i, j0, j1), ks in strips.items():
        ks.sort()
        a = ks[0]
        for previous, k in zip(ks, ks[1:] + [None]):
            if k != previous + 1 if k is not None else True:
                rows.setdefault((a, previous, j0, j1), []).append(i)
                if k is not None:
                    a = k
    boxes = []
    for (k0, k1, j0, j1), iss in rows.items():
        iss.sort()
        a = iss[0]
        for previous, i in zip(iss, iss[1:] + [None]):
            if i != previous + 1 if i is not None else True:
                boxes.append((a, k0, previous, k1, j0, j1))
                if i is not None:
                    a = i
    return boxes


class ShapeCache:
    """The surface of each model as small voxels in the model's own frame, computed once per model and kept in a file next to the unpacked assets."""

    LOCAL_CELL = 0.03

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.cells: dict[str, np.ndarray] = {}
        self.dirty = False
        if path is not None and path.exists():
            with np.load(path) as data:
                self.cells = {name: data[name] for name in data.files}

    def get(self, name: str, primitives) -> np.ndarray:
        """(N, 3) centres of the occupied voxels of the model, in its own frame."""
        key = name.replace("/", "__")  # a valid file name inside the .npz
        if key not in self.cells:
            points = _sample(_triangles(primitives, np.eye(4)), self.LOCAL_CELL * 0.7)
            voxels = np.unique(np.floor(points / self.LOCAL_CELL).astype(np.int64), axis=0)
            self.cells[key] = (voxels + 0.5) * self.LOCAL_CELL
            self.dirty = True
        return self.cells[key]

    def save(self) -> None:
        if self.path is not None and self.dirty:
            np.savez_compressed(self.path, **self.cells)
            self.dirty = False


def mesh_boxes(local_centres: np.ndarray, matrix: np.ndarray, cell: float = CELL) -> list[tuple[float, float, float, float, float, float]]:
    """(x_min, z_min, x_max, z_max, y_min, y_max) boxes that follow the shape of a model placed by `matrix`; `local_centres` is from `ShapeCache.get`."""
    points = local_centres @ matrix[:3, :3].T + matrix[:3, 3]
    voxels = np.unique(np.floor(points / cell).astype(np.int64), axis=0)  # (i, j, k) with j up
    boxes = _merge(_runs(voxels))
    return [(i0 * cell, k0 * cell, (i1 + 1) * cell, (k1 + 1) * cell, j0 * cell, (j1 + 1) * cell) for i0, k0, i1, k1, j0, j1 in boxes]
