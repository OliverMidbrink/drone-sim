"""Procedural terrain: rolling hills plus ridged mountains, as a heightmap.

Everything is built from a few layers of smooth random lattices (one `cv2.resize` per octave), so
a 800 x 800 terrain takes well under a second. A heightmap image can replace the generated terrain
(`load_heightmap`), and the generated one can be written out as an image (`save_heightmap`).

The grid is indexed heights[row, column]: column = +x (east), row = +z (south), y is up. The terrain
is centred on the origin and `size` metres wide in both directions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def smoothstep(low: float, high: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - low) / (high - low), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def fbm(rng: np.random.Generator, resolution: int, cells: int, octaves: int, gain: float = 0.5, ridged: bool = False) -> np.ndarray:
    """Fractal noise in roughly [0, 1], `cells` features across at the first octave, doubling every octave.

    `ridged` turns the valleys of the noise into sharp crests, each octave weighted by the crest
    height below it (Musgrave's ridged multifractal), which looks like eroded mountain ranges.
    """
    total = np.zeros((resolution, resolution), np.float32)
    weight = np.ones_like(total)
    amplitude, norm = 1.0, 0.0
    for _ in range(octaves):
        lattice = rng.random((min(cells, resolution), min(cells, resolution)), dtype=np.float32)
        layer = np.clip(cv2.resize(lattice, (resolution, resolution), interpolation=cv2.INTER_CUBIC), 0.0, 1.0)
        if ridged:
            layer = (1.0 - np.abs(2.0 * layer - 1.0)) ** 2 * weight
            weight = np.clip(layer * 2.0, 0.0, 1.0)
        total += amplitude * layer
        norm += amplitude
        amplitude *= gain
        cells *= 2
    return total / norm


def tileable_noise(rng: np.random.Generator, size: int, cells_list: tuple[int, ...]) -> np.ndarray:
    """size x size noise that repeats seamlessly, summing one lattice per entry of cells_list (equal weight)."""
    total = np.zeros((size, size), np.float32)
    for cells in cells_list:
        lattice = np.tile(rng.random((cells, cells), dtype=np.float32), (3, 3))
        big = cv2.resize(lattice, (3 * size, 3 * size), interpolation=cv2.INTER_CUBIC)
        total += big[size : 2 * size, size : 2 * size]
    return np.clip(total / len(cells_list), 0.0, 1.0)


def load_heightmap(path: Path, resolution: int, height_max: float) -> np.ndarray:
    """Greyscale image (8 or 16 bit; black = 0 m, white = height_max m) resampled to the terrain grid."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise SystemExit(f"Cannot read heightmap {path}")
    if image.ndim == 3:
        image = image[..., :3].mean(axis=2)
    scale = 65535.0 if image.dtype == np.uint16 else 255.0 if image.dtype == np.uint8 else 1.0
    image = cv2.resize(image.astype(np.float32) / scale, (resolution, resolution), interpolation=cv2.INTER_CUBIC)
    return np.clip(image, 0.0, 1.0) * height_max


def save_heightmap(path: Path, heights: np.ndarray, height_max: float) -> None:
    """16-bit PNG, black = 0 m and white = height_max m (the format `load_heightmap` reads)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.round(np.clip(heights / height_max, 0.0, 1.0) * 65535.0).astype(np.uint16))


LOW_PEAK = 40.0  # metres: the highest ground when the mountains are lowered, about twice a tall tree


def generate_heights(seed: int, resolution: int, low: bool = False) -> np.ndarray:
    """Heights in metres: gentle hills everywhere, a ring of ridged mountains near the edge and a few massifs inside.

    `low` keeps the same layout but squashes the mountains to LOW_PEAK metres and calms the hills.
    """
    rng = np.random.default_rng(seed)
    axis = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
    x, z = np.meshgrid(axis, axis)
    radius = np.sqrt(x * x + z * z)  # 0 in the middle, 1 at the middle of each edge

    hills = (fbm(rng, resolution, 3, 6) - 0.5) * 80.0
    hills *= 0.3 + 0.7 * smoothstep(0.0, 0.35, radius)  # calmer around the start

    ring = smoothstep(0.45, 0.95, radius + 0.35 * (fbm(rng, resolution, 2, 3) - 0.5))
    massifs = smoothstep(0.6, 0.78, fbm(rng, resolution, 2, 2)) * smoothstep(0.25, 0.5, radius)
    mask = np.clip(ring + 0.85 * massifs, 0.0, 1.0)
    ridges = fbm(rng, resolution, 3, 7, gain=0.55, ridged=True)
    ridges = np.clip(ridges / np.percentile(ridges, 99.5), 0.0, 1.0)
    mountains = (ridges**1.1 * 430.0 + 60.0) * mask**1.5

    # bend the mountains a little so the ridges do not follow the lattice
    pixels = np.arange(resolution, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(pixels, pixels)
    strength = 0.06 * resolution
    shift_x = (fbm(rng, resolution, 3, 3) - 0.5) * 2.0 * strength
    shift_y = (fbm(rng, resolution, 3, 3) - 0.5) * 2.0 * strength
    mountains = cv2.remap(mountains.astype(np.float32), grid_x + shift_x, grid_y + shift_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if low:
        hills, mountains = hills * 0.4, mountains * (LOW_PEAK / float(mountains.max()))
    return (hills + mountains).astype(np.float32)


@dataclass
class Terrain:
    size: float  # metres, both directions
    heights: np.ndarray  # (resolution, resolution) float32 metres
    normals: np.ndarray  # (resolution, resolution, 3) float32

    @classmethod
    def from_heights(cls, size: float, heights: np.ndarray) -> "Terrain":
        spacing = size / (heights.shape[0] - 1)
        gradient_z, gradient_x = np.gradient(heights, spacing)
        normals = np.stack([-gradient_x, np.ones_like(heights), -gradient_z], axis=-1)
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
        return cls(size, heights.astype(np.float32), normals.astype(np.float32))

    @classmethod
    def generate(
        cls, seed: int, size: float = 2000.0, resolution: int = 801, heightmap: Path | None = None, heightmap_max: float = 600.0, low: bool = False
    ) -> "Terrain":
        if heightmap is not None:
            heights = load_heightmap(heightmap, resolution, heightmap_max)
            if low:
                heights *= LOW_PEAK / max(float(heights.max()), 1e-6)
            return cls.from_heights(size, heights)
        return cls.from_heights(size, generate_heights(seed, resolution, low))

    @property
    def resolution(self) -> int:
        return self.heights.shape[0]

    @property
    def half(self) -> float:
        return self.size / 2.0

    def _grid(self, x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scale = self.resolution - 1
        return np.clip((np.asarray(x) / self.size + 0.5) * scale, 0.0, scale), np.clip((np.asarray(z) / self.size + 0.5) * scale, 0.0, scale)

    def height_at(self, x: np.ndarray | float, z: np.ndarray | float) -> np.ndarray:
        """Bilinearly interpolated ground height at world (x, z); clamped to the terrain's edge."""
        gx, gz = self._grid(x, z)
        top = self.resolution - 2
        ix, iz = np.minimum(gx.astype(int), top), np.minimum(gz.astype(int), top)
        fx, fz = gx - ix, gz - iz
        h = self.heights
        return (h[iz, ix] * (1 - fx) + h[iz, ix + 1] * fx) * (1 - fz) + (h[iz + 1, ix] * (1 - fx) + h[iz + 1, ix + 1] * fx) * fz

    def flatness_at(self, x: np.ndarray | float, z: np.ndarray | float) -> np.ndarray:
        """Up component of the surface normal: 1 on flat ground, 0.7 on a 45 degree slope."""
        gx, gz = self._grid(x, z)
        return self.normals[np.round(gz).astype(int), np.round(gx).astype(int), 1]

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """(vertices (N, 6) float32 = position + normal, triangle indices uint32)."""
        resolution = self.resolution
        axis = np.linspace(-self.half, self.half, resolution, dtype=np.float32)
        x, z = np.meshgrid(axis, axis)
        vertices = np.concatenate([np.stack([x, self.heights, z], axis=-1), self.normals], axis=-1).reshape(-1, 6)
        index = np.arange(resolution * resolution, dtype=np.uint32).reshape(resolution, resolution)
        a, b, c, d = index[:-1, :-1], index[:-1, 1:], index[1:, :-1], index[1:, 1:]
        indices = np.stack([a, c, b, b, c, d], axis=-1).reshape(-1)
        return np.ascontiguousarray(vertices, dtype=np.float32), indices
