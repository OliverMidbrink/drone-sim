"""The proving ground: virtual-sim's terrain, flattened round a skid pad and a test road, with bumps in the road.

World frame (as everywhere): x east, z south, y up. The grass terrain is the generated map of
virtual-sim (same seed, same generator, low mountains) squashed to y = 0 near the paved area and left
alone further away. On it lie

    the pad    a paved rectangle 80 x 60 m centred on the origin: skid pad with a 20 m circle
    the road   8 m wide, from the pad's east edge to x = 380 m, with test sections along it:
               sine waves, a washboard, speed humps, a frequency sweep, potholes and a kicker ramp.
    the track  a small RC race track on the grass north of the pad (see track.py)

The paving lies ASPHALT above the grass, with sloping edges EDGE wide, so everything on it is above
the grass. `Ground.sample` is the height everything (the car's wheels, the picture) agrees on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from terrain import Terrain, generate_heights, load_heightmap, smoothstep
from world import HEIGHTMAP_DIR

from track import Track

ASPHALT = 0.05  # m the paving lies above the grass
EDGE = 0.25  # m width of the sloping edge of the paving
PAD = (-40.0, 40.0, -30.0, 30.0)  # x0, x1, z0, z1
ROAD = (40.0, 380.0, -4.0, 4.0)
FLAT_RADIUS, FREE_RADIUS = 60.0, 160.0  # the grass is exactly flat within FLAT_RADIUS of the paving and natural beyond FREE_RADIUS
TABLE_STEP = (0.02, 0.05)  # m, the lookup table of the road: along x, across z
SKID_CIRCLE = 20.0  # m radius of the circle painted on the pad


@dataclass(frozen=True)
class Section:
    name: str
    x0: float
    x1: float
    detail: str


SECTIONS = (
    Section("Sine waves", 60.0, 120.0, "5 m long, 7 cm peak to peak"),
    Section("Washboard", 130.0, 170.0, "0.9 m long, 3 cm"),
    Section("Speed humps", 180.0, 230.0, "7 cm high, every 10 m"),
    Section("Frequency sweep", 240.0, 300.0, "7 m down to 0.8 m long, 5 cm"),
    Section("Potholes", 310.0, 335.0, "4 cm deep, left and right"),
    Section("Kicker ramp", 345.0, 360.0, "50 cm rise over 4 m"),
)
SINE_LENGTH, SINE_AMPLITUDE = 5.0, 0.035
WASH_LENGTH, WASH_AMPLITUDE = 0.9, 0.015
HUMP_HEIGHT, HUMP_WIDTH = 0.07, 1.6
SWEEP_LONG, SWEEP_SHORT, SWEEP_AMPLITUDE = 7.0, 0.8, 0.025
POTHOLES = ((313.0, -2.0, 0.55), (317.0, 2.0, 0.55), (321.0, -1.5, 0.45), (325.0, 1.5, 0.45), (329.0, 0.0, 0.6))  # x, z, radius
POTHOLE_DEPTH = 0.04
KICKER = (350.0, 4.0, 0.5, 0.35)  # start x, ramp length, rise, length of the drop at the end


def _box_distance(x: np.ndarray, z: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    """Distance outside a rectangle (0 inside)."""
    x0, x1, z0, z1 = box
    return np.hypot(np.maximum(np.maximum(x0 - x, x - x1), 0.0), np.maximum(np.maximum(z0 - z, z - z1), 0.0))


def _window(x: np.ndarray, x0: float, x1: float, taper: float = 3.0) -> np.ndarray:
    """1 inside [x0, x1], easing to 0 over `taper` metres at both ends (0 outside)."""
    return smoothstep(x0, x0 + taper, x) * (1.0 - smoothstep(x1 - taper, x1, x))


def road_bumps(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Height in metres that the test sections add to the road surface."""
    return np.broadcast_to(_profile(x), np.broadcast(x, z).shape) + _potholes(x, z)


def _profile(x: np.ndarray) -> np.ndarray:
    """The bumps that are the same all the way across the road."""
    h = np.zeros(np.shape(x))
    sine, wash, humps, sweep = SECTIONS[0], SECTIONS[1], SECTIONS[2], SECTIONS[3]
    if ((x >= sine.x0) & (x <= sine.x1)).any():
        h = h + SINE_AMPLITUDE * np.sin(2.0 * np.pi * (x - sine.x0) / SINE_LENGTH) * _window(x, sine.x0, sine.x1)
    if ((x >= wash.x0) & (x <= wash.x1)).any():
        h = h + WASH_AMPLITUDE * np.sin(2.0 * np.pi * (x - wash.x0) / WASH_LENGTH) * _window(x, wash.x0, wash.x1)
    if ((x >= humps.x0) & (x <= humps.x1)).any():
        centre = np.round((x - humps.x0 - 5.0) / 10.0) * 10.0 + humps.x0 + 5.0  # nearest hump
        offset = np.abs(x - np.clip(centre, humps.x0 + 5.0, humps.x1 - 5.0))
        h = h + HUMP_HEIGHT * np.where(offset < HUMP_WIDTH / 2.0, np.cos(np.pi * offset / HUMP_WIDTH) ** 2, 0.0)
    if ((x >= sweep.x0) & (x <= sweep.x1)).any():
        length = sweep.x1 - sweep.x0
        u = np.clip((x - sweep.x0) / length, 0.0, 1.0)
        ratio = SWEEP_SHORT / SWEEP_LONG
        phase = 2.0 * np.pi * (length / SWEEP_LONG) * (1.0 - ratio ** (-u)) / np.log(ratio)  # the wavelength shrinks exponentially
        h = h + SWEEP_AMPLITUDE * np.sin(phase) * _window(x, sweep.x0, sweep.x1)
    start, length, rise, drop = KICKER
    if ((x >= start) & (x <= start + length + drop)).any():
        ramp = rise * np.clip((x - start) / length, 0.0, 1.0)
        ramp = np.where(x < start + length, ramp, rise * (1.0 - np.clip((x - start - length) / drop, 0.0, 1.0)))
        h = h + np.where(x >= start, ramp, 0.0)
    return h


def _potholes(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    h = np.zeros(np.broadcast(x, z).shape)
    if ((x >= 310.0) & (x <= 335.0)).any():
        for px, pz, radius in POTHOLES:
            d = np.hypot(x - px, z - pz)
            h = h - POTHOLE_DEPTH * np.where(d < radius, np.cos(np.pi * d / (2.0 * radius)) ** 2, 0.0)
    return h


class Ground:
    """The terrain and paving as one height function, plus what is needed to draw it."""

    def __init__(self, seed: int = 0, size: float = 1600.0, resolution: int = 801, low: bool = True) -> None:
        if (HEIGHTMAP_DIR / "terrain.png").is_file():
            heights = load_heightmap(HEIGHTMAP_DIR / "terrain.png", resolution, 600.0)
            heights *= 40.0 / max(float(heights.max()), 1e-6) if low else 1.0
        else:
            heights = generate_heights(seed, resolution, low)
        self.track = Track(ASPHALT, EDGE)
        axis = np.linspace(-size / 2.0, size / 2.0, resolution, dtype=np.float32)
        gx, gz = np.meshgrid(axis, axis)
        near = np.minimum(np.minimum(_box_distance(gx, gz, PAD), _box_distance(gx, gz, ROAD)), _box_distance(gx, gz, self.track.box))
        heights = heights * smoothstep(FLAT_RADIUS, FREE_RADIUS, near).astype(np.float32)  # exactly 0 round the paving
        self.terrain = Terrain.from_heights(size, heights)
        self.size = size
        self.x_nodes = np.arange(ROAD[0] + EDGE, ROAD[1] + EDGE + 1e-9, TABLE_STEP[0])  # the road's lookup table
        self.z_nodes = np.arange(-ROAD[3] - EDGE, ROAD[3] + EDGE + 1e-9, TABLE_STEP[1])
        gx, gz = np.meshgrid(self.x_nodes, self.z_nodes, indexing="ij")
        road = np.clip(1.0 - _box_distance(gx, gz, ROAD) / EDGE, 0.0, 1.0)  # what _analytic gives here, where the grass is flat and the pad is out of reach
        bumps = np.broadcast_to(_profile(self.x_nodes)[:, None], gx.shape).copy()
        holes = (self.x_nodes >= 310.0) & (self.x_nodes <= 335.0)
        bumps[holes] += _potholes(gx[holes], gz[holes])
        self.table = ((ASPHALT + bumps * road) * road).astype(np.float32)

    # --- the height function -----------------------------------------------------------------

    def _grass(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Terrain height, without evaluating the terrain where it is known to be flat."""
        if float(x.min()) > -80.0 and float(x.max()) < 420.0 and float(np.abs(z).max()) < 40.0:
            return np.zeros(np.broadcast(x, z).shape)  # this box is within FLAT_RADIUS of the paving, where the grass is exactly flat
        x0, x1, z0, z1 = self.track.box
        if float(x.min()) > x0 - 50.0 and float(x.max()) < x1 + 50.0 and float(z.min()) > z0 - 50.0 and float(z.max()) < z1 + 50.0:
            return np.zeros(np.broadcast(x, z).shape)
        return self.terrain.height_at(x, z).astype(float)

    def _analytic(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        x, z = np.asarray(x, float), np.asarray(z, float)
        pad = np.clip(1.0 - _box_distance(x, z, PAD) / EDGE, 0.0, 1.0)
        road = np.clip(1.0 - _box_distance(x, z, ROAD) / EDGE, 0.0, 1.0)
        paved = np.maximum(pad, road)
        grass = self._grass(x, z)
        return grass * (1.0 - paved) + (ASPHALT + road_bumps(x, z) * road) * paved

    def sample(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Height of the ground at world (x, z), any array shape."""
        x, z = np.asarray(x, float), np.asarray(z, float)
        inside = (x >= self.x_nodes[0]) & (x <= self.x_nodes[-1]) & (np.abs(z) <= self.z_nodes[-1])
        if inside.all():
            return self._lookup(x, z)
        on_track = self.track.inside(x, z)
        if on_track.all():
            return self.track.sample(x, z)
        out = self._analytic(x, z)
        if inside.any():
            out = np.where(inside, self._lookup(np.clip(x, self.x_nodes[0], self.x_nodes[-1]), np.clip(z, self.z_nodes[0], self.z_nodes[-1])), out)
        if on_track.any():
            out = np.where(on_track, self.track.sample(x, z), out)
        return out

    def _lookup(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        fx = (x - self.x_nodes[0]) / TABLE_STEP[0]
        fz = (z - self.z_nodes[0]) / TABLE_STEP[1]
        ix = np.minimum(fx.astype(int), len(self.x_nodes) - 2)
        iz = np.minimum(fz.astype(int), len(self.z_nodes) - 2)
        tx, tz = fx - ix, fz - iz
        t = self.table
        return (t[ix, iz] * (1 - tx) + t[ix + 1, iz] * tx) * (1 - tz) + (t[ix, iz + 1] * (1 - tx) + t[ix + 1, iz + 1] * tx) * tz

    def normal(self, x: np.ndarray, z: np.ndarray, eps: float = 0.02) -> np.ndarray:
        """Unit surface normals at the points (flat arrays), by central differences."""
        x, z = np.asarray(x, float).ravel(), np.asarray(z, float).ravel()
        n = len(x)
        h = self.sample(np.concatenate([x + eps, x - eps, x, x]), np.concatenate([z, z, z + eps, z - eps]))
        gx, gz = (h[:n] - h[n : 2 * n]) / (2 * eps), (h[2 * n : 3 * n] - h[3 * n :]) / (2 * eps)
        normals = np.stack([-gx, np.ones(n), -gz], axis=1)
        return normals / np.linalg.norm(normals, axis=1, keepdims=True)

    def paved(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """1 on the paving, 0 on grass (the edge ramps are in between)."""
        x, z = np.asarray(x, float), np.asarray(z, float)
        slabs = np.maximum(np.clip(1.0 - _box_distance(x, z, PAD) / EDGE, 0.0, 1.0), np.clip(1.0 - _box_distance(x, z, ROAD) / EDGE, 0.0, 1.0))
        return np.maximum(slabs, self.track.paved(x, z))

    def clear_of_paving(self, x: np.ndarray, z: np.ndarray, margin: float = 70.0) -> np.ndarray:
        """True where a tree may stand: further than `margin` metres from the paving."""
        return np.minimum(np.minimum(_box_distance(x, z, PAD), _box_distance(x, z, ROAD)), _box_distance(x, z, self.track.box)) > margin

    # --- meshes for drawing --------------------------------------------------------------------

    def _mesh(self, xs: np.ndarray, zs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vertices (x, y, z, nx, ny, nz) and triangle indices of the height function sampled on the grid xs x zs."""
        gx, gz = np.meshgrid(xs, zs, indexing="ij")
        h = self._analytic(gx, gz)
        dh_dx = np.gradient(h, xs, axis=0)
        dh_dz = np.gradient(h, zs, axis=1)
        normals = np.stack([-dh_dx, np.ones_like(h), -dh_dz], axis=-1)
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
        vertices = np.concatenate([np.stack([gx, h, gz], axis=-1), normals], axis=-1).reshape(-1, 6).astype(np.float32)
        index = np.arange(h.size, dtype=np.uint32).reshape(h.shape)
        a, b, c, d = index[:-1, :-1], index[:-1, 1:], index[1:, :-1], index[1:, 1:]
        return vertices, np.stack([a, b, c, b, d, c], axis=-1).reshape(-1)

    def pad_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        x0, x1, z0, z1 = PAD
        xs = np.concatenate([[x0 - EDGE], np.arange(x0, x1 + 1e-9, 2.0), [x1 + EDGE]])
        zs = np.concatenate([[z0 - EDGE], np.arange(z0, z1 + 1e-9, 2.0), [z1 + EDGE]])
        return self._mesh(xs, zs)

    def road_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        x0, x1 = ROAD[0] + EDGE, ROAD[1] + EDGE
        fine = np.arange(SECTIONS[0].x0 - 2.0, SECTIONS[-1].x1 + 6.0, 0.05)
        xs = np.unique(np.round(np.concatenate([np.arange(x0, fine[0], 1.0), fine, np.arange(fine[-1], x1, 1.0), [x1]]), 4))
        zs = np.concatenate([[-ROAD[3] - EDGE], np.arange(-ROAD[3], ROAD[3] + 1e-9, 0.2), [ROAD[3] + EDGE]])
        return self._mesh(xs, zs)

    def spawn(self) -> tuple[float, float, float]:
        """x, z and heading (radians, 0 = looking along +x) to start on: the west side of the pad, pointing at the road."""
        return -30.0, 0.0, 0.0
