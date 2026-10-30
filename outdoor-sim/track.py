"""A small RC race track on the grass north of the skid pad: a 6 m wide asphalt loop of about 335 m with raised kerbs in the corners.

The centre line is a closed centripetal Catmull-Rom spline through CONTROL, resampled every STEP
metres; `s` is the distance along it from the start / finish line (at CONTROL[0], heading east) and
`offset` the distance to the right of it. Heights come from a fine lookup table over the track's
bounding box, so the wheels and the picture use the same surface.
"""

from __future__ import annotations

import numpy as np

HALF_WIDTH = 3.0  # m, asphalt each side of the centre line
KERB = 0.6  # m width of the kerbs outside the asphalt
KERB_HEIGHT = 0.025
KERB_RADIUS = 15.0  # m: corners tighter than this get kerbs
STEP = 0.25  # m between centre-line samples
TABLE_STEP = 0.05
CONTROL = (
    (0.0, -46.0), (28.0, -46.0), (38.0, -50.0), (40.0, -60.0), (32.0, -67.0), (18.0, -68.0), (6.0, -63.0), (-6.0, -68.0),
    (-10.0, -80.0), (-2.0, -89.0), (14.0, -88.0), (28.0, -82.0), (39.0, -86.0), (38.0, -96.0), (26.0, -99.0), (-20.0, -98.0),
    (-36.0, -90.0), (-40.0, -70.0), (-36.0, -52.0), (-26.0, -46.0),
)


def _catmull_rom(points: np.ndarray, per_segment: int = 80) -> np.ndarray:
    """Dense points on the closed centripetal Catmull-Rom spline through `points`."""
    n = len(points)
    out = []
    for i in range(n):
        p0, p1, p2, p3 = points[(i - 1) % n], points[i], points[(i + 1) % n], points[(i + 2) % n]
        t1 = np.linalg.norm(p1 - p0) ** 0.5
        t2 = t1 + np.linalg.norm(p2 - p1) ** 0.5
        t3 = t2 + np.linalg.norm(p3 - p2) ** 0.5
        t = np.linspace(t1, t2, per_segment, endpoint=False)[:, None]
        a1 = (t1 - t) / t1 * p0 + t / t1 * p1
        a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        b1 = (t2 - t) / t2 * a1 + t / t2 * a2
        b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
        out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    return np.concatenate(out)


class Track:
    def __init__(self, asphalt: float, edge: float) -> None:
        self.asphalt, self.edge = asphalt, edge
        dense = _catmull_rom(np.array(CONTROL, float))
        dense = np.vstack([dense, dense[:1]])
        along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))])
        self.length = float(along[-1])
        self.s = np.arange(0.0, self.length, STEP)
        self.points = np.stack([np.interp(self.s, along, dense[:, 0]), np.interp(self.s, along, dense[:, 1])], axis=1)
        ahead = np.roll(self.points, -1, axis=0) - np.roll(self.points, 1, axis=0)
        self.tangent = ahead / np.linalg.norm(ahead, axis=1, keepdims=True)
        self.right = np.stack([-self.tangent[:, 1], self.tangent[:, 0]], axis=1)  # x east, z south: +z is to the right of +x
        following = np.roll(self.tangent, -1, axis=0)
        turn = np.arctan2(self.tangent[:, 0] * following[:, 1] - self.tangent[:, 1] * following[:, 0], np.einsum("ij,ij->i", self.tangent, following))
        window = int(4.0 / STEP)
        curvature = np.convolve(np.concatenate([turn[-window:], turn, turn[:window]]) / STEP, np.ones(window) / window, "same")[window:-window]
        tight = (np.abs(curvature) > 1.0 / KERB_RADIUS).astype(float)
        self.kerb = self._grow(self._grow(tight, 5.0), -2.0) > 0.5  # from 3 m before to 3 m after each corner, with gaps under 4 m closed
        outer = HALF_WIDTH + KERB + edge + 0.5
        self.box = (float(self.points[:, 0].min() - outer), float(self.points[:, 0].max() + outer), float(self.points[:, 1].min() - outer), float(self.points[:, 1].max() + outer))
        self._build_table()

    @staticmethod
    def _grow(flags: np.ndarray, metres: float) -> np.ndarray:
        """Widen (metres > 0) or narrow (< 0) the runs of 1s in a closed loop of 0 / 1 flags."""
        n = int(abs(metres) / STEP)
        padded = np.concatenate([flags[-n:], flags, flags[:n]])
        window = np.convolve(padded, np.ones(2 * n + 1), "same")[n:-n]
        return (window > 0.0).astype(float) if metres > 0 else (window >= 2 * n + 1 - 1e-6).astype(float)

    # --- the surface ---------------------------------------------------------------------------

    def _profile(self, distance: np.ndarray, kerb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Height and paved share at `distance` from the centre line."""
        outside = distance - HALF_WIDTH
        bump = np.clip(np.minimum(outside, KERB - outside) / 0.1, 0.0, 1.0) * KERB_HEIGHT * kerb  # a raised strip with 10 cm ramps
        paved = np.clip(1.0 - (distance - HALF_WIDTH - KERB) / self.edge, 0.0, 1.0)
        return self.asphalt * paved + bump, paved

    def _build_table(self) -> None:
        x0, x1, z0, z1 = self.box
        self.x_nodes = np.arange(x0, x1 + 1e-9, TABLE_STEP)
        self.z_nodes = np.arange(z0, z1 + 1e-9, TABLE_STEP)
        distance = np.full((len(self.x_nodes), len(self.z_nodes)), 1e3, np.float32)
        kerb = np.zeros_like(distance)
        reach = HALF_WIDTH + KERB + self.edge + 0.3
        cells = int(np.ceil(reach / TABLE_STEP)) + 6
        stride = 2  # segments of 0.5 m
        for i in range(0, len(self.points), stride):
            a, b = self.points[i], self.points[(i + stride) % len(self.points)]
            ix, iz = int((a[0] - x0) / TABLE_STEP), int((a[1] - z0) / TABLE_STEP)
            sx = slice(max(ix - cells, 0), min(ix + cells, len(self.x_nodes)))
            sz = slice(max(iz - cells, 0), min(iz + cells, len(self.z_nodes)))
            px = (self.x_nodes[sx] - a[0]).astype(np.float32)[:, None]
            pz = (self.z_nodes[sz] - a[1]).astype(np.float32)[None, :]
            ab = (b - a).astype(np.float32)
            t = np.clip((px * ab[0] + pz * ab[1]) / float(ab @ ab), 0.0, 1.0)
            d = np.sqrt((px - t * ab[0]) ** 2 + (pz - t * ab[1]) ** 2)
            window = distance[sx, sz]
            closer = d < window
            window[closer] = d[closer]
            kerb[sx, sz][closer] = float(self.kerb[i])
        height, paved = self._profile(distance, kerb)
        self.height_table = height.astype(np.float32)
        self.paved_table = paved.astype(np.float32)

    def inside(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        return (x >= self.x_nodes[0]) & (x <= self.x_nodes[-1]) & (z >= self.z_nodes[0]) & (z <= self.z_nodes[-1])

    def sample(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Height at (x, z), which must be inside the box."""
        fx = np.clip((x - self.x_nodes[0]) / TABLE_STEP, 0.0, len(self.x_nodes) - 1.001)
        fz = np.clip((z - self.z_nodes[0]) / TABLE_STEP, 0.0, len(self.z_nodes) - 1.001)
        ix, iz = fx.astype(int), fz.astype(int)
        tx, tz = fx - ix, fz - iz
        t = self.height_table
        return (t[ix, iz] * (1 - tx) + t[ix + 1, iz] * tx) * (1 - tz) + (t[ix, iz + 1] * (1 - tx) + t[ix + 1, iz + 1] * tx) * tz

    def paved(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Paved share at (x, z): 0 outside the box."""
        x, z = np.asarray(x, float), np.asarray(z, float)
        ix = np.clip(np.round((x - self.x_nodes[0]) / TABLE_STEP).astype(int), 0, len(self.x_nodes) - 1)
        iz = np.clip(np.round((z - self.z_nodes[0]) / TABLE_STEP).astype(int), 0, len(self.z_nodes) - 1)
        return np.where(self.inside(x, z), self.paved_table[ix, iz], 0.0)

    # --- where the car is ----------------------------------------------------------------------

    def locate(self, x: float, z: float) -> tuple[float, float]:
        """(s, offset to the right) of the nearest centre-line point."""
        d = self.points - np.array([x, z])
        i = int(np.argmin(np.einsum("ij,ij->i", d, d)))
        return float(self.s[i]), float(-(d[i] @ self.right[i]))

    def pose(self, s: float) -> tuple[float, float, float]:
        """x, z and heading of the centre line at distance s."""
        i = int(round((s % self.length) / STEP)) % len(self.s)
        t = self.tangent[i]
        return float(self.points[i, 0]), float(self.points[i, 1]), float(np.arctan2(t[1], t[0]))

    # --- drawing -------------------------------------------------------------------------------

    def mesh(self, ground) -> tuple[np.ndarray, np.ndarray]:
        """Vertices (x, y, z, nx, ny, nz, s, offset, kerb) and triangle indices of the track strip."""
        wide = HALF_WIDTH + KERB
        offsets = np.concatenate([[-wide - self.edge], np.linspace(-wide, -HALF_WIDTH, 7), np.linspace(-HALF_WIDTH, HALF_WIDTH, 25)[1:-1], np.linspace(HALF_WIDTH, wide, 7), [wide + self.edge]])
        rows = np.append(np.arange(len(self.s)), 0)  # closed: the last row repeats the first
        xz = self.points[rows][:, None, :] + offsets[None, :, None] * self.right[rows][:, None, :]
        x, z = xz[..., 0].ravel(), xz[..., 1].ravel()
        y = ground.sample(x, z)
        normals = ground.normal(x, z)
        s = np.append(self.s, self.length)[:, None].repeat(len(offsets), axis=1).ravel()
        offset = np.tile(offsets, len(rows))
        kerb = self.kerb[rows][:, None].repeat(len(offsets), axis=1).ravel().astype(float)
        vertices = np.column_stack([x, y, z, normals, s, offset, kerb]).astype(np.float32)
        index = np.arange(len(rows) * len(offsets), dtype=np.uint32).reshape(len(rows), len(offsets))
        a, b, c, d = index[:-1, :-1], index[:-1, 1:], index[1:, :-1], index[1:, 1:]
        return vertices, np.stack([a, c, b, b, c, d], axis=-1).reshape(-1)


class LapTimer:
    """Lap times on the track: a lap counts when the car crosses the start line going forwards after passing three checkpoints."""

    CHECKPOINTS = 3

    def __init__(self, track: Track) -> None:
        self.track = track
        self.best: float | None = None
        self.last: float | None = None
        self.reset()

    def reset(self) -> None:
        self.started: float | None = None
        self.s: float | None = None
        self.passed: set[int] = set()

    def update(self, time: float, x: float, z: float) -> None:
        track = self.track
        s, offset = track.locate(x, z)
        if abs(offset) > HALF_WIDTH + KERB + 4.0:  # off in the grass: nothing changes until it is back
            self.s = None
            return
        length = track.length
        if self.s is not None:
            if self.s > length - 15.0 and s < 15.0:  # over the line, forwards
                if self.started is not None and len(self.passed) == self.CHECKPOINTS:
                    self.last = time - self.started
                    self.best = self.last if self.best is None else min(self.best, self.last)
                self.started, self.passed = time, set()
            elif self.s < 15.0 and s > length - 15.0:  # backwards over it
                self.started = None
        for k in range(1, self.CHECKPOINTS + 1):
            if abs(s - k * length / (self.CHECKPOINTS + 1)) < 8.0:
                self.passed.add(k)
        self.s = s

    def current(self, time: float) -> float | None:
        return None if self.started is None else time - self.started
