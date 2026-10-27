"""The virtual world: flat grass ground, scattered boxes, textures, and the figure-8 flight path.

Pure numpy, no OpenGL. World axes: x right, y UP, z toward the viewer (OpenGL convention),
units are metres. The ground is the plane y = 0.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------- specs
GROUND_SIZE_M = 200.0  # square, centred on the origin
PIXELS_PER_METRE = 16  # texture density (Minecraft: 16 px per 1 m block)
GRASS_TEXTURE_BLOCKS = 16  # the grass texture is 16 x 16 blocks (= 16 m), then repeats

BOX_COUNT = 35
BOX_WIDTH_RANGE_M = (0.6, 2.4)  # footprint sides, each drawn independently
BOX_HEIGHT_RANGE_M = (0.5, 3.0)
BOX_GAP_M = 0.5  # minimum clear distance between boxes (footprints treated as circles)
BOX_AREA_MARGIN_M = 3.0  # boxes may sit this far beyond the flight path's extent

PATH_HALF_WIDTH_M = 10.0  # figure 8 spans +-10 m in x ...
PATH_HALF_DEPTH_M = 6.0  # ... and +-6 m in z
CLEARANCE_M = 1.0  # flight altitude = tallest box top + this
SPEED_MS = 3.0  # constant speed along the path
CAMERA_PITCH_DEG = 25.0  # camera looks this far below the horizontal, along the direction of travel

DEFAULT_FOV_DEG = 90.0  # horizontal
DEFAULT_SIZE = (1280, 720)
DEFAULT_SEED = 7


# ---------------------------------------------------------------- boxes
@dataclass
class Box:
    x: float
    z: float
    yaw: float  # radians about the vertical axis
    width: float
    depth: float
    height: float
    tint: tuple[float, float, float]  # multiplies the stone texture

    @property
    def radius(self) -> float:
        return 0.5 * float(np.hypot(self.width, self.depth))


def generate_boxes(seed: int) -> list[Box]:
    rng = np.random.default_rng(seed)
    half_x = PATH_HALF_WIDTH_M + BOX_AREA_MARGIN_M
    half_z = PATH_HALF_DEPTH_M + BOX_AREA_MARGIN_M
    boxes: list[Box] = []
    attempts = 0
    while len(boxes) < BOX_COUNT and attempts < 20000:
        attempts += 1
        width, depth = rng.uniform(*BOX_WIDTH_RANGE_M, size=2)
        height = float(rng.uniform(*BOX_HEIGHT_RANGE_M))
        x = float(rng.uniform(-half_x, half_x))
        z = float(rng.uniform(-half_z, half_z))
        radius = 0.5 * float(np.hypot(width, depth))
        if any(np.hypot(x - b.x, z - b.z) < radius + b.radius + BOX_GAP_M for b in boxes):
            continue
        tint = colorsys.hsv_to_rgb(float(rng.uniform(0.0, 1.0)), 0.35, 1.0)
        boxes.append(Box(x, z, float(rng.uniform(0.0, 2.0 * np.pi)), float(width), float(depth), height, tint))
    return boxes


def flight_altitude(boxes: list[Box]) -> float:
    return max(b.height for b in boxes) + CLEARANCE_M


# ---------------------------------------------------------------- textures
def grass_texture(seed: int) -> np.ndarray:
    """Minecraft-style grass top: speckled greens, each 1 m block slightly lighter or darker. (H, W, 3) uint8."""
    rng = np.random.default_rng(seed + 100)
    palette = np.array(
        [(98, 160, 52), (88, 148, 46), (108, 170, 60), (80, 138, 42), (116, 178, 66), (72, 128, 40)], dtype=np.float64
    )
    size = GRASS_TEXTURE_BLOCKS * PIXELS_PER_METRE
    image = palette[rng.integers(0, len(palette), size=(size, size))]
    block_factor = rng.uniform(0.94, 1.06, size=(GRASS_TEXTURE_BLOCKS, GRASS_TEXTURE_BLOCKS))
    block_factor = np.kron(block_factor, np.ones((PIXELS_PER_METRE, PIXELS_PER_METRE)))
    return np.clip(image * block_factor[..., None], 0, 255).astype(np.uint8)


def box_texture(seed: int) -> np.ndarray:
    """One 1 m tile of speckled grey stone (tinted per box at draw time). (H, W, 3) uint8."""
    rng = np.random.default_rng(seed + 200)
    grey = rng.integers(95, 170, size=(PIXELS_PER_METRE, PIXELS_PER_METRE)).astype(np.float64)
    image = np.stack([grey, grey * 0.98, grey * 0.96], axis=-1)
    return np.clip(image, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- meshes
# Interleaved float32 rows: position(3) normal(3) uv(2) tint(3).
# For a cuboid face whose normal is local axis 0 / 1 / 2: which local axes give the texture (u, v).
_UV_AXES = {0: (2, 1), 1: (0, 2), 2: (0, 1)}


def ground_mesh() -> np.ndarray:
    half = GROUND_SIZE_M / 2.0
    corners = [(-half, -half), (half, -half), (half, half), (-half, half)]
    rows = []
    for index in (0, 1, 2, 0, 2, 3):
        x, z = corners[index]
        u, v = x / (GRASS_TEXTURE_BLOCKS), z / (GRASS_TEXTURE_BLOCKS)  # one repeat per 16 m
        rows.append([x, 0.0, z, 0.0, 1.0, 0.0, u, v, 1.0, 1.0, 1.0])
    return np.array(rows, dtype=np.float32)


def box_mesh(boxes: list[Box], seed: int) -> np.ndarray:
    """All boxes as triangles. Texture coordinates are in metres (one repeat per 1 m), planar per face."""
    rng = np.random.default_rng(seed + 300)
    rows = []
    for box in boxes:
        half = np.array([box.width / 2.0, box.height / 2.0, box.depth / 2.0])
        centre = np.array([0.0, box.height / 2.0, 0.0])  # box sits on the ground, local origin at its base centre
        c, s = np.cos(box.yaw), np.sin(box.yaw)
        rotation = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        uv_offset = rng.uniform(0.0, 16.0, size=2)  # so neighbouring boxes do not show the same texture patch
        for axis in range(3):
            u_axis, v_axis = _UV_AXES[axis]
            other = [a for a in range(3) if a != axis]
            for sign in (-1.0, 1.0):
                normal = np.zeros(3)
                normal[axis] = sign
                quad = []
                for su, sv in ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)):
                    local = centre + normal * half[axis]
                    local[other[0]] += su * half[other[0]]
                    local[other[1]] += sv * half[other[1]]
                    position = rotation @ local + np.array([box.x, 0.0, box.z])
                    quad.append((position, local[u_axis] + uv_offset[0], local[v_axis] + uv_offset[1]))
                world_normal = rotation @ normal
                for index in (0, 1, 2, 0, 2, 3):
                    position, u, v = quad[index]
                    rows.append([*position, *world_normal, u, v, *box.tint])
    return np.array(rows, dtype=np.float32)


# ---------------------------------------------------------------- flight path
class FigureEightPath:
    """Horizontal figure 8 (x = A sin t, z = C sin 2t) at a fixed altitude, flown at constant speed."""

    def __init__(self, altitude: float) -> None:
        self.altitude = altitude
        theta = np.linspace(0.0, 2.0 * np.pi, 4001)
        x = PATH_HALF_WIDTH_M * np.sin(theta)
        z = PATH_HALF_DEPTH_M * np.sin(2.0 * theta)
        arc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(z)))])
        self._theta, self._arc = theta, arc
        self.length = float(arc[-1])  # metres per lap
        self.lap_seconds = self.length / SPEED_MS

    def pose(self, seconds: float) -> tuple[np.ndarray, np.ndarray]:
        """(camera position, unit forward vector) after `seconds`; the camera looks along the path, pitched down."""
        distance = (SPEED_MS * seconds) % self.length
        theta = float(np.interp(distance, self._arc, self._theta))
        position = np.array(
            [PATH_HALF_WIDTH_M * np.sin(theta), self.altitude, PATH_HALF_DEPTH_M * np.sin(2.0 * theta)]
        )
        heading = np.array([PATH_HALF_WIDTH_M * np.cos(theta), 0.0, 2.0 * PATH_HALF_DEPTH_M * np.cos(2.0 * theta)])
        heading /= np.linalg.norm(heading)
        pitch = np.radians(CAMERA_PITCH_DEG)
        forward = heading * np.cos(pitch) + np.array([0.0, -1.0, 0.0]) * np.sin(pitch)
        return position, forward


# ---------------------------------------------------------------- specs text
def describe(boxes: list[Box], path: FigureEightPath, fov_deg: float, width: int, height: int) -> list[str]:
    fov_y = np.degrees(2.0 * np.arctan(np.tan(np.radians(fov_deg) / 2.0) * height / width))
    tallest = max(b.height for b in boxes)
    return [
        "Virtual world specs",
        f"  ground      flat plane, {GROUND_SIZE_M:g} x {GROUND_SIZE_M:g} m, Minecraft-style grass "
        f"({PIXELS_PER_METRE} px per metre, nearest-neighbour, {GRASS_TEXTURE_BLOCKS} x {GRASS_TEXTURE_BLOCKS} m texture repeating)",
        f"  boxes       {len(boxes)} on the ground, footprint {BOX_WIDTH_RANGE_M[0]}-{BOX_WIDTH_RANGE_M[1]} m, "
        f"height {BOX_HEIGHT_RANGE_M[0]}-{BOX_HEIGHT_RANGE_M[1]} m (tallest {tallest:.2f} m), random rotation, "
        f"min gap {BOX_GAP_M} m, stone texture with a random tint per box",
        f"  flight      figure 8, {2 * PATH_HALF_WIDTH_M:g} m x {2 * PATH_HALF_DEPTH_M:g} m, horizontal, altitude "
        f"{path.altitude:.2f} m (= tallest box + {CLEARANCE_M:g} m)",
        f"  motion      {SPEED_MS:g} m/s constant, lap {path.length:.1f} m = {path.lap_seconds:.1f} s",
        f"  camera      looks along the direction of travel, pitched {CAMERA_PITCH_DEG:g} deg down, no roll",
        f"  optics      pinhole, no distortion, {fov_deg:g} deg horizontal / {fov_y:.1f} deg vertical FOV, "
        f"{width} x {height}, near 0.05 m, far 500 m",
        "  lighting    one fixed sun (ambient 0.55 + diffuse 0.45), no shadows, no fog",
    ]
