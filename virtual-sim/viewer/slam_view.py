"""The 3D map pane: what the SLAM has built, drawn in the estimator's world (metres) from a free orbit camera.

Everything drawn from the SLAM or the estimator comes from the drone's own estimates. The one thing
here that does not is the magenta line, the true path, which is there so *you* can judge the
estimate; nothing reads it back.
"""

from __future__ import annotations

import moderngl
import numpy as np

from camera import Camera

_POINT_VERTEX = """
#version 330 core
uniform mat4 mvp;
uniform float point_size;
in vec3 position;
in vec3 color;
out vec3 v_color;
void main() {
    gl_Position = mvp * vec4(position, 1.0);
    gl_PointSize = point_size;
    v_color = color;
}
"""

_FRAGMENT = """
#version 330 core
in vec3 v_color;
out vec4 frag;
void main() { frag = vec4(v_color, 1.0); }
"""

BACKGROUND = (0.035, 0.04, 0.05)
MAX_POINTS = 400_000


class SlamMapView:
    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        self.program = ctx.program(vertex_shader=_POINT_VERTEX, fragment_shader=_FRAGMENT)
        self.point_buffer = ctx.buffer(reserve=MAX_POINTS * 24)
        self.point_vao = ctx.vertex_array(self.program, [(self.point_buffer, "3f 3f", "position", "color")])
        self.line_buffer = ctx.buffer(reserve=200_000 * 24)
        self.line_vao = ctx.vertex_array(self.program, [(self.line_buffer, "3f 3f", "position", "color")])
        self.point_count = 0
        self.yaw, self.pitch, self.distance = np.radians(35.0), np.radians(30.0), 22.0
        self.target = np.zeros(3)
        self.dragging = False
        self.last_cursor: tuple[float, float] | None = None
        self.rect = (0, 0, 1, 1)  # pane in window pixels, bottom-left origin

    # --- input ---------------------------------------------------------------------------

    def inside(self, x: float, y_from_top: float, window_height: int) -> bool:
        x0, y0, x1, y1 = self.rect
        y = window_height - y_from_top
        return x0 <= x <= x1 and y0 <= y <= y1

    def mouse_button(self, pressed: bool, x: float, y: float, window_height: int) -> None:
        self.dragging = pressed and self.inside(x, y, window_height)
        self.last_cursor = (x, y)

    def mouse_move(self, x: float, y: float) -> None:
        if self.dragging and self.last_cursor is not None:
            self.yaw -= (x - self.last_cursor[0]) * 0.006
            self.pitch = float(np.clip(self.pitch + (y - self.last_cursor[1]) * 0.006, -1.5, 1.5))
        self.last_cursor = (x, y)

    def scroll(self, amount: float) -> None:
        self.distance = float(np.clip(self.distance * 0.9**amount, 2.0, 400.0))

    # --- data ----------------------------------------------------------------------------

    def set_points(self, positions: np.ndarray, colors: np.ndarray) -> None:
        """Landmarks in world metres (N, 3) and their colours (N, 3) as 0..1 floats."""
        count = min(len(positions), MAX_POINTS)
        step = max(1, len(positions) // MAX_POINTS)
        data = np.hstack([positions[::step][:count], colors[::step][:count]]).astype(np.float32)
        self.point_count = len(data)
        if self.point_count:
            self.point_buffer.write(data.tobytes())

    # --- drawing -------------------------------------------------------------------------

    def _camera(self, aspect: float) -> Camera:
        offset = self.distance * np.array([np.sin(self.yaw) * np.cos(self.pitch), np.sin(self.pitch), np.cos(self.yaw) * np.cos(self.pitch)])
        eye = self.target + offset
        forward = -offset / np.linalg.norm(offset)
        right = np.cross(forward, [0.0, 1.0, 0.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        return Camera(eye, fov_deg=60.0, basis=(forward, right, up))

    def draw(self, target: moderngl.Framebuffer, rect: tuple[int, int, int, int], lines: np.ndarray, follow: np.ndarray) -> None:
        """`lines` is (M, 6) float32 rows of position + colour, two rows per segment."""
        self.rect = rect
        x0, y0, x1, y1 = rect
        width, height = x1 - x0, y1 - y0
        if width <= 0 or height <= 0:
            return
        self.target += (follow - self.target) * 0.15
        target.use()
        target.viewport = (x0, y0, width, height)
        ctx = self.ctx
        ctx.scissor = (x0, y0, width, height)
        target.clear(*BACKGROUND, viewport=(x0, y0, width, height))
        camera = self._camera(width / height)
        mvp = (camera.projection_matrix(width / height, 0.1, 2000.0) @ camera.view_matrix()).astype("f4").T.tobytes()
        self.program["mvp"].write(mvp)
        ctx.enable(moderngl.DEPTH_TEST | moderngl.PROGRAM_POINT_SIZE)
        target.clear(depth=1.0, viewport=(x0, y0, width, height))
        if self.point_count:
            self.program["point_size"].value = 3.0
            self.point_vao.render(moderngl.POINTS, vertices=self.point_count)
        if len(lines):
            lines = lines[:200_000]
            self.line_buffer.write(np.ascontiguousarray(lines, dtype=np.float32).tobytes())
            ctx.disable(moderngl.DEPTH_TEST)
            self.program["point_size"].value = 1.0
            self.line_vao.render(moderngl.LINES, vertices=len(lines))
        ctx.disable(moderngl.PROGRAM_POINT_SIZE)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.scissor = None


def segments(points: np.ndarray, color: tuple[float, float, float]) -> np.ndarray:
    """A polyline as line-segment rows (position + colour)."""
    if len(points) < 2:
        return np.empty((0, 6), np.float32)
    pairs = np.empty((2 * (len(points) - 1), 3))
    pairs[0::2], pairs[1::2] = points[:-1], points[1:]
    return np.hstack([pairs, np.tile(color, (len(pairs), 1))]).astype(np.float32)


def grid_lines(centre: np.ndarray, half: float = 10.0, step: float = 1.0, color=(0.16, 0.18, 0.2)) -> np.ndarray:
    ticks = np.arange(-half, half + 1e-6, step)
    rows = []
    for t in ticks:
        rows.append([centre[0] + t, centre[1], centre[2] - half])
        rows.append([centre[0] + t, centre[1], centre[2] + half])
        rows.append([centre[0] - half, centre[1], centre[2] + t])
        rows.append([centre[0] + half, centre[1], centre[2] + t])
    return np.hstack([np.array(rows), np.tile(color, (len(rows), 1))]).astype(np.float32)


def frame_axes(origin: np.ndarray, rotation: np.ndarray, length: float = 0.8) -> np.ndarray:
    """Body axes as three coloured segments: right red, up green, nose blue."""
    rows = []
    for axis, color in ((rotation[:, 0], (1.0, 0.3, 0.3)), (rotation[:, 1], (0.3, 1.0, 0.3)), (-rotation[:, 2], (0.4, 0.6, 1.0))):
        rows.append([*origin, *color])
        rows.append([*(origin + axis * length), *color])
    return np.array(rows, np.float32)


def cross_marker(centre: np.ndarray, size: float = 0.5, color=(1.0, 0.85, 0.2)) -> np.ndarray:
    rows = []
    for axis in np.eye(3):
        rows.append([*(centre - axis * size), *color])
        rows.append([*(centre + axis * size), *color])
    return np.array(rows, np.float32)
