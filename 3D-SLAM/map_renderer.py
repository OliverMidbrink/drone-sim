"""GPU 3D map view (OpenGL via moderngl): orbit, pan, WASD fly, Q/E roll, scroll forward/back."""

from __future__ import annotations

import time

import cv2
import glfw
import moderngl
import numpy as np

from map_store import MapStore

MAP_WINDOW = (
    "3D SLAM map (LMB look, MMB orbit, RMB pan, WASD fly, Space up, Ctrl down, Q/E roll, Shift = 3x speed)"
)

FLY_SPEED = 0.18  # WASD speed, in view-distances per second
SPRINT_MULTIPLIER = 3.0  # holding Shift
SCROLL_STEP = 0.1  # scroll notch, in view-distances

INTRO_FOLLOW_SECONDS = 3.0

# 180° rotation about X: SLAM (x right, y down, z forward) -> view (y up, z back).
SLAM_TO_VIEW = np.array([1.0, -1.0, -1.0])

# Map points are drawn as splats whose world size is ~the typical spacing between
# neighbouring points, so the cloud reads as a continuous surface.
SPLAT_SPACING_FACTOR = 0.26  # splat diameter / median nearest-neighbour spacing
SPLAT_MIN_PX = 1.5
SPLAT_MAX_PX = 24.0
SPACING_RECOMPUTE_SECONDS = 1.0
SPACING_SAMPLE_POINTS = 20_000
SPACING_QUERY_POINTS = 600
PROJECTION_FOV_Y_DEG = 45.0

MARKER_DIAMETER_PX = 16  # camera sphere size on screen (logical pixels)
MARKER_COLOR = (1.0, 0.25, 0.2)

AXIS_LENGTH_PX = 60  # camera axis arrows, on-screen length (logical pixels)
AXIS_HEAD_FRACTION = 0.22  # arrowhead length / arrow length
AXIS_COLORS = ((1.0, 0.15, 0.15), (0.2, 1.0, 0.2), (0.3, 0.5, 1.0))  # camera x, y, z = red, green, blue

DEFAULT_YAW = 0.85
DEFAULT_PITCH = 0.38
ROLL_SPEED = 1.2  # rad/s while Q / E is held

_VERTEX_SHADER = """
#version 330 core
in vec3 in_position;
in vec3 in_color;
uniform mat4 mvp;
uniform float splat_world_size;  // splat diameter in world units
uniform float focal_px;          // projection focal length in framebuffer pixels
uniform vec2 splat_px_limits;    // min / max on-screen diameter
out vec3 v_color;
void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
    v_color = in_color;
    // Perspective-scaled: a fixed world-space size, so splats grow as you
    // approach and fill the gaps between neighbouring points.
    float px = splat_world_size * focal_px / max(gl_Position.w, 1e-4);
    gl_PointSize = clamp(px, splat_px_limits.x, splat_px_limits.y);
}
"""

_FRAGMENT_SHADER = """
#version 330 core
in vec3 v_color;
out vec4 fragColor;
void main() {
    fragColor = vec4(v_color, 1.0);
}
"""

# Coloured triangles, flat-shaded with a headlight so relief reads without a light source.
_MESH_VERTEX_SHADER = """
#version 330 core
in vec3 in_position;
in vec3 in_color;
in vec3 in_normal;
uniform mat4 mvp;
out vec3 v_color;
out vec3 v_normal;
out vec3 v_world;
void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
    v_color = in_color;
    v_normal = in_normal;
    v_world = in_position;
}
"""

_MESH_FRAGMENT_SHADER = """
#version 330 core
in vec3 v_color;
in vec3 v_normal;
in vec3 v_world;
uniform vec3 eye;
out vec4 fragColor;
void main() {
    float facing = abs(dot(normalize(v_normal), normalize(eye - v_world)));
    fragColor = vec4(v_color * (0.45 + 0.55 * facing), 1.0);
}
"""

_MARKER_VERTEX_SHADER = """
#version 330 core
in vec3 in_position;
uniform mat4 mvp;
uniform float point_size;
void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
    gl_PointSize = point_size;
}
"""

# Round point sprite shaded like a lit sphere.
_MARKER_FRAGMENT_SHADER = """
#version 330 core
out vec4 fragColor;
uniform vec3 base_color;
void main() {
    vec2 d = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(d, d);
    if (r2 > 1.0) discard;
    vec3 normal = vec3(d.x, -d.y, sqrt(1.0 - r2));
    float light = 0.3 + 0.7 * max(dot(normalize(vec3(-0.4, 0.5, 0.75)), normal), 0.0);
    float rim = smoothstep(0.85, 1.0, sqrt(r2));
    fragColor = vec4(base_color * light * (1.0 - 0.35 * rim), 1.0);
}
"""

_UI_VERTEX_SHADER = """
#version 330 core
in vec2 in_position;
in vec2 in_uv;
out vec2 v_uv;
uniform vec2 u_resolution;
void main() {
    vec2 ndc = (in_position / u_resolution) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_uv = in_uv;
}
"""

_UI_FRAGMENT_SHADER = """
#version 330 core
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_texture;
void main() {
    fragColor = texture(u_texture, v_uv);
}
"""


def _perspective(fov_y_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / np.tan(np.radians(fov_y_deg) / 2.0)
    mat = np.zeros((4, 4), dtype=np.float32)
    mat[0, 0] = f / aspect
    mat[1, 1] = f
    mat[2, 2] = (far + near) / (near - far)
    mat[2, 3] = (2.0 * far * near) / (near - far)
    mat[3, 2] = -1.0
    return mat


def _look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = center - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    norm = np.linalg.norm(right)
    if norm < 1e-8:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
    else:
        right /= norm
    true_up = np.cross(right, forward)
    rotation = np.eye(4, dtype=np.float32)
    rotation[0, :3] = right
    rotation[1, :3] = true_up
    rotation[2, :3] = -forward
    translation = np.eye(4, dtype=np.float32)
    translation[:3, 3] = -eye.astype(np.float32)
    return rotation @ translation


MESH_REBUILD_SECONDS = 0.5  # minimum time between mesh geometry uploads
MESH_MAX_FACES = 400_000


class MapRenderer:
    _RESET_X = 10
    _RESET_Y = 10
    _RESET_W = 86
    _RESET_H = 30
    _VIEW_X = 104
    _VIEW_Y = 10
    _VIEW_W = 128
    _VIEW_H = 30

    def __init__(self, width: int, height: int) -> None:
        self.logical_width = width
        self.logical_height = height
        self.yaw = DEFAULT_YAW
        self.pitch = DEFAULT_PITCH
        self.roll = 0.0  # lean of the camera around its view direction, + = to the right
        self.distance = 4.0
        self.pan = np.zeros(3, dtype=np.float64)
        self.focus = np.zeros(3, dtype=np.float64)
        self._user_camera = False
        self._last_motion_time = time.monotonic()
        self._intro_start: float | None = None
        self._follow_released = False
        self._orbit_drag = False
        self._orbit_around_target = False  # MMB orbits instead of looking in place
        self._pan_drag = False
        self._last_cursor = (0.0, 0.0)

        self._trajectory: list[np.ndarray] = []
        self.camera_rotation: np.ndarray | None = None  # camera -> world (3, 3); set by the SLAM script
        self._axes_vbo: moderngl.Buffer | None = None
        self._map_store: MapStore | None = None
        self._render_points = np.empty((0, 3), dtype=np.float64)
        self._render_colors = np.empty((0, 3), dtype=np.float64)

        self._window = None
        self._ctx: moderngl.Context | None = None
        self._program: moderngl.Program | None = None
        self._ui_program: moderngl.Program | None = None
        self._marker_program: moderngl.Program | None = None
        self._marker_vbo: moderngl.Buffer | None = None
        self._reset_texture: moderngl.Texture | None = None
        self._ui_vbo: moderngl.Buffer | None = None
        self._point_vbo: moderngl.Buffer | None = None
        self._path_vbo: moderngl.Buffer | None = None
        self._point_count = 0
        self._path_count = 0
        self._point_spacing: float | None = None
        self._spacing_time = 0.0

        self.view_mode = "points"  # "points" | "mesh"
        self._mesh = None  # MeshBuilder, set by the SLAM script
        self._mesh_program: moderngl.Program | None = None
        self._mesh_vbo: moderngl.Buffer | None = None
        self._mesh_vertex_count = 0
        self._mesh_built_version = -1
        self._mesh_built_time = 0.0
        self._view_textures: dict[str, moderngl.Texture] = {}
        self._view_vbo: moderngl.Buffer | None = None

    def set_mesh(self, mesh) -> None:
        self._mesh = mesh

    def toggle_view_mode(self) -> None:
        if self._mesh is None:
            return
        self.view_mode = "mesh" if self.view_mode == "points" else "points"
        self._mesh_built_version = -1  # force a rebuild when entering mesh mode
        self.draw()

    def setup_window(self, margin_x: int = 40, margin_y: int = 40) -> None:
        if not glfw.init():
            raise SystemExit("GLFW failed to initialize (GPU map window unavailable).")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.SCALE_TO_MONITOR, glfw.TRUE)

        self._window = glfw.create_window(
            self.logical_width,
            self.logical_height,
            MAP_WINDOW,
            None,
            None,
        )
        if self._window is None:
            glfw.terminate()
            raise SystemExit("Could not create OpenGL map window.")

        glfw.set_window_pos(self._window, margin_x, margin_y)
        glfw.make_context_current(self._window)
        glfw.swap_interval(1)

        self._ctx = moderngl.create_context()
        self._program = self._ctx.program(
            vertex_shader=_VERTEX_SHADER,
            fragment_shader=_FRAGMENT_SHADER,
        )
        self._ui_program = self._ctx.program(
            vertex_shader=_UI_VERTEX_SHADER,
            fragment_shader=_UI_FRAGMENT_SHADER,
        )
        self._marker_program = self._ctx.program(
            vertex_shader=_MARKER_VERTEX_SHADER,
            fragment_shader=_MARKER_FRAGMENT_SHADER,
        )
        self._mesh_program = self._ctx.program(
            vertex_shader=_MESH_VERTEX_SHADER,
            fragment_shader=_MESH_FRAGMENT_SHADER,
        )
        self._marker_vbo = self._ctx.buffer(reserve=12)
        self._setup_reset_button()
        self._setup_view_button()

        glfw.set_framebuffer_size_callback(self._window, self._on_resize)
        glfw.set_mouse_button_callback(self._window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self._window, self._on_cursor_move)
        glfw.set_scroll_callback(self._window, self._on_scroll)

        fb_w, fb_h = glfw.get_framebuffer_size(self._window)
        self._ctx.viewport = (0, 0, fb_w, fb_h)
        self._intro_start = time.monotonic()

    def _setup_reset_button(self) -> None:
        assert self._ctx is not None

        label = np.zeros((self._RESET_H, self._RESET_W, 3), dtype=np.uint8)
        label[:] = (48, 48, 52)
        cv2.rectangle(label, (0, 0), (self._RESET_W - 1, self._RESET_H - 1), (90, 90, 98), 1)
        cv2.putText(
            label,
            "Reset",
            (12, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (230, 230, 235),
            1,
            cv2.LINE_AA,
        )
        rgb = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

        self._reset_texture = self._ctx.texture(
            (self._RESET_W, self._RESET_H),
            3,
            rgb.tobytes(),
        )
        self._reset_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

        x0, y0 = self._RESET_X, self._RESET_Y
        x1, y1 = x0 + self._RESET_W, y0 + self._RESET_H
        quad = np.array(
            [
                x0, y0, 0.0, 0.0,
                x1, y0, 1.0, 0.0,
                x0, y1, 0.0, 1.0,
                x1, y1, 1.0, 1.0,
            ],
            dtype="f4",
        )
        self._ui_vbo = self._ctx.buffer(quad.tobytes())

    def _setup_view_button(self) -> None:
        assert self._ctx is not None
        for mode, text in (("points", "View: Points"), ("mesh", "View: Mesh")):
            label = np.zeros((self._VIEW_H, self._VIEW_W, 3), dtype=np.uint8)
            label[:] = (48, 48, 52) if mode == "points" else (70, 120, 80)
            cv2.rectangle(label, (0, 0), (self._VIEW_W - 1, self._VIEW_H - 1), (90, 90, 98), 1)
            cv2.putText(
                label, text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (230, 230, 235), 1, cv2.LINE_AA,
            )
            rgb = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
            texture = self._ctx.texture((self._VIEW_W, self._VIEW_H), 3, rgb.tobytes())
            texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._view_textures[mode] = texture

        x0, y0 = self._VIEW_X, self._VIEW_Y
        x1, y1 = x0 + self._VIEW_W, y0 + self._VIEW_H
        quad = np.array(
            [x0, y0, 0.0, 0.0, x1, y0, 1.0, 0.0, x0, y1, 0.0, 1.0, x1, y1, 1.0, 1.0],
            dtype="f4",
        )
        self._view_vbo = self._ctx.buffer(quad.tobytes())

    def _in_view_button(self, x: float, y: float) -> bool:
        return (
            self._mesh is not None
            and self._VIEW_X <= x <= self._VIEW_X + self._VIEW_W
            and self._VIEW_Y <= y <= self._VIEW_Y + self._VIEW_H
        )

    def _draw_view_button(self) -> None:
        if (
            self._mesh is None
            or self._ctx is None
            or self._ui_program is None
            or self._view_vbo is None
            or self._window is None
        ):
            return
        win_w, win_h = glfw.get_window_size(self._window)
        self._ctx.disable(moderngl.DEPTH_TEST)
        self._view_textures[self.view_mode].use(0)
        self._ui_program["u_texture"].value = 0
        self._ui_program["u_resolution"].value = (float(win_w), float(win_h))
        vao = self._ctx.vertex_array(
            self._ui_program, [(self._view_vbo, "2f 2f", "in_position", "in_uv")]
        )
        vao.render(moderngl.TRIANGLE_STRIP)
        vao.release()

    def _upload_mesh(self) -> None:
        """Rebuild the triangle buffer when the mesh changed (throttled)."""
        assert self._ctx is not None
        if self._mesh is None or self._map_store is None:
            self._mesh_vertex_count = 0
            return
        now = time.monotonic()
        stale = self._mesh.version != self._mesh_built_version
        if not stale or (
            self._mesh_built_version >= 0 and now - self._mesh_built_time < MESH_REBUILD_SECONDS
        ):
            return
        self._mesh_built_version = self._mesh.version
        self._mesh_built_time = now

        pos, col, normal = self._mesh.mesh_arrays(MESH_MAX_FACES)
        if len(pos) == 0:
            self._mesh_vertex_count = 0
            return
        pos = pos * SLAM_TO_VIEW
        normal = np.repeat((normal * SLAM_TO_VIEW)[:, None, :], 3, axis=1)
        data = np.concatenate([pos, col, normal], axis=2).astype("f4")  # (F, 3, 9)
        raw = data.tobytes()
        if self._mesh_vbo is None or self._mesh_vbo.size < len(raw):
            if self._mesh_vbo is not None:
                self._mesh_vbo.release()
            self._mesh_vbo = self._ctx.buffer(raw)
        else:
            self._mesh_vbo.write(raw)
        self._mesh_vertex_count = len(pos) * 3

    def _target(self) -> np.ndarray:
        return self.focus + self.pan

    def _orbit_direction(self) -> np.ndarray:
        return np.array(
            [
                np.cos(self.pitch) * np.sin(self.yaw),
                np.sin(self.pitch),
                np.cos(self.pitch) * np.cos(self.yaw),
            ],
            dtype=np.float64,
        )

    def _camera_right_up(self) -> tuple[np.ndarray, np.ndarray]:
        direction = self._orbit_direction()
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        forward = -direction
        right = np.cross(forward, world_up)
        norm = np.linalg.norm(right)
        if norm < 1e-8:
            world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            right = np.cross(forward, world_up)
            right /= np.linalg.norm(right)
        else:
            right /= norm
        up = np.cross(right, forward)
        if self.roll != 0.0:
            # lean the camera around its view direction (E = to the right, Q = to the left)
            cos, sin = np.cos(self.roll), np.sin(self.roll)
            right, up = cos * right - sin * up, cos * up + sin * right
        return right, up

    def _in_reset_button(self, x: float, y: float) -> bool:
        return (
            self._RESET_X <= x <= self._RESET_X + self._RESET_W
            and self._RESET_Y <= y <= self._RESET_Y + self._RESET_H
        )

    def reset_view(self) -> None:
        self.yaw = DEFAULT_YAW
        self.pitch = DEFAULT_PITCH
        self.roll = 0.0
        self.pan[:] = 0.0
        self._user_camera = False
        self._follow_released = False
        self._intro_start = time.monotonic()

    def _on_resize(self, _window, width: int, height: int) -> None:
        if self._ctx is not None and height > 0:
            self._ctx.viewport = (0, 0, width, height)

    def _on_mouse_button(self, _window, button: int, action: int, _mods: int) -> None:
        x, y = glfw.get_cursor_pos(self._window)

        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
            if self._in_reset_button(x, y):
                self.reset_view()
                self.draw()
                return
            if self._in_view_button(x, y):
                self.toggle_view_mode()
                return
            self._orbit_drag = True
            self._orbit_around_target = False
            self._last_cursor = (x, y)
        elif button == glfw.MOUSE_BUTTON_LEFT and action == glfw.RELEASE:
            self._orbit_drag = False
        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            self._orbit_drag = action == glfw.PRESS
            self._orbit_around_target = True
            if self._orbit_drag:
                self._last_cursor = (x, y)
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self._pan_drag = action == glfw.PRESS
            if self._pan_drag:
                self._last_cursor = (x, y)

    def _on_cursor_move(self, _window, x: float, y: float) -> None:
        if not self._orbit_drag and not self._pan_drag:
            return

        dx = x - self._last_cursor[0]
        dy = y - self._last_cursor[1]
        self._last_cursor = (x, y)

        if self._orbit_drag:
            eye = self._target() + self._orbit_direction() * self.distance
            self.yaw -= dx * 0.008
            self.pitch += dy * 0.008
            self.pitch = float(np.clip(self.pitch, -1.45, 1.45))
            if not self._orbit_around_target:
                # Look around in place: keep the eye fixed and swing the target.
                new_target = eye - self._orbit_direction() * self.distance
                self.pan = new_target - self.focus
        elif self._pan_drag:
            right, up = self._camera_right_up()
            scale = self.distance * 0.0025
            self.pan += (-dx * scale) * right + (dy * scale) * up

        self._user_camera = True
        self.draw()

    def _on_scroll(self, _window, _xoffset: float, yoffset: float) -> None:
        if yoffset == 0:
            return
        # Scroll flies the camera forward/back along the view direction.
        self.pan += self._view_forward() * (self.distance * SCROLL_STEP * yoffset)
        self._user_camera = True
        self.draw()

    def _view_forward(self) -> np.ndarray:
        return -self._orbit_direction()

    def _key_down(self, key: int) -> bool:
        return glfw.get_key(self._window, key) == glfw.PRESS

    def _apply_keyboard_motion(self) -> None:
        """WASD = fly, Space = up, Ctrl = down, Shift = 3x speed, Q / E = roll left / right (while the map window has focus)."""
        now = time.monotonic()
        dt = min(now - self._last_motion_time, 0.1)
        self._last_motion_time = now

        forward = self._view_forward()
        right, _ = self._camera_right_up()
        world_up = np.array([0.0, 1.0, 0.0])

        move = np.zeros(3, dtype=np.float64)
        if self._key_down(glfw.KEY_W):
            move += forward
        if self._key_down(glfw.KEY_S):
            move -= forward
        if self._key_down(glfw.KEY_D):
            move += right
        if self._key_down(glfw.KEY_A):
            move -= right
        if self._key_down(glfw.KEY_SPACE):
            move += world_up
        if self._key_down(glfw.KEY_LEFT_CONTROL) or self._key_down(glfw.KEY_RIGHT_CONTROL):
            move -= world_up

        lean = float(self._key_down(glfw.KEY_E)) - float(self._key_down(glfw.KEY_Q))
        if lean != 0.0:
            self.roll += lean * ROLL_SPEED * dt
            self._user_camera = True

        length = float(np.linalg.norm(move))
        if length < 1e-9:
            if lean != 0.0:
                self.draw()
            return
        speed = max(self.distance, 0.5) * FLY_SPEED
        if self._key_down(glfw.KEY_LEFT_SHIFT) or self._key_down(glfw.KEY_RIGHT_SHIFT):
            speed *= SPRINT_MULTIPLIER
        self.pan += (move / length) * speed * dt
        self._user_camera = True
        self.draw()

    @property
    def context(self) -> moderngl.Context:
        """The window's moderngl context (after setup_window), for rendering other things alongside the map."""
        if self._ctx is None:
            raise RuntimeError("call setup_window() first")
        return self._ctx

    def poll(self) -> bool:
        if self._window is None:
            return False
        glfw.poll_events()
        self._apply_keyboard_motion()
        return not glfw.window_should_close(self._window)

    def show(self, trajectory: list[np.ndarray], map_store: MapStore) -> None:
        self._trajectory = trajectory
        self._map_store = map_store
        self.draw()

    def draw(self) -> None:
        if self._ctx is None or self._program is None or self._window is None:
            return

        traj = np.array(self._trajectory, dtype=np.float64)

        fb_w, fb_h = glfw.get_framebuffer_size(self._window)
        aspect = fb_w / max(fb_h, 1)
        projection = _perspective(PROJECTION_FOV_Y_DEG, aspect, 0.02, 500.0)

        target = self._target()
        direction = self._orbit_direction()
        eye = target + direction * self.distance

        # SLAM space is OpenCV-style (y down, z forward); the viewer is y up.
        # The map is culled in SLAM space and converted to view space for display.
        if self._map_store is not None:
            points, colors = self._map_store.render_subset(
                eye * SLAM_TO_VIEW,
                target * SLAM_TO_VIEW,
                aspect,
            )
            self._render_points = points * SLAM_TO_VIEW
            self._render_colors = colors
        else:
            self._render_points = np.empty((0, 3), dtype=np.float64)
            self._render_colors = np.empty((0, 3), dtype=np.float64)

        if len(traj):
            traj = traj * SLAM_TO_VIEW

        self._update_focus_and_distance(traj, self._render_points)
        self._upload_geometry(traj, self._render_points, self._render_colors)

        view = _look_at(eye, target, self._camera_right_up()[1])

        mvp = projection @ view
        self._program["mvp"].write(mvp.astype(np.float32).T.tobytes())
        mesh_mode = self.view_mode == "mesh" and self._mesh is not None
        if mesh_mode:
            self._upload_mesh()
        else:
            self._update_point_spacing(self._render_points)
        spacing = self._point_spacing if self._point_spacing is not None else 0.0
        focal_px = fb_h / (2.0 * np.tan(np.radians(PROJECTION_FOV_Y_DEG) / 2.0))
        self._program["splat_world_size"].value = float(spacing * SPLAT_SPACING_FACTOR)
        self._program["focal_px"].value = float(focal_px)
        self._program["splat_px_limits"].value = (SPLAT_MIN_PX, SPLAT_MAX_PX)

        self._ctx.enable(moderngl.DEPTH_TEST | moderngl.PROGRAM_POINT_SIZE)
        self._ctx.clear(0.11, 0.11, 0.12)

        if self._path_vbo is not None and self._path_count >= 2:
            path_vao = self._ctx.vertex_array(
                self._program,
                [(self._path_vbo, "3f 3f", "in_position", "in_color")],
            )
            path_vao.render(moderngl.LINE_STRIP, vertices=self._path_count)
            path_vao.release()

        if mesh_mode:
            if self._mesh_vbo is not None and self._mesh_vertex_count > 0:
                self._mesh_program["mvp"].write(mvp.astype(np.float32).T.tobytes())
                self._mesh_program["eye"].value = tuple(float(v) for v in eye)
                mesh_vao = self._ctx.vertex_array(
                    self._mesh_program,
                    [(self._mesh_vbo, "3f 3f 3f", "in_position", "in_color", "in_normal")],
                )
                mesh_vao.render(moderngl.TRIANGLES, vertices=self._mesh_vertex_count)
                mesh_vao.release()
        elif self._point_vbo is not None and self._point_count > 0:
            point_vao = self._ctx.vertex_array(
                self._program,
                [(self._point_vbo, "3f 3f", "in_position", "in_color")],
            )
            point_vao.render(moderngl.POINTS, vertices=self._point_count)
            point_vao.release()

        if len(traj) > 0:
            self._draw_camera_marker(traj[-1], mvp)
            if self.camera_rotation is not None:
                self._draw_camera_axes(traj[-1], mvp, self.camera_rotation)

        self._draw_reset_button()
        self._draw_view_button()

        glfw.swap_buffers(self._window)

    def _update_point_spacing(self, points: np.ndarray) -> None:
        """Median nearest-neighbour spacing of the drawn points (re-estimated ~1x/s)."""
        now = time.monotonic()
        if self._point_spacing is not None and now - self._spacing_time < SPACING_RECOMPUTE_SECONDS:
            return
        n = len(points)
        if n < 50:
            return
        self._spacing_time = now

        rng = np.random.default_rng(0)
        sample_n = min(n, SPACING_SAMPLE_POINTS)
        sample = points[rng.choice(n, sample_n, replace=False)].astype(np.float32)
        queries = sample[: min(sample_n, SPACING_QUERY_POINTS)]

        sample_sq = np.einsum("ij,ij->i", sample, sample)
        nearest = np.empty(len(queries), dtype=np.float32)
        for start in range(0, len(queries), 150):
            chunk = queries[start : start + 150]
            d2 = (
                np.einsum("ij,ij->i", chunk, chunk)[:, None]
                + sample_sq[None, :]
                - 2.0 * chunk @ sample.T
            )
            d2[d2 < 1e-12] = np.inf  # self matches
            nearest[start : start + 150] = d2.min(axis=1)
        spacing_sample = float(np.sqrt(max(np.median(nearest), 0.0)))
        if not np.isfinite(spacing_sample) or spacing_sample <= 0:
            return
        # Surfaces: spacing scales with 1/sqrt(density), correct for subsampling.
        spacing = spacing_sample * np.sqrt(sample_n / n)
        if self._point_spacing is None:
            self._point_spacing = spacing
        else:
            self._point_spacing = 0.5 * self._point_spacing + 0.5 * spacing

    def _draw_camera_marker(self, position: np.ndarray, mvp: np.ndarray) -> None:
        """Sphere at the current camera position, always drawn on top of the cloud."""
        if (
            self._ctx is None
            or self._marker_program is None
            or self._marker_vbo is None
            or self._window is None
        ):
            return

        # Constant on-screen size (the map scale is arbitrary), in framebuffer pixels.
        win_h = max(glfw.get_window_size(self._window)[1], 1)
        fb_h = glfw.get_framebuffer_size(self._window)[1]
        size_px = MARKER_DIAMETER_PX * fb_h / win_h

        self._marker_vbo.write(position.astype("f4").tobytes())
        self._marker_program["mvp"].write(mvp.astype(np.float32).T.tobytes())
        self._marker_program["point_size"].value = float(size_px)
        self._marker_program["base_color"].value = MARKER_COLOR

        self._ctx.disable(moderngl.DEPTH_TEST)
        marker_vao = self._ctx.vertex_array(
            self._marker_program, [(self._marker_vbo, "3f", "in_position")]
        )
        marker_vao.render(moderngl.POINTS, vertices=1)
        marker_vao.release()
        self._ctx.enable(moderngl.DEPTH_TEST)

    def _draw_camera_axes(self, position: np.ndarray, mvp: np.ndarray, rotation: np.ndarray) -> None:
        """Camera x / y / z axes as red / green / blue arrows from the camera marker.

        Camera axes are x right, y down, z forward (the SLAM convention); the columns of
        `rotation` are those axes in world space. Constant on-screen length, drawn on top.
        """
        if self._ctx is None or self._program is None or self._window is None:
            return

        win_h = max(glfw.get_window_size(self._window)[1], 1)
        fb_h = glfw.get_framebuffer_size(self._window)[1]
        focal_px = fb_h / (2.0 * np.tan(np.radians(PROJECTION_FOV_Y_DEG) / 2.0))
        depth = float((mvp @ np.append(position, 1.0))[3])
        if depth <= 1e-4:
            return  # camera is behind the viewer
        length = AXIS_LENGTH_PX * fb_h / win_h * depth / focal_px
        head = AXIS_HEAD_FRACTION * length

        vertices: list[np.ndarray] = []
        for axis in range(3):
            direction = rotation[:, axis] * SLAM_TO_VIEW
            direction = direction / np.linalg.norm(direction)
            tip = position + direction * length
            helper = np.array([0.0, 1.0, 0.0]) if abs(direction[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
            side_a = np.cross(direction, helper)
            side_a /= np.linalg.norm(side_a)
            side_b = np.cross(direction, side_a)
            barb_base = tip - direction * head
            color = np.array(AXIS_COLORS[axis])
            segments = [(position, tip)] + [(tip, barb_base + side * 0.4 * head) for side in (side_a, -side_a, side_b, -side_b)]
            for start, end in segments:
                vertices.append(np.concatenate([start, color]))
                vertices.append(np.concatenate([end, color]))

        data = np.array(vertices, dtype="f4")
        if self._axes_vbo is None:
            self._axes_vbo = self._ctx.buffer(reserve=data.nbytes)
        self._axes_vbo.write(data.tobytes())

        self._ctx.disable(moderngl.DEPTH_TEST)
        self._ctx.line_width = 2.0
        axes_vao = self._ctx.vertex_array(self._program, [(self._axes_vbo, "3f 3f", "in_position", "in_color")])
        axes_vao.render(moderngl.LINES, vertices=len(vertices))
        axes_vao.release()
        self._ctx.line_width = 1.0
        self._ctx.enable(moderngl.DEPTH_TEST)

    def _draw_reset_button(self) -> None:
        if (
            self._ctx is None
            or self._ui_program is None
            or self._ui_vbo is None
            or self._reset_texture is None
            or self._window is None
        ):
            return

        win_w, win_h = glfw.get_window_size(self._window)
        self._ctx.disable(moderngl.DEPTH_TEST)
        self._reset_texture.use(0)
        self._ui_program["u_texture"].value = 0
        self._ui_program["u_resolution"].value = (float(win_w), float(win_h))

        ui_vao = self._ctx.vertex_array(
            self._ui_program,
            [(self._ui_vbo, "2f 2f", "in_position", "in_uv")],
        )
        ui_vao.render(moderngl.TRIANGLE_STRIP)
        ui_vao.release()

    def _upload_geometry(
        self,
        trajectory: np.ndarray,
        map_points: np.ndarray,
        map_colors: np.ndarray,
    ) -> None:
        assert self._ctx is not None

        if len(map_points) > 0:
            colors = map_colors.astype(np.float32)
            points = map_points.astype(np.float32)
            interleaved = np.hstack([points, colors]).astype("f4")
            if self._point_vbo is None or self._point_vbo.size < interleaved.nbytes:
                if self._point_vbo is not None:
                    self._point_vbo.release()
                self._point_vbo = self._ctx.buffer(interleaved.tobytes())
            else:
                self._point_vbo.write(interleaved.tobytes())
            self._point_count = len(map_points)
        else:
            self._point_count = 0

        if len(trajectory) >= 2:
            path_color = np.tile(
                np.array([[1.0, 0.86, 0.31]], dtype=np.float32),
                (len(trajectory), 1),
            )
            path_pos = trajectory.astype(np.float32)
            path_data = np.hstack([path_pos, path_color]).astype("f4")
            if self._path_vbo is None:
                self._path_vbo = self._ctx.buffer(path_data.tobytes())
            else:
                if self._path_vbo.size < path_data.nbytes:
                    self._path_vbo.release()
                    self._path_vbo = self._ctx.buffer(path_data.tobytes())
                else:
                    self._path_vbo.write(path_data.tobytes())
            self._path_count = len(trajectory)
        else:
            self._path_count = 0

    def _update_focus_and_distance(
        self, trajectory: np.ndarray, map_points: np.ndarray
    ) -> None:
        if self._follow_released:
            return
        if self._intro_start is None:
            return
        if time.monotonic() - self._intro_start >= INTRO_FOLLOW_SECONDS:
            self._follow_released = True
            return

        if len(trajectory) == 0 and len(map_points) == 0:
            return

        if len(map_points) > 0:
            samples = (
                map_points
                if len(trajectory) == 0
                else np.vstack([trajectory, map_points])
            )
        else:
            samples = trajectory

        center = samples.mean(axis=0)
        self.focus = 0.92 * self.focus + 0.08 * center

        if self._user_camera:
            return

        radius = float(np.max(np.linalg.norm(samples - self.focus, axis=1)))
        radius = max(radius, 0.5)
        self.distance = max(1.5, radius * 2.2)

    def close(self) -> None:
        if self._point_vbo is not None:
            self._point_vbo.release()
        if self._path_vbo is not None:
            self._path_vbo.release()
        if self._ui_vbo is not None:
            self._ui_vbo.release()
        if self._marker_vbo is not None:
            self._marker_vbo.release()
        if self._axes_vbo is not None:
            self._axes_vbo.release()
        if self._reset_texture is not None:
            self._reset_texture.release()
        for texture in self._view_textures.values():
            texture.release()
        if self._view_vbo is not None:
            self._view_vbo.release()
        if self._mesh_vbo is not None:
            self._mesh_vbo.release()
        if self._window is not None:
            glfw.destroy_window(self._window)
        glfw.terminate()
