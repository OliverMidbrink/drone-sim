"""GPU 3D map view (OpenGL via moderngl): orbit, pan, dolly zoom."""

from __future__ import annotations

import cv2
import glfw
import moderngl
import numpy as np

from map_store import MapStore

MAP_WINDOW = "3D SLAM map (LMB orbit, RMB pan, scroll dolly)"

DEFAULT_YAW = 0.85
DEFAULT_PITCH = -0.38

_VERTEX_SHADER = """
#version 330 core
in vec3 in_position;
in vec3 in_color;
uniform mat4 mvp;
out vec3 v_color;
void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
    v_color = in_color;
    gl_PointSize = 2.5;
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


class MapRenderer:
    _RESET_X = 10
    _RESET_Y = 10
    _RESET_W = 86
    _RESET_H = 30

    def __init__(self, width: int, height: int) -> None:
        self.logical_width = width
        self.logical_height = height
        self.yaw = DEFAULT_YAW
        self.pitch = DEFAULT_PITCH
        self.distance = 4.0
        self.pan = np.zeros(3, dtype=np.float64)
        self.focus = np.zeros(3, dtype=np.float64)
        self._user_camera = False
        self._framed = False
        self._orbit_drag = False
        self._pan_drag = False
        self._last_cursor = (0.0, 0.0)

        self._trajectory: list[np.ndarray] = []
        self._map_store: MapStore | None = None
        self._render_points = np.empty((0, 3), dtype=np.float64)
        self._render_colors = np.empty((0, 3), dtype=np.float64)

        self._window = None
        self._ctx: moderngl.Context | None = None
        self._program: moderngl.Program | None = None
        self._ui_program: moderngl.Program | None = None
        self._reset_texture: moderngl.Texture | None = None
        self._ui_vbo: moderngl.Buffer | None = None
        self._point_vbo: moderngl.Buffer | None = None
        self._path_vbo: moderngl.Buffer | None = None
        self._point_count = 0
        self._path_count = 0

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
        self._setup_reset_button()

        glfw.set_framebuffer_size_callback(self._window, self._on_resize)
        glfw.set_mouse_button_callback(self._window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self._window, self._on_cursor_move)
        glfw.set_scroll_callback(self._window, self._on_scroll)

        fb_w, fb_h = glfw.get_framebuffer_size(self._window)
        self._ctx.viewport = (0, 0, fb_w, fb_h)

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
        return right, up

    def _in_reset_button(self, x: float, y: float) -> bool:
        return (
            self._RESET_X <= x <= self._RESET_X + self._RESET_W
            and self._RESET_Y <= y <= self._RESET_Y + self._RESET_H
        )

    def reset_view(self) -> None:
        self.yaw = DEFAULT_YAW
        self.pitch = DEFAULT_PITCH
        self.pan[:] = 0.0
        self._user_camera = False
        self._framed = False

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
            self._orbit_drag = True
            self._last_cursor = (x, y)
        elif button == glfw.MOUSE_BUTTON_LEFT and action == glfw.RELEASE:
            self._orbit_drag = False
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
            self.yaw -= dx * 0.008
            self.pitch += dy * 0.008
            self.pitch = float(np.clip(self.pitch, -1.45, 1.45))
        elif self._pan_drag:
            right, up = self._camera_right_up()
            scale = self.distance * 0.0025
            self.pan += (-dx * scale) * right + (dy * scale) * up

        self._user_camera = True
        self.draw()

    def _on_scroll(self, _window, _xoffset: float, yoffset: float) -> None:
        if yoffset == 0:
            return
        step = 0.9 if yoffset > 0 else 1.1
        self.distance = max(0.05, self.distance * step)
        self._user_camera = True
        self.draw()

    def poll(self) -> bool:
        if self._window is None:
            return False
        glfw.poll_events()
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
        projection = _perspective(45.0, aspect, 0.02, 500.0)

        target = self._target()
        direction = self._orbit_direction()
        eye = target + direction * self.distance

        if self._map_store is not None:
            self._render_points, self._render_colors = self._map_store.render_subset(
                eye,
                target,
                aspect,
            )
        else:
            self._render_points = np.empty((0, 3), dtype=np.float64)
            self._render_colors = np.empty((0, 3), dtype=np.float64)

        self._update_focus_and_distance(traj, self._render_points)
        self._upload_geometry(traj, self._render_points, self._render_colors)

        view = _look_at(eye, target, np.array([0.0, 1.0, 0.0]))

        mvp = projection @ view
        self._program["mvp"].write(mvp.astype(np.float32).T.tobytes())

        self._ctx.enable(moderngl.DEPTH_TEST | moderngl.PROGRAM_POINT_SIZE)
        self._ctx.clear(0.11, 0.11, 0.12)

        if self._path_vbo is not None and self._path_count >= 2:
            path_vao = self._ctx.vertex_array(
                self._program,
                [(self._path_vbo, "3f 3f", "in_position", "in_color")],
            )
            path_vao.render(moderngl.LINE_STRIP, vertices=self._path_count)
            path_vao.release()

        if self._point_vbo is not None and self._point_count > 0:
            point_vao = self._ctx.vertex_array(
                self._program,
                [(self._point_vbo, "3f 3f", "in_position", "in_color")],
            )
            point_vao.render(moderngl.POINTS, vertices=self._point_count)
            point_vao.release()

        self._draw_reset_button()

        glfw.swap_buffers(self._window)

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
        if not self._framed:
            self.distance = max(1.5, radius * 2.2)
            self._framed = True

    def close(self) -> None:
        if self._point_vbo is not None:
            self._point_vbo.release()
        if self._path_vbo is not None:
            self._path_vbo.release()
        if self._ui_vbo is not None:
            self._ui_vbo.release()
        if self._reset_texture is not None:
            self._reset_texture.release()
        if self._window is not None:
            glfw.destroy_window(self._window)
        glfw.terminate()
