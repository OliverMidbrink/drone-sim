"""A 3D body for the quadrotor: frame, motors, spinning propellers, landing legs, the two lens bumps.

Drawn in the body frame (x right, y up, nose along -z) at the real size of the physics model:
arm length from drone.QuadParams, collision sphere around it. The nose is marked orange and the
front motors red so the heading can be read from any side. Only for looking at: nothing reads it back.
"""

from __future__ import annotations

import moderngl
import numpy as np

from drone import QuadParams

_VERTEX = """
#version 330 core
uniform mat4 vp;
uniform mat4 model;
in vec3 position;
in vec3 normal;
in vec3 color;
out vec3 v_normal;
out vec3 v_color;
void main() {
    vec4 world = model * vec4(position, 1.0);
    gl_Position = vp * world;
    v_normal = mat3(model) * normal;
    v_color = color;
}
"""

_FRAGMENT = """
#version 330 core
uniform vec3 sun_dir;
in vec3 v_normal;
in vec3 v_color;
out vec4 frag;
void main() {
    vec3 n = normalize(v_normal);
    float diffuse = max(dot(n, sun_dir), 0.0);
    float sky = 0.5 + 0.5 * n.y;
    vec3 albedo = pow(v_color, vec3(2.2));
    vec3 lit = albedo * (1.05 * diffuse + 0.35 * sky + 0.08);
    frag = vec4(pow(clamp(lit, 0.0, 1.0), vec3(1.0 / 2.2)), 1.0);
}
"""

_CUBE_FACES = (  # (normal, two edge directions): a unit cube's six faces
    ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
    ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
    ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
    ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
    ((0, 0, -1), (0, 1, 0), (1, 0, 0)),
)


def _yaw_matrix(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def box(center, size, color, yaw: float = 0.0) -> np.ndarray:
    """A box as 36 vertices of (position, normal, colour), turned by `yaw` about its own vertical axis."""
    rows = []
    rotation = _yaw_matrix(yaw)
    half = np.asarray(size, dtype=float) / 2.0
    for normal, u, v in _CUBE_FACES:
        normal, u, v = (np.array(a, dtype=float) for a in (normal, u, v))
        corners = [normal + su * u + sv * v for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, -1), (1, 1), (-1, 1))]
        for corner in corners:
            position = np.asarray(center, dtype=float) + rotation @ (corner * half)
            rows.append([*position, *(rotation @ normal), *color])
    return np.array(rows, np.float32)


def cylinder(center, radius: float, height: float, color, segments: int = 14) -> np.ndarray:
    """A vertical cylinder with flat caps."""
    rows = []
    cx, cy, cz = center
    top, bottom = cy + height / 2.0, cy - height / 2.0
    angles = np.linspace(0.0, 2.0 * np.pi, segments + 1)
    for a0, a1 in zip(angles[:-1], angles[1:]):
        p0 = np.array([np.cos(a0), np.sin(a0)])
        p1 = np.array([np.cos(a1), np.sin(a1)])
        mid = 0.5 * (a0 + a1)
        side = (np.cos(mid), 0.0, np.sin(mid))
        for x, y, z, normal in (
            (cx + radius * p0[0], bottom, cz + radius * p0[1], side),
            (cx + radius * p1[0], bottom, cz + radius * p1[1], side),
            (cx + radius * p1[0], top, cz + radius * p1[1], side),
            (cx + radius * p0[0], bottom, cz + radius * p0[1], side),
            (cx + radius * p1[0], top, cz + radius * p1[1], side),
            (cx + radius * p0[0], top, cz + radius * p0[1], side),
            (cx, top, cz, (0, 1, 0)),
            (cx + radius * p0[0], top, cz + radius * p0[1], (0, 1, 0)),
            (cx + radius * p1[0], top, cz + radius * p1[1], (0, 1, 0)),
        ):
            rows.append([x, y, z, *normal, *color])
    return np.array(rows, np.float32)


class DroneModel:
    def __init__(self, ctx: moderngl.Context, params: QuadParams, sun_direction: np.ndarray) -> None:
        self.ctx, self.params = ctx, params
        self.program = ctx.program(vertex_shader=_VERTEX, fragment_shader=_FRAGMENT)
        self.program["sun_dir"].value = tuple(float(v) for v in sun_direction)
        d = params.arm / np.sqrt(2.0)
        # motor order of drone.mixer_matrix: front-left, front-right, rear-left, rear-right (nose is -z)
        self.motor_positions = np.array([[-d, 0.02, -d], [d, 0.02, -d], [-d, 0.02, d], [d, 0.02, d]])
        self.spin_direction = np.array([1.0, -1.0, -1.0, 1.0])
        self.phase = np.zeros(4)
        self.prop_radius = 0.062
        static = self._frame(d)
        self.static_count = len(static)
        self.static_buffer = ctx.buffer(static.tobytes())
        self.static_vao = ctx.vertex_array(self.program, [(self.static_buffer, "3f 3f 3f", "position", "normal", "color")])
        self.blade_buffer = ctx.buffer(reserve=4 * 2 * 36 * 9 * 4)
        self.blade_vao = ctx.vertex_array(self.program, [(self.blade_buffer, "3f 3f 3f", "position", "normal", "color")])

    def _frame(self, d: float) -> np.ndarray:
        dark, carbon = (0.10, 0.10, 0.12), (0.18, 0.18, 0.2)
        parts = [
            box((0.0, 0.0, 0.0), (0.075, 0.03, 0.115), carbon),  # body
            box((0.0, 0.027, 0.012), (0.05, 0.026, 0.085), (0.95, 0.75, 0.1)),  # battery
            box((0.0, 0.003, -0.066), (0.03, 0.02, 0.022), (1.0, 0.4, 0.05)),  # the nose
            box((-0.06, 0.0, 0.0), (0.03, 0.026, 0.026), (0.1, 0.35, 0.9)),  # left lens
            box((0.06, 0.0, 0.0), (0.03, 0.026, 0.026), (0.1, 0.35, 0.9)),  # right lens
        ]
        arm_length = float(np.hypot(d, d))
        for index, (x, _, z) in enumerate(self.motor_positions):
            yaw = np.arctan2(-z, x)  # the arm's long axis points from the body to the motor
            parts.append(box((x / 2.0, 0.0, z / 2.0), (arm_length, 0.012, 0.016), dark, yaw=yaw))
            front = index < 2
            parts.append(cylinder((x, 0.012, z), 0.018, 0.026, (0.85, 0.12, 0.1) if front else (0.8, 0.8, 0.82)))
        for sx in (-1.0, 1.0):  # landing legs
            for sz in (-1.0, 1.0):
                bottom = -(self.params.radius - 0.004)
                parts.append(box((sx * 0.055, (bottom - 0.01) / 2.0, sz * 0.06), (0.008, abs(bottom) - 0.01, 0.008), dark))
                parts.append(box((sx * 0.055, bottom + 0.004, sz * 0.06), (0.03, 0.008, 0.03), (0.08, 0.08, 0.09)))
        return np.vstack(parts)

    def _blades(self) -> np.ndarray:
        parts = []
        for index, position in enumerate(self.motor_positions):
            color = (0.9, 0.9, 0.92) if index < 2 else (0.3, 0.3, 0.33)
            for offset in (0.0, np.pi / 2.0):
                parts.append(box((position[0], position[1] + 0.02, position[2]), (2.0 * self.prop_radius, 0.003, 0.012), color, yaw=self.phase[index] + offset))
        return np.vstack(parts)

    def draw(self, view_projection: np.ndarray, position: np.ndarray, rotation: np.ndarray, motors: np.ndarray, dt: float) -> None:
        """Draw into whatever framebuffer / viewport is current. `motors` are the normalised motor speeds 0..1."""
        self.phase += self.spin_direction * motors * 55.0 * dt  # a calm visual rate: real propellers would just be a blur
        blades = self._blades()
        self.blade_buffer.write(blades.tobytes())
        model = np.eye(4)
        model[:3, :3] = rotation
        model[:3, 3] = position
        self.program["vp"].write(view_projection.astype("f4").T.tobytes())
        self.program["model"].write(model.astype("f4").T.tobytes())
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        self.static_vao.render(moderngl.TRIANGLES, vertices=self.static_count)
        self.blade_vao.render(moderngl.TRIANGLES, vertices=len(blades))
