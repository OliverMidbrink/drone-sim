"""A 360-camera style fisheye on the drone: one circular fisheye lens (160 degrees by default, 180 = a full hemisphere), square picture.

The scene is rendered into five 92 degree square views (front, left, right, up, down of the
camera) and a shader maps them to an equidistant fisheye picture: the distance from the image
centre is proportional to the angle from the optical axis. The picture is square and the image
circle touches its four edges, so it covers the lens' field of view across the picture in every direction;
the corners outside the circle are black. See lens.py for the geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import moderngl
import numpy as np

from camera import Camera
from lens import LENS_FOV, mount_rotation
from render import Renderer

FACE_FOV = 92.0  # degrees; a little more than 90 so the seams between faces blend cleanly
_TAN_HALF = float(np.tan(np.radians(FACE_FOV / 2.0)))

# (forward, up) of each face in the camera's own frame: x right, y up, -z forward.
_FACES = (
    ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ((0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),
    ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
)

_VERTEX = """
#version 330 core
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""


def _vec(v: tuple[float, float, float]) -> str:
    return f"vec3({v[0]:.1f}, {v[1]:.1f}, {v[2]:.1f})"


_FRAGMENT = (
    """
#version 330 core
uniform sampler2D face0, face1, face2, face3, face4;
uniform vec2 size;
out vec4 frag;
const float T = %f;

bool look(sampler2D s, vec3 f, vec3 u, vec3 d, inout vec3 colour) {
    float depth = dot(d, f);
    if (depth <= 0.0) return false;
    vec2 p = vec2(dot(d, cross(f, u)), dot(d, u)) / depth;
    if (abs(p.x) > T || abs(p.y) > T) return false;
    colour = texture(s, p / (2.0 * T) + 0.5).rgb;
    return true;
}

void main() {
    vec2 q = (gl_FragCoord.xy - 0.5 * size) / (0.5 * size.x);
    float rho = length(q);
    if (rho > 1.0) { frag = vec4(0.0, 0.0, 0.0, 1.0); return; }
    float theta = rho * LENS_HALF;
    vec2 dir = rho > 1e-6 ? q / rho : vec2(0.0);
    vec3 d = vec3(sin(theta) * dir, -cos(theta));
    vec3 colour = vec3(0.0);
    bool hit = look(face0, %s, %s, d, colour)
            || look(face1, %s, %s, d, colour)
            || look(face2, %s, %s, d, colour)
            || look(face3, %s, %s, d, colour)
            || look(face4, %s, %s, d, colour);
    frag = vec4(colour, 1.0);
}
"""
    % ((_TAN_HALF,) + tuple(_vec(v) for face in _FACES for v in face))
)


@dataclass
class CameraMount:
    name: str
    offset: np.ndarray  # body-frame position of the lens
    yaw_degrees: float  # where it looks: -90 left, +90 right, 0 forward


class FisheyeCamera:
    def __init__(self, ctx: moderngl.Context, renderer: Renderer, mount: CameraMount, size: tuple[int, int] = (1080, 1080), face_size: int = 1024, lens_fov: float = LENS_FOV) -> None:
        self.ctx, self.renderer, self.mount, self.size = ctx, renderer, mount, size
        self.lens_fov = lens_fov
        self.rotation = mount_rotation(mount.yaw_degrees)
        self.faces = []
        self.face_buffers = []
        depth = ctx.depth_renderbuffer((face_size, face_size))
        for _ in _FACES:
            texture = ctx.texture((face_size, face_size), 3)
            texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            texture.repeat_x = texture.repeat_y = False
            self.faces.append(texture)
            self.face_buffers.append(ctx.framebuffer([texture], depth))
        self.face_size = face_size
        self.output = ctx.texture(size, 3)
        self.output.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.output_buffer = ctx.framebuffer([self.output])
        self.program = ctx.program(vertex_shader=_VERTEX, fragment_shader=_FRAGMENT.replace("LENS_HALF", f"{np.radians(lens_fov / 2.0):.7f}"))
        for index in range(len(_FACES)):
            self.program[f"face{index}"].value = index
        self.program["size"].value = (float(size[0]), float(size[1]))
        self.vao = ctx.vertex_array(self.program, [])

    def render(self, body_position: np.ndarray, body_rotation: np.ndarray) -> None:
        """Take one picture from a drone at this position and orientation (body-to-world rotation)."""
        position = body_position + body_rotation @ self.mount.offset
        camera_to_world = body_rotation @ self.rotation
        for (forward, up), buffer in zip(_FACES, self.face_buffers):
            face_forward = camera_to_world @ np.array(forward)
            face_up = camera_to_world @ np.array(up)
            camera = Camera(position, fov_deg=FACE_FOV, basis=(face_forward, np.cross(face_forward, face_up), face_up))
            self.renderer.draw(buffer, (self.face_size, self.face_size), camera)
        self.output_buffer.use()
        self.output_buffer.viewport = (0, 0, *self.size)
        self.ctx.disable(moderngl.DEPTH_TEST)
        for index, texture in enumerate(self.faces):
            texture.use(index)
        self.vao.render(moderngl.TRIANGLES, vertices=3)
        self.ctx.enable(moderngl.DEPTH_TEST)

    def read_bgr(self) -> np.ndarray:
        """The latest picture as a (height, width, 3) BGR uint8 array (this waits for the GPU)."""
        width, height = self.size
        rgb = np.frombuffer(self.output_buffer.read(components=3), dtype=np.uint8).reshape(height, width, 3)
        return cv2.cvtColor(cv2.flip(rgb, 0), cv2.COLOR_RGB2BGR)  # GL rows run bottom-up

    def release(self) -> None:
        for buffer in (*self.face_buffers, self.output_buffer):
            buffer.release()
        for texture in (*self.faces, self.output):
            texture.release()
        self.vao.release()
        self.program.release()
