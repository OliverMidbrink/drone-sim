"""OpenGL side of the virtual world, shared by the on-screen viewer, the SLAM run and the calibration.

Needs a current GL context (glfw window). `create_hidden_context()` makes one with no visible
window, for rendering to an off-screen framebuffer.
"""

from __future__ import annotations

import glfw
import moderngl
import numpy as np

import world

SKY_COLOR = (0.53, 0.81, 0.92)
SUN_DIRECTION = np.array([0.4, 0.8, 0.45])
NEAR, FAR = 0.05, 500.0
UP = np.array([0.0, 1.0, 0.0])
VERTEX_LAYOUT = ("3f 3f 2f 3f", "in_position", "in_normal", "in_uv", "in_tint")  # matches world.py's mesh rows

_VERTEX_SHADER = """
#version 330 core
in vec3 in_position;
in vec3 in_normal;
in vec2 in_uv;
in vec3 in_tint;
uniform mat4 mvp;
out vec3 v_normal;
out vec2 v_uv;
out vec3 v_tint;
void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
    v_normal = in_normal;
    v_uv = in_uv;
    v_tint = in_tint;
}
"""

_FRAGMENT_SHADER = """
#version 330 core
in vec3 v_normal;
in vec2 v_uv;
in vec3 v_tint;
uniform sampler2D tex;
uniform vec3 sun_dir;
out vec4 fragColor;
void main() {
    vec3 base = texture(tex, v_uv).rgb * v_tint;
    float diffuse = max(dot(normalize(v_normal), sun_dir), 0.0);
    fragColor = vec4(base * (0.55 + 0.45 * diffuse), 1.0);
}
"""


def perspective(fov_y_rad: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / np.tan(fov_y_rad / 2.0)
    mat = np.zeros((4, 4), dtype=np.float32)
    mat[0, 0] = f / aspect
    mat[1, 1] = f
    mat[2, 2] = (far + near) / (near - far)
    mat[2, 3] = (2.0 * far * near) / (near - far)
    mat[3, 2] = -1.0
    return mat


def vertical_fov(fov_x_deg: float, aspect: float) -> float:
    """Vertical FOV in radians for a given horizontal FOV (degrees) and width / height."""
    return float(2.0 * np.arctan(np.tan(np.radians(fov_x_deg) / 2.0) / aspect))


def look_along(eye: np.ndarray, forward: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    rotation = np.eye(4, dtype=np.float32)
    rotation[0, :3] = right
    rotation[1, :3] = true_up
    rotation[2, :3] = -forward
    translation = np.eye(4, dtype=np.float32)
    translation[:3, 3] = -eye.astype(np.float32)
    return rotation @ translation


def make_texture(ctx: moderngl.Context, image: np.ndarray) -> moderngl.Texture:
    height, width = image.shape[:2]
    texture = ctx.texture((width, height), 3, np.ascontiguousarray(image).tobytes())
    texture.build_mipmaps()
    texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.NEAREST)  # crisp blocky close-up, smooth far away
    texture.anisotropy = 8.0  # keeps the ground sharp at grazing angles
    return texture


def create_hidden_context() -> tuple[object, moderngl.Context]:
    """A GL 3.3 core context on an invisible window. Returns (glfw window, moderngl context)."""
    if not glfw.init():
        raise SystemExit("GLFW failed to initialize")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(64, 64, "offscreen", None, None)
    glfw.window_hint(glfw.VISIBLE, glfw.TRUE)  # windows created afterwards (e.g. the map view) must show
    if window is None:
        glfw.terminate()
        raise SystemExit("Could not create an OpenGL context")
    glfw.make_context_current(window)
    return window, moderngl.create_context()


class OffscreenTarget:
    """Colour + depth framebuffer (4x multisampled) whose picture can be read back as a BGR image."""

    def __init__(self, ctx: moderngl.Context, size: tuple[int, int], samples: int = 4) -> None:
        self.ctx = ctx
        self.size = size
        self.msaa = ctx.framebuffer(ctx.renderbuffer(size, samples=samples), ctx.depth_renderbuffer(size, samples=samples))
        self._resolved = ctx.framebuffer(ctx.renderbuffer(size))

    def read_bgr(self) -> np.ndarray:
        width, height = self.size
        self.ctx.copy_framebuffer(self._resolved, self.msaa)  # resolves the multisampling
        rgb = np.frombuffer(self._resolved.read(components=3), dtype=np.uint8).reshape(height, width, 3)
        return np.ascontiguousarray(rgb[::-1, :, ::-1])  # GL rows start at the bottom; RGB -> BGR


class SceneRenderer:
    """Draws the ground and boxes from a given camera pose into a framebuffer."""

    def __init__(self, ctx: moderngl.Context, boxes: list[world.Box], seed: int) -> None:
        self.ctx = ctx
        self.program = ctx.program(vertex_shader=_VERTEX_SHADER, fragment_shader=_FRAGMENT_SHADER)
        self.program["tex"].value = 0
        self.program["sun_dir"].value = tuple(float(v) for v in SUN_DIRECTION / np.linalg.norm(SUN_DIRECTION))
        self.grass = make_texture(ctx, world.grass_texture(seed))
        self.stone = make_texture(ctx, world.box_texture(seed))
        ground = world.ground_mesh()
        box_data = world.box_mesh(boxes, seed)
        self.box_vertices = len(box_data)
        self.ground_vao = ctx.vertex_array(self.program, [(ctx.buffer(ground.tobytes()), *VERTEX_LAYOUT)])
        self.box_vao = ctx.vertex_array(self.program, [(ctx.buffer(box_data.tobytes()), *VERTEX_LAYOUT)])

    def draw(
        self,
        target: moderngl.Framebuffer,
        size: tuple[int, int],
        position: np.ndarray,
        forward: np.ndarray,
        fov_deg: float,
    ) -> None:
        """Render the camera's view (horizontal FOV `fov_deg`) into `target`, a `size` = (width, height) framebuffer."""
        width, height = size
        aspect = width / height
        target.viewport = (0, 0, width, height)
        target.use()
        self.ctx.enable(moderngl.DEPTH_TEST)
        target.clear(*SKY_COLOR, 1.0)
        mvp = perspective(vertical_fov(fov_deg, aspect), aspect, NEAR, FAR) @ look_along(position, forward, UP)
        self.program["mvp"].write(mvp.astype(np.float32).T.tobytes())
        self.grass.use(0)
        self.ground_vao.render(moderngl.TRIANGLES)
        self.stone.use(0)
        self.box_vao.render(moderngl.TRIANGLES, vertices=self.box_vertices)
