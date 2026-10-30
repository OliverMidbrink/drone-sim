"""OpenGL 3.3 rendering (moderngl): sky, shaded terrain and instanced trees with distance + frustum culling.

One draw call for the terrain and one per tree material. Lighting is a sun plus hemisphere
ambient in linear colour, with an exponential fog that matches the sky at the horizon. No textures
are needed for the ground: its colours come from height, slope and a small tiling noise texture.
"""

from __future__ import annotations

import cv2
import moderngl
import numpy as np

from camera import Camera
from gltf import Primitive
from terrain import Terrain, tileable_noise
from trees import TreeModel

NEAR, FAR = 0.3, 6000.0
SUN_DIRECTION = np.array([0.5, 0.7, 0.35]) / np.linalg.norm([0.5, 0.7, 0.35])
SUN_COLOR = (1.10, 1.03, 0.93)
HORIZON_COLOR = (0.62, 0.76, 0.92)  # sRGB; the fog fades into this
ZENITH_COLOR = (0.20, 0.42, 0.78)
INSTANCE_FLOATS = 6  # x, y, z, yaw, height, brightness


def to_linear(color: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(float(c) ** 2.2 for c in color)


_LIGHTING = """
uniform vec3 sun_dir;
uniform vec3 sun_color;
uniform vec3 fog_color;
uniform float fog_density;
uniform vec3 cam_pos;

vec3 lit(vec3 albedo, vec3 n, float wrap) {
    float diffuse = clamp((dot(n, sun_dir) + wrap) / (1.0 + wrap), 0.0, 1.0);
    vec3 ambient = mix(vec3(0.10, 0.09, 0.07), vec3(0.34, 0.44, 0.60), n.y * 0.5 + 0.5);
    return albedo * (ambient + sun_color * diffuse);
}

vec3 fogged(vec3 color, vec3 world) {
    float d = length(world - cam_pos) * fog_density;
    return mix(color, fog_color, 1.0 - exp(-d * d));
}

vec3 encode(vec3 c) { return pow(clamp(c, 0.0, 1.0), vec3(1.0 / 2.2)); }
"""

_SKY_VERTEX = """
#version 330 core
out vec2 v_ndc;
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    v_ndc = p * 2.0 - 1.0;
    gl_Position = vec4(v_ndc, 0.0, 1.0);
}
"""

_SKY_FRAGMENT = """
#version 330 core
in vec2 v_ndc;
uniform mat4 inv_vp;
uniform vec3 cam_pos;
uniform vec3 sun_dir;
uniform vec3 fog_color;
uniform vec3 zenith_color;
out vec4 frag;
void main() {
    vec4 p = inv_vp * vec4(v_ndc, 1.0, 1.0);
    vec3 dir = normalize(p.xyz / p.w - cam_pos);
    vec3 color = mix(fog_color, zenith_color, pow(max(dir.y, 0.0), 0.45));
    float s = max(dot(dir, sun_dir), 0.0);
    color += vec3(1.0, 0.85, 0.6) * (0.22 * pow(s, 8.0) + 6.0 * pow(s, 1200.0));
    frag = vec4(pow(clamp(color, 0.0, 1.0), vec3(1.0 / 2.2)), 1.0);
}
"""

_TERRAIN_VERTEX = """
#version 330 core
in vec3 in_position;
in vec3 in_normal;
uniform mat4 vp;
out vec3 v_world;
out vec3 v_normal;
void main() {
    v_world = in_position;
    v_normal = in_normal;
    gl_Position = vp * vec4(in_position, 1.0);
}
"""

_TERRAIN_FRAGMENT = (
    """
#version 330 core
in vec3 v_world;
in vec3 v_normal;
uniform sampler2D detail;
out vec4 frag;
"""
    + _LIGHTING
    + """
void main() {
    vec3 n = normalize(v_normal);
    vec2 p = v_world.xz;
    float fine = texture(detail, p / 3.0).r;
    float medium = texture(detail, p / 23.0).g;
    // two incommensurate scales so the 256 px tile never shows as a repeating pattern
    float broad = 0.5 * (texture(detail, p / 170.0).b + texture(detail, vec2(p.x * 0.8 - p.y * 0.6, p.x * 0.6 + p.y * 0.8) / 437.0).b);
    float h = v_world.y;
    float slope = 1.0 - n.y;

    vec3 meadow = mix(vec3(0.20, 0.33, 0.10), vec3(0.34, 0.44, 0.15), medium);
    vec3 forest_floor = vec3(0.12, 0.21, 0.08);
    vec3 alpine = mix(vec3(0.36, 0.38, 0.22), vec3(0.45, 0.42, 0.29), medium);
    vec3 rock = mix(vec3(0.34, 0.32, 0.30), vec3(0.50, 0.47, 0.43), 0.5 * fine + 0.5 * medium);
    vec3 snow = vec3(0.92, 0.94, 0.97);

    vec3 color = mix(meadow, forest_floor, smoothstep(0.35, 0.65, broad) * (1.0 - smoothstep(120.0, 230.0, h)));
    color = mix(color, alpine, smoothstep(150.0, 300.0, h + 60.0 * (broad - 0.5)));
    float rock_amount = max(smoothstep(0.17, 0.38, slope + 0.07 * (fine - 0.5)), smoothstep(300.0, 420.0, h + 90.0 * (broad - 0.5)));
    color = mix(color, rock, rock_amount);
    float snow_amount = smoothstep(290.0, 360.0, h + 70.0 * (broad - 0.5)) * (1.0 - smoothstep(0.35, 0.65, slope + 0.1 * (fine - 0.5)));
    color = mix(color, snow, snow_amount);
    color *= 0.88 + 0.24 * fine;

    frag = vec4(encode(fogged(lit(color * color, n, 0.0), v_world)), 1.0);
}
"""
)

_TREE_VERTEX = """
#version 330 core
in vec3 in_position;
in vec3 in_normal;
in vec2 in_uv;
in vec3 in_offset;   // per instance: where the trunk stands
in vec3 in_params;   // per instance: yaw, height in metres, brightness
uniform mat4 vp;
out vec3 v_world;
out vec3 v_normal;
out vec2 v_uv;
out float v_brightness;
void main() {
    float c = cos(in_params.x), s = sin(in_params.x);
    mat3 turn = mat3(c, 0.0, -s, 0.0, 1.0, 0.0, s, 0.0, c);
    v_world = turn * (in_position * in_params.y) + in_offset;
    v_normal = turn * in_normal;
    v_uv = in_uv;
    v_brightness = in_params.z;
    gl_Position = vp * vec4(v_world, 1.0);
}
"""

_TREE_FRAGMENT = (
    """
#version 330 core
in vec3 v_world;
in vec3 v_normal;
in vec2 v_uv;
in float v_brightness;
uniform sampler2D tex;
uniform vec4 base_color;
uniform float cutoff;
out vec4 frag;
"""
    + _LIGHTING
    + """
void main() {
    vec4 t = texture(tex, v_uv) * base_color;
    if (t.a < cutoff) discard;
    vec3 n = normalize(v_normal);
    if (dot(n, cam_pos - v_world) < 0.0) n = -n;  // leaves are lit from whichever side we see
    vec3 albedo = pow(t.rgb * v_brightness, vec3(2.2));
    frag = vec4(encode(fogged(lit(albedo, n, 0.35), v_world)), 1.0);
}
"""
)


def _set(program: moderngl.Program, name: str, value) -> None:
    if name in program:  # a uniform the shader does not use is optimised away
        program[name].value = value


def make_texture(ctx: moderngl.Context, rgba: np.ndarray) -> moderngl.Texture:
    height, width = rgba.shape[:2]
    texture = ctx.texture((width, height), rgba.shape[2], np.ascontiguousarray(rgba).tobytes())
    texture.build_mipmaps()
    texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
    texture.anisotropy = 8.0
    return texture


def frustum_planes(view_projection: np.ndarray) -> np.ndarray:
    """(6, 4) normalised planes (a, b, c, d); a point is inside when a*x + b*y + c*z + d >= 0 for all of them."""
    rows = view_projection
    planes = np.stack([rows[3] + rows[0], rows[3] - rows[0], rows[3] + rows[1], rows[3] - rows[1], rows[3] + rows[2], rows[3] - rows[2]])
    return planes / np.linalg.norm(planes[:, :3], axis=1, keepdims=True)


def create_hidden_context():
    """(glfw window, moderngl context) on an invisible window, for rendering without showing anything."""
    import glfw

    if not glfw.init():
        raise SystemExit("GLFW failed to initialize")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(64, 64, "virtual-sim offscreen", None, None)
    if window is None:
        glfw.terminate()
        raise SystemExit("Could not create an OpenGL context")
    glfw.make_context_current(window)
    return window, moderngl.create_context()


class OffscreenTarget:
    """4x multisampled colour + depth framebuffer whose picture can be read back as a BGR image."""

    def __init__(self, ctx: moderngl.Context, size: tuple[int, int], samples: int = 4) -> None:
        self.ctx, self.size = ctx, size
        self.framebuffer = ctx.framebuffer(ctx.renderbuffer(size, samples=samples), ctx.depth_renderbuffer(size, samples=samples))
        self._resolved = ctx.framebuffer(ctx.renderbuffer(size))

    def read_bgr(self) -> np.ndarray:
        width, height = self.size
        self.ctx.copy_framebuffer(self._resolved, self.framebuffer)
        rgb = np.frombuffer(self._resolved.read(components=3), dtype=np.uint8).reshape(height, width, 3)
        return cv2.cvtColor(cv2.flip(rgb, 0), cv2.COLOR_RGB2BGR)


class _TreeBatch:
    """The GPU side of one tree model: a VAO per material and one shared per-frame instance buffer."""

    def __init__(self, ctx: moderngl.Context, program: moderngl.Program, tree: TreeModel, textures: dict[int, moderngl.Texture]) -> None:
        self.tree = tree
        self.ctx = ctx
        self.instances = np.empty((0, INSTANCE_FLOATS), np.float32)
        self.centres = np.empty((0, 3))
        self.radii = np.empty(0)
        self.instance_buffer: moderngl.Buffer | None = None
        self.parts: list[tuple[Primitive, moderngl.Texture, moderngl.Buffer, moderngl.Buffer]] = []
        self.vaos: list[moderngl.VertexArray] = []
        self.program = program
        self._textures = textures
        self._white = make_texture(ctx, np.full((2, 2, 4), 255, np.uint8))

    def set_instances(self, instances: np.ndarray) -> None:
        self.instances = np.ascontiguousarray(instances, dtype=np.float32)
        height = self.instances[:, 4].astype(np.float64)
        self.centres = np.stack([self.instances[:, 0], self.instances[:, 1] + 0.5 * height, self.instances[:, 2]], axis=1)
        self.radii = self.tree.radius * height
        self._release_vaos()
        if len(instances) == 0:
            return
        self.instance_buffer = self.ctx.buffer(reserve=len(instances) * INSTANCE_FLOATS * 4, dynamic=True)
        for primitive in self.tree.model.primitives:
            vertices = np.concatenate([primitive.positions, primitive.normals, primitive.uvs], axis=1).astype(np.float32)
            vertex_buffer = self.ctx.buffer(vertices.tobytes())
            index_buffer = self.ctx.buffer(primitive.indices.astype(np.uint32).tobytes())
            vao = self.ctx.vertex_array(
                self.program,
                [(vertex_buffer, "3f 3f 2f", "in_position", "in_normal", "in_uv"), (self.instance_buffer, "3f 3f/i", "in_offset", "in_params")],
                index_buffer=index_buffer,
                index_element_size=4,
            )
            texture = self._white
            if primitive.texture is not None:
                key = id(primitive.texture)
                if key not in self._textures:
                    self._textures[key] = make_texture(self.ctx, primitive.texture)
                texture = self._textures[key]
            self.parts.append((primitive, texture, vertex_buffer, index_buffer))
            self.vaos.append(vao)

    def _release_vaos(self) -> None:
        for vao, (_, _, vertex_buffer, index_buffer) in zip(self.vaos, self.parts):
            vao.release()
            vertex_buffer.release()
            index_buffer.release()
        if self.instance_buffer is not None:
            self.instance_buffer.release()
            self.instance_buffer = None
        self.vaos, self.parts = [], []

    def visible(self, camera_position: np.ndarray, planes: np.ndarray) -> np.ndarray:
        """The instances inside the draw distance and the view frustum."""
        if len(self.instances) == 0:
            return self.instances
        reach = self.tree.species.draw_distance + self.radii
        near = np.sum((self.centres - camera_position) ** 2, axis=1) < reach**2
        inside = np.all(self.centres @ planes[:, :3].T + planes[:, 3] > -self.radii[:, None], axis=1)
        return self.instances[near & inside]

    def draw(self, camera_position: np.ndarray, planes: np.ndarray) -> int:
        shown = self.visible(camera_position, planes)
        if len(shown) == 0:
            return 0
        self.instance_buffer.write(shown.tobytes())
        for (primitive, texture, _, _), vao in zip(self.parts, self.vaos):
            texture.use(0)
            _set(self.program, "base_color", primitive.color)
            _set(self.program, "cutoff", 0.5 if primitive.cutout else 0.0)
            vao.render(moderngl.TRIANGLES, instances=len(shown))
        return len(shown)

    def release(self) -> None:
        self._release_vaos()


class Renderer:
    def __init__(self, ctx: moderngl.Context, models: list[TreeModel], visibility: float = 3500.0) -> None:
        self.ctx = ctx
        self.fog_density = 2.0 / visibility
        self.fog_color = to_linear(HORIZON_COLOR)
        self.sky = ctx.program(vertex_shader=_SKY_VERTEX, fragment_shader=_SKY_FRAGMENT)
        self.ground = ctx.program(vertex_shader=_TERRAIN_VERTEX, fragment_shader=_TERRAIN_FRAGMENT)
        self.tree_program = ctx.program(vertex_shader=_TREE_VERTEX, fragment_shader=_TREE_FRAGMENT)
        for program in (self.sky, self.ground, self.tree_program):
            _set(program, "sun_dir", tuple(float(v) for v in SUN_DIRECTION))
            _set(program, "sun_color", SUN_COLOR)
            _set(program, "fog_color", self.fog_color)
            _set(program, "fog_density", self.fog_density)
            _set(program, "zenith_color", to_linear(ZENITH_COLOR))
        self.ground["detail"].value = 0
        self.tree_program["tex"].value = 0
        rng = np.random.default_rng(1234)
        noise = np.stack([tileable_noise(rng, 256, (16, 32, 64)), tileable_noise(rng, 256, (6, 12, 24)), tileable_noise(rng, 256, (2, 4, 8))], axis=-1)
        self.detail = make_texture(ctx, (noise * 255).astype(np.uint8))
        self.sky_vao = ctx.vertex_array(self.sky, [])
        self.terrain_vao: moderngl.VertexArray | None = None
        self._terrain_buffers: list[moderngl.Buffer] = []
        self._textures: dict[int, moderngl.Texture] = {}
        self.batches = [_TreeBatch(ctx, self.tree_program, tree, self._textures) for tree in models]
        self.last_tree_count = 0
        self.tiles = None

    def set_terrain(self, terrain: Terrain) -> None:
        if self.terrain_vao is not None:
            self.terrain_vao.release()
            for buffer in self._terrain_buffers:
                buffer.release()
        self.tiles = terrain.tiled_mesh()
        vertex_buffer, index_buffer = self.ctx.buffer(self.tiles.vertices.tobytes()), self.ctx.buffer(self.tiles.indices.tobytes())
        self._terrain_buffers = [vertex_buffer, index_buffer]
        self.terrain_vao = self.ctx.vertex_array(self.ground, [(vertex_buffer, "3f 3f", "in_position", "in_normal")], index_buffer=index_buffer, index_element_size=4)

    def set_trees(self, instances: list[np.ndarray]) -> None:
        for batch, data in zip(self.batches, instances):
            batch.set_instances(data)

    def draw(self, target: moderngl.Framebuffer, size: tuple[int, int], camera: Camera, origin: tuple[int, int] = (0, 0)) -> None:
        """Render the camera's view into `target`, into the `size` = (width, height) pixel rectangle whose bottom-left corner is `origin`."""
        width, height = size
        target.use()
        target.viewport = (*origin, width, height)
        self.ctx.scissor = (*origin, width, height)
        target.clear(depth=1.0, viewport=(*origin, width, height))
        view_projection = camera.projection_matrix(width / height, NEAR, FAR) @ camera.view_matrix()
        position = tuple(float(v) for v in camera.position)
        for program in (self.sky, self.ground, self.tree_program):
            _set(program, "cam_pos", position)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.sky["inv_vp"].write(np.linalg.inv(view_projection).astype("f4").T.tobytes())
        self.sky_vao.render(moderngl.TRIANGLES, vertices=3)
        self.ctx.enable(moderngl.DEPTH_TEST)
        packed = view_projection.astype("f4").T.tobytes()
        self.ground["vp"].write(packed)
        self.tree_program["vp"].write(packed)
        self.detail.use(0)
        planes = frustum_planes(view_projection)
        if self.terrain_vao is not None:
            self._draw_terrain(planes)
        self.last_tree_count = sum(batch.draw(camera.position, planes) for batch in self.batches)
        self.ctx.scissor = None

    def _draw_terrain(self, planes: np.ndarray) -> None:
        """Draw the terrain tiles inside the frustum, neighbouring tiles in one call."""
        tiles = self.tiles
        inside = np.all(tiles.centres @ planes[:, :3].T + planes[:, 3] > -tiles.radii[:, None], axis=1)
        visible = np.flatnonzero(inside)
        if len(visible) == 0:
            return
        for run in np.split(visible, np.flatnonzero(np.diff(visible) != 1) + 1):
            first = int(tiles.starts[run[0]])
            self.terrain_vao.render(moderngl.TRIANGLES, vertices=int(tiles.ends[run[-1]]) - first, first=first)

    def release(self) -> None:
        for batch in self.batches:
            batch.release()
