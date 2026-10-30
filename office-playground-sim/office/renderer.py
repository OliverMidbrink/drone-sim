"""Indoor rendering (moderngl): a baked static scene, a few moving models, and ceiling lights that stay in their room.

The building is merged on the CPU into one vertex buffer per texture, so a frame is a handful of
draw calls however many chairs there are. Moving things (the robot) are small `DynamicModel`s
drawn with their own model matrix. Lighting is per fragment: a hemisphere ambient plus point lights,
each limited to a rectangle on the floor plan (its room), because there are no shadows to keep the
light of one room out of the next. Surfaces are lit from whichever side the camera sees, since
many wall models are single sheets. Linear colour inside, filmic tone mapping and gamma at the end.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import moderngl
import numpy as np

from camera import Camera
from gltf import Primitive
from render import make_texture

NEAR, FAR = 0.04, 300.0
MAX_LIGHTS = 64
SKY_COLOR = (0.55, 0.72, 0.92)  # sRGB, what a window looks out on
VERTEX_FLOATS = 11  # position 3, normal 3, uv 2, tint 3

_VERTEX = """
#version 330 core
in vec3 in_position;
in vec3 in_normal;
in vec2 in_uv;
in vec3 in_tint;
uniform mat4 vp;
uniform mat4 model;
out vec3 v_world;
out vec3 v_normal;
out vec2 v_uv;
out vec3 v_tint;
void main() {
    vec4 world = model * vec4(in_position, 1.0);
    v_world = world.xyz;
    v_normal = mat3(model) * in_normal;
    v_uv = in_uv;
    v_tint = in_tint;
    gl_Position = vp * world;
}
"""

_FRAGMENT = """
#version 330 core
#define MAX_LIGHTS 64
in vec3 v_world;
in vec3 v_normal;
in vec2 v_uv;
in vec3 v_tint;
uniform sampler2D tex;
uniform vec3 cam_pos;
uniform int light_count;
uniform vec4 light_pos[MAX_LIGHTS];    // xyz, reach in metres
uniform vec4 light_color[MAX_LIGHTS];  // rgb (linear, includes intensity), unused
uniform vec4 light_rect[MAX_LIGHTS];   // xmin, zmin, xmax, zmax: where this light shines
uniform vec3 ambient_sky;
uniform vec3 ambient_ground;
uniform float exposure;
uniform float emissive;
out vec4 frag;

vec3 tonemap(vec3 c) {
    c *= exposure;
    return clamp((c * (2.51 * c + 0.03)) / (c * (2.43 * c + 0.59) + 0.14), 0.0, 1.0);
}

void main() {
    vec4 t = texture(tex, v_uv);
    vec3 albedo = pow(t.rgb, vec3(2.2)) * v_tint;
    vec3 n = normalize(v_normal);
    vec3 view = normalize(cam_pos - v_world);
    if (dot(n, view) < 0.0) n = -n;
    vec3 light = mix(ambient_ground, ambient_sky, n.y * 0.5 + 0.5);
    // a little darkening near the floor, where bounce light does not reach: a cheap stand-in for ambient occlusion
    light *= 0.62 + 0.38 * smoothstep(0.0, 0.7, v_world.y);
    vec3 specular = vec3(0.0);
    for (int i = 0; i < light_count; ++i) {
        vec4 rect = light_rect[i];
        if (v_world.x < rect.x || v_world.x > rect.z || v_world.z < rect.y || v_world.z > rect.w) continue;
        vec3 to_light = light_pos[i].xyz - v_world;
        float d = length(to_light);
        vec3 l = to_light / max(d, 1e-4);
        float falloff = 1.0 / (1.0 + (d * d) / (light_pos[i].w * light_pos[i].w));
        float lambert = max((dot(n, l) + 0.25) / 1.25, 0.0);
        light += light_color[i].rgb * falloff * lambert;
        specular += light_color[i].rgb * falloff * 0.12 * pow(max(dot(n, normalize(l + view)), 0.0), 24.0);
    }
    vec3 color = albedo * light + specular + albedo * emissive;
    frag = vec4(pow(tonemap(color), vec3(1.0 / 2.2)), 1.0);
}
"""


@dataclass
class Light:
    position: tuple[float, float, float]
    color: tuple[float, float, float] = (1.0, 0.96, 0.9)  # linear, scaled by intensity
    reach: float = 5.0  # metres where the light has fallen to half
    rect: tuple[float, float, float, float] = (-1e6, -1e6, 1e6, 1e6)  # xmin, zmin, xmax, zmax of the area it lights


def _vertices(primitive: Primitive, matrix: np.ndarray, tint: tuple[float, float, float]) -> np.ndarray:
    rotation = matrix[:3, :3]
    scale = np.cbrt(abs(np.linalg.det(rotation)))
    positions = primitive.positions.astype(np.float64) @ rotation.T + matrix[:3, 3]
    normals = primitive.normals.astype(np.float64) @ (rotation / max(scale, 1e-9)).T
    color = np.array(primitive.color[:3]) * np.array(tint)
    out = np.empty((len(positions), VERTEX_FLOATS), np.float32)
    out[:, 0:3], out[:, 3:6], out[:, 6:8], out[:, 8:11] = positions, normals, primitive.uvs, color
    return out


class MeshBuilder:
    """Collects placed primitives and merges them into one mesh per texture."""

    def __init__(self) -> None:
        self._vertices: dict[int, list[np.ndarray]] = {}
        self._indices: dict[int, list[np.ndarray]] = {}
        self._counts: dict[int, int] = {}
        self.textures: dict[int, np.ndarray | None] = {}
        self.triangles = 0

    def add(self, primitive: Primitive, matrix: np.ndarray, tint: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> None:
        key = id(primitive.texture) if primitive.texture is not None else 0
        self.textures[key] = primitive.texture
        offset = self._counts.get(key, 0)
        vertices = _vertices(primitive, matrix, tint)
        self._vertices.setdefault(key, []).append(vertices)
        flip = np.linalg.det(matrix[:3, :3]) < 0.0  # a mirrored placement turns the triangles inside out
        indices = primitive.indices.reshape(-1, 3)[:, ::-1].reshape(-1) if flip else primitive.indices
        self._indices.setdefault(key, []).append(indices + np.uint32(offset))
        self._counts[key] = offset + len(vertices)
        self.triangles += len(indices) // 3

    def groups(self) -> list[tuple[np.ndarray | None, np.ndarray, np.ndarray]]:
        return [(self.textures[k], np.concatenate(self._vertices[k]), np.concatenate(self._indices[k]).astype(np.uint32)) for k in self._vertices]

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        points = np.concatenate([v[:, :3] for chunks in self._vertices.values() for v in chunks])
        return points.min(axis=0), points.max(axis=0)


class _GpuMesh:
    def __init__(self, ctx: moderngl.Context, program: moderngl.Program, vertices: np.ndarray, indices: np.ndarray, texture: moderngl.Texture) -> None:
        self.texture = texture
        self.vertex_buffer = ctx.buffer(np.ascontiguousarray(vertices, np.float32).tobytes())
        self.index_buffer = ctx.buffer(np.ascontiguousarray(indices, np.uint32).tobytes())
        self.vao = ctx.vertex_array(
            program, [(self.vertex_buffer, "3f 3f 2f 3f", "in_position", "in_normal", "in_uv", "in_tint")], index_buffer=self.index_buffer, index_element_size=4
        )

    def draw(self) -> None:
        self.texture.use(0)
        self.vao.render(moderngl.TRIANGLES)

    def release(self) -> None:
        self.vao.release()
        self.vertex_buffer.release()
        self.index_buffer.release()


class DynamicModel:
    """A set of primitives uploaded once, drawn anywhere with a model matrix (a robot, a ball ...)."""

    def __init__(self, meshes: list[_GpuMesh]) -> None:
        self.meshes = meshes
        self.emissive = 0.0

    def release(self) -> None:
        for mesh in self.meshes:
            mesh.release()


@dataclass
class OfficeRenderer:
    ctx: moderngl.Context
    exposure: float = 1.0
    ambient_sky: tuple[float, float, float] = (0.17, 0.18, 0.20)
    ambient_ground: tuple[float, float, float] = (0.07, 0.065, 0.06)
    program: moderngl.Program = field(init=False)
    _layers: dict[str, list[_GpuMesh]] = field(default_factory=dict, init=False)
    hidden_layers: set[str] = field(default_factory=set)  # layers that are not drawn (the ceiling, for a view from above)
    _textures: dict[int, moderngl.Texture] = field(default_factory=dict, init=False)
    _white: moderngl.Texture = field(init=False)
    _light_count: int = field(default=0, init=False)
    last_draw_calls: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.program = self.ctx.program(vertex_shader=_VERTEX, fragment_shader=_FRAGMENT)
        self._white = make_texture(self.ctx, np.full((2, 2, 4), 255, np.uint8))
        self.program["tex"].value = 0
        self.set_lights([])

    def _texture(self, array: np.ndarray | None) -> moderngl.Texture:
        if array is None:
            return self._white
        key = id(array)
        if key not in self._textures:
            self._textures[key] = make_texture(self.ctx, array)
        return self._textures[key]

    def set_static(self, builder: MeshBuilder, layer: str = "building") -> None:
        """Upload a baked scene as a named layer (replacing that layer); layers can be hidden by name."""
        for mesh in self._layers.get(layer, []):
            mesh.release()
        self._layers[layer] = [_GpuMesh(self.ctx, self.program, vertices, indices, self._texture(texture)) for texture, vertices, indices in builder.groups()]

    def set_lights(self, lights: list[Light]) -> None:
        lights = lights[:MAX_LIGHTS]
        self._light_count = len(lights)
        position = np.zeros((MAX_LIGHTS, 4), np.float32)
        color = np.zeros((MAX_LIGHTS, 4), np.float32)
        rect = np.zeros((MAX_LIGHTS, 4), np.float32)
        for i, light in enumerate(lights):
            position[i] = (*light.position, light.reach)
            color[i, :3] = light.color
            rect[i] = light.rect
        self.program["light_count"].value = len(lights)
        self.program["light_pos"].write(position.tobytes())
        self.program["light_color"].write(color.tobytes())
        self.program["light_rect"].write(rect.tobytes())

    def make_model(self, primitives: list[Primitive]) -> DynamicModel:
        builder = MeshBuilder()
        for primitive in primitives:
            builder.add(primitive, np.eye(4))
        return DynamicModel([_GpuMesh(self.ctx, self.program, v, i, self._texture(t)) for t, v, i in builder.groups()])

    def draw(
        self,
        target: moderngl.Framebuffer,
        size: tuple[int, int],
        camera: Camera,
        models: list[tuple[DynamicModel, np.ndarray]] | None = None,
        origin: tuple[int, int] = (0, 0),
    ) -> None:
        """Render the scene as seen by `camera` into the `size` pixel rectangle of `target` whose bottom-left corner is `origin`.

        `models` are (model, 4x4 matrix to world) pairs drawn on top of the building.
        """
        width, height = size
        target.use()
        target.viewport = (*origin, width, height)
        self.ctx.scissor = (*origin, width, height)
        sky = tuple(c**2.2 for c in SKY_COLOR)
        target.clear(*sky, depth=1.0, viewport=(*origin, width, height))
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        view_projection = camera.projection_matrix(width / height, NEAR, FAR) @ camera.view_matrix()
        program = self.program
        program["vp"].write(view_projection.astype("f4").T.tobytes())
        program["cam_pos"].value = tuple(float(v) for v in camera.position)
        program["ambient_sky"].value = self.ambient_sky
        program["ambient_ground"].value = self.ambient_ground
        program["exposure"].value = self.exposure
        program["emissive"].value = 0.0
        program["model"].write(np.eye(4, dtype="f4").tobytes())
        self.last_draw_calls = 0
        for name, meshes in self._layers.items():
            if name in self.hidden_layers:
                continue
            for mesh in meshes:
                mesh.draw()
            self.last_draw_calls += len(meshes)
        for model, matrix in models or []:
            program["model"].write(matrix.astype("f4").T.tobytes())
            program["emissive"].value = model.emissive
            for mesh in model.meshes:
                mesh.draw()
            self.last_draw_calls += len(model.meshes)
        self.ctx.scissor = None

    def release(self) -> None:
        for meshes in self._layers.values():
            for mesh in meshes:
                mesh.release()
