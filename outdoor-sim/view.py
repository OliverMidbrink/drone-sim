"""Drawing the proving ground: virtual-sim's sky, grass terrain and trees, plus the paving, the car's parts, skid marks and a shadow.

`virtual_sim.render.Renderer` draws the sky, the terrain (grass, hills) and the trees; this adds a
second pass on the same framebuffer with the paved road (fine mesh, painted with section marks,
edge lines and the skid-pad circle in the shader), the RC track (a strip mesh that knows its
distance along and across the track, for the kerbs and the start line), solid-coloured parts with
the same sun and fog, and the tyre marks the car leaves.
"""

from __future__ import annotations

import moderngl
import numpy as np

from camera import Camera
from render import _LIGHTING, FAR, NEAR, Renderer, _set
from trees import TreeModel, place_trees

from ground import SECTIONS, Ground
from track import HALF_WIDTH, KERB

_ROAD_VERTEX = """
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

_ROAD_FRAGMENT = (
    """
#version 330 core
in vec3 v_world;
in vec3 v_normal;
uniform sampler2D detail;
uniform float marks[16];
uniform int mark_count;
out vec4 frag;
"""
    + _LIGHTING
    + """
float line(float distance, float half_width) {
    float w = fwidth(distance);
    return 1.0 - smoothstep(half_width - w, half_width + w, abs(distance));
}

void main() {
    vec3 n = normalize(v_normal);
    vec2 p = v_world.xz;
    float grain = texture(detail, p / 0.9).r;
    float patch = texture(detail, p / 11.0).g;
    vec3 color = vec3(0.30, 0.305, 0.32) * (0.85 + 0.3 * grain) * (0.92 + 0.16 * patch);

    bool on_road = p.x > 40.0 && abs(p.y) < 4.0;
    bool on_pad = abs(p.x) < 40.0 && abs(p.y) < 30.0;
    float paint = 0.0;
    vec3 paint_color = vec3(0.85);
    if (on_road) {
        paint = max(paint, line(abs(p.y) - 3.65, 0.06));
        float dash = step(mod(p.x, 6.0), 3.0);
        paint = max(paint, line(p.y, 0.06) * dash);
        for (int i = 0; i < mark_count; i++) {
            float stripe = line(p.x - marks[i], 0.3) * step(0.5, mod(floor(p.y / 0.5), 2.0));
            if (stripe > paint) { paint = stripe; paint_color = vec3(0.95, 0.62, 0.10); }
        }
    }
    if (on_pad || (on_road && p.x < 42.0)) {
        float ring = line(length(p) - 20.0, 0.07) + line(length(p) - 10.0, 0.05);
        vec2 cell = abs(mod(p + 5.0, 10.0) - 5.0);
        float grid = max(line(cell.x, 0.035), line(cell.y, 0.035)) * 0.35;
        paint = max(paint, max(ring, grid));
        paint = max(paint, line(abs(p.y) - 29.5, 0.08));
        paint = max(paint, line(abs(p.x) - 39.5, 0.08) * step(abs(p.y), 29.9));
    }
    color = mix(color, paint_color * 0.8, clamp(paint, 0.0, 1.0) * 0.9);
    // the sloping border of the paving is concrete
    bool core = on_pad || (p.x >= 40.0 && p.x <= 380.0 && abs(p.y) <= 4.0);
    if (!core) color = vec3(0.62, 0.61, 0.58);
    frag = vec4(encode(fogged(lit(color * color, n, 0.0), v_world)), 1.0);
}
"""
)

_TRACK_VERTEX = """
#version 330 core
in vec3 in_position;
in vec3 in_normal;
in vec3 in_track;  // s along the track, offset to the right, kerb (0 / 1)
uniform mat4 vp;
out vec3 v_world;
out vec3 v_normal;
out vec3 v_track;
void main() {
    v_world = in_position;
    v_normal = in_normal;
    v_track = in_track;
    gl_Position = vp * vec4(in_position, 1.0);
}
"""

_TRACK_FRAGMENT = (
    """
#version 330 core
in vec3 v_world;
in vec3 v_normal;
in vec3 v_track;
uniform sampler2D detail;
out vec4 frag;
"""
    + _LIGHTING
    + f"""
const float HALF_WIDTH = {HALF_WIDTH:.3f};
const float KERB = {KERB:.3f};
"""
    + """
float line(float distance, float half_width) {
    float w = fwidth(distance);
    return 1.0 - smoothstep(half_width - w, half_width + w, abs(distance));
}

void main() {
    vec3 n = normalize(v_normal);
    vec2 p = v_world.xz;
    float s = v_track.x;
    float side = abs(v_track.y);
    float grain = texture(detail, p / 0.9).r;
    float patch = texture(detail, p / 11.0).g;
    vec3 color = vec3(0.27, 0.275, 0.29) * (0.85 + 0.3 * grain) * (0.92 + 0.16 * patch);
    color = mix(color, vec3(0.85), line(side - (HALF_WIDTH - 0.15), 0.05) * 0.9);
    if (s < 1.2) {  // chequered start / finish line
        vec2 cell = floor(vec2(s, v_track.y) / 0.4);
        color = mix(color, vec3(mod(cell.x + cell.y, 2.0) * 0.85 + 0.03), step(side, HALF_WIDTH - 0.2));
    }
    if (side > HALF_WIDTH) {
        if (v_track.z > 0.5 && side <= HALF_WIDTH + KERB) {
            color = mod(floor(s / 1.0), 2.0) < 1.0 ? vec3(0.72, 0.07, 0.05) : vec3(0.88, 0.88, 0.86);
            color *= 0.9 + 0.15 * grain;
        } else if (side > HALF_WIDTH + KERB) {
            color = vec3(0.62, 0.61, 0.58);
        }
    }
    frag = vec4(encode(fogged(lit(color * color, n, 0.0), v_world)), 1.0);
}
"""
)

_SOLID_VERTEX = """
#version 330 core
in vec3 in_position;
in vec3 in_normal;
uniform mat4 vp;
uniform mat4 model;
out vec3 v_world;
out vec3 v_normal;
void main() {
    vec4 world = model * vec4(in_position, 1.0);
    v_world = world.xyz;
    v_normal = normalize(mat3(model) * in_normal);
    gl_Position = vp * world;
}
"""

_SOLID_FRAGMENT = (
    """
#version 330 core
in vec3 v_world;
in vec3 v_normal;
uniform vec3 base_color;  // linear
uniform float emissive;
out vec4 frag;
"""
    + _LIGHTING
    + """
void main() {
    vec3 n = normalize(v_normal);
    if (dot(n, cam_pos - v_world) < 0.0) n = -n;
    vec3 color = lit(base_color, n, 0.15) + base_color * emissive;
    // a little sheen from the sun so the paint is not flat
    vec3 view = normalize(cam_pos - v_world);
    float spec = pow(max(dot(reflect(-sun_dir, n), view), 0.0), 40.0) * 0.25;
    frag = vec4(encode(fogged(color + sun_color * spec, v_world)), 1.0);
}
"""
)

_SHADOW_VERTEX = """
#version 330 core
in vec2 in_corner;
uniform mat4 vp;
uniform vec3 centre;
uniform vec2 axis;      // unit vector along the car (x, z)
uniform vec2 half_size;
out vec2 v_local;
void main() {
    v_local = in_corner;
    vec2 side = vec2(-axis.y, axis.x);
    vec2 offset = axis * in_corner.x * half_size.x + side * in_corner.y * half_size.y;
    gl_Position = vp * vec4(centre + vec3(offset.x, 0.0, offset.y), 1.0);
}
"""

_SHADOW_FRAGMENT = """
#version 330 core
in vec2 v_local;
out vec4 frag;
void main() {
    vec2 q = max(abs(v_local) - vec2(0.45, 0.35), 0.0);
    float d = length(q) / 0.55;
    frag = vec4(0.0, 0.0, 0.0, 0.5 * (1.0 - smoothstep(0.0, 1.0, d)));
}
"""

MAX_SKID_QUADS = 16000


class Part:
    """A model made of single-coloured meshes (geometry primitives), drawn with any model matrix."""

    def __init__(self, ctx: moderngl.Context, program: moderngl.Program, primitives, emissive: float = 0.0) -> None:
        self.emissive = emissive
        self.meshes: list[tuple[moderngl.VertexArray, tuple[float, float, float]]] = []
        for primitive in primitives:
            data = np.concatenate([primitive.positions, primitive.normals], axis=1).astype(np.float32)
            vertex_buffer = ctx.buffer(data.tobytes())
            index_buffer = ctx.buffer(primitive.indices.astype(np.uint32).tobytes())
            vao = ctx.vertex_array(program, [(vertex_buffer, "3f 3f", "in_position", "in_normal")], index_buffer=index_buffer, index_element_size=4)
            self.meshes.append((vao, tuple(primitive.color[:3])))


class SkidMarks:
    """Dark ribbons where tyres slide, in a ring buffer of quads."""

    def __init__(self, ctx: moderngl.Context, program: moderngl.Program) -> None:
        self.vertices = np.zeros((MAX_SKID_QUADS * 4, 6), np.float32)
        self.vertex_buffer = ctx.buffer(reserve=self.vertices.nbytes, dynamic=True)
        quad = np.arange(MAX_SKID_QUADS, dtype=np.uint32)[:, None] * 4 + np.array([0, 1, 2, 1, 3, 2], np.uint32)
        self.index_buffer = ctx.buffer(quad.reshape(-1).tobytes())
        self.vao = ctx.vertex_array(program, [(self.vertex_buffer, "3f 3f", "in_position", "in_normal")], index_buffer=self.index_buffer, index_element_size=4)
        self.count = 0
        self.head = 0
        self.last = [None] * 4
        self.dirty = False

    def clear(self) -> None:
        self.count = self.head = 0
        self.last = [None] * 4

    def add(self, wheel: int, point: np.ndarray, axle: np.ndarray, normal: np.ndarray, width: float, strength: float) -> None:
        """Extend wheel `wheel`'s mark to `point` (strength > 0), or end it."""
        last = self.last[wheel]
        if strength <= 0.0:
            self.last[wheel] = None
            return
        edge = axle / max(float(np.linalg.norm(axle)), 1e-9) * width / 2.0
        lifted = point + normal * 0.006
        current = (lifted - edge, lifted + edge)
        if last is not None and float(np.linalg.norm(lifted - last[2])) > 0.04:
            if float(np.linalg.norm(lifted - last[2])) < 1.0:
                n = self.head
                self.vertices[n * 4 : n * 4 + 4, :3] = (last[0], last[1], current[0], current[1])
                self.vertices[n * 4 : n * 4 + 4, 3:] = normal
                self.head = (n + 1) % MAX_SKID_QUADS
                self.count = min(self.count + 1, MAX_SKID_QUADS)
                self.dirty = True
            self.last[wheel] = (*current, lifted)
        elif last is None:
            self.last[wheel] = (*current, lifted)

    def upload(self) -> None:
        if self.dirty:
            self.vertex_buffer.write(self.vertices.tobytes())
            self.dirty = False


class OutdoorRenderer:
    def __init__(self, ctx: moderngl.Context, ground: Ground, models: list[TreeModel], seed: int = 0, trees: int = 6000, visibility: float = 1800.0) -> None:
        self.ctx = ctx
        self.base = Renderer(ctx, models, visibility)
        self.base.set_terrain(ground.terrain)
        instances = place_trees(ground.terrain, models, seed, trees)
        kept = []
        for batch in instances:  # leave the proving ground free of trees
            kept.append(batch[ground.clear_of_paving(batch[:, 0], batch[:, 2])] if len(batch) else batch)
        self.base.set_trees(kept)
        self.road = ctx.program(vertex_shader=_ROAD_VERTEX, fragment_shader=_ROAD_FRAGMENT)
        self.track = ctx.program(vertex_shader=_TRACK_VERTEX, fragment_shader=_TRACK_FRAGMENT)
        self.solid = ctx.program(vertex_shader=_SOLID_VERTEX, fragment_shader=_SOLID_FRAGMENT)
        self.shadow = ctx.program(vertex_shader=_SHADOW_VERTEX, fragment_shader=_SHADOW_FRAGMENT)
        for program in (self.road, self.track, self.solid):
            for name in ("sun_dir", "sun_color", "fog_color", "fog_density"):
                _set(program, name, self.base.ground[name].value)
        self.road["detail"].value = 0
        self.track["detail"].value = 0
        marks = np.zeros(16, np.float32)
        edges = [x for section in SECTIONS for x in (section.x0, section.x1)]
        marks[: len(edges)] = edges
        self.road["marks"].write(marks.tobytes())
        self.road["mark_count"].value = len(edges)
        self.road_vaos: list[moderngl.VertexArray] = []
        self._buffers: list[moderngl.Buffer] = []
        for vertices, indices in (ground.pad_mesh(), ground.road_mesh()):
            vertex_buffer, index_buffer = ctx.buffer(vertices.tobytes()), ctx.buffer(indices.astype(np.uint32).tobytes())
            self._buffers += [vertex_buffer, index_buffer]
            self.road_vaos.append(ctx.vertex_array(self.road, [(vertex_buffer, "3f 3f", "in_position", "in_normal")], index_buffer=index_buffer, index_element_size=4))
        vertices, indices = ground.track.mesh(ground)
        vertex_buffer, index_buffer = ctx.buffer(vertices.tobytes()), ctx.buffer(indices.astype(np.uint32).tobytes())
        self._buffers += [vertex_buffer, index_buffer]
        self.track_vao = ctx.vertex_array(self.track, [(vertex_buffer, "3f 3f 3f", "in_position", "in_normal", "in_track")], index_buffer=index_buffer, index_element_size=4)
        corners = np.array([(-1, -1), (1, -1), (-1, 1), (1, 1)], np.float32)
        self.shadow_vao = ctx.vertex_array(self.shadow, [(ctx.buffer(corners.tobytes()), "2f", "in_corner")])
        self.skids = SkidMarks(ctx, self.solid)
        self.solid_parts: list[tuple[Part, np.ndarray]] = []
        self.static: list[tuple[Part, np.ndarray]] = []

    def make_part(self, primitives, emissive: float = 0.0) -> Part:
        return Part(self.ctx, self.solid, primitives, emissive)

    def draw(
        self,
        target: moderngl.Framebuffer,
        size: tuple[int, int],
        camera: Camera,
        parts: list[tuple[Part, np.ndarray]],
        shadow: tuple[np.ndarray, np.ndarray, tuple[float, float]] | None = None,
    ) -> None:
        """`parts` are (part, 4x4 matrix) pairs; `shadow` is (centre, along-the-car (x, z), half size) of a soft blob under the car."""
        self.base.draw(target, size, camera)
        width, height = size
        view_projection = camera.projection_matrix(width / height, NEAR, FAR) @ camera.view_matrix()
        packed = view_projection.astype("f4").T.tobytes()
        position = tuple(float(v) for v in camera.position)
        for program in (self.road, self.track, self.solid):
            program["vp"].write(packed)
            _set(program, "cam_pos", position)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.base.detail.use(0)
        for vao in self.road_vaos:
            vao.render(moderngl.TRIANGLES)
        self.track_vao.render(moderngl.TRIANGLES)
        self.skids.upload()
        if self.skids.count:
            self.solid["model"].write(np.eye(4, dtype="f4").tobytes())
            self.solid["base_color"].value = (0.004, 0.004, 0.004)
            self.solid["emissive"].value = 0.0
            self.skids.vao.render(moderngl.TRIANGLES, vertices=6 * self.skids.count)
        if shadow is not None:
            centre, axis, half = shadow
            self.shadow["vp"].write(packed)
            self.shadow["centre"].value = tuple(float(v) for v in centre)
            self.shadow["axis"].value = (float(axis[0]), float(axis[1]))
            self.shadow["half_size"].value = half
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            self.ctx.disable(moderngl.DEPTH_TEST)
            self.shadow_vao.render(moderngl.TRIANGLE_STRIP, vertices=4)
            self.ctx.enable(moderngl.DEPTH_TEST)
            self.ctx.disable(moderngl.BLEND)
        for part, matrix in self.static + parts:
            self.solid["model"].write(np.asarray(matrix, "f4").T.tobytes())
            self.solid["emissive"].value = part.emissive
            for vao, color in part.meshes:
                self.solid["base_color"].value = color
                vao.render(moderngl.TRIANGLES)
        self.ctx.scissor = None
