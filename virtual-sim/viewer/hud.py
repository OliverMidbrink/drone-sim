"""On-screen display: textured quads on the GL screen plus CPU-drawn panels (graphs and text) uploaded as textures."""

from __future__ import annotations

import cv2
import moderngl
import numpy as np

# BGR for OpenCV: x red, y green, z blue, and a fourth for the motors
AXIS_COLORS = ((70, 70, 255), (80, 230, 80), (255, 160, 60), (60, 220, 255))
PANEL_BORDER = (60, 60, 60)
FONT = cv2.FONT_HERSHEY_SIMPLEX

_VERTEX = """
#version 330 core
uniform vec4 rect;  // x0, y0, x1, y1 in clip space
uniform float flip;
out vec2 uv;
void main() {
    vec2 p = vec2(gl_VertexID & 1, (gl_VertexID >> 1) & 1);
    uv = vec2(p.x, mix(1.0 - p.y, p.y, flip));
    gl_Position = vec4(mix(rect.xy, rect.zw, p), 0.0, 1.0);
}
"""

_FRAGMENT = """
#version 330 core
uniform sampler2D tex;
in vec2 uv;
out vec4 frag;
void main() { frag = texture(tex, uv); }
"""


class Overlay:
    """Draws textures as screen-aligned rectangles, in pixels measured from the bottom-left corner."""

    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        self.program = ctx.program(vertex_shader=_VERTEX, fragment_shader=_FRAGMENT)
        self.vao = ctx.vertex_array(self.program, [])
        self.program["tex"].value = 0
        self._textures: dict[str, moderngl.Texture] = {}

    def upload(self, name: str, image: np.ndarray) -> moderngl.Texture:
        """Store a top-down RGB or RGBA array as the texture called `name`."""
        height, width, channels = image.shape
        texture = self._textures.get(name)
        if texture is None or texture.size != (width, height) or texture.components != channels:
            if texture is not None:
                texture.release()
            texture = self.ctx.texture((width, height), channels)
            texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._textures[name] = texture
        texture.write(np.ascontiguousarray(image).tobytes())
        return texture

    def draw(self, texture: moderngl.Texture, rect: tuple[float, float, float, float], screen: tuple[int, int], flip: bool = False, blend: bool = False) -> None:
        """`flip` is for textures rendered by GL (bottom row first) rather than uploaded from images."""
        width, height = screen
        x0, y0, x1, y1 = rect
        self.program["rect"].value = (x0 / width * 2 - 1, y0 / height * 2 - 1, x1 / width * 2 - 1, y1 / height * 2 - 1)
        self.program["flip"].value = 1.0 if flip else 0.0
        texture.use(0)
        self.ctx.disable(moderngl.DEPTH_TEST)
        if blend:
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.vao.render(moderngl.TRIANGLE_STRIP, vertices=4)
        if blend:
            self.ctx.disable(moderngl.BLEND)
        self.ctx.enable(moderngl.DEPTH_TEST)


def fit_rect(box: tuple[float, float, float, float], aspect: float) -> tuple[float, float, float, float]:
    """The largest rectangle of this width / height ratio centred in the box."""
    x0, y0, x1, y1 = box
    width, height = x1 - x0, y1 - y0
    if width / height > aspect:
        inner = height * aspect
        return x0 + (width - inner) / 2, y0, x0 + (width + inner) / 2, y1
    inner = width / aspect
    return x0, y0 + (height - inner) / 2, x1, y0 + (height + inner) / 2


class History:
    """The last `length` samples of a few channels."""

    def __init__(self, channels: int, length: int = 240) -> None:
        self.data = np.zeros((length, channels))

    def push(self, values) -> None:
        self.data = np.roll(self.data, -1, axis=0)
        self.data[-1] = values

    def clear(self) -> None:
        self.data[:] = 0.0


def draw_graph(canvas: np.ndarray, box: tuple[int, int, int, int], title: str, history: History, limit: float, unit: str, scale: float, colors=AXIS_COLORS, labels: str = "", show_values: bool = True) -> None:
    """A small strip chart: black background, one coloured trace per channel, the latest values printed underneath."""
    x0, y0, x1, y1 = box
    cv2.rectangle(canvas, (x0, y0), (x1, y1), PANEL_BORDER, 1)
    cv2.putText(canvas, title, (x0 + 4, y0 + int(13 * scale)), FONT, 0.38 * scale, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.putText(canvas, unit, (x1 - int(38 * scale), y0 + int(13 * scale)), FONT, 0.3 * scale, (130, 130, 130), 1, cv2.LINE_AA)
    top, bottom = y0 + int(19 * scale), y1 - int(15 * scale)
    middle = (top + bottom) // 2
    cv2.line(canvas, (x0 + 1, middle), (x1 - 1, middle), (55, 55, 55), 1)
    data = history.data
    xs = np.linspace(x0 + 2, x1 - 2, len(data))
    for channel in range(data.shape[1]):
        ys = middle - np.clip(data[:, channel] / limit, -1.0, 1.0) * (bottom - top) / 2
        cv2.polylines(canvas, [np.stack([xs, ys], axis=1).astype(np.int32)], False, colors[channel], 1, cv2.LINE_AA)
    if not show_values:
        return
    latest = data[-1]
    cursor = x0 + 4
    for channel in range(data.shape[1]):
        text = f"{labels[channel] if channel < len(labels) else ''}{latest[channel]:+.1f}"
        cv2.putText(canvas, text, (cursor, y1 - int(4 * scale)), FONT, 0.3 * scale, colors[channel], 1, cv2.LINE_AA)
        cursor += int((x1 - x0 - 6) / data.shape[1])
