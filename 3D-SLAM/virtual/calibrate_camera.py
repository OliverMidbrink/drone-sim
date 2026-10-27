#!/usr/bin/env python3
"""Calibrate the virtual camera at pixel level: trace the viewing ray (angle) of every pixel.

This is the generic "two-plane" calibration (Grossberg & Nayar's raxel model, Sturm &
Ramalingam): look at a target whose points carry their own coordinates, at two known
distances. Pixel p sees point P1 on the near plane and P2 on the far plane; the line P1 -> P2
is that pixel's viewing ray. Nothing about the lens is assumed, so it would also measure a
distorted lens, and it is the same procedure you would run on a real camera facing a
monitor that shows coordinate patterns.

Here the target is rendered: a plane whose fragments output their own (x, y, z) as 32-bit
floats, once at 1 m and once at 2 m. The measured rays are then
  - compared with the ideal pinhole rays for the camera's declared FOV,
  - fitted with a pinhole model (fx, fy, cx, cy) to show how pinhole the camera really is.

Angles saved / shown for every pixel (camera axes x right, y down, z forward):
  horizontal angle = atan2(x, z)   the ray's angle left (-) / right (+) of the optical axis
  vertical angle   = atan2(-y, z)  the ray's angle below (-) / above (+) the optical axis
  off-axis angle   = acos(z)       the true angle between the ray and the optical axis

Window: 1 / 2 / 3 choose the map, hover shows the angles at the pixel, white lines every
10 degrees, Esc closes. Results go to virtual/out/camera_rays.npz and camera_angles.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import moderngl
import numpy as np

import renderer
import world

NEAR_PLANE_M, FAR_PLANE_M = 1.0, 2.0
OUT_DIR = Path(__file__).resolve().parent / "out"
CALIBRATION_WINDOW = "Camera calibration: pixel angles"

_VERTEX_SHADER = """
#version 330 core
in vec3 in_position;
uniform mat4 mvp;
out vec3 v_position;
void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
    v_position = in_position;
}
"""

_FRAGMENT_SHADER = """
#version 330 core
in vec3 v_position;
out vec4 fragColor;
void main() {
    fragColor = vec4(v_position, 1.0);
}
"""


def plane_hits(ctx: moderngl.Context, program: moderngl.Program, size: tuple[int, int], fov_deg: float, distance: float) -> np.ndarray:
    """For every pixel, the camera-space (x right, y up, z backward) point it sees on the plane at `distance`. (H, W, 3)."""
    width, height = size
    colour = ctx.texture(size, 4, dtype="f4")
    framebuffer = ctx.framebuffer(color_attachments=[colour])
    framebuffer.viewport = (0, 0, width, height)
    framebuffer.use()
    framebuffer.clear(0.0, 0.0, 0.0, 0.0)
    ctx.disable(moderngl.DEPTH_TEST)

    extent = 10.0 * distance  # far wider than any FOV up to 150 degrees needs
    corners = [(-extent, -extent), (extent, -extent), (extent, extent), (-extent, extent)]
    vertices = np.array([[*corners[i], -distance] for i in (0, 1, 2, 0, 2, 3)], dtype=np.float32)
    buffer = ctx.buffer(vertices.tobytes())
    array = ctx.vertex_array(program, [(buffer, "3f", "in_position")])

    aspect = width / height
    view = renderer.look_along(np.zeros(3), np.array([0.0, 0.0, -1.0]), renderer.UP)  # GL camera at the origin, looking down -z
    mvp = renderer.perspective(renderer.vertical_fov(fov_deg, aspect), aspect, renderer.NEAR, renderer.FAR) @ view
    program["mvp"].write(mvp.astype(np.float32).T.tobytes())
    array.render(moderngl.TRIANGLES)

    rgba = np.frombuffer(colour.read(), dtype=np.float32).reshape(height, width, 4)[::-1]  # GL rows start at the bottom
    if not np.all(rgba[..., 3] == 1.0):
        raise SystemExit("The calibration plane does not cover the whole image")
    hits = rgba[..., :3].copy()
    for resource in (array, buffer, framebuffer, colour):
        resource.release()
    return hits


def trace_rays(ctx: moderngl.Context, size: tuple[int, int], fov_deg: float) -> np.ndarray:
    """Measured unit viewing ray of every pixel, (H, W, 3), camera axes x right, y down, z forward."""
    program = ctx.program(vertex_shader=_VERTEX_SHADER, fragment_shader=_FRAGMENT_SHADER)
    near = plane_hits(ctx, program, size, fov_deg, NEAR_PLANE_M)
    far = plane_hits(ctx, program, size, fov_deg, FAR_PLANE_M)
    direction = (far - near).astype(np.float64)  # near point -> far point, GL camera axes (x right, y up, looking down -z); float64 keeps the angle report free of float32 rounding noise
    rays = np.stack([direction[..., 0], -direction[..., 1], -direction[..., 2]], axis=-1)  # -> x right, y down, z forward
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


def ideal_pinhole_rays(size: tuple[int, int], fov_deg: float) -> np.ndarray:
    """Rays of a perfect pinhole camera with this horizontal FOV (pixel centres at index + 0.5)."""
    width, height = size
    focal = (width / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    rays = np.stack([(u - width / 2.0) / focal, (v - height / 2.0) / focal, np.ones_like(u)], axis=-1)
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


def fit_pinhole(rays: np.ndarray) -> tuple[float, float, float, float, float]:
    """Least-squares pinhole fit u = fx * x/z + cx, v = fy * y/z + cy -> (fx, fy, cx, cy, worst residual in px)."""
    height, width = rays.shape[:2]
    u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    x_over_z = rays[..., 0] / rays[..., 2]
    y_over_z = rays[..., 1] / rays[..., 2]
    fx, cx = np.polyfit(x_over_z.ravel(), u.ravel(), 1)
    fy, cy = np.polyfit(y_over_z.ravel(), v.ravel(), 1)
    residual = max(float(np.abs(u - (fx * x_over_z + cx)).max()), float(np.abs(v - (fy * y_over_z + cy)).max()))
    return float(fx), float(fy), float(cx), float(cy), residual


def angle_maps(rays: np.ndarray) -> dict[str, np.ndarray]:
    x, y, z = rays[..., 0], rays[..., 1], rays[..., 2]
    return {
        "horizontal angle": np.degrees(np.arctan2(x, z)),
        "vertical angle": np.degrees(np.arctan2(-y, z)),
        "off-axis angle": np.degrees(np.arccos(np.clip(z, -1.0, 1.0))),
    }


def angle_panel(angle_deg: np.ndarray, title: str, step_deg: float = 10.0) -> np.ndarray:
    """Colour picture of one angle map with white iso-lines every `step_deg` degrees."""
    low, high = float(angle_deg.min()), float(angle_deg.max())
    scaled = ((angle_deg - low) / max(high - low, 1e-9) * 255.0).astype(np.uint8)
    image = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    band = np.floor(angle_deg / step_deg).astype(int)
    edge = np.zeros(band.shape, dtype=bool)
    edge[:, 1:] |= band[:, 1:] != band[:, :-1]
    edge[1:, :] |= band[1:, :] != band[:-1, :]
    image[edge] = (255, 255, 255)
    label = f"{title}  ({low:.1f} to {high:.1f} deg, lines every {step_deg:g} deg)"
    cv2.putText(image, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def window_closed() -> bool:
    """True once the user closed the window (the Qt backend raises instead of returning 0)."""
    try:
        return cv2.getWindowProperty(CALIBRATION_WINDOW, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fov-deg", type=float, default=world.DEFAULT_FOV_DEG, help="horizontal FOV of the virtual camera (default: %(default)g)")
    parser.add_argument("--width", type=int, default=960, help="image width (default: %(default)d)")
    parser.add_argument("--height", type=int, default=540, help="image height (default: %(default)d)")
    parser.add_argument("--no-show", action="store_true", help="only write the files, do not open the window")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    size = (args.width, args.height)
    _window, ctx = renderer.create_hidden_context()

    measured = trace_rays(ctx, size, args.fov_deg)
    ideal = ideal_pinhole_rays(size, args.fov_deg)
    error_deg = np.degrees(np.arccos(np.clip(np.einsum("hwc,hwc->hw", measured, ideal), -1.0, 1.0)))
    fx, fy, cx, cy, residual_px = fit_pinhole(measured)
    maps = angle_maps(measured)

    print(f"Calibration of the virtual camera: {args.width} x {args.height}, declared horizontal FOV {args.fov_deg:g} deg")
    print(f"  measured vs ideal pinhole rays   mean {error_deg.mean():.5f} deg, max {error_deg.max():.5f} deg")
    print(f"  pinhole fit                      fx {fx:.3f}  fy {fy:.3f}  cx {cx:.3f}  cy {cy:.3f}   (pixel corner = 0, centre of pixel i = i + 0.5)")
    print(f"  in OpenCV index convention       cx {cx - 0.5:.3f}  cy {cy - 0.5:.3f}   (3d-slam.py assumes cx = {args.width / 2:g}, cy = {args.height / 2:g})")
    print(f"  worst deviation from the fit     {residual_px:.5f} px   (0 = a perfect pinhole, lens distortion would show here)")
    print(f"  horizontal FOV from the fit      {np.degrees(2 * np.arctan(args.width / 2 / fx)):.3f} deg")
    print(f"  vertical FOV from the fit        {np.degrees(2 * np.arctan(args.height / 2 / fy)):.3f} deg")
    print(f"  corner pixel off-axis angle      {maps['off-axis angle'][0, 0]:.2f} deg")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_DIR / "camera_rays.npz",
        rays=measured.astype(np.float32),
        horizontal_deg=maps["horizontal angle"].astype(np.float32),
        vertical_deg=maps["vertical angle"].astype(np.float32),
        off_axis_deg=maps["off-axis angle"].astype(np.float32),
        fov_deg=args.fov_deg,
        size=np.array(size),
    )
    panels = {name: angle_panel(values, name) for name, values in maps.items()}
    cv2.imwrite(str(OUT_DIR / "camera_angles.png"), np.vstack(list(panels.values())))
    print(f"  wrote {OUT_DIR / 'camera_rays.npz'} and {OUT_DIR / 'camera_angles.png'}")
    if args.no_show:
        return

    names = list(maps)
    state = {"map": 0, "mouse": (0, 0)}

    def on_mouse(_event: int, x: int, y: int, _flags: int, _param) -> None:
        state["mouse"] = (x, y)

    cv2.namedWindow(CALIBRATION_WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(CALIBRATION_WINDOW, on_mouse)
    while True:
        image = panels[names[state["map"]]].copy()
        x, y = state["mouse"]
        if 0 <= x < args.width and 0 <= y < args.height:
            text = (
                f"pixel ({x}, {y})  horizontal {maps['horizontal angle'][y, x]:+.2f}  "
                f"vertical {maps['vertical angle'][y, x]:+.2f}  off-axis {maps['off-axis angle'][y, x]:.2f} deg"
            )
            cv2.putText(image, text, (10, args.height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(image, text, (10, args.height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(CALIBRATION_WINDOW, image)
        key = cv2.waitKey(30) & 0xFF
        if key == 27 or window_closed():
            break
        if key in (ord("1"), ord("2"), ord("3")):
            state["map"] = key - ord("1")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
