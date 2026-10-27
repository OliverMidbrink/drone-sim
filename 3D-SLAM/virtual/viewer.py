#!/usr/bin/env python3
"""Fly a virtual camera in a figure 8 over a grass field with boxes (OpenGL via moderngl + glfw).

The window shows what the drone camera sees. Esc or closing the window quits.
The scene and the path are defined in world.py; the specs are printed at start-up.
"""

from __future__ import annotations

import argparse

import glfw
import moderngl

import renderer
import world


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fov-deg", type=float, default=world.DEFAULT_FOV_DEG, help="horizontal field of view (default: %(default)g)")
    parser.add_argument("--width", type=int, default=world.DEFAULT_SIZE[0], help="window width (default: %(default)d)")
    parser.add_argument("--height", type=int, default=world.DEFAULT_SIZE[1], help="window height (default: %(default)d)")
    parser.add_argument("--seed", type=int, default=world.DEFAULT_SEED, help="seed for box layout and textures (default: %(default)d)")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    boxes = world.generate_boxes(args.seed)
    if not boxes:
        raise SystemExit("No boxes could be placed")
    path = world.FigureEightPath(world.flight_altitude(boxes))
    print("\n".join(world.describe(boxes, path, args.fov_deg, args.width, args.height)))

    if not glfw.init():
        raise SystemExit("GLFW failed to initialize")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.SAMPLES, 4)
    window = glfw.create_window(args.width, args.height, "Virtual drone camera", None, None)
    if window is None:
        glfw.terminate()
        raise SystemExit("Could not create the OpenGL window")
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    ctx = moderngl.create_context()
    scene = renderer.SceneRenderer(ctx, boxes, args.seed)
    start = glfw.get_time()
    try:
        while not glfw.window_should_close(window):
            glfw.poll_events()
            if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
                break
            fb_w, fb_h = glfw.get_framebuffer_size(window)
            if fb_w == 0 or fb_h == 0:
                continue  # minimised
            position, forward = path.pose(glfw.get_time() - start)
            scene.draw(ctx.screen, (fb_w, fb_h), position, forward, args.fov_deg)
            glfw.swap_buffers(window)
    finally:
        glfw.destroy_window(window)
        glfw.terminate()


if __name__ == "__main__":
    main()
