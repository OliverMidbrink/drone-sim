"""Lightweight drone flight simulator: a random mountain-and-forest world you can fly around in.

    python virtual-sim/simulator.py                    # random world
    python virtual-sim/simulator.py --seed 7           # the same world again
    python virtual-sim/simulator.py --screenshot a.png # render one frame, no window

Controls (same feel as the 3D-SLAM map viewer):
    W / S        fly forward / back along the view direction
    A / D        strafe left / right
    Space / Ctrl up / down
    Shift        3x faster
    Q / E        roll left / right
    Left or right mouse drag   look around (Tab: capture the mouse instead)
    Scroll       change the flying speed
    M            tall / low mountains (low = about twice a tree's height)
    G            keep above the ground on / off
    R            back to the start pose
    N            generate a new random world
    Esc          quit
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import glfw
import moderngl
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from camera import Camera  # noqa: E402
from render import OffscreenTarget, Renderer, create_hidden_context  # noqa: E402
from terrain import Terrain, save_heightmap  # noqa: E402
from trees import load_models  # noqa: E402
from world import World  # noqa: E402

START_SPEED = 15.0  # m/s
BOOST = 3.0
MIN_SPEED, MAX_SPEED = 1.0, 500.0
ROLL_SPEED = np.radians(70.0)
MOUSE_SENSITIVITY = 0.0025  # radians per pixel
GROUND_CLEARANCE = 1.0  # metres kept above the terrain while the ground clamp is on


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=None, help="world seed (default: random)")
    parser.add_argument("--size", type=float, default=2000.0, help="terrain width in metres")
    parser.add_argument("--resolution", type=int, default=801, help="terrain grid points per side (lower = faster start)")
    parser.add_argument("--trees", type=int, default=6000, help="most trees to plant (0 = none)")
    parser.add_argument("--godot-dir", type=Path, default=None, help="the Godot forest demo (default: godot-forest-demo-* in the project root)")
    parser.add_argument("--heightmap", type=Path, default=None, help="greyscale image to use as the terrain instead of generating one")
    parser.add_argument("--heightmap-max", type=float, default=600.0, help="metres for a white pixel of --heightmap")
    parser.add_argument("--export-heightmap", type=Path, default=None, help="write the generated terrain as a 16-bit PNG and exit")
    parser.add_argument("--low-mountains", action="store_true", help="start with the mountains lowered to about twice a tree's height (M toggles)")
    parser.add_argument("--fov", type=float, default=90.0, help="horizontal field of view in degrees")
    parser.add_argument("--visibility", type=float, default=3500.0, help="fog distance in metres")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no-vsync", action="store_true", help="uncap the frame rate")
    parser.add_argument("--screenshot", type=Path, default=None, help="render one frame to this image and exit (no window)")
    parser.add_argument("--pose", type=float, nargs=5, metavar=("X", "Y", "Z", "YAW_DEG", "PITCH_DEG"), help="camera pose for --screenshot (y is the altitude)")
    return parser.parse_args()


def build_world(args: argparse.Namespace, seed: int, models) -> World:
    return World.generate(seed, models, args.size, args.resolution, args.trees, args.heightmap, args.heightmap_max, args.low_mountains)


def export_heightmap(args: argparse.Namespace, seed: int) -> None:
    heights = Terrain.generate(seed, args.size, args.resolution, low=args.low_mountains).heights
    height_max = float(np.ceil(heights.max() / 10.0) * 10.0)
    save_heightmap(args.export_heightmap, heights, height_max)
    print(f"Wrote {args.export_heightmap} ({heights.shape[1]} px); use it with --heightmap {args.export_heightmap} --heightmap-max {height_max:.0f}")


def take_screenshot(args: argparse.Namespace, seed: int) -> None:
    _window, ctx = create_hidden_context()
    models = load_models(args.godot_dir) if args.trees > 0 else []
    world = build_world(args, seed, models)
    renderer = Renderer(ctx, models, args.visibility)
    renderer.set_terrain(world.terrain)
    renderer.set_trees(world.trees)
    camera = world.spawn_camera(args.fov)
    if args.pose:
        x, y, z, yaw, pitch = args.pose
        camera.position, camera.yaw, camera.pitch = np.array([x, y, z]), np.radians(yaw), np.radians(pitch)
    target = OffscreenTarget(ctx, (args.width, args.height))
    renderer.draw(target.framebuffer, target.size, camera)
    args.screenshot.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.screenshot), target.read_bgr())
    print(f"Wrote {args.screenshot} ({renderer.last_tree_count} trees drawn)")
    glfw.terminate()


class Simulator:
    def __init__(self, args: argparse.Namespace, seed: int) -> None:
        self.args = args
        if not glfw.init():
            raise SystemExit("GLFW failed to initialize")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.SAMPLES, 4)
        self.window = glfw.create_window(args.width, args.height, "virtual-sim", None, None)
        if self.window is None:
            glfw.terminate()
            raise SystemExit("Could not create an OpenGL window")
        glfw.make_context_current(self.window)
        glfw.swap_interval(0 if args.no_vsync else 1)
        self.ctx = moderngl.create_context()

        self.models = load_models(args.godot_dir) if args.trees > 0 else []
        self.renderer = Renderer(self.ctx, self.models, args.visibility)
        self.speed = START_SPEED
        self.ground_clamp = True
        self.captured = False
        self.dragging = False
        self.last_cursor: tuple[float, float] | None = None
        self.camera = Camera()
        self.start_pose = Camera()
        self.world: World
        self.load_world(seed)

        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor)
        glfw.set_mouse_button_callback(self.window, self._on_button)
        glfw.set_scroll_callback(self.window, self._on_scroll)

    # --- world -----------------------------------------------------------------------------

    def load_world(self, seed: int, keep_pose: bool = False) -> None:
        glfw.set_window_title(self.window, f"virtual-sim - generating world {seed} ...")
        self.world = build_world(self.args, seed, self.models)
        self.renderer.set_terrain(self.world.terrain)
        self.renderer.set_trees(self.world.trees)
        self.start_pose = self.world.spawn_camera(self.args.fov)
        if not keep_pose:
            self.reset_pose()

    def reset_pose(self) -> None:
        start = self.start_pose
        self.camera = Camera(start.position.copy(), start.yaw, start.pitch, 0.0, start.fov_deg)

    # --- input -----------------------------------------------------------------------------

    def _on_key(self, window, key, _scancode, action, _mods) -> None:
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_TAB:
            self.captured = not self.captured
            glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED if self.captured else glfw.CURSOR_NORMAL)
            self.last_cursor = None
        elif key == glfw.KEY_G:
            self.ground_clamp = not self.ground_clamp
        elif key == glfw.KEY_R:
            self.reset_pose()
        elif key == glfw.KEY_M:
            self.args.low_mountains = not self.args.low_mountains
            self.load_world(self.world.seed, keep_pose=True)  # same world, other mountain height
        elif key == glfw.KEY_N:
            self.load_world(int(np.random.default_rng().integers(0, 1_000_000)))

    def _on_button(self, window, button, action, _mods) -> None:
        if button in (glfw.MOUSE_BUTTON_LEFT, glfw.MOUSE_BUTTON_RIGHT):
            self.dragging = action == glfw.PRESS
            self.last_cursor = None

    def _on_cursor(self, window, x, y) -> None:
        if self.last_cursor is not None and (self.dragging or self.captured):
            self.camera.look((x - self.last_cursor[0]) * MOUSE_SENSITIVITY, -(y - self.last_cursor[1]) * MOUSE_SENSITIVITY)
        self.last_cursor = (x, y)

    def _on_scroll(self, window, _dx, dy) -> None:
        self.speed = float(np.clip(self.speed * 1.15**dy, MIN_SPEED, MAX_SPEED))

    def _down(self, key: int) -> float:
        return 1.0 if glfw.get_key(self.window, key) == glfw.PRESS else 0.0

    def _fly(self, dt: float) -> None:
        camera = self.camera
        forward, right, _ = camera.axes()
        ahead = self._down(glfw.KEY_W) - self._down(glfw.KEY_S)
        side = self._down(glfw.KEY_D) - self._down(glfw.KEY_A)
        lift = self._down(glfw.KEY_SPACE) - (self._down(glfw.KEY_LEFT_CONTROL) + self._down(glfw.KEY_RIGHT_CONTROL))
        boost = BOOST if self._down(glfw.KEY_LEFT_SHIFT) or self._down(glfw.KEY_RIGHT_SHIFT) else 1.0
        camera.position = camera.position + (forward * ahead + right * side + np.array([0.0, lift, 0.0])) * self.speed * boost * dt
        camera.roll += (self._down(glfw.KEY_E) - self._down(glfw.KEY_Q)) * ROLL_SPEED * dt

        terrain = self.world.terrain
        limit = terrain.half - 2.0
        camera.position[[0, 2]] = np.clip(camera.position[[0, 2]], -limit, limit)
        if self.ground_clamp:
            camera.position[1] = max(camera.position[1], float(terrain.height_at(camera.position[0], camera.position[2])) + GROUND_CLEARANCE)

    # --- main loop -------------------------------------------------------------------------

    def run(self) -> None:
        last, title_at, frames = time.perf_counter(), 0.0, 0
        while not glfw.window_should_close(self.window):
            now = time.perf_counter()
            dt, last = min(now - last, 0.1), now
            glfw.poll_events()
            self._fly(dt)
            width, height = glfw.get_framebuffer_size(self.window)
            if width > 0 and height > 0:
                self.renderer.draw(self.ctx.screen, (width, height), self.camera)
            glfw.swap_buffers(self.window)
            frames += 1
            if now - title_at > 0.4:
                self._update_title(frames / max(now - title_at, 1e-6) if title_at else 0.0)
                title_at, frames = now, 0
        glfw.terminate()

    def _update_title(self, fps: float) -> None:
        x, y, z = self.camera.position
        agl = y - float(self.world.terrain.height_at(x, z))
        glfw.set_window_title(
            self.window,
            f"virtual-sim | seed {self.world.seed} | {fps:.0f} fps | x {x:.0f} z {z:.0f} | alt {y:.0f} m, {agl:.0f} m above ground | "
            f"speed {self.speed:.0f} m/s | trees drawn {self.renderer.last_tree_count} | mountains {'low' if self.args.low_mountains else 'tall'} | ground clamp {'on' if self.ground_clamp else 'off'}",
        )


def main() -> None:
    args = parse_args()
    seed = args.seed if args.seed is not None else int(np.random.default_rng().integers(0, 1_000_000))
    if args.export_heightmap:
        export_heightmap(args, seed)
    elif args.screenshot:
        take_screenshot(args, seed)
    else:
        print(__doc__)
        Simulator(args, seed).run()


if __name__ == "__main__":
    main()
