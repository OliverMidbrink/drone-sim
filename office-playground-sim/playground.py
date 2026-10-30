"""Office playground: walk around the office and watch a two-wheeled robot balance in it.

    python office-playground-sim/playground.py
    python office-playground-sim/playground.py --seed 3            # another arrangement of the furniture
    python office-playground-sim/playground.py --screenshot a.png --mode follow

Walking (click the window to capture the mouse, Esc to let go, Esc again to quit):
    W A S D        walk            Shift  sprint
    Space          jump (you can jump onto desks)
    Mouse          look around
    V              camera: walk / follow the robot / ride on the robot
    X              ceiling on / off (off for a view from above: jump onto a desk and look down)
    H              HUD on / off

The robot (arrow keys work in every camera mode; W A S D too while following or riding):
    Up / Down      drive forward / back          Left / Right   turn
    P / O          shove it forward / backward   U              stand it up again after a fall
    T              bring it to the front of you  R              back to where it started
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
import paths  # noqa: E402,F401

from camera import Camera  # noqa: E402
from hud import FONT, History, Overlay, draw_graph  # noqa: E402
from office.layout import SPAWN, OfficeLayout  # noqa: E402
from office.renderer import OfficeRenderer  # noqa: E402
from office.walker import Walker  # noqa: E402
from render import OffscreenTarget, create_hidden_context  # noqa: E402
from robot.firmware import Command  # noqa: E402
from robot.model import RobotModel  # noqa: E402
from robot.sim import RobotSim  # noqa: E402

MOUSE_SENSITIVITY = 0.0023
DRIVE_SPEED = 0.7  # m/s
DRIVE_TURN = 1.8  # rad/s
MODES = ("walk", "follow", "ride")
HUD_SIZE = (460, 250)


def open_window(width: int, height: int, title: str, vsync: bool):
    if not glfw.init():
        raise SystemExit("GLFW failed to initialize")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.SAMPLES, 4)
    window = glfw.create_window(width, height, title, None, None)
    if window is None:
        glfw.terminate()
        raise SystemExit("Could not create an OpenGL window")
    glfw.make_context_current(window)
    glfw.swap_interval(1 if vsync else 0)
    return window, moderngl.create_context()


class Scene:
    """The office, the robot, the person walking, and the cameras. Independent of any window."""

    def __init__(self, ctx: moderngl.Context, seed: int, fov: float, robot: bool = True) -> None:
        self.ctx, self.fov = ctx, fov
        self.layout = OfficeLayout(seed)
        self.renderer = OfficeRenderer(ctx)
        self.renderer.set_static(self.layout.building, "building")
        self.renderer.set_static(self.layout.ceiling, "ceiling")
        self.renderer.set_lights(self.layout.lights)
        self.walker = Walker(self.layout.boxes, 1.2, SPAWN[1], np.pi / 2.0)  # camera yaw 90 degrees looks along +x
        self.robot: RobotSim | None = None
        if robot:
            self.robot = RobotSim(SPAWN[0] + 1.2, SPAWN[1], SPAWN[2], seed=seed, obstacles=self.layout.boxes)
            self.model = RobotModel(self.renderer, self.robot.params)
        self.mode = "walk"
        self.chase_offset = np.array([0.0, np.radians(-14.0)])  # yaw relative to the robot's heading, pitch
        self.chase_heading = SPAWN[2]

    # --- the robot ---------------------------------------------------------------------------

    def bring_robot(self) -> None:
        if not self.robot:
            return
        yaw = self.walker.yaw
        ahead = np.array([np.sin(yaw), -np.cos(yaw)])
        target = self.walker.position + ahead * 1.6
        if not self.layout.is_free(float(target[0]), float(target[1]), 0.2):
            target = self.walker.position + ahead * 0.9
        self.robot.place(float(target[0]), float(target[1]), float(np.arctan2(-ahead[1], -ahead[0])))  # facing the walker

    def reset_robot(self) -> None:
        if self.robot:
            self.robot.place(SPAWN[0] + 1.2, SPAWN[1], SPAWN[2])

    # --- cameras -----------------------------------------------------------------------------

    def camera(self) -> Camera:
        robot = self.robot
        if self.mode == "walk" or robot is None:
            w = self.walker
            return Camera(w.eye, w.yaw, w.pitch, 0.0, self.fov)
        plant = robot.plant
        heading = plant.heading
        if self.mode == "ride":
            p = robot.params
            c, s = np.cos(heading), np.sin(heading)
            theta = plant.pitch
            lean_forward = np.array([np.cos(theta), -np.sin(theta), 0.0])  # the body's forward and up while it leans
            lean_up = np.array([np.sin(theta), np.cos(theta), 0.0])

            def world(v: np.ndarray) -> np.ndarray:
                return np.array([v[0] * c - v[2] * s, v[1], v[0] * s + v[2] * c])

            mast = world(lean_forward * 0.04 + lean_up * 0.20)
            position = np.array([plant.x, p.wheel_radius, plant.z]) + mast
            right = world(np.array([0.0, 0.0, 1.0]))
            return Camera(position, basis=(world(lean_forward), right, world(lean_up)), fov_deg=self.fov)
        # follow: behind the robot, the mouse swings round it
        delta = (heading - self.chase_heading + np.pi) % (2.0 * np.pi) - np.pi
        self.chase_heading += delta * 0.08
        yaw = self.chase_heading + np.pi / 2.0 + self.chase_offset[0]
        pitch = self.chase_offset[1]
        forward = np.array([np.sin(yaw) * np.cos(pitch), np.sin(pitch), -np.cos(yaw) * np.cos(pitch)])
        target = np.array([plant.x, 0.13, plant.z])
        distance = self._clear_distance(target, -forward, 0.85)
        return Camera(target - forward * distance, yaw, pitch, 0.0, self.fov)

    def _clear_distance(self, origin: np.ndarray, direction: np.ndarray, wanted: float) -> float:
        """How far a camera may sit from `origin` along `direction` before it would be inside a wall or furniture."""
        bounds = self.walker.bounds
        for d in np.linspace(0.1, wanted, 12):
            p = origin + direction * d
            if p[1] < 0.05 or p[1] > 2.9:
                return max(d - 0.12, 0.15)
            inside = (bounds[:, 0] < p[0]) & (p[0] < bounds[:, 2]) & (bounds[:, 1] < p[2]) & (p[2] < bounds[:, 3]) & (bounds[:, 4] > p[1])
            if inside.any():
                return max(d - 0.12, 0.15)
        return wanted

    def models(self) -> list:
        return self.model.draw_list(self.robot.plant) if self.robot else []

    def draw(self, target, size: tuple[int, int], origin: tuple[int, int] = (0, 0)) -> None:
        self.renderer.draw(target, size, self.camera(), self.models(), origin)


# --- window ------------------------------------------------------------------------------------


class Playground:
    def __init__(self, args: argparse.Namespace) -> None:
        self.window, self.ctx = open_window(args.width, args.height, "office playground", not args.no_vsync)
        self.scene = Scene(self.ctx, args.seed, args.fov, not args.no_robot)
        self.overlay = Overlay(self.ctx)
        self.captured = False
        self.last_cursor: tuple[float, float] | None = None
        self.hud = True
        self.jump_requested = False
        self.charts = [History(1, 200) for _ in range(4)]
        self.hud_image: np.ndarray | None = None
        self.hud_at = 0.0
        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor)
        glfw.set_mouse_button_callback(self.window, self._on_button)
        self._capture(True)

    # --- input -------------------------------------------------------------------------------

    def _capture(self, on: bool) -> None:
        self.captured = on
        glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_DISABLED if on else glfw.CURSOR_NORMAL)
        self.last_cursor = None

    def _on_button(self, window, button, action, _mods) -> None:
        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS and not self.captured:
            self._capture(True)

    def _on_cursor(self, window, x, y) -> None:
        if self.captured and self.last_cursor is not None:
            dx, dy = (x - self.last_cursor[0]) * MOUSE_SENSITIVITY, -(y - self.last_cursor[1]) * MOUSE_SENSITIVITY
            scene = self.scene
            if scene.mode == "walk" or scene.robot is None:
                scene.walker.look(dx, dy)
            elif scene.mode == "follow":
                scene.chase_offset[0] = (scene.chase_offset[0] + dx + np.pi) % (2.0 * np.pi) - np.pi
                scene.chase_offset[1] = float(np.clip(scene.chase_offset[1] + dy, np.radians(-80), np.radians(60)))
        self.last_cursor = (x, y)

    def _on_key(self, window, key, _scancode, action, _mods) -> None:
        if action != glfw.PRESS:
            return
        scene = self.scene
        if key == glfw.KEY_ESCAPE:
            if self.captured:
                self._capture(False)
            else:
                glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_SPACE:
            self.jump_requested = True
        elif key == glfw.KEY_V and scene.robot:
            scene.mode = MODES[(MODES.index(scene.mode) + 1) % len(MODES)]
            scene.chase_offset[0] = 0.0
        elif key == glfw.KEY_X:
            hidden = scene.renderer.hidden_layers
            hidden.symmetric_difference_update({"ceiling"})
        elif key == glfw.KEY_H:
            self.hud = not self.hud
        elif scene.robot:
            if key == glfw.KEY_P:
                scene.robot.push(3.0, 0.12)
            elif key == glfw.KEY_O:
                scene.robot.push(-3.0, 0.12)
            elif key == glfw.KEY_U:
                scene.robot.stand_up()
            elif key == glfw.KEY_T:
                scene.bring_robot()
            elif key == glfw.KEY_R:
                scene.reset_robot()

    def _down(self, *keys: int) -> float:
        return 1.0 if any(glfw.get_key(self.window, k) == glfw.PRESS for k in keys) else 0.0

    def _walk_input(self) -> np.ndarray:
        w = self.scene.walker
        ahead = self._down(glfw.KEY_W) - self._down(glfw.KEY_S)
        side = self._down(glfw.KEY_D) - self._down(glfw.KEY_A)
        forward = np.array([np.sin(w.yaw), -np.cos(w.yaw)])
        right = np.array([np.cos(w.yaw), np.sin(w.yaw)])
        move = forward * ahead + right * side
        length = float(np.hypot(*move))
        return move / length if length > 1.0 else move

    def _robot_command(self) -> Command:
        if self.scene.mode == "walk":
            ahead = self._down(glfw.KEY_UP) - self._down(glfw.KEY_DOWN)
            turn = self._down(glfw.KEY_RIGHT) - self._down(glfw.KEY_LEFT)
        else:
            ahead = self._down(glfw.KEY_UP, glfw.KEY_W) - self._down(glfw.KEY_DOWN, glfw.KEY_S)
            turn = self._down(glfw.KEY_RIGHT, glfw.KEY_D) - self._down(glfw.KEY_LEFT, glfw.KEY_A)
        boost = 1.6 if self._down(glfw.KEY_LEFT_SHIFT, glfw.KEY_RIGHT_SHIFT) else 1.0
        return Command(ahead * DRIVE_SPEED * boost, turn * DRIVE_TURN)

    # --- hud ---------------------------------------------------------------------------------

    def _draw_hud(self, fps: float) -> np.ndarray:
        scene = self.scene
        width, height = HUD_SIZE
        canvas = np.zeros((height, width, 3), np.uint8)
        x, z = scene.walker.position
        lines = [f"{fps:3.0f} fps   {scene.layout.room_at(x, z)}   [{scene.mode}]"]
        if scene.robot:
            fw, plant = scene.robot.firmware, scene.robot.plant
            lines.append(f"robot: {'FALLEN - press U' if plant.fallen else fw.state}   room: {scene.layout.room_at(plant.x, plant.z)}")
        for i, text in enumerate(lines):
            cv2.putText(canvas, text, (8, 18 + 17 * i), FONT, 0.46, (235, 235, 235), 1, cv2.LINE_AA)
        if scene.robot:
            top = 56
            half = (width - 12) // 2
            boxes = [(4, top, 4 + half, top + 94), (8 + half, top, width - 4, top + 94), (4, top + 98, 4 + half, top + 192), (8 + half, top + 98, width - 4, top + 192)]
            titles = (("pitch", "deg", 20.0), ("speed", "m/s", 1.0), ("motor current", "A", 6.0), ("battery - 12 V", "V", 1.0))
            for box, chart, (title, unit, limit) in zip(boxes, self.charts, titles):
                draw_graph(canvas, box, title, chart, limit, unit, 1.0)
            cv2.putText(canvas, "arrows drive  P/O shove  U stand up  T bring  V camera  X ceiling", (6, height - 8), FONT, 0.34, (170, 170, 170), 1, cv2.LINE_AA)
        else:
            cv2.putText(canvas, "WASD walk  Space jump  Shift sprint  X ceiling", (6, height - 8), FONT, 0.36, (170, 170, 170), 1, cv2.LINE_AA)
        rgba = np.zeros((height, width, 4), np.uint8)
        rgba[..., :3] = canvas[..., ::-1]
        rgba[..., 3] = np.where(canvas.any(axis=2), 255, 150)
        return rgba

    def _record_charts(self) -> None:
        robot = self.scene.robot
        if robot and robot.history:
            sample = robot.history[-1]
            for chart, value in zip(self.charts, (np.degrees(sample.pitch), sample.speed, sample.current, sample.battery_voltage - 12.0)):
                chart.push([value])

    # --- main loop ---------------------------------------------------------------------------

    def run(self) -> None:
        print(__doc__)
        scene = self.scene
        last, title_at, frames, fps = time.perf_counter(), 0.0, 0, 0.0
        while not glfw.window_should_close(self.window):
            now = time.perf_counter()
            dt, last = min(now - last, 0.05), now
            glfw.poll_events()
            scene.walker.update(dt, self._walk_input() if scene.mode == "walk" or not scene.robot else np.zeros(2), bool(self._down(glfw.KEY_LEFT_SHIFT)), self.jump_requested and scene.mode == "walk")
            self.jump_requested = False
            if scene.robot:
                scene.robot.command = self._robot_command()
                scene.robot.advance(dt)
                self._record_charts()
            width, height = glfw.get_framebuffer_size(self.window)
            if width > 0 and height > 0:
                scene.draw(self.ctx.screen, (width, height))
                if self.hud:
                    if now - self.hud_at > 0.07 or self.hud_image is None:
                        self.hud_image, self.hud_at = self._draw_hud(fps), now
                    texture = self.overlay.upload("hud", self.hud_image)
                    scale = max(1.0, height / 720.0)
                    self.overlay.draw(texture, (10, height - 10 - HUD_SIZE[1] * scale, 10 + HUD_SIZE[0] * scale, height - 10), (width, height), blend=True)
            glfw.swap_buffers(self.window)
            frames += 1
            if now - title_at > 0.5:
                fps = frames / max(now - title_at, 1e-6) if title_at else 0.0
                title_at, frames = now, 0
        glfw.terminate()


def screenshot(args: argparse.Namespace) -> None:
    window, ctx = create_hidden_context()
    scene = Scene(ctx, args.seed, args.fov, not args.no_robot)
    scene.mode = args.mode if scene.robot else "walk"
    if args.pose:
        x, y, z, yaw, pitch = args.pose
        scene.walker.position = np.array([x, z])
        scene.walker.feet = y - scene.walker.EYE
        scene.walker.yaw, scene.walker.pitch = np.radians(yaw), np.radians(pitch)
    if scene.robot:
        scene.robot.advance(args.settle)
    if args.no_ceiling:
        scene.renderer.hidden_layers.add("ceiling")
    target = OffscreenTarget(ctx, (args.width, args.height))
    scene.draw(target.framebuffer, target.size)
    args.screenshot.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.screenshot), target.read_bgr())
    print(f"Wrote {args.screenshot}")
    glfw.terminate()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0, help="furniture arrangement and the robot's build tolerances")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov", type=float, default=85.0, help="horizontal field of view in degrees")
    parser.add_argument("--no-robot", action="store_true", help="just the office")
    parser.add_argument("--no-vsync", action="store_true")
    parser.add_argument("--screenshot", type=Path, default=None, help="render one frame to this image and exit (no window)")
    parser.add_argument("--mode", choices=MODES, default="walk", help="camera for --screenshot")
    parser.add_argument("--pose", type=float, nargs=5, metavar=("X", "EYE_Y", "Z", "YAW_DEG", "PITCH_DEG"), help="walking camera for --screenshot")
    parser.add_argument("--settle", type=float, default=3.0, help="seconds the robot balances before a --screenshot")
    parser.add_argument("--no-ceiling", action="store_true", help="hide the ceiling in a --screenshot")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.screenshot:
        screenshot(args)
    else:
        Playground(args).run()


if __name__ == "__main__":
    main()
