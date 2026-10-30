"""Outdoor playground: drive a half-scale RC buggy round a grass proving ground and over a road of test bumps.

    python outdoor-sim/drive.py
    python outdoor-sim/drive.py --seed 3 --trees 3000
    python outdoor-sim/drive.py --screenshot a.png --start 1 --speed 9 --seconds 2.5 --mode side
    python outdoor-sim/drive.py --screenshot b.png --track --speed 8 --throttle 0.5 --seconds 3

Driving:
    Up / Down      throttle / brake, and reverse once stopped      Left / Right   steer
    Space          handbrake (locks the rear wheels: flick the tail out)
    T              drivetrain: all-wheel / front-wheel / rear-wheel drive
    G              counter-steer assist on / off (off = you catch the slides yourself)

Tuning while you drive (each press changes the value by 12 %, shown in the panel):
    1 / 2          motor power less / more        3 / 4   tyre grip less / more (low grip = ice, high = slicks)
    5 / 6          damping less / more            7 / 8   spring stiffness softer / stiffer
    0              back to the standard tune

Elsewhere:
    R              put the car back on the skid pad          N / B   next / previous test section (at a running start)
    F              to the start line of the RC race track (lap times are shown in the panel)
    C              chase camera distance: close / normal / far / high
    V              camera: chase / bumper / side view of the suspension / trackside
    Mouse drag     orbit the chase camera, wheel zooms       H   panel on / off          Esc   quit

The road (x = 40 .. 380 m, east of the skid pad) has, in order: sine waves, washboard, speed humps,
a frequency sweep, potholes and a kicker ramp. The race track lies on the grass north of the pad.
Skid marks show where the tyres slid.
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
from hud import Overlay  # noqa: E402
from render import OffscreenTarget, create_hidden_context  # noqa: E402
from trees import load_models  # noqa: E402

from car.model import CarModel  # noqa: E402
from car.params import CarParams  # noqa: E402
from car.physics import CarSim  # noqa: E402
from ground import PAD, ROAD, SECTIONS, Ground  # noqa: E402
from panel import CHARTS, PANEL, Charts, draw_panel  # noqa: E402
from track import LapTimer  # noqa: E402
from view import OutdoorRenderer  # noqa: E402

CAMERAS = ("chase", "bumper", "side", "trackside")
DISTANCES = (3.4, 5.2, 8.0, 12.0)  # m, the chase camera's distances that C steps through
CHASE_PITCH = -14.0  # deg the chase camera looks down
RUN_UP = 25.0  # m before a test section the car is put at, doing RUN_SPEED
RUN_SPEED = 9.0
THROTTLE_RATE = 4.0  # per second: the throttle takes a quarter of a second to open fully
SLIDING = 0.35  # tyre slip above which a mark is left
WHEEL_KEYS = {glfw.KEY_1: ("power", -1), glfw.KEY_2: ("power", 1), glfw.KEY_3: ("grip", -1), glfw.KEY_4: ("grip", 1), glfw.KEY_5: ("damping", -1), glfw.KEY_6: ("damping", 1), glfw.KEY_7: ("stiffness", -1), glfw.KEY_8: ("stiffness", 1)}


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


def aim(position: np.ndarray, target: np.ndarray, fov: float) -> Camera:
    """A camera at `position` looking at `target`."""
    d = target - position
    d = d / np.linalg.norm(d)
    return Camera(position.copy(), float(np.arctan2(d[0], -d[2])), float(np.arcsin(d[1])), 0.0, fov)


class Scene:
    """The ground, the car, the cameras and the skid marks. Independent of any window."""

    def __init__(self, ctx: moderngl.Context, seed: int, fov: float, trees: int) -> None:
        self.ctx, self.fov = ctx, fov
        self.ground = Ground(seed)
        models = load_models() if trees > 0 else []
        self.renderer = OutdoorRenderer(ctx, self.ground, models, seed, trees)
        self.sim = CarSim(self.ground, CarParams(), *self.ground.spawn())
        self.model = CarModel(self.renderer, self.sim.params)
        self.camera_mode = "chase"
        self.orbit = np.array([0.0, np.radians(CHASE_PITCH)])  # chase camera: yaw relative to the car, pitch
        self.distance = DISTANCES[1]
        self.chase_heading = self.sim.heading
        self.side_heading = self.sim.heading
        self.trackside: np.ndarray | None = None
        self.charts = Charts()
        self.laps = LapTimer(self.ground.track)

    # --- the car -----------------------------------------------------------------------------

    def reset(self) -> None:
        self.sim.place(*self.ground.spawn())
        self.clear()

    def jump_to_section(self, step: int, speed: float = RUN_SPEED) -> None:
        """Start a few metres before the next (step=1) or previous (step=-1) test section, already moving."""
        x = self.sim.pos[0]
        starts = [s.x0 for s in SECTIONS]
        current = int(np.argmin([abs(x - (s - RUN_UP)) for s in starts]))
        self.start_at((current + step) % len(starts), speed)

    def to_track(self, speed: float = 0.0) -> None:
        """Put the car on the race track a few metres before the start line."""
        track = self.ground.track
        self.sim.place(*track.pose(track.length - 6.0))
        self.sim.vel = self.sim.forward * speed
        self.sim.spin[:] = speed / self.sim.params.wheel_radius
        self.laps.reset()
        self.clear()

    def next_distance(self) -> None:
        i = int(np.argmin([abs(d - self.distance) for d in DISTANCES]))
        self.distance = DISTANCES[(i + 1) % len(DISTANCES)]

    def start_at(self, section: int, speed: float) -> None:
        self.sim.place(SECTIONS[section].x0 - RUN_UP, 0.0, 0.0)
        self.sim.vel = self.sim.forward * speed
        self.sim.spin[:] = speed / self.sim.params.wheel_radius
        self.clear()

    def clear(self) -> None:
        self.renderer.skids.clear()
        self.charts.clear()
        self.chase_heading = self.side_heading = self.sim.heading
        self.trackside = None

    def where(self) -> str:
        x, z = float(self.sim.pos[0]), float(self.sim.pos[2])
        track = self.ground.track
        if track.inside(np.array(x), np.array(z)):
            laps = self.laps
            text = "RC track"
            for label, value in (("lap", laps.current(self.sim.time)), ("last", laps.last), ("best", laps.best)):
                if value is not None:
                    text += f"   {label} {value:5.2f} s"
            return text if laps.started is not None or laps.best is not None else text + "   (cross the start line)"
        for s in SECTIONS:
            if s.x0 - 4.0 <= x <= s.x1 + 4.0 and abs(z) <= ROAD[3] + 1.0:
                return f"{s.name}: {s.detail}"
        if PAD[0] <= x <= PAD[1] and PAD[2] <= z <= PAD[3]:
            return "Skid pad"
        if ROAD[0] <= x <= ROAD[1] and abs(z) <= ROAD[3]:
            return "Test road"
        return "Grass" if not self.ground.paved(np.array([x]), np.array([z]))[0] else "Paving"

    def update(self, dt: float) -> None:
        sim = self.sim
        sim.advance(dt)
        self.charts.record(sim)
        self.laps.update(sim.time, float(sim.pos[0]), float(sim.pos[2]))
        t = sim.telemetry
        for i in range(4):  # skid marks where a tyre slides on the paving (and dark scuffs on the grass too)
            sliding = t.touching[i] and t.slip[i] > SLIDING and t.load[i] > 40.0
            self.renderer.skids.add(i, t.tread[i], t.axle[i], t.normal[i], sim.params.wheel_width * 0.85, 1.0 if sliding else 0.0)
        heading = sim.heading
        velocity = np.array([sim.vel[0], sim.vel[2]])
        if np.linalg.norm(velocity) > 4.0 and sim.speed > 0.0:  # the chase camera follows where the car goes, not where it points
            heading = float(np.arctan2(velocity[1], velocity[0]))
        k = 1.0 - np.exp(-2.5 * dt)
        self.chase_heading += k * ((heading - self.chase_heading + np.pi) % (2.0 * np.pi) - np.pi)
        k = 1.0 - np.exp(-4.0 * dt)
        self.side_heading += k * ((sim.heading - self.side_heading + np.pi) % (2.0 * np.pi) - np.pi)

    # --- cameras -----------------------------------------------------------------------------

    def camera(self) -> Camera:
        sim = self.sim
        R = sim.R
        fov = self.fov
        if self.camera_mode == "bumper":
            return Camera(sim.pos + R @ np.array([0.62, 0.02, 0.0]), basis=(R[:, 0], R[:, 2], R[:, 1]), fov_deg=fov)
        if self.camera_mode == "side":
            h = self.side_heading
            side = np.array([-np.sin(h), 0.0, np.cos(h)])
            ahead = np.array([np.cos(h), 0.0, np.sin(h)])
            focus = sim.pos + ahead * 0.65 + side * 0.25 + np.array([0.0, -0.1, 0.0])  # the front axle, seen from the front quarter
            return aim(focus + side * 1.5 + ahead * 1.4 + np.array([0.0, 0.5, 0.0]), focus, fov)
        if self.camera_mode == "trackside":  # a camera on a tripod ahead of the car, moved on once the car is past and far from it
            if self.trackside is None or np.linalg.norm(self.trackside[[0, 2]] - sim.pos[[0, 2]]) > 28.0:
                h = self.chase_heading
                ahead = np.array([np.cos(h), 0.0, np.sin(h)])
                side = np.array([-np.sin(h), 0.0, np.cos(h)])
                position = sim.pos + ahead * 16.0 + side * 9.0
                position[1] = float(self.ground.sample(position[None, 0], position[None, 2])[0]) + 1.6
                self.trackside = position
            return aim(self.trackside, sim.pos, fov * 0.7)
        yaw = self.chase_heading + self.orbit[0]
        pitch = self.orbit[1]
        fwd = np.array([np.cos(yaw) * np.cos(pitch), np.sin(pitch), np.sin(yaw) * np.cos(pitch)])
        focus = sim.pos + np.array([0.0, 0.35, 0.0])
        position = focus - fwd * self.distance
        ground = float(self.ground.sample(position[None, 0], position[None, 2])[0])
        position[1] = max(position[1], ground + 0.35)
        return aim(position, focus, fov)

    # --- drawing -----------------------------------------------------------------------------

    def draw(self, target: moderngl.Framebuffer, size: tuple[int, int]) -> None:
        sim = self.sim
        ground = float(self.ground.sample(sim.pos[None, 0], sim.pos[None, 2])[0])
        shadow = (np.array([sim.pos[0], ground + 0.03, sim.pos[2]]), (sim.R[0, 0], sim.R[2, 0]), (1.4, 0.8))
        shadow = (shadow[0], shadow[1] / np.maximum(np.hypot(*shadow[1]), 1e-9), shadow[2])
        self.renderer.draw(target, size, self.camera(), self.model.draw_list(sim), shadow)


class Drive:
    def __init__(self, args: argparse.Namespace) -> None:
        self.window, self.ctx = open_window(args.width, args.height, "outdoor-sim", not args.no_vsync)
        self.scene = Scene(self.ctx, args.seed, args.fov, args.trees)
        self.overlay = Overlay(self.ctx)
        self.hud = True
        self.hud_at, self.panel_image, self.chart_image = 0.0, None, None
        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor)
        glfw.set_scroll_callback(self.window, self._on_scroll)
        self.last_cursor = None

    # --- input -------------------------------------------------------------------------------

    def _on_cursor(self, window, x, y) -> None:
        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS and self.last_cursor is not None and self.scene.camera_mode == "chase":
            orbit = self.scene.orbit
            orbit[0] = (orbit[0] + (x - self.last_cursor[0]) * 0.006 + np.pi) % (2.0 * np.pi) - np.pi
            orbit[1] = float(np.clip(orbit[1] - (y - self.last_cursor[1]) * 0.004, np.radians(-80), np.radians(5)))
        self.last_cursor = (x, y)

    def _on_scroll(self, window, _dx, dy) -> None:
        self.scene.distance = float(np.clip(self.scene.distance * 0.9**dy, 2.5, 14.0))

    def _on_key(self, window, key, _scancode, action, _mods) -> None:
        if action == glfw.RELEASE:
            return
        scene, sim = self.scene, self.scene.sim
        if key in WHEEL_KEYS:
            name, direction = WHEEL_KEYS[key]
            sim.tuning.nudge(name, direction)
        elif action != glfw.PRESS:
            return
        elif key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_T:
            sim.tuning.cycle_drivetrain()
        elif key == glfw.KEY_G:
            sim.controls.assist = not sim.controls.assist
        elif key == glfw.KEY_0:
            sim.tuning.reset()
        elif key == glfw.KEY_R:
            scene.reset()
        elif key == glfw.KEY_N:
            scene.jump_to_section(1)
        elif key == glfw.KEY_B:
            scene.jump_to_section(-1)
        elif key == glfw.KEY_F:
            scene.to_track()
        elif key == glfw.KEY_C:
            scene.next_distance()
        elif key == glfw.KEY_V:
            scene.camera_mode = CAMERAS[(CAMERAS.index(scene.camera_mode) + 1) % len(CAMERAS)]
            scene.trackside = None
        elif key == glfw.KEY_H:
            self.hud = not self.hud

    def _down(self, *keys: int) -> float:
        return 1.0 if any(glfw.get_key(self.window, k) == glfw.PRESS for k in keys) else 0.0

    def _controls(self, dt: float) -> None:
        sim = self.scene.sim
        c = sim.controls
        up, down = self._down(glfw.KEY_UP), self._down(glfw.KEY_DOWN)
        throttle, c.brake = up, 0.0
        if down and not up:
            if sim.speed > 0.8:
                c.brake = 1.0
            else:
                throttle = -1.0
        rate = THROTTLE_RATE * dt if abs(throttle) > abs(c.throttle) and throttle * c.throttle >= 0.0 else 4.0 * THROTTLE_RATE * dt  # squeezed on, let go of at once
        c.throttle = float(np.clip(c.throttle + np.clip(throttle - c.throttle, -rate, rate), -1.0, 1.0))
        c.steer = self._down(glfw.KEY_RIGHT) - self._down(glfw.KEY_LEFT)
        c.handbrake = bool(self._down(glfw.KEY_SPACE))

    # --- main loop ---------------------------------------------------------------------------

    def run(self) -> None:
        print(__doc__)
        scene = self.scene
        last, title_at, frames, fps = time.perf_counter(), 0.0, 0, 0.0
        while not glfw.window_should_close(self.window):
            now = time.perf_counter()
            dt, last = min(now - last, 0.05), now
            glfw.poll_events()
            self._controls(dt)
            scene.update(dt)
            if scene.sim.upside_down and scene.sim.speed < 0.5 and abs(scene.sim.omega).max() < 0.5:
                scene.reset()
            width, height = glfw.get_framebuffer_size(self.window)
            if width > 0 and height > 0:
                scene.draw(self.ctx.screen, (width, height))
                if self.hud:
                    if now - self.hud_at > 0.05 or self.panel_image is None:
                        self.panel_image = draw_panel(scene.sim, scene.sim.tuning, scene.where(), fps, scene.sim.controls.assist)
                        self.chart_image = scene.charts.draw()
                        self.hud_at = now
                    draw_hud(self.overlay, self.panel_image, self.chart_image, (width, height))
            glfw.swap_buffers(self.window)
            frames += 1
            if now - title_at > 0.5:
                fps = frames / max(now - title_at, 1e-6) if title_at else 0.0
                title_at, frames = now, 0
        glfw.terminate()


def draw_hud(overlay: Overlay, panel: np.ndarray, charts: np.ndarray, size: tuple[int, int]) -> None:
    width, height = size
    scale = max(1.0, height / 720.0)
    texture = overlay.upload("panel", panel)
    overlay.draw(texture, (10, height - 10 - PANEL[1] * scale, 10 + PANEL[0] * scale, height - 10), size, blend=True)
    texture = overlay.upload("charts", charts)
    overlay.draw(texture, (width - 10 - CHARTS[0] * scale, 10, width - 10, 10 + CHARTS[1] * scale), size, blend=True)


def screenshot(args: argparse.Namespace) -> None:
    _window, ctx = create_hidden_context()
    scene = Scene(ctx, args.seed, args.fov, args.trees)
    sim = scene.sim
    scene.camera_mode = args.mode
    sim.tuning.drivetrain = args.drivetrain
    for name in ("power", "grip", "damping", "stiffness"):
        setattr(sim.tuning, name, getattr(args, name))
    scene.distance = DISTANCES[args.distance]
    if args.track:
        scene.to_track(args.speed)
    elif args.start is not None:
        scene.start_at(args.start, args.speed)
    elif args.speed:
        sim.vel = sim.forward * args.speed
        sim.spin[:] = args.speed / sim.params.wheel_radius
    sim.controls.throttle, sim.controls.steer, sim.controls.handbrake = args.throttle, args.steer, args.handbrake
    sim.controls.assist = not args.no_assist
    dt, t = 1.0 / 60.0, 0.0
    while t < args.seconds:
        if args.handbrake_for and t > args.handbrake_for:
            sim.controls.handbrake = False
        sim.controls.steer = args.steer if t >= args.steer_at else 0.0
        scene.update(dt)
        t += dt
    target = OffscreenTarget(ctx, (args.width, args.height))
    scene.draw(target.framebuffer, target.size)
    if args.hud:
        draw_hud(Overlay(ctx), draw_panel(sim, sim.tuning, scene.where(), 60.0, sim.controls.assist), scene.charts.draw(), target.size)
    args.screenshot.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.screenshot), target.read_bgr())
    print(f"Wrote {args.screenshot}  (x={sim.pos[0]:.1f} z={sim.pos[2]:.1f} speed={sim.speed:.1f} m/s slip angle={sim.slip_angle():.0f} deg)")
    glfw.terminate()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0, help="terrain and trees")
    parser.add_argument("--trees", type=int, default=6000, help="number of trees (0 for none)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov", type=float, default=80.0, help="horizontal field of view in degrees")
    parser.add_argument("--no-vsync", action="store_true")
    shot = parser.add_argument_group("screenshots (no window): drive the car for a while, then take one picture")
    shot.add_argument("--screenshot", type=Path, default=None, help="render one frame to this image and exit")
    shot.add_argument("--mode", choices=CAMERAS, default="chase")
    shot.add_argument("--start", type=int, default=None, help="start before test section N (0 = sine waves)")
    shot.add_argument("--track", action="store_true", help="start on the race track, just before the start line")
    shot.add_argument("--distance", type=int, choices=range(len(DISTANCES)), default=1, help="chase camera distance (as C steps through)")
    shot.add_argument("--speed", type=float, default=0.0, help="starting speed, m/s")
    shot.add_argument("--seconds", type=float, default=0.5, help="how long to drive before the picture")
    shot.add_argument("--throttle", type=float, default=0.0)
    shot.add_argument("--steer", type=float, default=0.0, help="-1 .. 1")
    shot.add_argument("--steer-at", type=float, default=0.0, help="start steering after this many seconds")
    shot.add_argument("--handbrake", action="store_true")
    shot.add_argument("--handbrake-for", type=float, default=0.0, help="release the handbrake after this many seconds")
    shot.add_argument("--no-assist", action="store_true")
    shot.add_argument("--drivetrain", choices=("awd", "fwd", "rwd"), default="awd")
    shot.add_argument("--power", type=float, default=1.0)
    shot.add_argument("--grip", type=float, default=1.0)
    shot.add_argument("--damping", type=float, default=1.0)
    shot.add_argument("--stiffness", type=float, default=1.0)
    shot.add_argument("--hud", action="store_true", help="draw the panel and charts too")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.screenshot:
        screenshot(args)
    else:
        Drive(args).run()


if __name__ == "__main__":
    main()
