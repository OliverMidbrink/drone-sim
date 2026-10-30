"""FPV drone mode: fly a quadrotor with real thruster physics through the virtual world.

    python virtual-sim/fpv.py                       # random world, both side cameras
    python virtual-sim/fpv.py --seed 7 --low-mountains
    python virtual-sim/fpv.py --cams none           # no side cameras (fastest)
    python virtual-sim/fpv.py --record out/         # also save left/right mp4, imu.csv and poses.csv
    python virtual-sim/fpv.py --screenshot hud.png  # fly a short scripted hop and save the screen, no window

    python virtual-sim/fpv.py --slam                # camera SLAM + IMU hold the drone in place (split screen)
    python virtual-sim/fpv.py --slam --hold-test 60 # same, no window: fly 60 s and report how far it wandered

You sit in the pilot's seat (the main view is the nose camera, tilted up like a real FPV
quad). The drone levels itself and holds its height when you let go of the sticks.

    W / S        fly forward / back (tilts the drone)
    A / D        fly left / right
    Q / E        turn left / right
    Space / Ctrl climb / descend (no key = hold height)
    Shift        steeper tilts
    V            switch between the FPV view and a chase view (3rd person: follows the
                 drone's position and heading from behind, looking down 20 degrees, ignoring roll/pitch)
    1 / 2        left / right side camera on or off
    R            put the drone back at the start (also after a crash)
    M            tall / low mountains        N  new random world
    Esc          quit
  with --slam only:
    P            position hold at the current estimated position (any stick takes the control back)
    H            velocity assist: sticks command speed (the drone brakes by itself) instead of tilt
    mouse drag / scroll in the right half: orbit / zoom the 3D map

The sensors are an IMU (gyro + accelerometer) and, unless --baro off, a barometer. No magnetometer,
no GPS. The two side cameras are 360-camera style: each a circular fisheye (--cam-fov, 160 degrees by default) in a square
picture, 30 fps of simulated time, shown as the two squares in the bottom right. The strip charts
show the fused attitude, gyroscope, accelerometer, motor thrusts and climb rate.

--slam puts the drone on the ground in a clearing with forest around it and runs the position-hold
experiment: it calibrates the IMU sitting there, takes off on the IMU alone, then wiggles a little
(the cameras start now) so the accelerometer and the SLAM can agree on metric scale, and finally holds
a point 2 m above where it started, using nothing but the two cameras and the IMU (no barometer).
The left half of the screen is the drone (chase view by default, V for FPV), the right half the SLAM's
3D map. Wind pushes the drone around once it is off the ground (--wind). Hook into the data with FpvApp.imu_callbacks and FpvApp.frame_callbacks, or record it with --record.
"""

from __future__ import annotations

import argparse
import csv
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Callable

import cv2
import glfw
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402,F401  (puts flight/, slam/, viewer/ on the path)

from autopilot import FlightController, PilotInput  # noqa: E402
from camera import Camera  # noqa: E402
from collision import CollisionWorld  # noqa: E402
from drone import Quadrotor, attitude_angles  # noqa: E402
from fisheye import CameraMount, FisheyeCamera  # noqa: E402
from hud import AXIS_COLORS, FONT, History, Overlay, draw_graph, fit_rect  # noqa: E402
from lens import GL_TO_CV, mount_rotation  # noqa: E402
from route import Route, square  # noqa: E402
from drone_model import DroneModel  # noqa: E402
from render import FAR, NEAR, SUN_DIRECTION, OffscreenTarget, Renderer, create_hidden_context  # noqa: E402
from sensors import SensorNoise, SensorSample, SensorSuite, StateEstimator  # noqa: E402
from simulator import add_world_arguments, build_world, open_window  # noqa: E402
from trees import load_models  # noqa: E402

STATE_EVERY = 250  # physics ticks between rows of the --diag state log (0.5 s)
PHYSICS_DT = 1.0 / 500.0
CONTROL_EVERY = 2  # physics steps per flight-controller / IMU sample: 250 Hz
HISTORY_EVERY = 4  # control ticks per chart sample
CAMERA_FPS = 30.0
SPAWN_CLEARANCE = 0.0
CRASH_BANNER_SECONDS = 4.0

DEBUG = bool(__import__("os").environ.get("DRONE_DEBUG"))
HOVER_HEIGHT = 2.0  # m above the ground: where the SLAM flight takes off to and holds
SETTLE_SECONDS = 3.0  # hold before the route starts
AUTO_MISSIONS = ("takeoff", "wiggle", "hold", "route")
TAKEOFF_CLIMB = 0.4  # stick
TAKEOFF_MAX_SECONDS = 8.0
CHASE_PITCH_DEG = 20.0

WIGGLE_AMPLITUDE = 0.3  # stick, during the start-up wiggle
WIGGLE_HZ = 0.3
WIGGLE_MIN_SECONDS = 4.0
WIGGLE_MAX_SECONDS = 25.0
GUST_SIGMA = 0.7  # m/s
GUST_TIME = 3.0  # s


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_world_arguments(parser)
    parser.add_argument("--cams", choices=["both", "left", "right", "none"], default="both", help="which side cameras start enabled")
    parser.add_argument("--cam-layout", choices=["sides", "forward"], default="sides", help="sides: look out of the left and right; forward: a forward-looking stereo pair")
    parser.add_argument("--cam-size", default=None, help="side camera picture size, square (default 1080x1080, 768x768 with --slam): the 180 degree image circle fills it")
    parser.add_argument("--cam-face", type=int, default=None, help="resolution of the five cube faces behind each fisheye picture (lower = faster; default 1024, 640 with --slam)")
    parser.add_argument("--cam-fov", type=float, default=160.0, help="field of view of the side cameras' fisheye lenses in degrees (image circle across the square picture; 180 = hemisphere)")
    parser.add_argument("--cam-fps", type=float, default=CAMERA_FPS)
    parser.add_argument("--fov", type=float, default=110.0, help="pilot (FPV) view horizontal field of view")
    parser.add_argument("--view", choices=["fpv", "chase"], default=None, help="starting view (default chase with --slam, else fpv); V toggles")
    parser.add_argument("--chase-distance", type=float, default=3.0, help="metres the chase camera follows behind the drone")
    parser.add_argument("--chase-fov", type=float, default=75.0, help="chase view horizontal field of view")
    parser.add_argument("--view-tilt", type=float, default=10.0, help="degrees the pilot camera is tilted up")
    parser.add_argument("--assist", action="store_true", help="start with the velocity assist on (H toggles; needs --slam)")
    parser.add_argument("--baro", choices=["auto", "on", "off"], default="auto", help="barometer: auto = on for manual flight, off with --slam")
    parser.add_argument("--imu-errors", type=float, default=1.0, help="multiplies every IMU error (bias, drift, scale factor, coupling, noise, vibration): 1 = consumer MEMS, 0.3 = good, 0 = perfect apart from rounding")
    parser.add_argument("--cam-time-offset", type=float, default=0.006, help="seconds the camera timestamps lag the IMU clock (exposure and readout latency; --slam)")
    parser.add_argument("--slam", action="store_true", help="run camera SLAM + IMU position hold, with the 3D map on the right half of the screen")
    parser.add_argument("--slam-size", type=int, default=512, help="pinhole picture (px, square) the SLAM works on, cut out of the fisheye picture")
    parser.add_argument("--slam-fov", type=float, default=110.0, help="field of view of that pinhole picture")
    parser.add_argument("--slam-fixes", choices=["on", "off"], default="on", help="let the SLAM snap its pose to old landmarks (loop-closure-like)")
    parser.add_argument("--vio-align", choices=["on", "off"], default="on", help="turn each SLAM position by how far the SLAM's frame has turned from the filter's attitude")
    parser.add_argument("--vio-turn-trust", choices=["on", "off"], default="on", help="believe pictures less while the drone turns fast (the IMU carries the filter instead)")
    parser.add_argument("--slam-latency", type=float, default=0.12, help="simulated seconds the SLAM may lag before the simulation waits for it")
    parser.add_argument("--route", choices=["on", "off"], default="on", help="after the start-up, fly round four waypoints (on) instead of holding one point (off); T starts / P stops it")
    parser.add_argument("--route-size", type=float, default=4.0, help="side (m) of the square the four waypoints make")
    parser.add_argument("--route-speed", type=float, default=2.0, help="fastest the route may fly (m/s)")
    parser.add_argument("--route-accel", type=float, default=2.0, help="largest acceleration the route may ask for (m/s^2); 9.81 * tan(tilt)")
    parser.add_argument("--route-dwell", type=float, default=1.0, help="seconds to hover at each waypoint")
    parser.add_argument("--wind", type=float, default=None, help="mean wind m/s (default 1.0 with --slam, else 0); gusts come on top")
    parser.add_argument("--anchor-window", type=float, default=0.0, metavar="SECONDS", help="make new SLAM anchors only during this many seconds after a camera's first picture (0 = always); old anchors keep being used")
    parser.add_argument("--diag", type=Path, default=None, metavar="PREFIX", help="with --slam: log truth vs estimate and every SLAM pose to PREFIX_state.csv / PREFIX_slam.csv (tools/diag_report.py summarises them)")
    parser.add_argument("--hold-test", type=float, default=None, metavar="SECONDS", help="with --slam: no window, fly this long, print how far the drone wandered")
    parser.add_argument("--record", type=Path, default=None, help="folder for left.mp4, right.mp4, imu.csv and poses.csv")
    parser.add_argument("--run-seconds", type=float, default=None, help="quit after this long and print frame rates (for benchmarking)")
    parser.add_argument("--screenshot", type=Path, default=None, help="fly a short scripted hop (or --hold-test), save the screen to this image and exit (no window)")
    parser.add_argument("--screenshot-seconds", type=float, default=8.0, help="how long the scripted flight for --screenshot lasts")
    args = parser.parse_args(argv)
    if args.cam_size is None:
        args.cam_size = "768x768" if args.slam else "1080x1080"
    if args.cam_face is None:
        args.cam_face = 640 if args.slam else 1024
    if args.wind is None:
        args.wind = 1.0 if args.slam else 0.0
    if args.slam and (args.width, args.height) == (1280, 720):
        args.width, args.height = 1600, 900
    return args


class FpvApp:
    """The drone, its sensors and flight controller, the cameras and the screen.

    Subscribe to the data streams:
        app.imu_callbacks.append(lambda sample: ...)            # SensorSample, 250 Hz
        app.frame_callbacks.append(lambda name, t, bgr: ...)    # "left" / "right", seconds, (size, size, 3) uint8, 30 fps
    Frames are only read back from the GPU while at least one frame callback (or --record or --slam) is set.
    """

    def __init__(self, args: argparse.Namespace, seed: int, window, ctx, headless: bool = False) -> None:
        self.args, self.window, self.ctx, self.headless = args, window, ctx, headless
        self.slam = args.slam
        self.models = load_models(args.godot_dir) if args.trees > 0 else []
        self.renderer = Renderer(ctx, self.models, args.visibility)
        self.overlay = Overlay(ctx)
        self.quad = Quadrotor()
        self.drone_model = DroneModel(ctx, self.quad.params, SUN_DIRECTION)
        self.view = args.view or ("chase" if args.slam else "fpv")
        self.last_draw = time.perf_counter()
        use_baro = args.baro == "on" or (args.baro == "auto" and not self.slam)
        self.sensors = SensorSuite(SensorNoise().scaled(args.imu_errors), seed=seed, barometer=use_baro)

        width, height = (int(v) for v in args.cam_size.lower().split("x"))
        forward = args.cam_layout == "forward"
        mounts = [
            CameraMount("left", np.array([-0.06, 0.0, 0.0]), 0.0 if forward else -90.0),
            CameraMount("right", np.array([0.06, 0.0, 0.0]), 0.0 if forward else 90.0),
        ]
        self.cameras = {m.name: FisheyeCamera(ctx, self.renderer, m, (width, height), args.cam_face, args.cam_fov) for m in mounts}
        self.camera_on = {"left": args.cams in ("both", "left"), "right": args.cams in ("both", "right")}
        self.feed_times: dict[str, list[float]] = {"left": [], "right": []}

        self.link = None
        self.map_view = None
        if self.slam:
            from slam_view import SlamMapView
            from slam_worker import CameraConfig, SlamLink
            from vio import CameraModel, VioConfig, VisualInertialEstimator

            if width != height:
                raise SystemExit("--slam needs square camera pictures (--cam-size NxN)")
            models = [CameraModel(m.name, mount_rotation(m.yaw_degrees) @ GL_TO_CV, m.offset.copy()) for m in mounts]
            vio = VioConfig(align=args.vio_align == "on", turn_rate=0.6 if args.vio_turn_trust == "on" else 1e9)
            estimator = VisualInertialEstimator(models, vio, use_baro=use_baro)
            self.link = SlamLink([CameraConfig(m.name, width, args.slam_size, args.slam_fov, mm.body_from_cv, args.cam_fov, args.slam_fixes == "on", args.anchor_window) for m, mm in zip(mounts, models)])
            self.map_view = SlamMapView(ctx)
        else:
            estimator = StateEstimator()
        self.estimator = estimator
        self.controller = FlightController(self.quad.params, self.quad.mixer_inverse, estimator)
        self.controller.velocity_assist = args.assist and self.slam
        self.pilot = PilotInput()
        self.auto_pilot = PilotInput()
        self.imu_callbacks: list[Callable[[SensorSample], None]] = []
        self.frame_callbacks: list[Callable[[str, float, np.ndarray], None]] = []

        self.histories = {"att": History(3), "gyro": History(3), "accel": History(3), "motors": History(4), "climb": History(2), "pos": History(3)}
        self.yaw_unwrapped = 0.0
        self.last_sample: SensorSample | None = None
        self.banner = ("", -1e9)
        self.recorder: Recorder | None = Recorder(args.record, (width, height), args.cam_fps) if args.record else None
        self.diag = None
        if args.diag and self.slam:
            from diagnostics import FlightDiagnostics

            self.diag = FlightDiagnostics(args.diag)
        self.accumulator = 0.0
        self.tick = 0
        self.next_frame = 0.0
        self.strip_at = self.top_at = self.map_text_at = -1e9
        self.world = None
        self.rng = np.random.default_rng(seed + 1)
        self.sim_rate = 1.0
        self.load_world(seed)
        if window is not None and not headless:
            glfw.set_key_callback(window, self._on_key)
            if self.slam:
                glfw.set_mouse_button_callback(window, self._on_mouse_button)
                glfw.set_cursor_pos_callback(window, lambda _w, x, y: self.map_view.mouse_move(x, y))
                glfw.set_scroll_callback(window, lambda _w, _x, y: self.map_view.scroll(y))

    # --- world and drone -------------------------------------------------------------------

    def load_world(self, seed: int) -> None:
        if self.window is not None and not self.headless:
            glfw.set_window_title(self.window, f"virtual-sim FPV - generating world {seed} ...")
        self.world = build_world(self.args, seed, self.models)
        self.renderer.set_terrain(self.world.terrain)
        self.renderer.set_trees(self.world.trees)
        self.collisions = CollisionWorld(self.world.terrain, self.world.trees, self.models)
        self.spawn = None
        if self.slam:
            from spawn import find_spawn

            self.spawn = find_spawn(self.world.terrain, self.collisions)
            print(f"Spawn (on the ground) {np.round(self.spawn.position, 1)}: nearest tree {self.spawn.nearest_tree:.1f} m, {self.spawn.trees_in_view_range} tree trunks within sight")
        self.reset_drone()

    def reset_drone(self) -> None:
        terrain = self.world.terrain
        if self.slam:
            yaw = 0.0
            x, z = float(self.spawn.position[0]), float(self.spawn.position[2])
            self.start_position = np.array([x, float(terrain.height_at(x, z)) + self.quad.params.radius, z])  # sitting on its legs
            self.quad.reset(self.start_position, yaw)
        else:
            yaw = self.world.spawn_camera().yaw
            self.start_position = np.array([0.0, float(terrain.height_at(0.0, 0.0)) + self.quad.params.radius + SPAWN_CLEARANCE, 0.0])
            self.quad.reset(self.start_position, yaw)
        self.sensors.reset(self.quad.velocity)
        if self.slam:
            print(self.sensors.describe())
        self.controller.reset(airborne=False)
        if self.diag is not None:
            self.diag.reset()
        # the estimator's world W: origin at the start, nose heading at the start is "forward"; true = start + W_axes @ p_W
        forward = np.array([np.sin(yaw), 0.0, -np.cos(yaw)])
        right = np.array([np.cos(yaw), 0.0, np.sin(yaw)])
        self.w_axes = np.stack([right, np.array([0.0, 1.0, 0.0]), -forward], axis=1)
        for history in self.histories.values():
            history.clear()
        self.yaw_unwrapped = yaw
        self.next_frame = 0.0
        self.tick = 0
        self.impacts_seen = 0
        self.banner = ("", -1e9)
        self.mission = "boot" if self.slam else "manual"
        self.mission_started = 0.0
        self.auto_pilot = PilotInput()
        self.gyro_rows: dict[str, list] = {"left": [], "right": []}
        self.trail_estimate: list[np.ndarray] = []
        self.trail_truth: list[np.ndarray] = []
        self.hold_log: list[tuple[float, float, float]] = []  # t, true error, estimated error (m) while holding
        self.route: Route | None = None
        self.debug_rows: list[tuple] = []
        self.route_log: list[tuple[float, float, float, float]] = []  # t, true distance to the planned point, estimator error, speed (m/s)
        self.map_points_at = -1e9
        self.gust = np.zeros(3)
        angle = self.rng.uniform(0.0, 2.0 * np.pi)
        self.wind_mean = self.args.wind * np.array([np.cos(angle), 0.0, np.sin(angle)])
        if self.link is not None:
            self.link.restart()
        if self.map_view is not None:
            self.map_view.set_points(np.empty((0, 3)), np.empty((0, 3)))

    def target_point(self) -> np.ndarray:
        """The point the drone is being flown to: the hold point, or where the route says it should be now."""
        if self.controller.reference is not None:
            return self.controller.reference[0]
        return self.controller.hold_target if self.controller.hold_target is not None else np.zeros(3)

    def truth_in_w(self) -> np.ndarray:
        """Where the drone really is, in the estimator's world axes (for judging the estimate; nothing reads this back)."""
        return self.w_axes.T @ (self.quad.position - self.start_position)

    # --- input -----------------------------------------------------------------------------

    def _on_key(self, window, key, _scancode, action, _mods) -> None:
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_R:
            self.reset_drone()
        elif key == glfw.KEY_H and self.slam:
            self.controller.velocity_assist = not self.controller.velocity_assist
            for pid in self.controller.velocity_pid:
                pid.reset()
        elif key == glfw.KEY_P and self.slam:
            self.start_hold(self.estimator.position.copy())
        elif key == glfw.KEY_T and self.slam and self.estimator.initialised:
            self.start_route(self.last_sample.t)
        elif key == glfw.KEY_V:
            self.view = "fpv" if self.view == "chase" else "chase"
        elif key == glfw.KEY_1:
            self.camera_on["left"] = not self.camera_on["left"]
        elif key == glfw.KEY_2:
            self.camera_on["right"] = not self.camera_on["right"]
        elif key == glfw.KEY_M:
            self.args.low_mountains = not self.args.low_mountains
            self.load_world(self.world.seed)
        elif key == glfw.KEY_N:
            self.load_world(int(np.random.default_rng().integers(0, 1_000_000)))

    def _on_mouse_button(self, window, button, action, _mods) -> None:
        x, y = glfw.get_cursor_pos(window)
        self.map_view.mouse_button(action == glfw.PRESS, x, y, glfw.get_window_size(window)[1])

    def _down(self, key: int) -> float:
        return 1.0 if glfw.get_key(self.window, key) == glfw.PRESS else 0.0

    def read_pilot(self, dt: float) -> None:
        """Keys -> stick positions, eased a little like a real stick."""
        d = self._down
        target = PilotInput(
            forward=d(glfw.KEY_W) - d(glfw.KEY_S),
            right=d(glfw.KEY_D) - d(glfw.KEY_A),
            yaw=d(glfw.KEY_E) - d(glfw.KEY_Q),
            climb=d(glfw.KEY_SPACE) - max(d(glfw.KEY_LEFT_CONTROL), d(glfw.KEY_RIGHT_CONTROL)),
            sport=bool(d(glfw.KEY_LEFT_SHIFT) or d(glfw.KEY_RIGHT_SHIFT)),
        )
        blend = 1.0 - np.exp(-dt / 0.08)
        for name in ("forward", "right", "yaw", "climb"):
            setattr(self.pilot, name, getattr(self.pilot, name) + blend * (getattr(target, name) - getattr(self.pilot, name)))
        self.pilot.sport = target.sport
        if self.mission in AUTO_MISSIONS and max(abs(target.forward), abs(target.right), abs(target.yaw), abs(target.climb)) > 0.5:
            self.mission = "manual"  # a stick moved: the pilot takes over
            self.controller.hold_target = None
            self.controller.reference = None

    def start_route(self, now: float) -> None:
        a = self.args
        center = np.array([0.0, HOVER_HEIGHT, 0.0])
        waypoints = square(center, a.route_size, height_step=0.8)
        self.route = Route(waypoints, self.estimator.position.copy(), now, a.route_speed, a.route_accel, a.route_dwell)
        self.mission = "route"
        self.controller.hold_target = None
        self.controller.velocity_assist = False
        for pid in self.controller.velocity_pid:
            pid.reset()
        self.auto_pilot = PilotInput()

    def start_hold(self, target: np.ndarray) -> None:
        self.mission = "hold"
        self.route = None
        self.controller.reference = None
        self.controller.hold_target = np.array(target, dtype=float)
        self.controller.velocity_assist = False
        for pid in self.controller.velocity_pid:
            pid.reset()
        self.controller.climb_pid.reset()
        self.auto_pilot = PilotInput()

    # --- simulation ------------------------------------------------------------------------

    def advance(self, dt: float) -> None:
        """Run the physics (500 Hz) and flight controller / IMU (250 Hz) for dt seconds of simulated time."""
        self.accumulator += dt
        steps = 0
        while self.accumulator >= PHYSICS_DT and steps < 100:
            self._step()
            self.accumulator -= PHYSICS_DT
            steps += 1
        if steps == 100:
            self.accumulator = 0.0
        self._check_impacts()
        if self.quad.time >= self.next_frame:
            self.shoot_feeds()
            self.next_frame += 1.0 / self.args.cam_fps
            if self.quad.time - self.next_frame > 0.25:  # fell behind: skip ahead instead of catching up
                self.next_frame = self.quad.time
        if self.link is not None:
            self._sync_slam()

    def _step(self) -> None:
        quad = self.quad
        if self.tick % CONTROL_EVERY == 0:
            self._weather(CONTROL_EVERY * PHYSICS_DT)
            sample = self.sensors.sample(quad)
            self.last_sample = sample
            if self.slam:
                self._mission(sample)
            pilot = self.auto_pilot if self.mission in AUTO_MISSIONS else self.pilot
            if self.mission == "route" and self.route is not None:
                self.controller.reference = self.route.sample(sample.t)
            self.controller.update(sample, pilot, quad.on_ground)
            if self.diag is not None and self.tick % STATE_EVERY == 0:
                self.diag.state_row(self)
            self._publish(sample)
            if (self.tick // CONTROL_EVERY) % HISTORY_EVERY == 0:
                self._record_history(sample)
        quad.step(PHYSICS_DT, self.controller.thrust_command, self.collisions, self.world.terrain.half - 2.0)
        self.tick += 1

    def _weather(self, dt: float) -> None:
        if self.args.wind <= 0.0:
            return
        shelter = float(np.clip((self.quad.position[1] - self.start_position[1] - 0.2) / 1.0, 0.0, 1.0))  # no wind on the ground, full wind above 1.2 m
        decay = np.exp(-dt / GUST_TIME)
        self.gust = self.gust * decay + self.rng.normal(0.0, GUST_SIGMA * np.sqrt(1.0 - decay * decay), 3) * np.array([1.0, 0.3, 1.0])
        self.quad.wind = shelter * (self.wind_mean + self.gust)

    def _mission(self, sample: SensorSample) -> None:
        """Start-up: calibrate, wiggle so the SLAM can learn the scale, then hold the start position."""
        est, t = self.estimator, sample.t
        if self.mission == "boot":
            if est.initialised:
                self.mission, self.mission_started = "takeoff", t
        elif self.mission == "takeoff":
            # climb on the IMU alone (no baro, the cameras have not started): the filter's height is a double integral of the accelerometer
            self.auto_pilot = PilotInput(climb=TAKEOFF_CLIMB)
            if est.position[1] > HOVER_HEIGHT - 0.6 or t - self.mission_started > TAKEOFF_MAX_SECONDS:
                self.mission, self.mission_started = "wiggle", t
                self.auto_pilot = PilotInput()
        elif self.mission == "wiggle":
            s = t - self.mission_started
            w = 2.0 * np.pi * WIGGLE_HZ
            ramp = min(s / 1.5, 1.0)
            self.auto_pilot = PilotInput(
                forward=WIGGLE_AMPLITUDE * ramp * np.sin(w * s),
                right=WIGGLE_AMPLITUDE * ramp * np.sin(w * s + 1.2),
                climb=0.2 * ramp * np.sin(0.7 * w * s),
            )
            active = [n for n in self.cameras if self.camera_on[n]]
            calibrated = active and all(est.scale_ready(n) for n in active)
            if (s > WIGGLE_MIN_SECONDS and calibrated) or s > WIGGLE_MAX_SECONDS:
                self.start_hold(np.array([0.0, HOVER_HEIGHT, 0.0]))
                self.hold_started = t
        elif self.mission == "hold" and self.args.route == "on" and t - self.hold_started > SETTLE_SECONDS:
            self.start_route(t)

    def _publish(self, sample: SensorSample) -> None:
        for callback in self.imu_callbacks:
            callback(sample)
        if self.recorder:
            self.recorder.imu(sample, self.quad)
        if self.link is not None and self.estimator.initialised:
            row = (sample.t, *sample.gyro)
            for rows in self.gyro_rows.values():
                rows.append(row)

    def _record_history(self, sample: SensorSample) -> None:
        est = self.controller.estimator
        roll, pitch, yaw = attitude_angles(est.rotation)
        self.yaw_unwrapped += ((yaw - self.yaw_unwrapped + np.pi) % (2 * np.pi)) - np.pi
        h = self.histories
        h["att"].push([np.degrees(roll), np.degrees(pitch), np.degrees(self.yaw_unwrapped)])
        h["gyro"].push(sample.gyro - est.gyro_bias)
        h["accel"].push(sample.accel)
        h["motors"].push(self.quad.thrust / self.quad.params.max_thrust * 100.0)
        h["climb"].push([self.quad.velocity[1], est.climb_rate])
        if self.slam:
            error = self.truth_in_w()
            h["pos"].push(error)
            if self.tick % 50 == 0 or not self.trail_truth:
                self.trail_truth.append(error)
                self.trail_estimate.append(est.position.copy())
                del self.trail_truth[:-3000], self.trail_estimate[:-3000]
            if self.mission == "route" and self.controller.reference is not None:
                self.route_log.append((sample.t, float(np.linalg.norm(error - self.controller.reference[0])), float(np.linalg.norm(est.position - error)), float(np.linalg.norm(self.quad.velocity))))
            if self.mission == "hold" and self.controller.hold_target is not None:
                self.hold_log.append((sample.t, float(np.linalg.norm(error - self.controller.hold_target)), float(np.linalg.norm(est.position - error))))

    def _check_impacts(self) -> None:
        impacts = self.quad.impacts
        if len(impacts) > self.impacts_seen:
            hit = impacts[-1]
            self.impacts_seen = len(impacts)
            text = f"{'CRASH' if self.quad.crashed else 'HIT'}: {hit.kind} at {hit.speed:.1f} m/s" + (" - press R to reset" if self.quad.crashed else "")
            self.banner = (text, time.perf_counter())

    # --- cameras ---------------------------------------------------------------------------

    def shoot_feeds(self) -> None:
        quad = self.quad
        rotation = quad.rotation
        wanted = bool(self.frame_callbacks) or self.recorder is not None
        slam_ready = self.link is not None and self.estimator.initialised and self.mission in ("wiggle", "hold", "route", "manual")  # the cameras start once airborne (after the IMU-only climb, which the accelerometer tracks to a few cm)
        for name, camera in self.cameras.items():
            if not self.camera_on[name]:
                continue
            camera.render(quad.position, rotation)
            times = self.feed_times[name]
            times.append(time.perf_counter())
            del times[:-31]
            if wanted or slam_ready:
                frame = camera.read_bgr()
                for callback in self.frame_callbacks:
                    callback(name, quad.time, frame)
                if self.recorder:
                    self.recorder.frame(name, frame)
                if slam_ready:
                    self._submit_slam(name, frame)
        if self.recorder:
            self.recorder.pose(quad)

    def _submit_slam(self, name: str, frame: np.ndarray) -> None:
        """Hand the SLAM what a real drone's computer has: the picture and the gyro since the last picture."""
        est = self.estimator
        stamp = self.quad.time - self.args.cam_time_offset  # the camera's clock runs a little behind the IMU's: the picture is stamped early
        buffer = self.gyro_rows[name]
        if name not in self.link.started:
            est.begin_camera(name)
            buffer.clear()
        cut = sum(1 for row in buffer if row[0] <= stamp)
        rows = np.array(buffer[:cut]).reshape(-1, 4)
        del buffer[:cut]
        self.link.submit(name, stamp, frame, rows, est.gyro_bias.copy())

    def _sync_slam(self) -> None:
        """Feed finished SLAM poses to the estimator; wait if the SLAM lags more than --slam-latency behind the simulation."""
        for result in self.link.poll():
            self._apply_slam(result)
        self.link.wait_until(self.quad.time - self.args.slam_latency)
        for result in self.link.poll():
            self._apply_slam(result)

    def _apply_slam(self, result) -> None:
        if result.changed and result.stats["steps"] > 0:
            if DEBUG:
                self._debug_pose(result)
            before = self.diag.before(self.estimator, result.name) if self.diag is not None else None
            self.estimator.slam_pose(result.name, result.t, result.rotation, result.position)
            if self.diag is not None:
                self.diag.pose(self, result, before)

    def _debug_pose(self, result) -> None:
        """Developer check (DRONE_DEBUG=1): how wrong the SLAM's and the filter's attitude and position are against the truth. Nothing reads this back."""
        est = self.estimator
        cam = est.cameras[result.name]
        r_ws = est.start_rotation[result.name] @ cam.body_from_cv
        slam_body = r_ws @ result.rotation @ cam.body_from_cv.T
        truth = self.w_axes.T @ self.quad.rotation
        angle = lambda a, b: float(np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1.0) / 2.0, -1.0, 1.0))))
        lam = max(est.scale(result.name), 0.02)
        camera_truth = self.truth_in_w() - est.start_position[result.name] + (truth - est.start_rotation[result.name]) @ cam.lever_arm
        slam_position_error = float(np.linalg.norm(r_ws @ result.position / lam - camera_truth))
        self.debug_rows.append((self.quad.time, angle(slam_body, truth), angle(est.rotation, truth), slam_position_error, float(np.linalg.norm(self.quad.omega))))

    def feed_rate(self, name: str) -> float:
        times = self.feed_times[name]
        if not self.camera_on[name] or len(times) < 2 or time.perf_counter() - times[-1] > 0.5:
            return 0.0
        return (len(times) - 1) / (times[-1] - times[0])

    # --- the map pane -----------------------------------------------------------------------

    def _refresh_map(self) -> None:
        """The SLAM's landmarks, moved into the estimator's world with the estimated scale of each camera."""
        est = self.estimator
        positions, colors = [], []
        for name, (points, rgb) in self.link.maps.items():
            if name not in est.start_rotation or len(points) == 0:
                continue
            cam = est.cameras[name]
            r_ws = est.start_rotation[name] @ cam.body_from_cv
            lam = max(est.scale(name), 0.02)
            positions.append((points.astype(np.float64) @ r_ws.T) / lam + est.start_position[name] + est.start_rotation[name] @ cam.lever_arm)
            colors.append(rgb.astype(np.float32) / 255.0)
        if positions:
            self.map_view.set_points(np.vstack(positions), np.vstack(colors))

    def _map_lines(self) -> np.ndarray:
        from slam_view import cross_marker, frame_axes, grid_lines, segments

        est = self.estimator
        target = self.target_point()
        parts = [
            grid_lines(np.array([0.0, 0.0, 0.0]), 12.0, 1.0),
            segments(np.array(self.trail_truth), (1.0, 0.25, 0.9)),
            segments(np.array(self.trail_estimate), (0.3, 1.0, 0.4)),
            cross_marker(target),
            frame_axes(est.position, est.rotation),
        ]
        return np.vstack(parts)

    # --- drawing ---------------------------------------------------------------------------

    def pilot_camera(self) -> Camera:
        rotation = self.quad.rotation
        right, up, forward = rotation[:, 0], rotation[:, 1], -rotation[:, 2]
        tilt = np.radians(self.args.view_tilt)
        tilted_forward = np.cos(tilt) * forward + np.sin(tilt) * up
        tilted_up = np.cos(tilt) * up - np.sin(tilt) * forward
        return Camera(self.quad.position + rotation @ np.array([0.0, 0.03, -0.08]), fov_deg=self.args.fov, basis=(tilted_forward, right, tilted_up))

    def chase_camera(self) -> Camera:
        """Third person: follows the drone's position and heading only, never its roll or pitch; looks down 20 degrees."""
        nose = -self.quad.rotation[:, 2]
        heading = float(np.arctan2(nose[0], -nose[2]))
        camera = Camera(self.quad.position.copy(), yaw=heading, pitch=-np.radians(CHASE_PITCH_DEG), fov_deg=self.args.chase_fov)
        forward = camera.axes()[0]
        camera.position = self.quad.position - forward * self.args.chase_distance
        ground = float(self.world.terrain.height_at(camera.position[0], camera.position[2]))
        camera.position[1] = max(camera.position[1], ground + 0.3)
        return camera

    def _draw_world(self, target, size: tuple[int, int], origin: tuple[int, int]) -> None:
        """The 3D world into a pane: the pilot's view, or the chase view with the drone's body in it."""
        now = time.perf_counter()
        dt, self.last_draw = min(now - self.last_draw, 0.1), now
        if self.view == "fpv":
            self.renderer.draw(target, size, self.pilot_camera(), origin=origin)
            return
        camera = self.chase_camera()
        self.renderer.draw(target, size, camera, origin=origin)
        target.viewport = (*origin, *size)
        view_projection = camera.projection_matrix(size[0] / size[1], NEAR, FAR) @ camera.view_matrix()
        self.drone_model.draw(view_projection, self.quad.position, self.quad.rotation, self.quad.motors, dt)

    def draw(self, target, size: tuple[int, int]) -> None:
        width, height = size
        scale = max(1.0, height / 720.0)
        now = time.perf_counter()
        strip_height = int(max(150.0, height * 0.28))
        if self.slam:
            pane = (width // 2, height - strip_height)
            self._draw_world(target, pane, (0, strip_height))
            if now - self.map_points_at > 0.5:
                self._refresh_map()
                self.map_points_at = now
            self.map_view.draw(target, (width // 2, strip_height, width, height), self._map_lines(), self.estimator.position)
        else:
            self._draw_world(target, size, (0, 0))
        if self.headless:
            self.strip_at = self.top_at = self.map_text_at = -1e9
        layout = self._strip_layout(width, strip_height, scale)
        if now - self.strip_at > 1 / 25:
            self.overlay.upload("strip", self._draw_strip(width, strip_height, scale, layout))
            self.strip_at = now
        self.overlay.draw(self.overlay._textures["strip"], (0, 0, width, strip_height), size)
        for name in ("left", "right"):
            if self.camera_on[name]:
                cam = self.cameras[name]
                tile = layout["tiles"][name]
                box = (tile[0], tile[1], tile[2], tile[3])
                self.overlay.draw(cam.output, fit_rect(box, cam.size[0] / cam.size[1]), size, flip=True)
        if now - self.top_at > 1 / 10:
            self.overlay.upload("top", self._draw_top(scale))
            self.top_at = now
        top = self.overlay._textures["top"]
        self.overlay.draw(top, (0, height - top.size[1], top.size[0], height), size, blend=True)
        if self.slam:
            if now - self.map_text_at > 1 / 5:
                self.overlay.upload("map_text", self._draw_map_text(scale))
                self.map_text_at = now
            text = self.overlay._textures["map_text"]
            x0 = width // 2 + 4
            self.overlay.draw(text, (x0, height - text.size[1], x0 + text.size[0], height), size, blend=True)

    def _strip_layout(self, width: int, height: int, scale: float) -> dict:
        gap = int(6 * scale)
        tile = height - int(26 * scale)
        right_x0 = width - gap - tile
        left_x0 = right_x0 - gap - tile
        bottom = gap
        return {
            "gap": gap,
            "tile": tile,
            "panels_right": left_x0 - gap,
            "tiles": {"left": (left_x0, bottom, left_x0 + tile, bottom + tile), "right": (right_x0, bottom, right_x0 + tile, bottom + tile)},
        }

    def _draw_strip(self, width: int, height: int, scale: float, layout: dict) -> np.ndarray:
        canvas = np.full((height, width, 3), 12, np.uint8)
        gap = layout["gap"]
        h = self.histories
        attitude = History(3, len(h["att"].data))
        attitude.data = h["att"].data.copy()
        attitude.data[:, 2] -= attitude.data[-1, 2]  # heading drawn relative to now
        panels = [
            ("ATTITUDE", attitude, 60.0, "deg", "rpy"[0:3], None),
            ("GYRO", h["gyro"], 8.0, "rad/s", "xyz", None),
            ("ACCEL", h["accel"], 25.0, "m/s2", "xyz", None),
            ("MOTORS", h["motors"], 100.0, "% thrust", "", AXIS_COLORS),
            ("CLIMB", h["climb"], 6.0, "m/s", "te", None),
        ]
        if self.slam:
            panels.append(("DRIFT FROM START (truth)", h["pos"], 3.0, "m", "xyz", None))
        count = len(panels)
        panel_width = (layout["panels_right"] - gap * (count + 1)) // count
        top, bottom = gap, height - gap
        for index, (title, history, limit, unit, labels, colors) in enumerate(panels):
            x0 = gap + index * (panel_width + gap)
            box = (x0, top, x0 + panel_width, bottom)
            if title == "MOTORS":
                # motor thrusts are 0..100: draw them from the bottom line up
                shifted = History(4, len(history.data))
                shifted.data = history.data - 50.0
                draw_graph(canvas, box, title, shifted, 50.0, unit, scale, AXIS_COLORS, show_values=False)
                latest = history.data[-1]
                cv2.putText(canvas, " ".join(f"{v:.0f}" for v in latest), (x0 + 4, bottom - int(4 * scale)), FONT, 0.3 * scale, (200, 200, 200), 1, cv2.LINE_AA)
            else:
                draw_graph(canvas, box, title, history, limit, unit, scale, colors or AXIS_COLORS, labels)
        for name, label in (("left", "LEFT CAM"), ("right", "RIGHT CAM")):
            x0, y0, x1, y1 = layout["tiles"][name]
            cy0, cy1 = height - y1, height - y0
            cv2.rectangle(canvas, (x0, cy0), (x1, cy1), (0, 0, 0), -1)
            cv2.rectangle(canvas, (x0, cy0), (x1, cy1), (70, 70, 70), 1)
            fps = self.feed_rate(name)
            cam = self.cameras[name]
            status = f"{cam.size[0]}x{cam.size[1]} {fps:4.1f} fps" if self.camera_on[name] else "off"
            cv2.putText(canvas, f"{label} {status}", (x0, cy0 - int(6 * scale)), FONT, 0.34 * scale, (200, 200, 200) if self.camera_on[name] else (110, 110, 110), 1, cv2.LINE_AA)
            if not self.camera_on[name]:
                key = "1" if name == "left" else "2"
                cv2.putText(canvas, f"press {key}", (x0 + int(8 * scale), (cy0 + cy1) // 2), FONT, 0.4 * scale, (110, 110, 110), 1, cv2.LINE_AA)
        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    def _draw_top(self, scale: float) -> np.ndarray:
        quad, ctl = self.quad, self.controller
        terrain = self.world.terrain
        agl = quad.position[1] - float(terrain.height_at(quad.position[0], quad.position[2]))
        speed = float(np.linalg.norm(quad.velocity))
        roll, pitch, yaw = np.degrees(attitude_angles(quad.rotation))
        er, ep, ey = np.degrees(attitude_angles(ctl.estimator.rotation)) if ctl.estimator.initialised else (0.0, 0.0, 0.0)
        if self.slam:
            mode = {"boot": "ON THE GROUND, CALIBRATING THE IMU", "takeoff": "TAKING OFF (IMU + cameras starting)", "wiggle": "START-UP WIGGLE (learning scale)", "hold": "POSITION HOLD (SLAM + IMU)", "route": f"ROUTE: waypoint {self.route.leg.target + 1 if self.route else 0}/4 (SLAM + IMU)", "manual": "MANUAL"}[self.mission]
            if ctl.velocity_assist:
                mode += " + VELOCITY ASSIST"
        else:
            mode = "ANGLE MODE"
        lines = [
            (f"{mode}   alt {quad.position[1]:.1f} m  ({agl:.1f} m above ground)   speed {speed:.1f} m/s   heading {yaw % 360:.0f}", (255, 255, 255)),
            (f"attitude true r {roll:+.1f} p {pitch:+.1f}   fused r {er:+.1f} p {ep:+.1f}   gyro bias est {np.round(ctl.estimator.gyro_bias, 3)}", (200, 220, 200)),
            ("W/S A/D Space/Ctrl Q/E fly   1/2 cameras   R reset   M mountains   N new world" + ("   P hold   H assist" if self.slam else ""), (170, 170, 170)),
        ]
        text, when = self.banner
        if text and time.perf_counter() - when < CRASH_BANNER_SECONDS:
            lines.insert(0, (text, (80, 80, 255) if quad.crashed else (60, 200, 255)))
        elif quad.crashed:
            lines.insert(0, ("CRASHED - press R to reset", (80, 80, 255)))
        line_height = int(20 * scale)
        width, height = int(780 * scale), line_height * len(lines) + int(10 * scale)
        canvas = np.zeros((height, width, 4), np.uint8)
        canvas[:, :, 3] = 110
        for i, (line, color) in enumerate(lines):
            y = int(18 * scale) + i * line_height
            cv2.putText(canvas, line, (int(8 * scale), y), FONT, 0.45 * scale, (*color, 255), 1, cv2.LINE_AA)
        return cv2.cvtColor(canvas, cv2.COLOR_BGRA2RGBA)

    def _draw_map_text(self, scale: float) -> np.ndarray:
        est = self.estimator
        truth = self.truth_in_w()
        target = self.target_point()
        lines = [
            ("SLAM MAP (metres, drone's own estimate)    green: estimated path   magenta: true path   yellow +: hold point", (230, 230, 230)),
            (f"drift from the hold point (truth)  {np.linalg.norm(truth - target):.2f} m     estimate error {np.linalg.norm(est.position - truth):.2f} m     sim x{self.sim_rate:.2f}", (200, 255, 200)),
        ]
        for name in self.cameras:
            stats = self.link.stats.get(name)
            if stats is None:
                lines.append((f"{name}: waiting for the first picture", (150, 150, 150)))
                continue
            lines.append(
                (
                    f"{name}: {stats['tracks']} tracks  {stats['landmarks']} landmarks  {stats['steps']} steps  {stats['fixes']} fixes  {stats['ms']:.1f} ms/pic   "
                    f"scale {est.scale(name):.2f}/m +-{100 * est.scale_uncertainty(name):.0f}%  pose updates {est.updates[name]} (rejected {est.rejected[name]})",
                    (200, 220, 255),
                )
            )
        line_height = int(20 * scale)
        width, height = int(900 * scale), line_height * len(lines) + int(10 * scale)
        canvas = np.zeros((height, width, 4), np.uint8)
        canvas[:, :, 3] = 110
        for i, (line, color) in enumerate(lines):
            cv2.putText(canvas, line, (int(8 * scale), int(18 * scale) + i * line_height), FONT, 0.42 * scale, (*color, 255), 1, cv2.LINE_AA)
        return cv2.cvtColor(canvas, cv2.COLOR_BGRA2RGBA)

    # --- main loops ------------------------------------------------------------------------

    def run(self) -> None:
        started = last = time.perf_counter()
        title_at, frames, total_frames = 0.0, 0, 0
        while not glfw.window_should_close(self.window):
            now = time.perf_counter()
            if self.args.run_seconds and now - started > self.args.run_seconds:
                break
            total_frames += 1
            dt, last = min(now - last, 0.1), now
            glfw.poll_events()
            self.read_pilot(dt)
            sim_before = self.quad.time
            self.advance(dt)
            if dt > 0:
                self.sim_rate += 0.05 * ((self.quad.time - sim_before) / (time.perf_counter() - now + dt) - self.sim_rate)
            width, height = glfw.get_framebuffer_size(self.window)
            if width > 0 and height > 0:
                self.draw(self.ctx.screen, (width, height))
            glfw.swap_buffers(self.window)
            frames += 1
            if now - title_at > 0.5:
                fps = frames / max(now - title_at, 1e-6) if title_at else 0.0
                glfw.set_window_title(self.window, f"virtual-sim FPV | seed {self.world.seed} | {fps:.0f} fps | mountains {'low' if self.args.low_mountains else 'tall'}")
                title_at, frames = now, 0
        if self.args.run_seconds:
            span = time.perf_counter() - started
            print(f"{total_frames / span:.1f} screen fps; side cameras {self.feed_rate('left'):.1f} / {self.feed_rate('right'):.1f} fps; {span:.1f} s")
        self.close()
        glfw.terminate()

    def close(self) -> None:
        if self.diag is not None:
            self.diag.close()
        if self.recorder:
            self.recorder.close()
        if self.link is not None:
            self.link.close()

    def scripted_hop(self, seconds: float) -> None:
        """Take off, hover, fly forward, turn and brake, with no keyboard: for --screenshot."""
        plan = [(0.0, PilotInput(climb=1.0)), (2.5, PilotInput()), (3.5, PilotInput(forward=1.0)), (5.5, PilotInput(forward=0.6, right=0.4, yaw=0.5)), (7.0, PilotInput())]
        total = int(seconds / PHYSICS_DT)
        start_shots = max(0.0, seconds - 0.4)
        for i in range(total):
            elapsed = i * PHYSICS_DT
            for begin, stick in plan:
                if elapsed >= begin:
                    self.pilot = PilotInput(**vars(stick))
            self._step()
            if elapsed >= start_shots and self.quad.time >= self.next_frame:
                self.shoot_feeds()
                self.next_frame += 1.0 / self.args.cam_fps
            self._check_impacts()

    def hold_test(self, seconds: float, report_every: float = 2.0) -> None:
        """--hold-test: fly the SLAM + IMU position-hold experiment with no window and print how it went."""
        began = time.perf_counter()
        next_report = report_every
        print(f"{'t':>6} {'mission':8} {'truth drift x y z (m)':>26} {'|drift|':>8} {'|est err|':>9}  scale L/R (per m)   pose updates L/R  tracks L/R  steps L/R")
        frame_dt = 1.0 / 60.0
        while self.quad.time < seconds and not self.quad.crashed:
            self.advance(frame_dt)
            if self.quad.time >= next_report:
                next_report += report_every
                self._print_row()
        wall = time.perf_counter() - began
        if DEBUG and self.debug_rows:
            rows = np.array(self.debug_rows)
            for lo in np.arange(0.0, seconds, 10.0):
                m = (rows[:, 0] >= lo) & (rows[:, 0] < lo + 10.0)
                if m.any():
                    print(f"[debug {lo:4.0f}-{lo + 10:4.0f}s] SLAM attitude err {rows[m, 1].mean():5.2f} deg, filter attitude err {rows[m, 2].mean():5.2f} deg, SLAM position err (est. scale) {rows[m, 3].mean():5.2f} m, turn rate {rows[m, 4].mean():4.2f} rad/s, n={m.sum()}")
        for name, stats in self.link.stats.items():
            print(f"[{name}] " + "  ".join(f"{k} {v:.1f}" if isinstance(v, float) else f"{k} {v}" for k, v in stats.items()) + f"  attitude corrections {self.estimator.attitude_corrections} rejected {self.estimator.attitude_rejections}")
        print(f"\nsimulated {self.quad.time:.1f} s in {wall:.1f} s wall (x{self.quad.time / wall:.2f} real time)")
        if self.quad.crashed:
            print("CRASHED")
        if self.hold_log:
            log = np.array(self.hold_log)
            held = log[:, 0] - log[0, 0]
            print(f"position hold lasted {held[-1]:.1f} s: true distance from the hold point  mean {log[:, 1].mean():.2f} m   rms {np.sqrt((log[:, 1] ** 2).mean()):.2f} m   max {log[:, 1].max():.2f} m   at the end {log[-1, 1]:.2f} m")
            half = len(log) // 2
            print(f"   first half mean {log[:half, 1].mean():.2f} m, second half mean {log[half:, 1].mean():.2f} m; estimator error mean {log[:, 2].mean():.2f} m, max {log[:, 2].max():.2f} m")
        else:
            print("position hold never started (the SLAM did not learn its scale in time)")
        if self.route_log:
            log = np.array(self.route_log)
            print(f"route flown for {log[-1, 0] - log[0, 0]:.1f} s ({self.route.laps} full laps): true distance from the planned point  mean {log[:, 1].mean():.2f} m  rms {np.sqrt((log[:, 1] ** 2).mean()):.2f} m  max {log[:, 1].max():.2f} m;  estimator error mean {log[:, 2].mean():.2f} m  max {log[:, 2].max():.2f} m;  top speed {log[:, 3].max():.1f} m/s")

    def _print_row(self) -> None:
        est = self.estimator
        truth = self.truth_in_w()
        target = self.target_point()
        drift = truth - target
        pair = lambda f: "/".join(f(n) for n in ("left", "right"))
        print(
            f"{self.quad.time:6.1f} {self.mission:8} {drift[0]:+8.2f} {drift[1]:+8.2f} {drift[2]:+8.2f} {np.linalg.norm(drift):8.2f} {np.linalg.norm(est.position - truth):9.2f}  "
            f"{pair(lambda n: f'{est.scale(n):.2f}({100 * est.scale_uncertainty(n):.0f}%)'):>20}  {pair(lambda n: str(est.updates[n])):>14}  "
            f"{pair(lambda n: str(self.link.stats.get(n, {}).get('tracks', '-'))):>10}  {pair(lambda n: str(self.link.stats.get(n, {}).get('steps', '-'))):>10}"
        )


class Recorder:
    """--record: the two camera streams as mp4, plus IMU and ground-truth pose logs for SLAM experiments."""

    def __init__(self, folder: Path, size: tuple[int, int], fps: float) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        self.writers = {name: cv2.VideoWriter(str(folder / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, size) for name in ("left", "right")}
        self.imu_file = open(folder / "imu.csv", "w", newline="")
        self.pose_file = open(folder / "poses.csv", "w", newline="")
        self.imu_writer, self.pose_writer = csv.writer(self.imu_file), csv.writer(self.pose_file)
        self.imu_writer.writerow(["t", "gx", "gy", "gz", "ax", "ay", "az", "baro"])
        self.pose_writer.writerow(["t", "x", "y", "z", "qw", "qx", "qy", "qz"])
        self.queues = {name: queue.Queue(maxsize=8) for name in self.writers}
        self.threads = [threading.Thread(target=self._encode, args=(name,), daemon=True) for name in self.writers]  # cv2 releases the GIL while encoding
        for thread in self.threads:
            thread.start()
        print(f"Recording to {folder}/")

    def _encode(self, name: str) -> None:
        while (frame := self.queues[name].get()) is not None:
            self.writers[name].write(frame)

    def frame(self, name: str, bgr: np.ndarray) -> None:
        self.queues[name].put(bgr)  # waits when the encoder is behind, so no frame is dropped

    def imu(self, s: SensorSample, quad: Quadrotor) -> None:
        self.imu_writer.writerow([f"{s.t:.4f}", *map("{:.5f}".format, s.gyro), *map("{:.5f}".format, s.accel), "" if s.baro is None else f"{s.baro:.3f}"])

    def pose(self, quad: Quadrotor) -> None:
        self.pose_writer.writerow([f"{quad.time:.4f}", *map("{:.4f}".format, quad.position), *map("{:.6f}".format, quad.q)])

    def close(self) -> None:
        for name, frames in self.queues.items():
            frames.put(None)
        for thread in self.threads:
            thread.join()
        for writer in self.writers.values():
            writer.release()
        self.imu_file.close()
        self.pose_file.close()


def main() -> None:
    args = parse_args()
    seed = args.seed if args.seed is not None else int(np.random.default_rng().integers(0, 1_000_000))
    if args.hold_test is not None and not args.slam:
        raise SystemExit("--hold-test needs --slam")
    if args.screenshot or args.hold_test is not None:
        window, ctx = create_hidden_context()
        app = FpvApp(args, seed, window, ctx, headless=True)
        if args.hold_test is not None:
            app.hold_test(args.hold_test)
        else:
            app.scripted_hop(args.screenshot_seconds)
        if args.screenshot:
            size = (args.width, args.height)
            target = OffscreenTarget(ctx, size)
            app.draw(target.framebuffer, size)
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.screenshot), target.read_bgr())
            print(f"Wrote {args.screenshot}")
        app.close()
        glfw.terminate()
        return
    print(__doc__)
    window, ctx = open_window(args.width, args.height, "virtual-sim FPV", not args.no_vsync)
    FpvApp(args, seed, window, ctx).run()


if __name__ == "__main__":
    main()
