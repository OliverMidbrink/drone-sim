"""Visual-inertial state estimator: accelerometer + gyro (the IMU) fused with camera SLAM poses.

What goes in is only what a real drone has: IMU samples (`update`) and, whenever a SLAM worker
finishes a picture, that camera's pose in its own frame (`slam_pose`). Nothing absolute.

Frames
    W      the estimator's world: gravity-aligned (found from the accelerometer while sitting still at
           power-up) with the nose heading at power-up as heading 0, origin at the power-up position.
           "Hold position" means hold the point where the drone was when it turned on.
    S_i    camera i's SLAM frame: the camera at its first picture is the origin and the axes of the
           camera at that moment are the axes. Its unit of length is arbitrary (a single camera
           cannot know metres). Since the body's orientation in W is known at that moment, S_i's
           orientation in W is known too; only its scale lambda_i (SLAM units per metre) is not.

The filter
    * Attitude: gyro integration (the stock Mahony filter of sensors.StateEstimator, a weak pull on
      roll/pitch from the accelerometer), plus a small pull toward each SLAM orientation. The camera
      is what stops the heading from wandering off, since there is no compass.
    * Position, velocity, accelerometer bias and each camera's scale: an extended Kalman filter.
      The accelerometer drives it (rotated into W by the attitude, gravity removed). A SLAM pose is
      a measurement of  a = R_WS * c  (the camera centre in SLAM units, expressed in W axes) and the
      model says  a = lambda * (p - p0 + (R_wb - R_wb0) * lever_arm)  where p is the body position and p0 where the body was at that first picture. Metric
      scale is therefore learned from how far the IMU says the drone moved versus how far the SLAM
      says it moved, which needs the drone to move a little (see FpvApp's start-up wiggle).
    * Pictures arrive late (the SLAM takes a few milliseconds and the pipeline some frames), so the
      state history is kept and the innovation is computed against the state at the picture's time.
    * Optional barometer (off by default): a height measurement.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from drone import GRAVITY, quat_multiply, quat_to_matrix
from sensors import SensorSample, StateEstimator

HISTORY = 500  # IMU ticks kept for late pictures (2 s at 250 Hz)


@dataclass
class CameraModel:
    name: str
    body_from_cv: np.ndarray  # OpenCV camera axes -> body axes (an extrinsic calibration)
    lever_arm: np.ndarray  # body-frame position of the lens


@dataclass
class VioConfig:
    accel_noise: float = 0.25  # m/s^2: the accelerometer, plus the attitude error that leaks gravity into it
    bias_walk: float = 0.004  # m/s^2 per sqrt(s)
    accel_bias_online: bool = True  # False: the accelerometer is taken as calibrated, its offset is never estimated (nothing for the SLAM to drag around)
    accel_gravity_gain: bool = False  # True: at power-up, scale the accelerometer so it reads exactly 1 g while the drone sits still (fixes the vertical axis's gain)
    scale_walk: float = 0.02  # relative scale change per sqrt(s): a monocular SLAM's unit slowly drifts
    slam_position_noise: float = 0.15  # m: how well a SLAM pose is trusted
    gate: float = 25.0  # squared Mahalanobis distance above which a pose is not believed (3 degrees of freedom)
    gate_patience: int = 6  # ... until this many in a row disagree, then the filter believes them (they may be right)
    attitude_gain: float = 0.1  # share of a SLAM orientation error corrected per picture
    attitude_gate_deg: float = 20.0  # a SLAM attitude further than this from the filter's is not used at all
    attitude_cap_deg: float = 4.0  # ... and the error counted for the pull never exceeds this (a robust, Huber-like pull: bounded, never locked out)
    attitude_bias_gain: float = 0.004  # rad/s of gyro yaw-bias change per rad of heading disagreement, per picture (0 = off): heading is the only axis the accelerometer cannot see
    scale_prior: float = 3.0  # SLAM units per metre before anything is known
    scale_prior_sigma: float = 4.0
    baro_noise: float = 0.15
    scale_ready_sigma: float = 0.25  # relative scale uncertainty below which a camera counts as calibrated
    scale_ready_updates: int = 8
    align: bool = True  # realign each SLAM position by the SLAM-vs-filter attitude disagreement
    align_max_deg: float = 25.0  # a SLAM frame turned further than this from the filter's is not realigned, the picture is not believed
    turn_rate: float = 0.6  # rad/s: at this turn rate a SLAM pose counts twice as unreliable (variance x2); at 3x this, x26
    turn_rate_cap: float = 30.0  # the most a pose's variance is ever inflated by


class VisualInertialEstimator(StateEstimator):
    def __init__(self, cameras: list[CameraModel], config: VioConfig | None = None, use_baro: bool = False) -> None:
        self.cameras = {c.name: c for c in cameras}
        self.camera_index = {c.name: 9 + i for i, c in enumerate(cameras)}
        self.config = config or VioConfig()
        self.use_baro = use_baro
        super().__init__()

    # ------------------------------------------------------------------ lifecycle
    def reset(self) -> None:
        super().reset()
        n = 9 + len(self.cameras)
        cfg = self.config
        self.x = np.zeros(n)  # p (3), v (3), accelerometer bias (3), scale per camera
        self.x[9:] = cfg.scale_prior
        bias_sigma = 0.1 if cfg.accel_bias_online else 0.0  # a calibrated accelerometer has no unknown offset: zero variance, so no measurement ever moves it
        self.P = np.diag([0.01**2] * 3 + [0.05**2] * 3 + [bias_sigma**2] * 3 + [cfg.scale_prior_sigma**2] * len(self.cameras))
        self.accel_gain = 1.0
        self.position = self.x[0:3]
        self.velocity = self.x[3:6]
        self.history: deque = deque(maxlen=HISTORY)
        self.start_rotation: dict[str, np.ndarray] = {}  # R_wb at each camera's first picture
        self.start_position: dict[str, np.ndarray] = {}  # the filter's body position at that moment
        self.updates = {name: 0 for name in self.cameras}
        self.rejected = {name: 0 for name in self.cameras}
        self._streak = {name: 0 for name in self.cameras}
        self.last_innovation = {name: 0.0 for name in self.cameras}
        self.attitude_corrections = 0
        self.attitude_rejections = 0
        self._baro0: float | None = None
        self._ticks = 0
        self.turn_rate = 0.0  # |gyro - bias|, rad/s, smoothed over ~0.1 s
        self.last_distrust = {name: 1.0 for name in self.cameras}
        self.locked: set[int] = set()  # scale states held fixed (lock_scale)

    def lock_scale(self, name: str, value: float, relative_sigma: float = 0.03) -> None:
        """Hold a camera's scale at `value` from now on (a "consider" state: its uncertainty still enters every update, but no update moves it).

        Measured: once calibrated, the SLAM map keeps its unit within about 10% (its steps are held to the accelerometer at the calibrated
        scale), but a free scale state in this filter wandered off to twice that over two minutes. Scale and position are only told
        apart by the accelerometer's double integral, a weak signal, so any jump in the SLAM pose leaks partly into the scale and
        nothing pulls it back. The map's unit is set by the calibrated scale; the filter should read it with that same unit."""
        i = self.camera_index[name]
        self.x[i] = value
        self.P[i, :] = 0.0
        self.P[:, i] = 0.0
        self.P[i, i] = (relative_sigma * value) ** 2
        self.locked.add(i)

    def begin_camera(self, name: str) -> None:
        """Call at the moment a camera's SLAM takes its first picture: that picture's pose defines the frame S_i."""
        self.start_rotation[name] = self.rotation.copy()
        self.start_position[name] = self.x[0:3].copy()
        self.updates[name] = 0
        self.x[self.camera_index[name]] = self.config.scale_prior
        i = self.camera_index[name]
        self.locked.discard(i)
        self.P[i, :] = 0.0
        self.P[:, i] = 0.0
        self.P[i, i] = self.config.scale_prior_sigma**2

    # ------------------------------------------------------------------ IMU
    def update(self, sample: SensorSample) -> None:
        if not self.initialised:
            self._calibrate(sample)
            if self.initialised:
                self._baro0 = sample.baro
                self._store(sample.t)
            return
        dt = sample.dt
        if dt <= 0.0:
            return
        self.propagate_attitude(sample)
        self.turn_rate += (float(np.linalg.norm(sample.gyro - self.gyro_bias)) - self.turn_rate) * (1.0 - np.exp(-dt / 0.1))
        rotation = quat_to_matrix(self.q)
        x, cfg = self.x, self.config
        n = len(x)
        acceleration = rotation @ (self.accel_gain * sample.accel - x[6:9]) + np.array([0.0, -GRAVITY, 0.0])
        x[0:3] += x[3:6] * dt + 0.5 * acceleration * dt * dt
        x[3:6] += acceleration * dt
        f = np.eye(n)
        f[0:3, 3:6] = np.eye(3) * dt
        f[3:6, 6:9] = -rotation * dt
        q = np.zeros(n)
        q[3:6] = (cfg.accel_noise**2) * dt
        q[6:9] = (cfg.bias_walk**2) * dt if cfg.accel_bias_online else 0.0
        q[9:] = (cfg.scale_walk * x[9:]) ** 2 * dt
        for i in self.locked:
            q[i] = 0.0
        self.P = f @ self.P @ f.T + np.diag(q)
        self._ticks += 1
        if self.use_baro and sample.baro is not None and self._baro0 is not None and self._ticks % 5 == 0:
            self._scalar_update(1, sample.baro - self._baro0, cfg.baro_noise**2)
        self.altitude, self.climb_rate = float(x[1]), float(x[4])
        self._store(sample.t, rotation)

    def _calibrate(self, sample: SensorSample) -> None:
        super()._calibrate(sample)
        if self.initialised and self.config.accel_gravity_gain:
            self.accel_gain = GRAVITY / float(np.linalg.norm(self._cal_accel / self._cal_n))  # sitting still, the accelerometer feels exactly 1 g

    def update_vertical(self, sample: SensorSample) -> None:  # the base class's baro/accel blend is replaced by the filter
        pass

    def _store(self, t: float, rotation: np.ndarray | None = None) -> None:
        self.history.append((t, self.x[0:3].copy(), quat_to_matrix(self.q) if rotation is None else rotation, self.turn_rate))

    def _scalar_update(self, index: int, measurement: float, variance: float) -> None:
        h = np.zeros(len(self.x))
        h[index] = 1.0
        s = float(h @ self.P @ h) + variance
        k = (self.P @ h) / s
        self.x += k * (measurement - self.x[index])
        self.P = self.P - np.outer(k, h @ self.P)

    # ------------------------------------------------------------------ SLAM
    def scale(self, name: str) -> float:
        return float(self.x[self.camera_index[name]])

    def scale_uncertainty(self, name: str) -> float:
        i = self.camera_index[name]
        return float(np.sqrt(max(self.P[i, i], 0.0)) / max(self.x[i], 1e-3))

    def scale_ready(self, name: str) -> bool:
        return self.updates[name] >= self.config.scale_ready_updates and self.scale_uncertainty(name) < self.config.scale_ready_sigma

    def _state_at(self, t: float):
        """The stored (p, R_wb) closest to time t."""
        best = None
        for entry in reversed(self.history):
            best = entry
            if entry[0] <= t:
                break
        return best

    def slam_pose(self, name: str, t: float, rotation_sc: np.ndarray, position_s: np.ndarray, extra_sigma: float = 0.0) -> None:
        """A SLAM pose of camera `name` for the picture taken at time t: the camera (OpenCV axes) -> S_i rotation and its centre in S_i."""
        if name not in self.start_rotation or not self.initialised or not self.history:
            return
        cam, cfg = self.cameras[name], self.config
        entry = self._state_at(t)
        if entry is None:
            return
        _, p_then, r_then, _ = entry
        distrust = self._turn_distrust(t)
        self.last_distrust[name] = distrust
        r_ws = self.start_rotation[name] @ cam.body_from_cv  # S_i axes -> W
        measured_body = r_ws @ rotation_sc @ cam.body_from_cv.T  # the body's orientation in W as this camera sees it
        self._attitude_update(r_then, measured_body, 1.0 / distrust)
        # The SLAM's frame S slowly turns away from the one the filter believes in: every step of its rotation estimate adds a little
        # error, and a map-anchoring fix adds more, so its rotation random-walks (8-10 degrees after half a minute) while the gyro-led
        # attitude stays within about a degree. Both are looking at the same body, so the disagreement is how far S has turned: undo
        # it on the position as well (the camera centre swings about S's origin by exactly that much).
        align = r_then @ measured_body.T

        i = self.camera_index[name]
        lam = max(float(self.x[i]), 0.02)
        if not self._alignment_ok(align):
            self.rejected[name] += 1
            return
        measured = (align if cfg.align else np.eye(3)) @ (r_ws @ position_s)
        base = p_then - self.start_position[name] + (r_then - self.start_rotation[name]) @ cam.lever_arm  # the camera relative to where it started, in W
        innovation = measured - lam * base
        n = len(self.x)
        h = np.zeros((3, n))
        h[:, 0:3] = lam * np.eye(3)
        h[:, i] = base
        variance = ((lam * cfg.slam_position_noise) ** 2 + (0.03 * np.linalg.norm(measured)) ** 2 + 1e-8) * distrust
        variance += (lam * extra_sigma) ** 2  # a pose that came with an anchor fix carries the anchors' frozen errors: believed less
        covariance = variance * np.eye(3)
        s = h @ self.P @ h.T + covariance
        distance = float(innovation @ np.linalg.solve(s, innovation))
        self.last_innovation[name] = float(np.sqrt(distance))
        if distance > cfg.gate and self.updates[name] >= 3:
            self._streak[name] += 1
            self.rejected[name] += 1
            if self._streak[name] < cfg.gate_patience:
                return
            covariance = covariance * 25.0  # many in a row: perhaps the filter is the one that is wrong; listen, softly
            s = h @ self.P @ h.T + covariance
        self._streak[name] = 0
        k = self.P @ h.T @ np.linalg.inv(s)
        for locked in self.locked:
            k[locked, :] = 0.0
        self.x += k @ innovation
        identity = np.eye(n)
        joseph = identity - k @ h
        self.P = joseph @ self.P @ joseph.T + k @ covariance @ k.T
        self.P = 0.5 * (self.P + self.P.T)
        self.x[i] = max(self.x[i], 0.02)
        self.updates[name] += 1

    def _alignment_ok(self, align: np.ndarray) -> bool:
        angle = float(np.arccos(np.clip((np.trace(align) - 1.0) / 2.0, -1.0, 1.0)))
        return angle < np.radians(self.config.align_max_deg)

    def _turn_distrust(self, t: float) -> float:
        """How many times less a picture is believed because the drone was turning while it was taken (variance factor, >= 1).

        Turning is when the camera is worst: the optical flow is dominated by rotation (which carries no position
        information; the gyro-derotation has to subtract it exactly, and a clock offset of dt leaves omega*dt of it),
        features slide out of view, blur and rolling shutter grow. The gyro is at its best there for the very same
        reason: a gyro measures turning directly. So the pictures count less and the inertial prediction more."""
        rates = [e[3] for e in self.history if t - 0.3 <= e[0] <= t + 0.05]
        if not rates:
            return 1.0
        omega = float(np.sqrt(np.mean(np.square(rates))))
        return float(min(1.0 + (omega / self.config.turn_rate) ** 2, self.config.turn_rate_cap))

    def _attitude_update(self, r_then: np.ndarray, measured: np.ndarray, weight: float = 1.0) -> None:
        """Pull the attitude a little toward the SLAM's, as the filter saw it when the picture was taken."""
        error_body = r_then.T @ measured
        angle = float(np.arccos(np.clip((np.trace(error_body) - 1.0) / 2.0, -1.0, 1.0)))
        if angle > np.radians(self.config.attitude_gate_deg):
            self.attitude_rejections += 1
            return
        if angle < 1e-9:
            return
        axis = np.array([error_body[2, 1] - error_body[1, 2], error_body[0, 2] - error_body[2, 0], error_body[1, 0] - error_body[0, 1]]) / (2.0 * np.sin(angle))
        pulled = min(angle, np.radians(self.config.attitude_cap_deg))
        correction_world = r_then @ (axis * pulled * self.config.attitude_gain * weight)
        if self.config.attitude_bias_gain > 0.0:
            # A heading that the pictures keep correcting the same way is the gyro's yaw bias: hand it to the gyro integrator (about the vertical only).
            up_body = r_then[1, :]
            yaw_error = float(np.clip(np.dot(axis * angle, up_body), -pulled, pulled))
            self._integral += self.config.attitude_bias_gain * weight * yaw_error * up_body
        half = 0.5 * correction_world
        dq = np.array([1.0, *half])
        dq /= np.linalg.norm(dq)
        self.q = quat_multiply(dq, self.q)
        self.q /= np.linalg.norm(self.q)
        self.attitude_corrections += 1
