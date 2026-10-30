"""Flight controller: PID loops that hold the drone's attitude and altitude, flown by stick-like inputs.

Cascade, all driven by the *estimated* state from sensors.StateEstimator:

    stick -> desired tilt  --P-->  body rate setpoint  --PID-->  torque ----+
    stick -> climb rate    --------------------------PID-->  total thrust --+--> mixer --> 4 motor thrusts
    stick -> yaw rate (or heading hold) -------------PID-->  yaw torque ----+

Without stick input the drone levels itself and holds its height (hover). When the estimator has a
position and velocity (vio.VisualInertialEstimator: camera SLAM + accelerometer + gyro) two more
modes exist: the velocity assist turns the tilt sticks into horizontal speed commands, and position
hold flies back to `hold_target` (x, y, z in the estimator's world) and stays there.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drone import GRAVITY, UP, QuadParams, attitude_angles, wrap_angle
from sensors import SensorSample, StateEstimator


class PID:
    """PID on an error, with the derivative taken on the measurement (no kick when the setpoint jumps) and low-pass filtered."""

    def __init__(self, kp: float, ki: float = 0.0, kd: float = 0.0, integral_limit: float = np.inf, derivative_cutoff: float = 60.0) -> None:
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral_limit = integral_limit
        self.derivative_cutoff = derivative_cutoff  # Hz-ish, in rad/s
        self.reset()

    def reset(self) -> None:
        self.integral = 0.0
        self.derivative = 0.0
        self._last_measurement: float | None = None

    def update(self, setpoint: float, measurement: float, dt: float) -> float:
        error = setpoint - measurement
        self.integral = float(np.clip(self.integral + self.ki * error * dt, -self.integral_limit, self.integral_limit))
        if self._last_measurement is not None and dt > 0.0:
            raw = -(measurement - self._last_measurement) / dt
            blend = 1.0 - np.exp(-self.derivative_cutoff * dt)
            self.derivative += blend * (raw - self.derivative)
        self._last_measurement = measurement
        return self.kp * error + self.integral + self.kd * self.derivative


@dataclass
class PilotInput:
    forward: float = 0.0  # -1 .. 1, positive flies forward
    right: float = 0.0  # -1 .. 1, positive flies right
    yaw: float = 0.0  # -1 .. 1, positive turns right
    climb: float = 0.0  # -1 .. 1, positive climbs
    sport: bool = False  # allows a steeper tilt


@dataclass
class ControllerLimits:
    tilt: float = np.radians(26.0)
    sport_tilt: float = np.radians(42.0)
    yaw_rate: float = np.radians(120.0)
    climb_rate: float = 4.0  # m/s up
    descent_rate: float = 3.0  # m/s down
    max_rate: float = 9.0  # rad/s, roll and pitch
    assist_speed: float = 9.0  # m/s horizontal at full stick with the velocity assist
    hold_speed: float = 1.5  # m/s, fastest the position hold flies back to its target
    hold_climb: float = 1.0  # m/s
    hold_tilt: float = np.radians(18.0)
    route_tilt: float = np.radians(24.0)  # the steepest tilt the route planner's accelerations may ask for
    route_speed_margin: float = 1.0  # m/s above the planned speed the position loop may add to catch up


class FlightController:
    def __init__(self, params: QuadParams, mixer_inverse: np.ndarray, estimator: StateEstimator | None = None) -> None:
        self.params = params
        self.mixer_inverse = mixer_inverse
        self.estimator = estimator or StateEstimator()
        self.limits = ControllerLimits()
        self.velocity_assist = False
        self.hold_target: np.ndarray | None = None  # world position (x, y, z) the position hold flies to
        self.reference: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None  # (position, velocity, acceleration) of a planned trajectory to track, world frame
        self.position_gain = 0.9  # 1/s: speed per metre of position error
        self.climb_position_gain = 1.2  # 1/s
        self.angle_gain = 8.0  # 1/s: rate setpoint per radian of tilt error
        self.heading_gain = 3.0  # 1/s
        self.rate_pid = [PID(24.0, 14.0, 0.45, 4.0), PID(14.0, 6.0, 0.0, 3.0), PID(24.0, 14.0, 0.45, 4.0)]  # body x (pitch), y (yaw), z (roll), rad/s^2 out
        self.climb_pid = PID(4.5, 2.5, 0.0, 6.0)  # m/s^2 out
        self.velocity_pid = [PID(2.2, 0.6, 0.0, 3.0), PID(2.2, 0.6, 0.0, 3.0)]  # m/s^2 out, x and z
        self.reset()

    def reset(self, airborne: bool = False) -> None:
        """`airborne`: the drone starts in the air, so command hover thrust while the estimator calibrates."""
        self.estimator.reset()
        for pid in (*self.rate_pid, self.climb_pid, *self.velocity_pid):
            pid.reset()
        self.heading_target: float | None = None
        self.rate_setpoint = np.zeros(3)
        self.thrust_command = np.full(4, self.params.mass * GRAVITY / 4.0) if airborne else np.zeros(4)
        self.hold_target = None
        self.reference = None

    def update(self, sample: SensorSample, pilot: PilotInput, on_ground: bool) -> np.ndarray:
        """One control tick: new sensor sample in, four motor thrust commands (N) out."""
        est, lim, p = self.estimator, self.limits, self.params
        est.update(sample)
        dt = sample.dt
        if not est.initialised or dt <= 0.0:
            return self.thrust_command
        rotation = est.rotation
        _, _, yaw = attitude_angles(rotation)
        forward_h = np.array([np.sin(yaw), 0.0, -np.cos(yaw)])
        right_h = np.array([np.cos(yaw), 0.0, np.sin(yaw)])
        max_tilt = lim.sport_tilt if pilot.sport else lim.tilt

        # horizontal sticks -> the thrust direction we want
        holding = self.hold_target is not None and est.position is not None
        tracking = self.reference is not None and est.position is not None and est.velocity is not None
        feedforward = np.zeros(3)  # planned acceleration (+ the drag it will meet), m/s^2
        climb_setpoint = pilot.climb * (lim.climb_rate if pilot.climb > 0 else lim.descent_rate)
        if tracking:
            # Trajectory tracking by differential flatness. The planner hands us where the drone should be, how fast and with
            # what acceleration. The thrust vector that produces that motion is  m (a + g up) + drag(v)  (the direction of it is
            # the attitude to fly, its length the collective thrust). The position / velocity loops only trim what the model misses
            # (wind, an off-nominal thrust, estimator error).
            p_d, v_d, a_d = self.reference
            speed = np.linalg.norm(v_d)
            drag = (p.drag_quadratic * speed + p.drag_linear) * v_d / p.mass
            feedforward = a_d + drag
            to_target = p_d - est.position
            wanted = v_d + np.clip(self.position_gain * to_target, -lim.route_speed_margin, lim.route_speed_margin)
            wanted[1] = 0.0
            climb_setpoint = float(v_d[1] + np.clip(self.climb_position_gain * to_target[1], -lim.hold_climb, lim.hold_climb))
            max_tilt = lim.route_tilt
        elif holding:
            to_target = self.hold_target - est.position
            wanted = np.clip(self.position_gain * to_target, -lim.hold_speed, lim.hold_speed)
            wanted[1] = 0.0
            if np.linalg.norm(wanted) > lim.hold_speed:
                wanted *= lim.hold_speed / np.linalg.norm(wanted)
            climb_setpoint = float(np.clip(self.climb_position_gain * to_target[1], -lim.hold_climb, lim.hold_climb))
            max_tilt = lim.hold_tilt
        elif self.velocity_assist and est.velocity is not None:
            wanted = (forward_h * pilot.forward + right_h * pilot.right) * lim.assist_speed
        if (tracking or holding or self.velocity_assist) and est.velocity is not None:
            accel = np.array([self.velocity_pid[0].update(wanted[0], est.velocity[0], dt) + feedforward[0], 0.0, self.velocity_pid[1].update(wanted[2], est.velocity[2], dt) + feedforward[2]])
            limit = np.tan(max_tilt) * GRAVITY
            if np.linalg.norm(accel) > limit:
                accel *= limit / np.linalg.norm(accel)
            desired_up = UP * (GRAVITY + feedforward[1]) + accel
        else:
            desired_up = UP + np.tan(max_tilt) * (forward_h * pilot.forward + right_h * pilot.right)
        desired_up = desired_up / np.linalg.norm(desired_up)

        # attitude: tilt error -> body rate setpoints
        error = np.cross(UP, rotation.T @ desired_up)  # axis (body frame) that swings our up toward the wanted up
        rate_sp = np.clip(self.angle_gain * error, -lim.max_rate, lim.max_rate)
        if abs(pilot.yaw) > 0.01:
            self.heading_target = None
            rate_sp[1] = -pilot.yaw * lim.yaw_rate  # body +y turns left
        else:
            if self.heading_target is None:
                self.heading_target = yaw
            rate_sp[1] = -float(np.clip(self.heading_gain * wrap_angle(self.heading_target - yaw), -2.0, 2.0))
        self.rate_setpoint = rate_sp

        # rates: PID -> angular acceleration -> torque
        rates = sample.gyro - est.gyro_bias
        alpha = np.array([self.rate_pid[i].update(rate_sp[i], rates[i], dt) for i in range(3)])
        torque = alpha * np.array(p.inertia)

        # height: climb-rate PID, then compensate for the tilt
        if on_ground and climb_setpoint <= 0.0 and abs(est.climb_rate) < 0.5:
            self.climb_pid.reset()
            self.thrust_command = np.zeros(4)  # sitting on the ground, motors idle
            return self.thrust_command
        lift = p.mass * (GRAVITY + feedforward[1] + self.climb_pid.update(climb_setpoint, est.climb_rate, dt)) / max(float(rotation[1, 1]), 0.5)
        lift = float(np.clip(lift, 0.0, 3.6 * p.max_thrust))

        thrust = self.mixer_inverse @ np.array([lift, *torque])
        self.thrust_command = np.clip(thrust, 0.03 * p.max_thrust, p.max_thrust)
        return self.thrust_command
