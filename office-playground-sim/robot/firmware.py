"""The robot's control firmware: attitude estimator, speed observer and an LQR balance controller.

What runs on the robot's microcontroller, at 200 Hz, using only what its sensors give it:

1. Power-up: the robot is held upright and still for a second and a half while the gyro bias is measured.
2. Attitude: a complementary filter fuses the gyro (good for fast changes, drifts) with the accelerometer's
   gravity direction (good on average, fooled while the robot accelerates). The accelerometer is trusted less
   when its length is not 1 g.
3. Travel: an observer turns the encoder counts into position and speed along the floor.
4. Balance: a linear-quadratic regulator on (position error, speed error, pitch, pitch rate, position integral),
   designed on start from the linearised model of the robot (`design_lqr`) with the *datasheet* parameters,
   not the true ones. The integral term trims out the pitch offset of a centre of mass that is not where
   the model thinks it is, and of the chip's mounting angle.
5. Steering: a PI loop on the gyro's yaw rate plus a feed-forward for the back-EMF.
6. Output: left and right H-bridge duty, quantized to 10 bits. The motors are cut when the robot leans
   past 40 degrees (it has fallen) and only come back after it is picked up and held upright.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dynamics import TwipPlant
from .params import GRAVITY, RobotParams
from .sensors import EncoderSpec, ImuSample

CONTROL_RATE = 200.0  # Hz


# --- design -----------------------------------------------------------------------------------


def expm(matrix: np.ndarray) -> np.ndarray:
    """Matrix exponential by scaling and squaring of a Taylor series."""
    norm = np.linalg.norm(matrix, np.inf)
    squarings = max(0, int(np.ceil(np.log2(max(norm, 1e-12)))) + 2)
    scaled = matrix / 2.0**squarings
    result, term = np.eye(len(matrix)), np.eye(len(matrix))
    for k in range(1, 20):
        term = term @ scaled / k
        result = result + term
    for _ in range(squarings):
        result = result @ result
    return result


def discretize(a: np.ndarray, b: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    n, m = b.shape
    block = np.zeros((n + m, n + m))
    block[:n, :n], block[:n, n:] = a * dt, b * dt
    exp = expm(block)
    return exp[:n, :n], exp[:n, n:]


def solve_dare(a: np.ndarray, b: np.ndarray, q: np.ndarray, r: np.ndarray, iterations: int = 20000) -> np.ndarray:
    """Riccati iteration for the discrete infinite-horizon LQR cost; returns the feedback gain K (u = -K x)."""
    p = q.copy()
    for _ in range(iterations):
        gain = np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)
        updated = q + a.T @ p @ (a - b @ gain)
        if np.max(np.abs(updated - p)) < 1e-10 * max(1.0, np.max(np.abs(p))):
            p = updated
            break
        p = updated
    return np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)


def linear_pitch_model(params: RobotParams) -> tuple[np.ndarray, np.ndarray]:
    """(A, B) of  x' = A x + B V  for x = (position, speed, pitch, pitch rate), V = voltage on both motors, linearised upright.

    Both tyres grip, the winding current follows the voltage instantly (the electrical time
    constant is 0.5 ms) and Coulomb friction is left out; viscous friction and the back-EMF stay.
    """
    plant = TwipPlant(params)
    m, r = params.motor, params.wheel_radius
    viscous = m.gearbox_viscous + m.rotor_friction * m.gear_ratio**2

    def derivative(x: np.ndarray, v: float) -> np.ndarray:
        relative = x[1] / r - x[3]  # wheel speed relative to the chassis
        torque = m.gear_ratio * m.torque_constant * (v - m.torque_constant * m.gear_ratio * relative) / m.resistance - viscous * relative
        applied = np.zeros(5)
        applied[2] = -2.0 * torque
        applied[3] = applied[4] = torque
        a = plant.stuck_acceleration(x[2], np.array([x[1], 0.0, x[3], x[1] / r, x[1] / r]), applied)
        return np.array([x[1], a[0], x[3], a[2]])

    a_matrix, b_matrix = np.zeros((4, 4)), np.zeros((4, 1))
    x0 = np.zeros(4)
    step = 1e-5
    for i in range(4):
        d = np.zeros(4)
        d[i] = step
        a_matrix[:, i] = (derivative(x0 + d, 0.0) - derivative(x0 - d, 0.0)) / (2.0 * step)
    b_matrix[:, 0] = (derivative(x0, step) - derivative(x0, -step)) / (2.0 * step)
    return a_matrix, b_matrix


@dataclass
class LqrWeights:
    position: float = 0.5
    speed: float = 300.0
    pitch: float = 100.0
    pitch_rate: float = 3.0
    integral: float = 0.3
    voltage: float = 0.1  # cost of volts: high enough that the loop keeps its margin against the sample delay and the sensor filter

    @classmethod
    def for_robot(cls, params: RobotParams) -> "LqrWeights":
        """The same weights, with the cost of volts raised for heavier robots (found by trying: 0.1 for the 1 kg robot, about 1 for the 6 kg one).

        Without it the big robot's loop sits at the edge of its delay margin and chatters at full motor current.
        """
        return cls(voltage=0.1 * params.body_mass**1.25)


def design_lqr(params: RobotParams, weights: LqrWeights | None = None, rate: float = CONTROL_RATE) -> np.ndarray:
    """Feedback gain K (1x5) on (position error, speed error, pitch, pitch rate, integral of position error), volts per unit."""
    w = weights or LqrWeights.for_robot(params)
    a, b = linear_pitch_model(params)
    dt = 1.0 / rate
    # augment with the integral of the position: z' = position
    a_aug = np.zeros((5, 5))
    a_aug[:4, :4] = a
    a_aug[4, 0] = 1.0
    b_aug = np.vstack([b, [[0.0]]])
    ad, bd = discretize(a_aug, b_aug, dt)
    q = np.diag([w.position, w.speed, w.pitch, w.pitch_rate, w.integral])
    return solve_dare(ad, bd, q, np.array([[w.voltage]]))


# --- the firmware -------------------------------------------------------------------------------


@dataclass
class Command:
    speed: float = 0.0  # m/s along the floor
    yaw_rate: float = 0.0  # rad/s, positive turns right


@dataclass
class FirmwareTelemetry:
    pitch: float = 0.0
    pitch_rate: float = 0.0
    position: float = 0.0
    speed: float = 0.0
    yaw_rate: float = 0.0
    voltage_left: float = 0.0
    voltage_right: float = 0.0
    state: str = "calibrating"


class BalanceFirmware:
    def __init__(self, nominal: RobotParams, encoder: EncoderSpec | None = None, weights: LqrWeights | None = None, calibration_time: float = 1.5) -> None:
        self.params = nominal
        self.encoder = encoder or EncoderSpec()
        self.dt = 1.0 / CONTROL_RATE
        self.gain = design_lqr(nominal, weights)[0]
        m = nominal.motor
        self.counts_per_wheel_radian = self.encoder.counts_per_motor_rev * m.gear_ratio / (2.0 * np.pi)
        self.steer_feed_forward = m.torque_constant * m.gear_ratio * (nominal.track / 2.0) / nominal.wheel_radius  # V per rad/s of yaw rate
        self.speed_feed_forward = m.torque_constant * m.gear_ratio / nominal.wheel_radius  # V per m/s: the back-EMF at that speed
        self.calibration_time = calibration_time
        # the pitch feeds the position (through the wheel angle), the position the acceleration estimate, and that the pitch:
        # the loop gain grows with the wheel radius, so the acceleration filter is slowed down for bigger wheels
        self.acceleration_gain = 0.06 * min(1.0, 0.04 / nominal.wheel_radius)
        self.accel_limit = 0.8  # m/s^2: the fastest the reference speed may change
        self.lean_gain = 0.6
        self.reset()

    def reset(self) -> None:
        self.state = "calibrating"
        self.time = 0.0
        self.gyro_bias = np.zeros(3)
        self._bias_sum = np.zeros(3)
        self._bias_count = 0
        self.pitch = 0.0
        self.pitch_initialised = False
        self.position = 0.0
        self.speed = 0.0
        self.acceleration = 0.0
        self.position_ref = 0.0
        self.speed_ref = 0.0
        self.integral = 0.0
        self.yaw_integral = 0.0
        self.pitch_rate = 0.0
        self.yaw_rate = 0.0
        self.counts_previous: np.ndarray | None = None
        self.upright_time = 0.0
        self.output = np.zeros(2)
        self.telemetry = FirmwareTelemetry()

    # --- estimation ----------------------------------------------------------------------------

    def _estimate_attitude(self, imu: ImuSample) -> None:
        gyro = imu.gyro - self.gyro_bias
        self.pitch_rate += 0.6 * (gyro[1] - self.pitch_rate)  # a light smoothing of the gyro for the D term
        self.yaw_rate = -gyro[2]  # the gyro turns counter-clockwise, the heading clockwise
        accel = imu.accel
        # the accelerometer feels the robot's own acceleration as well as gravity: take out the acceleration seen by the wheels
        pitch_from_gravity = float(np.arctan2(-(accel[0] - self.acceleration), accel[2]))
        if not self.pitch_initialised:
            self.pitch, self.pitch_initialised = pitch_from_gravity, True
            return
        # trust the accelerometer less when it does not measure 1 g: the robot is accelerating and the reading is not gravity
        excess = abs(float(np.linalg.norm(accel)) / GRAVITY - 1.0)
        tau = 0.7 + 12.0 * excess if self.state == "balancing" else 0.05  # motors off: nothing but gravity (and a hand) is acting, so follow the accelerometer quickly
        alpha = tau / (tau + self.dt)
        self.pitch = alpha * (self.pitch + gyro[1] * self.dt) + (1.0 - alpha) * pitch_from_gravity

    def _estimate_travel(self, counts: np.ndarray) -> None:
        wheel_angle = counts.astype(float) / self.counts_per_wheel_radian  # relative to the chassis
        mean_angle = float(np.mean(wheel_angle))
        measured_position = self.params.wheel_radius * (mean_angle + self.pitch)
        if self.counts_previous is None:
            self.position, self.speed = measured_position, 0.0
        else:
            # position-speed observer: the position is measured (coarsely), the speed follows from it
            error = measured_position - self.position
            previous_speed = self.speed
            self.position += self.speed * self.dt + 0.55 * error
            self.speed += 0.09 * error / self.dt
            self.acceleration += self.acceleration_gain * ((self.speed - previous_speed) / self.dt - self.acceleration)
        self.counts_previous = counts

    # --- control -------------------------------------------------------------------------------

    def update(self, imu: ImuSample, counts: np.ndarray, command: Command, picked_up: bool = False) -> np.ndarray:
        """One 200 Hz cycle: returns the duty (-1..1) of the left and right motor."""
        self.time += self.dt
        t = self.telemetry
        if self.state == "calibrating":
            self._bias_sum += imu.gyro * np.array([1.0, 1.0, 1.0])
            self._bias_count += 1
            self.gyro_bias = self._bias_sum / self._bias_count
            self._estimate_attitude(imu)
            self._estimate_travel(counts)
            if self.time >= self.calibration_time:
                self.state = "balancing"
                self.position_ref = self.position
            self.output = np.zeros(2)
            return self.output

        self._estimate_attitude(imu)
        self._estimate_travel(counts)

        if self.state == "cutoff":
            if abs(self.pitch) < np.radians(6.0) and abs(self.pitch_rate) < 0.3:
                self.upright_time += self.dt
            else:
                self.upright_time = 0.0
            if self.upright_time > 0.5:
                self.state = "balancing"
                self.integral = self.yaw_integral = 0.0
                self.speed_ref = 0.0
                self.position_ref = self.position
            self.output = np.zeros(2)
            self._fill(t)
            return self.output
        if abs(self.pitch) > np.radians(40.0):
            self.state = "cutoff"
            self.upright_time = 0.0
            self.output = np.zeros(2)
            self._fill(t)
            return self.output

        # reference speed with limited acceleration; the reference position follows it but never runs far ahead of the robot
        limit = self.accel_limit * self.dt
        step = float(np.clip(command.speed - self.speed_ref, -limit, limit))
        self.speed_ref += step
        pitch_ref = self.lean_gain * step / self.dt / GRAVITY  # lean into the acceleration being asked for
        self.position_ref += self.speed_ref * self.dt
        self.position_ref = float(np.clip(self.position_ref, self.position - 0.35, self.position + 0.35))

        position_error = self.position - self.position_ref
        speed_error = self.speed - self.speed_ref
        if abs(self.speed_ref) < 0.02 and abs(command.speed) < 0.02:  # the trim only integrates while standing still
            self.integral = float(np.clip(self.integral + position_error * self.dt, -0.5, 0.5))
        state = np.array([position_error, speed_error, self.pitch - pitch_ref, self.pitch_rate, self.integral])
        common = float(-self.gain @ state) + self.speed_feed_forward * self.speed_ref  # the feed-forward carries the back-EMF, so cruising does not need a speed error

        yaw_error = command.yaw_rate - self.yaw_rate
        self.yaw_integral = float(np.clip(self.yaw_integral + yaw_error * self.dt, -3.0, 3.0))
        differential = self.steer_feed_forward * command.yaw_rate + 1.0 * yaw_error + 3.0 * self.yaw_integral

        voltage = np.array([common + differential, common - differential])  # a right turn: the left wheel drives harder
        duty = np.clip(voltage / max(self.params.battery.cells * 4.0, 1.0), -1.0, 1.0)
        self.output = np.round(duty * 1023.0) / 1023.0
        t.voltage_left, t.voltage_right = float(voltage[0]), float(voltage[1])
        self._fill(t)
        return self.output

    def _fill(self, t: FirmwareTelemetry) -> None:
        t.pitch, t.pitch_rate, t.position, t.speed, t.yaw_rate, t.state = self.pitch, self.pitch_rate, self.position, self.speed, self.yaw_rate, self.state
