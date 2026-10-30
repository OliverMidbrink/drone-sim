"""Simulated flight sensors and a basic attitude / height estimator.

The drone carries an IMU (3-axis gyroscope + 3-axis accelerometer) and, optionally, a barometer.
No magnetometer and no GPS: heading comes from integrating the gyro alone.

`SensorSuite.sample` produces what a flight controller would read from a consumer MEMS IMU: turn-on
bias, bias drift (Gauss-Markov wander, random walk, temperature), scale-factor error, axis coupling,
gyro g-sensitivity, white noise, propeller vibration (broadband plus an aliased tone), quantization
and saturation. See SensorNoise.
`StateEstimator` is the stock fusion: it calibrates the gyro bias while the drone sits still at
power-up, integrates the gyro, and nudges roll and pitch toward the accelerometer's gravity
direction (a Mahony complementary filter). Height comes from the barometer blended with the
accelerometer. Without a barometer, height and climb rate are plain integrals and drift.
vio.py replaces it with an estimator that also uses a camera SLAM.
The flight controller needs `q`, `gyro_bias`, `altitude`, `climb_rate`, `position` and `velocity`
(the last two may be None when the estimator has none).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

import numpy as np

from drone import GRAVITY, Quadrotor, matrix_to_quat, quat_multiply, quat_to_matrix


@dataclass
class SensorSample:
    t: float  # seconds of simulated time
    dt: float  # since the previous sample
    gyro: np.ndarray  # rad/s, body frame
    accel: np.ndarray  # m/s^2 specific force (what an accelerometer feels: +9.81 up when resting), body frame
    baro: float | None = None  # metres above the start, noisy; None when the drone has no barometer


DEG = np.pi / 180.0
G_UNIT = GRAVITY  # 1 g in m/s^2


@dataclass
class SensorNoise:
    """Error budget of a consumer MEMS IMU (MPU-6000 / ICM-20602 class) in a vibrating quadrotor, all 1-sigma unless noted.

    Everything that makes a real IMU wrong is here. The first group is random per power-up (each
    run draws its own sensor), the second evolves while flying, the third is what the electronics do
    to the signal. `scaled(k)` multiplies the errors by k (0 = a perfect IMU, <1 a better one).
    """

    # --- fixed at power-up: a different sensor every time
    gyro_bias: float = 0.012  # rad/s turn-on offset (0.7 deg/s)
    accel_bias: float = 0.06  # m/s^2 turn-on offset (6 mg)
    gyro_scale: float = 0.005  # scale-factor error, 0.5 %
    accel_scale: float = 0.008
    gyro_misalignment: float = 0.003  # rad, cross-axis coupling between every pair of axes (0.17 deg)
    accel_misalignment: float = 0.003
    gyro_g_sensitivity: float = 1.2e-4  # rad/s of gyro error per m/s^2 of specific force (about 0.07 deg/s per g)
    gyro_tempco: float = 2.6e-4  # rad/s per degree C (0.015 deg/s/C)
    accel_tempco: float = 0.0025  # m/s^2 per degree C (0.25 mg/C)
    # --- drift while flying
    gyro_bias_instability: float = 2.5e-4  # rad/s, wandering bias (Gauss-Markov, ~50 deg/h)
    accel_bias_instability: float = 0.012  # m/s^2
    bias_correlation_time: float = 300.0  # s (not scaled)
    gyro_walk: float = 5e-5  # rad/s per sqrt(s), rate random walk
    accel_walk: float = 5e-4  # m/s^2 per sqrt(s)
    warm_up: float = 14.0  # degrees C the electronics heat above ambient at full throttle (not scaled)
    thermal_time: float = 90.0  # s (not scaled)
    # --- noise, vibration, converters
    gyro_density: float = 0.012 * DEG  # rad/s per sqrt(Hz) (0.012 deg/s/rtHz)
    accel_density: float = 400e-6 * G_UNIT  # m/s^2 per sqrt(Hz) (400 ug/rtHz)
    bandwidth: float = 125.0  # Hz of the digital low-pass (not scaled)
    vibration: float = 0.35  # m/s^2 of accelerometer shake at full throttle: broadband and a tone at the propeller rate
    gyro_vibration: float = 0.012  # rad/s of the same on the gyro
    gyro_lsb: float = 2000.0 * DEG / 32768.0  # rad/s: 16 bit over +-2000 deg/s (not scaled)
    accel_lsb: float = 16.0 * G_UNIT / 32768.0  # m/s^2: 16 bit over +-16 g (not scaled)
    gyro_range: float = 2000.0 * DEG  # saturation (not scaled)
    accel_range: float = 16.0 * G_UNIT
    baro_noise: float = 0.12  # m
    # --- a factory-calibrated accelerometer (SensorSuite(accel_factory=True)): what is left of its static errors after a calibration
    accel_cal_gain: float = 0.025  # axis gain error, 2.5 %: the calibration is good to 2-3 %
    accel_cal_cross: float = 0.005  # axis coupling, rad
    accel_cal_bias: float = 0.015  # m/s^2 offset (1.5 mg) at the calibration temperature

    _NOT_SCALED = ("bias_correlation_time", "warm_up", "thermal_time", "bandwidth", "gyro_lsb", "accel_lsb", "gyro_range", "accel_range")

    def scaled(self, k: float) -> "SensorNoise":
        values = {f.name: getattr(self, f.name) * (1.0 if f.name in self._NOT_SCALED else k) for f in fields(self)}
        return SensorNoise(**values)


class SensorSuite:
    AMBIENT = 25.0  # degrees C; the sensors' temperature coefficients are zero here

    def __init__(self, noise: SensorNoise | None = None, seed: int = 0, barometer: bool = True, accel_factory: bool = False) -> None:
        self.noise = noise or SensorNoise()
        self.rng = np.random.default_rng(seed)
        self.calibration_rng = np.random.default_rng(seed + 7919)  # its own stream: the gyro and everything else draw the same sensor with or without calibration
        self.barometer = barometer
        self.accel_factory = accel_factory
        self.reset(np.zeros(3))

    def _misalignment(self, scale_sigma: float, cross_sigma: float, rng: np.random.Generator | None = None) -> np.ndarray:
        """Scale-factor error on the diagonal and axis coupling off it: the matrix between the true signal and what the chip reports."""
        rng = rng or self.rng
        matrix = np.eye(3) + np.diag(rng.normal(0.0, scale_sigma, 3))
        off = ~np.eye(3, dtype=bool)
        matrix[off] = rng.normal(0.0, cross_sigma, 6)
        return matrix

    def reset(self, velocity: np.ndarray) -> None:
        n, rng = self.noise, self.rng
        self.gyro_turn_on = rng.normal(0.0, n.gyro_bias, 3)
        self.accel_turn_on = rng.normal(0.0, n.accel_bias, 3)
        self.gyro_matrix = self._misalignment(n.gyro_scale, n.gyro_misalignment)
        self.accel_matrix = self._misalignment(n.accel_scale, n.accel_misalignment)
        self.gyro_g_matrix = rng.normal(0.0, n.gyro_g_sensitivity, (3, 3))
        if self.accel_factory:
            # Calibrated once, at the factory, and never touched again: the accelerometer's turn-on errors are replaced by what a
            # 2-3 % calibration leaves. The drift (temperature, wander, walk) and the noise below are still the chip's own.
            self.accel_matrix = self._misalignment(n.accel_cal_gain, n.accel_cal_cross, self.calibration_rng)
            self.accel_turn_on = self.calibration_rng.normal(0.0, n.accel_cal_bias, 3)
        self.gyro_tempco = rng.normal(0.0, n.gyro_tempco, 3)
        self.accel_tempco = rng.normal(0.0, n.accel_tempco, 3)
        self.gyro_wander = rng.normal(0.0, n.gyro_bias_instability, 3)  # Gauss-Markov, started in its stationary distribution
        self.accel_wander = rng.normal(0.0, n.accel_bias_instability, 3)
        self.gyro_walk = np.zeros(3)
        self.accel_walk = np.zeros(3)
        self.temperature = self.AMBIENT + rng.normal(0.0, 2.0)
        self.vibration_phase = rng.uniform(0.0, 2.0 * np.pi, 6)
        self.last_velocity = np.array(velocity, dtype=float)
        self.last_time: float | None = None
        self.last_force = np.zeros(3)  # the true specific force of the latest sample (for logs)

    @property
    def gyro_bias(self) -> np.ndarray:
        """The gyro's current total offset: what it would read standing still (for inspection only)."""
        return self.gyro_turn_on + self.gyro_wander + self.gyro_walk + self.gyro_tempco * (self.temperature - self.AMBIENT)

    @property
    def accel_bias(self) -> np.ndarray:
        return self.accel_turn_on + self.accel_wander + self.accel_walk + self.accel_tempco * (self.temperature - self.AMBIENT)

    def describe(self) -> str:
        """One line on the sensor this run drew."""
        g, a = self.gyro_matrix, self.accel_matrix
        return (
            f"IMU drawn for this run: gyro turn-on bias {np.round(self.gyro_turn_on / DEG, 2)} deg/s, scale error {np.round(100 * (np.diag(g) - 1), 2)} %, "
            f"max axis coupling {np.degrees(np.abs(g - np.diag(np.diag(g))).max()):.2f} deg; accel turn-on bias {np.round(self.accel_turn_on / G_UNIT * 1000, 0)} mg, "
            f"scale error {np.round(100 * (np.diag(a) - 1), 2)} %, start temperature {self.temperature:.1f} C"
        )

    def sample(self, quad: Quadrotor) -> SensorSample:
        n, rng = self.noise, self.rng
        dt = quad.time - self.last_time if self.last_time is not None else 0.0
        throttle = float(np.mean(quad.motors))
        if dt > 0.0:
            decay = np.exp(-dt / n.bias_correlation_time)  # the wandering part of each bias: first-order Gauss-Markov
            kick = np.sqrt(1.0 - decay * decay)
            self.gyro_wander = decay * self.gyro_wander + kick * rng.normal(0.0, n.gyro_bias_instability, 3)
            self.accel_wander = decay * self.accel_wander + kick * rng.normal(0.0, n.accel_bias_instability, 3)
            self.gyro_walk += rng.normal(0.0, n.gyro_walk * np.sqrt(dt), 3)
            self.accel_walk += rng.normal(0.0, n.accel_walk * np.sqrt(dt), 3)
            target = self.AMBIENT + n.warm_up * throttle
            self.temperature += (target - self.temperature) * (1.0 - np.exp(-dt / n.thermal_time))
            motor_hz = float(np.mean(quad.rpm)) / 60.0
            self.vibration_phase += 2.0 * np.pi * motor_hz * dt * np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])  # sampled at 250 Hz: aliases like the real thing
            acceleration = (quad.velocity - self.last_velocity) / dt  # includes the jolt of any collision
        else:
            acceleration = np.zeros(3)
        self.last_velocity, self.last_time = quad.velocity.copy(), quad.time

        force = quad.rotation.T @ (acceleration + np.array([0.0, GRAVITY, 0.0]))  # true specific force, body frame
        self.last_force = force
        shake = n.vibration * throttle**2
        gyro_shake = n.gyro_vibration * throttle**2
        tone = np.sin(self.vibration_phase[:3]) * np.array([1.0, 0.7, 1.3]) + 0.4 * np.sin(self.vibration_phase[3:])
        white = np.sqrt(n.bandwidth)
        gyro = (
            self.gyro_matrix @ quad.omega
            + self.gyro_g_matrix @ force
            + self.gyro_bias
            + rng.normal(0.0, n.gyro_density * white + 0.3 * gyro_shake, 3)
            + gyro_shake * tone[[1, 2, 0]]
        )
        accel = (
            self.accel_matrix @ force
            + self.accel_bias
            + rng.normal(0.0, n.accel_density * white + 0.5 * shake, 3)
            + shake * tone
        )
        if n.gyro_lsb > 0.0:  # the converter: round to its step and clip at its range
            gyro = np.clip(np.round(gyro / n.gyro_lsb) * n.gyro_lsb, -n.gyro_range, n.gyro_range)
            accel = np.clip(np.round(accel / n.accel_lsb) * n.accel_lsb, -n.accel_range, n.accel_range)
        return SensorSample(
            t=quad.time,
            dt=dt,
            gyro=gyro,
            accel=accel,
            baro=float(quad.position[1] + rng.normal(0.0, n.baro_noise)) if self.barometer else None,
        )


def level_orientation(up_in_body: np.ndarray) -> np.ndarray:
    """Quaternion (body -> world) of a drone whose accelerometer says `up_in_body`, with the nose defining heading 0.

    The world has y up and heading 0 along -z, the same axes the physics uses, but a drone with no
    compass can only say "forward is where I was pointing at power-up".
    """
    up = up_in_body / np.linalg.norm(up_in_body)
    nose = np.array([0.0, 0.0, -1.0])
    forward = nose - np.dot(nose, up) * up
    forward /= np.linalg.norm(forward)
    world_z = -forward
    world_x = np.cross(up, world_z)
    return matrix_to_quat(np.stack([world_x, up, world_z]))  # rows: world axes in body coordinates


@dataclass
class StateEstimator:
    """Gyro + accelerometer attitude, and height from the barometer (when there is one)."""

    calibration_time: float = 0.6  # s sitting still at power-up to learn the gyro bias and which way is up
    kp: float = 0.06  # 1/s, pull of the accelerometer on roll and pitch. Small: a quadrotor's accelerometer
    ki: float = 0.002  # only points up when it is not accelerating, so it is trusted slowly
    baro_position_gain: float = 5.0  # 1/s
    baro_velocity_gain: float = 6.25  # 1/s^2
    q: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    altitude: float = 0.0
    climb_rate: float = 0.0
    position: np.ndarray | None = None  # world position, only estimators with a position source set it
    velocity: np.ndarray | None = None
    initialised: bool = False

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.initialised = False
        self.gyro_bias = np.zeros(3)  # estimated sensor bias: the true rate is gyro - gyro_bias
        self._integral = np.zeros(3)
        self._cal_time = 0.0
        self._cal_gyro = np.zeros(3)
        self._cal_accel = np.zeros(3)
        self._cal_n = 0
        self._baro_zero: float | None = None

    def _calibrate(self, sample: SensorSample) -> None:
        self._cal_gyro += sample.gyro
        self._cal_accel += sample.accel
        self._cal_n += 1
        self._cal_time += sample.dt
        if self._cal_time >= self.calibration_time and self._cal_n >= 10:
            self.gyro_bias = self._cal_gyro / self._cal_n  # what the gyro reads while the drone is not rotating
            self.q = level_orientation(self._cal_accel / self._cal_n)
            self._integral = -self.gyro_bias.copy()
            self.altitude, self.climb_rate = (sample.baro if sample.baro is not None else 0.0), 0.0
            self.initialised = True

    def update(self, sample: SensorSample) -> None:
        if not self.initialised:
            self._calibrate(sample)
            return
        dt = sample.dt
        if dt <= 0.0:
            return
        self.propagate_attitude(sample)
        self.update_vertical(sample)

    def propagate_attitude(self, sample: SensorSample) -> None:
        dt = sample.dt
        rotation = quat_to_matrix(self.q)
        error = np.zeros(3)
        accel_norm = float(np.linalg.norm(sample.accel))
        if accel_norm > 1e-3:
            trust = 1.0 - min(abs(accel_norm - GRAVITY) / (0.3 * GRAVITY), 1.0)  # not while the reading is implausible
            error += trust * np.cross(sample.accel / accel_norm, rotation[1, :])
        self._integral += self.ki * error * dt
        self.gyro_bias = -self._integral
        corrected = sample.gyro + self.kp * error + self._integral
        dq = 0.5 * quat_multiply(self.q, np.array([0.0, *corrected])) * dt
        self.q = (self.q + dq) / np.linalg.norm(self.q + dq)

    def update_vertical(self, sample: SensorSample) -> None:
        dt = sample.dt
        rotation = quat_to_matrix(self.q)
        vertical_acceleration = float((rotation @ sample.accel)[1]) - GRAVITY
        self.climb_rate += vertical_acceleration * dt
        self.altitude += self.climb_rate * dt
        if sample.baro is not None:
            baro_error = sample.baro - self.altitude
            self.altitude += self.baro_position_gain * baro_error * dt
            self.climb_rate += self.baro_velocity_gain * baro_error * dt

    @property
    def rotation(self) -> np.ndarray:
        return quat_to_matrix(self.q)
