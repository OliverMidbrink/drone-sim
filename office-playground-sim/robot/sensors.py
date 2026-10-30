"""What the robot's electronics measure: a 6-axis MEMS IMU (MPU-6050 class) and quadrature encoders on the motor shafts.

The IMU model carries the errors that matter to a balancing robot: turn-on bias and slowly wandering
bias, scale-factor error, a mounting tilt (the chip is never exactly square to the chassis), white
noise, motor vibration that grows with wheel speed, the chip's digital low-pass filter (which is also
a delay), and quantization. The encoders count quarter-phase edges, so speed from them is coarse at
low speed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dynamics import TwipPlant
from .params import GRAVITY

DEG = np.pi / 180.0


@dataclass
class ImuSpec:
    rate: float = 1000.0  # Hz, output data rate
    filter_hz: float = 94.0  # digital low-pass corner (a one-pole stand-in for the chip's DLPF)
    gyro_noise_density: float = 0.005 * DEG  # rad/s per sqrt(Hz)
    accel_noise_density: float = 400e-6 * GRAVITY  # m/s^2 per sqrt(Hz)
    gyro_turn_on_bias: float = 0.04  # rad/s (about 2.3 deg/s), different every power-up
    gyro_bias_walk: float = 2.0e-4  # rad/s per sqrt(s)
    accel_turn_on_bias: float = 0.08  # m/s^2
    scale_error: float = 0.005
    mount_tilt: float = 0.5 * DEG  # rad, 1-sigma, the chip's mounting angle error about each axis
    vibration_accel: float = 0.25  # m/s^2 of motor and gearbox shake at standstill ...
    vibration_per_speed: float = 0.5  # ... plus this much per m/s of travel
    vibration_gyro: float = 0.004  # rad/s
    gyro_range: float = 500.0 * DEG
    accel_range: float = 4.0 * GRAVITY

    def scaled(self, k: float) -> "ImuSpec":
        """All error sources multiplied by k (0 = an ideal sensor)."""
        names = ("gyro_noise_density", "accel_noise_density", "gyro_turn_on_bias", "gyro_bias_walk", "accel_turn_on_bias", "scale_error", "mount_tilt", "vibration_accel", "vibration_per_speed", "vibration_gyro")
        values = {name: getattr(self, name) * k for name in names}
        return ImuSpec(**{**self.__dict__, **values})


@dataclass
class ImuSample:
    t: float
    gyro: np.ndarray  # rad/s in the chip's frame (x forward, y left, z up)
    accel: np.ndarray  # m/s^2 specific force: +g on z when the robot stands still


class Imu:
    def __init__(self, spec: ImuSpec | None = None, seed: int = 0) -> None:
        self.spec = spec or ImuSpec()
        self.rng = np.random.default_rng(seed)
        s, rng = self.spec, self.rng
        self.gyro_bias = rng.normal(0.0, s.gyro_turn_on_bias, 3)
        self.accel_bias = rng.normal(0.0, s.accel_turn_on_bias, 3)
        self.gyro_scale = 1.0 + rng.normal(0.0, s.scale_error, 3)
        self.accel_scale = 1.0 + rng.normal(0.0, s.scale_error, 3)
        tilt = rng.normal(0.0, s.mount_tilt, 3)
        self.mount = _rotation(tilt)  # chip frame = mount @ body frame
        self.state = np.zeros((2, 3))  # low-pass states (gyro, accel)
        self.primed = False
        self.dt = 1.0 / s.rate

    def sample(self, plant: TwipPlant) -> ImuSample:
        s, rng = self.spec, self.rng
        force, rates = plant.imu_truth()
        speed = abs(plant.speed)
        self.gyro_bias += rng.normal(0.0, s.gyro_bias_walk * np.sqrt(self.dt), 3)
        bandwidth = s.rate / 2.0
        shake = s.vibration_accel + s.vibration_per_speed * speed
        gyro = self.mount @ rates * self.gyro_scale + self.gyro_bias + rng.normal(0.0, s.gyro_noise_density * np.sqrt(bandwidth) + s.vibration_gyro * (1.0 + speed), 3)
        accel = self.mount @ force * self.accel_scale + self.accel_bias + rng.normal(0.0, s.accel_noise_density * np.sqrt(bandwidth) + 0.3 * shake, 3)
        raw = np.stack([gyro, accel])
        if not self.primed:
            self.state, self.primed = raw.copy(), True
        alpha = 1.0 - np.exp(-2.0 * np.pi * s.filter_hz * self.dt)
        self.state += alpha * (raw - self.state)
        gyro_out = np.round(np.clip(self.state[0], -s.gyro_range, s.gyro_range) / (s.gyro_range / 32768.0)) * (s.gyro_range / 32768.0)
        accel_out = np.round(np.clip(self.state[1], -s.accel_range, s.accel_range) / (s.accel_range / 32768.0)) * (s.accel_range / 32768.0)
        return ImuSample(plant.time, gyro_out, accel_out)


def _rotation(angles: np.ndarray) -> np.ndarray:
    ax, ay, az = angles
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return rz @ ry @ rx


@dataclass
class EncoderSpec:
    counts_per_motor_rev: int = 44  # 11 pulses per revolution, both channels, both edges


class Encoders:
    """Incremental counts of each motor shaft relative to the chassis."""

    def __init__(self, plant: TwipPlant, spec: EncoderSpec | None = None) -> None:
        self.spec = spec or EncoderSpec()
        self.plant = plant
        self.offset = np.zeros(2)
        self.zero()

    def zero(self) -> None:
        self.offset = self._angles()

    def _angles(self) -> np.ndarray:
        p = self.plant
        return p.n * (p.q[3:5] - p.q[2])

    def counts(self) -> np.ndarray:
        """Integer counts of the left and right motor, positive when the wheel turns forward relative to the chassis."""
        return np.floor((self._angles() - self.offset) * self.spec.counts_per_motor_rev / (2.0 * np.pi)).astype(np.int64)

    @property
    def counts_per_wheel_radian(self) -> float:
        return self.spec.counts_per_motor_rev * self.plant.n / (2.0 * np.pi)
