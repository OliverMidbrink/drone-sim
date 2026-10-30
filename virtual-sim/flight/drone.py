"""Quadrotor rigid-body physics: four thrusters with motor lag, gravity, drag and collisions.

Body frame: x right, y up (the thrust axis), z back, so the nose points along -z, like the camera.
Motors (X layout, seen from above):     0 front-left   1 front-right
                                        2 back-left    3 back-right
Positive body rates follow the right-hand rule: +x pitches the nose up, +y turns left, +z rolls left.

The mixer matrix is the single source of truth for how motor thrusts become total thrust and
torques; the physics uses it forwards and the flight controller inverts it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from collision import CollisionWorld

GRAVITY = 9.81
UP = np.array([0.0, 1.0, 0.0])


@dataclass
class QuadParams:
    mass: float = 0.9  # kg
    arm: float = 0.13  # m, centre to each motor
    inertia: tuple[float, float, float] = (0.0060, 0.0100, 0.0060)  # kg m^2 about body x (pitch), y (yaw), z (roll)
    max_thrust: float = 9.0  # N per motor: about 4:1 thrust to weight
    motor_tau: float = 0.03  # s, how fast a motor's speed follows its command
    yaw_coefficient: float = 0.016  # m, reaction torque per newton of thrust
    drag_quadratic: float = 0.015  # N / (m/s)^2
    drag_linear: float = 0.03  # N / (m/s)
    angular_drag: float = 0.002  # N m / (rad/s)
    radius: float = 0.18  # m, collision sphere around the body: the propeller tips and the landing legs touch it
    restitution: float = 0.25
    crash_speed: float = 4.5  # m/s impact that kills the motors
    max_rpm: float = 28000.0  # only for display


def mixer_matrix(params: QuadParams) -> np.ndarray:
    """Rows: total thrust, torque about x, about y, about z; columns: the four motors' thrusts."""
    d = params.arm / np.sqrt(2.0)
    c = params.yaw_coefficient
    return np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [d, d, -d, -d],  # front motors up -> nose up
            [c, -c, -c, c],  # reaction torque of the spinning props
            [-d, d, -d, d],  # right motors up -> roll left
        ]
    )


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw])


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix taking body coordinates to world coordinates."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat(m: np.ndarray) -> np.ndarray:
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    return q / np.linalg.norm(q)


def quat_from_yaw(yaw: float) -> np.ndarray:
    """Level orientation facing `yaw` (radians, turning right is positive, 0 faces -z like the Camera)."""
    half = -yaw / 2.0
    return np.array([np.cos(half), 0.0, np.sin(half), 0.0])


def attitude_angles(rotation: np.ndarray) -> tuple[float, float, float]:
    """(roll, pitch, yaw) in radians with the Camera's conventions: right / up / right are positive."""
    forward, right = -rotation[:, 2], rotation[:, 0]
    pitch = float(np.arcsin(np.clip(forward[1], -1.0, 1.0)))
    yaw = float(np.arctan2(forward[0], -forward[2]))
    level_right = np.cross(forward, UP)
    norm = np.linalg.norm(level_right)
    if norm < 1e-6:
        return 0.0, pitch, yaw
    level_right /= norm
    level_up = np.cross(level_right, forward)
    return float(np.arctan2(-np.dot(right, level_up), np.dot(right, level_right))), pitch, yaw


def wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class Impact:
    time: float
    kind: str
    speed: float  # m/s into the obstacle


@dataclass
class Quadrotor:
    params: QuadParams = field(default_factory=QuadParams)

    def __post_init__(self) -> None:
        self.mixer = mixer_matrix(self.params)
        self.mixer_inverse = np.linalg.inv(self.mixer)
        self.inertia = np.array(self.params.inertia)
        self.reset(np.zeros(3), 0.0)

    def reset(self, position: np.ndarray, yaw: float) -> None:
        self.position = np.array(position, dtype=float)
        self.velocity = np.zeros(3)
        self.wind = np.zeros(3)  # world-frame air velocity, set by whoever simulates weather
        self.q = quat_from_yaw(yaw)
        self.omega = np.zeros(3)  # body rates, rad/s
        self.motors = np.zeros(4)  # normalised motor speeds 0..1, thrust = max_thrust * speed^2
        self.thrust = np.zeros(4)
        self.time = 0.0
        self.crashed = False
        self.on_ground = False
        self.impacts: list[Impact] = []

    @property
    def rotation(self) -> np.ndarray:
        return quat_to_matrix(self.q)

    @property
    def rpm(self) -> np.ndarray:
        return self.motors * self.params.max_rpm

    def step(self, dt: float, thrust_command: np.ndarray, collisions: CollisionWorld | None = None, bound: float = np.inf) -> None:
        """Advance by dt seconds with the four motors commanded to these thrusts (newtons)."""
        p = self.params
        command = np.zeros(4) if self.crashed else np.clip(thrust_command, 0.0, p.max_thrust)
        self.motors += (np.sqrt(command / p.max_thrust) - self.motors) * (1.0 - np.exp(-dt / p.motor_tau))
        self.thrust = p.max_thrust * self.motors**2
        total, torque = self.mixer @ self.thrust, None
        rotation = quat_to_matrix(self.q)

        airspeed = self.velocity - self.wind
        speed = float(np.linalg.norm(airspeed))
        force = rotation[:, 1] * total[0] - (p.drag_quadratic * speed + p.drag_linear) * airspeed
        self.velocity = self.velocity + (force / p.mass + np.array([0.0, -GRAVITY, 0.0])) * dt
        torque = total[1:] - np.cross(self.omega, self.inertia * self.omega) - p.angular_drag * self.omega
        self.omega = self.omega + torque / self.inertia * dt
        self.position = self.position + self.velocity * dt
        dq = 0.5 * quat_multiply(self.q, np.array([0.0, *self.omega])) * dt
        self.q = (self.q + dq) / np.linalg.norm(self.q + dq)
        self.time += dt

        self.on_ground = False
        for axis in (0, 2):  # the edge of the map is a wall
            if abs(self.position[axis]) > bound:
                self.position[axis] = np.sign(self.position[axis]) * bound
                self.velocity[axis] = 0.0
        if collisions is not None:
            for contact in collisions.resolve(self.position, p.radius):
                self._collide(contact)

    def _collide(self, contact) -> None:
        p = self.params
        self.position = self.position + contact.normal * contact.depth
        into = -float(np.dot(self.velocity, contact.normal))
        if contact.kind == "ground":
            self.on_ground = True
        if into <= 0.0:
            return
        bounce = p.restitution if into > 1.0 else 0.0  # no jitter when resting
        sideways = self.velocity - np.dot(self.velocity, contact.normal) * contact.normal
        friction = 0.3 if contact.kind == "ground" else 0.05
        self.velocity = self.velocity - (1.0 + bounce) * np.dot(self.velocity, contact.normal) * contact.normal - sideways * friction
        self.omega = self.omega * 0.85
        if into > 1.0:
            self.impacts.append(Impact(self.time, contact.kind, into))
            if into >= p.crash_speed:
                self.crashed = True
