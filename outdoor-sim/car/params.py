"""The car's numbers: a half-scale RC buggy (a compact car at 1/2 scale) and the settings the player can change.

Body frame: x forward, y up, z to the right, origin at the centre of mass. Wheels are numbered
FL, FR, RL, RR. Suspension travel `s` is how far a hub is pushed up relative to the body, 0 = fully
extended, `travel` = on the bump stop; the car rests in the middle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

G = 9.81
DRIVETRAINS = ("awd", "fwd", "rwd")
WHEEL_NAMES = ("FL", "FR", "RL", "RR")


@dataclass
class CarParams:
    # body
    sprung_mass: float = 140.0  # kg: body, battery, motor, rider-less
    wheel_mass: float = 3.0  # kg each: wheel, tyre, hub, brake, share of the arms (unsprung)
    inertia: tuple[float, float, float] = (13.0, 54.0, 52.0)  # kg m^2 about x (roll), y (yaw), z (pitch)
    com_height: float = 0.28  # m above the ground at rest
    front_weight: float = 0.5  # share of the weight on the front axle
    body_half: tuple[float, float, float] = (0.95, 0.24, 0.27)  # half length, half height, half width of the body box
    body_centre_y: float = 0.06  # m, the box's centre above the centre of mass: the floor is 0.10 m above the ground
    # wheels and tyres
    wheelbase: float = 1.30
    track: float = 0.84
    wheel_radius: float = 0.165
    wheel_width: float = 0.11
    wheel_inertia: float = 0.05  # kg m^2 about the axle
    tyre_stiffness: float = 100e3  # N/m
    tyre_damping: float = 300.0  # N s/m
    grip: float = 1.05  # peak friction coefficient on asphalt
    grass_grip: float = 0.6  # share of that on grass
    rolling_resistance: float = 0.015  # on asphalt; 0.05 more on grass
    # suspension: one parallel-arm (parallelogram) axle per wheel, so the wheel stays upright while it travels
    travel: float = 0.14  # m
    arm_length: float = 0.26  # m, of the two parallel arms
    spring_rate: float = 6000.0  # N/m (about 2 Hz ride)
    damping_compression: float = 330.0  # N s/m
    damping_rebound: float = 470.0
    stop_stiffness: float = 60e3  # N/m of the bump stops (and the droop straps)
    stop_damping: float = 800.0
    anti_roll: float = 2500.0  # N/m, between the two wheels of an axle
    # driveline and steering
    thrust: float = 0.45  # forward force at full power, in weights of the car (shared by the driven wheels)
    top_speed: float = 24.0  # m/s; the motor runs out of torque here
    reverse_speed: float = 6.0
    brake: float = 1.1  # braking force in weights, 60 % on the front axle
    handbrake_torque: float = 400.0  # N m on each rear wheel
    steer_max: float = 0.62  # rad at the wheels at standstill
    steer_fade: float = 20.0  # m/s: the steering angle limit falls as 1 / (1 + (speed / steer_fade)^2)
    assist_gain: float = 1.0  # rad of counter-steer per rad the front axle slides sideways
    assist_fade: float = 30.0  # deg of front slide at which the player's steering is down to a quarter
    assist_slide: float = 20.0  # deg of slip angle above which the assist starts to lift the throttle
    assist_slide_range: float = 25.0  # deg further on it is down to 15 %
    steer_rate: float = 2.5  # rad/s the wheels turn at when a key is pressed (they return 1.6x as fast)
    drag_area: float = 0.16  # Cd * frontal area, m^2

    @property
    def mass(self) -> float:
        return self.sprung_mass + 4.0 * self.wheel_mass

    @property
    def hub_y(self) -> float:
        """Hub height relative to the centre of mass at mid travel."""
        deflection = 0.25 * self.mass * G / self.tyre_stiffness
        return -(self.com_height - (self.wheel_radius - deflection))

    @property
    def wheel_x(self) -> np.ndarray:
        """Hub x of each wheel in the body frame, from the weight distribution (the centre of mass is the origin)."""
        front = self.wheelbase * (1.0 - self.front_weight)
        return np.array([front, front, front - self.wheelbase, front - self.wheelbase])

    @property
    def corner_load(self) -> np.ndarray:
        """Sprung weight carried by each spring at rest, N."""
        axle = np.array([self.front_weight, self.front_weight, 1 - self.front_weight, 1 - self.front_weight]) / 2.0
        return self.sprung_mass * G * axle


@dataclass
class Tuning:
    """What the player changes while driving; all multipliers except the drivetrain."""

    drivetrain: str = "awd"
    power: float = 1.0
    grip: float = 1.0
    damping: float = 1.0
    stiffness: float = 1.0

    LIMITS = {"power": (0.2, 3.0), "grip": (0.2, 2.0), "damping": (0.1, 4.0), "stiffness": (0.3, 3.0)}

    def nudge(self, name: str, direction: int, factor: float = 1.12) -> None:
        low, high = self.LIMITS[name]
        value = getattr(self, name) * factor**direction
        setattr(self, name, float(np.clip(round(value, 3), low, high)))

    def cycle_drivetrain(self) -> None:
        self.drivetrain = DRIVETRAINS[(DRIVETRAINS.index(self.drivetrain) + 1) % len(DRIVETRAINS)]

    def reset(self) -> None:
        self.drivetrain, self.power, self.grip, self.damping, self.stiffness = "awd", 1.0, 1.0, 1.0, 1.0


@dataclass
class Controls:
    throttle: float = 0.0  # -1 (reverse) .. 1
    brake: float = 0.0  # 0 .. 1
    steer: float = 0.0  # -1 (left) .. 1 (right)
    handbrake: bool = False
    assist: bool = True  # counter-steer: the wheels follow the direction the car is sliding in, on top of the player's steering


@dataclass
class Telemetry:
    """What the last physics step produced, for the HUD and the skid marks."""

    travel: np.ndarray = field(default_factory=lambda: np.zeros(4))
    load: np.ndarray = field(default_factory=lambda: np.zeros(4))  # N, tyre normal force
    slip: np.ndarray = field(default_factory=lambda: np.zeros(4))  # combined slip of each tyre (0 rolling, ~0.15 at the grip limit, >0.5 sliding)
    contact: np.ndarray = field(default_factory=lambda: np.zeros((4, 3)))  # world position of each tyre's contact
    touching: np.ndarray = field(default_factory=lambda: np.zeros(4, bool))
    tread: np.ndarray = field(default_factory=lambda: np.zeros((4, 3)))  # world position of the middle of each tyre's contact patch
    axle: np.ndarray = field(default_factory=lambda: np.tile([0.0, 0.0, 1.0], (4, 1)))
    normal: np.ndarray = field(default_factory=lambda: np.tile([0.0, 1.0, 0.0], (4, 1)))
    body_acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))  # in the body frame, m/s^2 without gravity
