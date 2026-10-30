"""Physical parameters of the self-balancing robot: chassis, wheels, geared DC motors, battery, floor.

The defaults describe a hobby-size robot: a 1 kg body on two 80 mm wheels driven by 12 V brushed
gearmotors (37 mm "D" type, 30:1) from a 3S lithium pack. Everything here is SI.

Frames. Plan coordinates are the world's x and z (y is up). The heading psi points the robot along
(cos psi, sin psi) in (x, z); with y up that is a clockwise turn seen from above, so a positive yaw
rate turns the robot to its right. The pitch theta is positive when the body leans forward.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

GRAVITY = 9.81
_CELL_CURVE = ((0.0, 3.00), (0.05, 3.35), (0.15, 3.60), (0.5, 3.78), (0.85, 4.08), (1.0, 4.20))  # (charge, volts per cell) of a Li-ion cell at rest


@dataclass
class MotorParams:
    """A brushed DC motor with a gearbox, as the datasheet gives it, plus the friction a datasheet leaves out."""

    resistance: float = 2.2  # ohm, winding plus the H-bridge's on-resistance
    inductance: float = 1.2e-3  # H
    torque_constant: float = 0.0105  # N m / A at the rotor (equals the back-EMF constant in V s / rad)
    rotor_inertia: float = 3.0e-6  # kg m^2: small, but reflected through the gearbox it is N^2 times bigger
    gear_ratio: float = 30.0
    rotor_friction: float = 2.4e-6  # N m s / rad of viscous drag at the rotor
    coulomb_friction: float = 0.06  # N m at the output shaft: brush and gearbox stiction. This is the dead band the controller fights
    gearbox_viscous: float = 4.0e-4  # N m s / rad at the output shaft
    stall_current_limit: float = 6.0  # A: the driver's current limit
    friction_speed: float = 0.05  # rad/s over which Coulomb friction ramps from zero to full


@dataclass
class BatteryParams:
    """A 3S lithium-ion pack: open-circuit voltage falls with charge and the voltage sags under load."""

    cells: int = 3
    capacity_ah: float = 2.2
    internal_resistance: float = 0.12  # ohm for the whole pack
    initial_charge: float = 1.0  # 0 to 1
    cutoff: float = 9.0  # V: the pack protection board opens below this

    def open_circuit(self, charge: float) -> float:
        """Pack voltage at rest for a state of charge between 0 and 1 (a typical Li-ion curve)."""
        charge = min(max(float(charge), 0.0), 1.0)
        for (c0, v0), (c1, v1) in zip(_CELL_CURVE, _CELL_CURVE[1:]):
            if charge <= c1:
                return self.cells * (v0 + (v1 - v0) * (charge - c0) / (c1 - c0))
        return self.cells * _CELL_CURVE[-1][1]


@dataclass
class RobotParams:
    # --- chassis: everything above the axle, motors and battery included
    body_mass: float = 1.0
    com_height: float = 0.10  # m, centre of mass above the axle
    com_offset: float = 0.0  # m, centre of mass ahead of the vertical through the axle (a real robot is never perfectly balanced)
    inertia_pitch: float = 0.0042  # kg m^2 about the axle direction, through the centre of mass
    inertia_roll: float = 0.0090  # about the forward direction
    inertia_yaw: float = 0.0065  # about the body's vertical
    body_size: tuple[float, float, float] = (0.10, 0.26, 0.20)  # depth, width over the motors, height, for collisions and drawing
    # --- wheels
    wheel_radius: float = 0.04
    wheel_mass: float = 0.05  # each, tyre and hub
    wheel_inertia: float = 6.0e-5  # kg m^2 about the axle, each
    track: float = 0.20  # m between the two wheels' centres
    # --- ground
    friction_static: float = 0.9  # tyre on office flooring: breakaway
    friction_kinetic: float = 0.75
    rolling_resistance: float = 0.012  # coefficient (force per newton of load)
    scrub_torque: float = 0.012  # N m of yaw resistance from the tyres twisting on the floor
    # --- fall
    fall_angle: float = np.radians(62.0)  # the lean at which the chassis rests on the floor
    # --- IMU mount
    imu_height: float = 0.13  # m above the axle
    motor: MotorParams = field(default_factory=MotorParams)
    battery: BatteryParams = field(default_factory=BatteryParams)

    @property
    def total_mass(self) -> float:
        return self.body_mass + 2.0 * self.wheel_mass

    def perturbed(self, rng: np.random.Generator, amount: float = 1.0) -> "RobotParams":
        """The real robot as opposed to the datasheet: masses, centre of mass, motors and friction all a few percent off.

        `amount` scales the spread (1 = a plausible hobby build, 0 = exactly nominal).
        """

        def factor(sigma: float) -> float:
            return float(1.0 + amount * rng.normal(0.0, sigma))

        motor = replace(
            self.motor,
            resistance=self.motor.resistance * factor(0.05),
            torque_constant=self.motor.torque_constant * factor(0.04),
            coulomb_friction=self.motor.coulomb_friction * factor(0.25),
            gearbox_viscous=self.motor.gearbox_viscous * factor(0.25),
        )
        return replace(
            self,
            body_mass=self.body_mass * factor(0.05),
            com_height=self.com_height * factor(0.08),
            com_offset=self.com_offset + amount * rng.normal(0.0, 0.004),
            inertia_pitch=self.inertia_pitch * factor(0.10),
            wheel_radius=self.wheel_radius * factor(0.01),
            friction_static=self.friction_static * factor(0.1),
            friction_kinetic=self.friction_kinetic * factor(0.1),
            motor=motor,
        )
