"""Rigid-body dynamics of a two-wheeled inverted pendulum, with motors, battery, tyre friction and contacts.

Generalised coordinates  q = (s, psi, theta, phi_L, phi_R)

    s        distance the axle midpoint has travelled along the heading
    psi      heading (turning right is positive, see params.py)
    theta    body pitch, positive leaning forward
    phi_i    absolute spin angle of wheel i (forward positive); the encoders see N * (phi_i - theta)

The equations are the Lagrange equations of the kinetic and potential energy of the chassis (mass at
its centre of mass, three principal inertias), the two wheels, and the two motor rotors, whose
inertia is N^2 larger than it looks at the wheel. Nothing is linearised: the sin and cos of the
pitch and the yaw-pitch gyroscopic coupling are all in. Writing them out (L = distance of the centre of
mass from the axle, alpha = theta + the offset angle of the centre of mass):

    (M+2m) s'' + M L cos(alpha) theta'' - M L sin(alpha) theta'^2             = Q_s
    M_psi psi'' + D theta' psi'                                                 = Q_psi
    M_th theta'' + M L cos(alpha) s'' + Jr N (1-N) (phi_L'' + phi_R'')
                 - D psi'^2 / 2 - M g L sin(alpha)                              = Q_theta
    (Jw + Jr N^2) phi_i'' + Jr N (1-N) theta''                                  = Q_phi_i

The ground pushes on each wheel with a longitudinal force F_i. While the tyre grips, F_i is whatever
force keeps the contact point still (a Lagrange multiplier: that is the rolling constraint).
When that force would exceed mu_s N_i the tyre lets go and pushes with the kinetic friction
mu_k N_i against the slip instead, and the wheel spins up or locks independently. Lateral tyre
forces are an ideal non-holonomic constraint: the robot does not slide sideways.

Motors: voltage from the H-bridge duty and the sagging battery, back-EMF, winding current with its
inductance (integrated exactly, since the electrical time constant is shorter than a step), a
current limit, and Coulomb plus viscous friction in the gearbox.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .params import GRAVITY, RobotParams

SLIP_EPS = 0.02  # m/s of contact-point speed below which a tyre counts as gripping
SIDE = (-1.0, 1.0)  # lateral side of wheel 0 (left) and wheel 1 (right) in the direction of the heading's derivative


@dataclass
class Contact:
    """A wall or an obstacle touching the robot: unit normal pointing at the robot, in plan (x, z), and how deep it is."""

    normal: tuple[float, float]
    depth: float


@dataclass
class PlantTelemetry:
    currents: np.ndarray = field(default_factory=lambda: np.zeros(2))
    motor_torque: np.ndarray = field(default_factory=lambda: np.zeros(2))  # N m at the wheels (electromagnetic, before friction)
    battery_voltage: float = 12.0
    battery_current: float = 0.0
    traction_use: np.ndarray = field(default_factory=lambda: np.zeros(2))  # |F| / (mu_s N) per wheel; 1 = about to slip
    slipping: np.ndarray = field(default_factory=lambda: np.zeros(2, bool))


class TwipPlant:
    def __init__(self, params: RobotParams, x: float = 0.0, z: float = 0.0, heading: float = 0.0, pitch: float = 0.0) -> None:
        self.p = params
        p = params
        self.com_distance = float(np.hypot(p.com_height, p.com_offset))
        self.com_angle = float(np.arctan2(p.com_offset, p.com_height))  # of the centre of mass from the body's vertical
        m = p.motor
        self.n = m.gear_ratio
        self.reflected = p.wheel_inertia + m.rotor_inertia * m.gear_ratio**2
        self.coupling = m.rotor_inertia * m.gear_ratio * (1.0 - m.gear_ratio)
        self.m_ss = p.total_mass
        self.m_theta = p.body_mass * self.com_distance**2 + p.inertia_pitch + 2.0 * m.rotor_inertia * (1.0 - m.gear_ratio) ** 2
        self.psi_const = 2.0 * p.wheel_mass * (p.track / 2.0) ** 2 + p.wheel_inertia  # two wheels' yaw inertia: m (d/2)^2 each, plus the discs' diametral inertia
        self.G = np.zeros((5, 2))  # how each tyre force enters the generalised forces
        for i, side in enumerate(SIDE):
            self.G[0, i] = 1.0
            self.G[1, i] = -side * p.track / 2.0
            self.G[3 + i, i] = -p.wheel_radius
        self.reset(x, z, heading, pitch)

    # --- state ---------------------------------------------------------------------------------

    def reset(self, x: float = 0.0, z: float = 0.0, heading: float = 0.0, pitch: float = 0.0) -> None:
        self.q = np.array([0.0, heading, pitch, 0.0, 0.0])
        self.qd = np.zeros(5)
        self.qdd = np.zeros(5)
        self.x, self.z = x, z
        self.currents = np.zeros(2)
        self.charge = self.p.battery.initial_charge
        self.fallen = False
        self.time = 0.0
        self.telemetry = PlantTelemetry()
        self.telemetry.battery_voltage = self.p.battery.open_circuit(self.charge)
        self._normal_load = 0.5 * self.p.total_mass * GRAVITY

    @property
    def heading(self) -> float:
        return float(self.q[1])

    @property
    def pitch(self) -> float:
        return float(self.q[2])

    @property
    def speed(self) -> float:
        return float(self.qd[0])

    @property
    def yaw_rate(self) -> float:
        return float(self.qd[1])

    @property
    def pitch_rate(self) -> float:
        return float(self.qd[2])

    @property
    def motor_speed(self) -> np.ndarray:
        """Rotor speeds relative to the chassis, rad/s: what the back-EMF and the encoders see."""
        return self.n * (self.qd[3:5] - self.qd[2])

    # --- equations of motion -------------------------------------------------------------------

    def mass_matrix(self, theta: float) -> np.ndarray:
        p, m = self.p, self.p.motor
        alpha = theta + self.com_angle
        mass, length = p.body_mass, self.com_distance
        matrix = np.zeros((5, 5))
        matrix[0, 0] = self.m_ss
        matrix[0, 2] = matrix[2, 0] = mass * length * np.cos(alpha)
        matrix[2, 2] = self.m_theta
        matrix[1, 1] = mass * length**2 * np.sin(alpha) ** 2 + p.inertia_roll * np.sin(theta) ** 2 + p.inertia_yaw * np.cos(theta) ** 2 + self.psi_const
        for i in (3, 4):
            matrix[i, i] = self.reflected
            matrix[i, 2] = matrix[2, i] = self.coupling
        return matrix

    def bias_forces(self, theta: float, q_dot: np.ndarray) -> np.ndarray:
        """Everything on the right of M q'' except the applied forces: gravity, centrifugal and gyroscopic terms."""
        p = self.p
        alpha = theta + self.com_angle
        mass, length = p.body_mass, self.com_distance
        s_dot, psi_dot, theta_dot = q_dot[0], q_dot[1], q_dot[2]
        d_term = 2.0 * mass * length**2 * np.sin(alpha) * np.cos(alpha) + 2.0 * (p.inertia_roll - p.inertia_yaw) * np.sin(theta) * np.cos(theta)
        bias = np.zeros(5)
        bias[0] = mass * length * np.sin(alpha) * theta_dot**2
        bias[1] = -d_term * theta_dot * psi_dot
        bias[2] = 0.5 * d_term * psi_dot**2 + mass * GRAVITY * length * np.sin(alpha)
        return bias

    def _solve(self, matrix: np.ndarray, rhs: np.ndarray, stuck: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """Accelerations and the grip forces of the tyres listed in `stuck` (their contact points held still)."""
        if not stuck:
            return np.linalg.solve(matrix, rhs), np.zeros(0)
        g = self.G[:, stuck]
        k = len(stuck)
        system = np.zeros((5 + k, 5 + k))
        system[:5, :5] = matrix
        system[:5, 5:] = -g
        system[5:, :5] = g.T
        solution = np.linalg.solve(system, np.concatenate([rhs, np.zeros(k)]))
        return solution[:5], solution[5:]

    def stuck_acceleration(self, theta: float, q_dot: np.ndarray, applied: np.ndarray) -> np.ndarray:
        """Accelerations with both tyres gripping, for the generalised forces `applied` (s, psi, theta, phi_L, phi_R).

        Used to linearise the plant for the controller design.
        """
        return self._solve(self.mass_matrix(theta), applied + self.bias_forces(theta, q_dot), [0, 1])[0]

    # --- motors and battery --------------------------------------------------------------------

    def _motors(self, dt: float, duty: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Advance the winding currents and the battery; return (output torque of each motor, friction torque of each gearbox) at the wheels."""
        p, m, b = self.p, self.p.motor, self.p.battery
        open_circuit = b.open_circuit(self.charge)
        terminal = open_circuit - b.internal_resistance * self.telemetry.battery_current
        if terminal < b.cutoff:
            duty = np.zeros(2)  # the protection board has opened
            terminal = open_circuit
        duty = np.minimum(np.maximum(duty, -1.0), 1.0)
        voltage = duty * terminal
        rotor = self.motor_speed
        back_emf = m.torque_constant * rotor
        steady = (voltage - back_emf) / m.resistance
        decay = np.exp(-m.resistance * dt / m.inductance)
        self.currents = np.minimum(np.maximum(steady + (self.currents - steady) * decay, -m.stall_current_limit), m.stall_current_limit)
        torque = self.n * m.torque_constant * self.currents
        relative = rotor / self.n  # output shaft speed relative to the chassis
        friction = -(m.coulomb_friction * np.tanh(relative / m.friction_speed) + (m.gearbox_viscous + m.rotor_friction * self.n**2) * relative)
        battery_current = float(duty @ self.currents)
        self.telemetry.battery_current = battery_current
        self.telemetry.battery_voltage = terminal
        self.charge = max(self.charge - max(battery_current, 0.0) * dt / (3600.0 * b.capacity_ah), 0.0)
        return torque, friction

    # --- stepping ------------------------------------------------------------------------------

    def step(
        self,
        dt: float,
        duty: np.ndarray,
        push_force: float = 0.0,
        push_height: float = 0.12,
        yaw_torque: float = 0.0,
        contacts: list[Contact] | None = None,
        contact_height: float = 0.05,
        held: bool = False,
    ) -> None:
        """Advance by dt seconds. `duty` is the H-bridge duty (-1..1) of the left and right motor.

        `held` models a hand holding the robot at its pitch (during power-up): the chassis does not move.
        """
        p = self.p
        theta = float(self.q[2])
        torque, friction = self._motors(dt, np.asarray(duty, float))
        total_load = p.total_mass * GRAVITY - p.body_mass * self.com_distance * (self.qdd[2] * np.sin(theta + self.com_angle) + self.qd[2] ** 2 * np.cos(theta + self.com_angle))
        total_load = max(total_load, 0.1 * p.total_mass * GRAVITY)
        load = np.full(2, 0.5 * total_load)
        self._normal_load = float(load[0])

        wheel_axle = torque + friction  # torque the gearbox puts on each wheel, reacted on the chassis
        applied = np.zeros(5)
        applied[3:5] = wheel_axle
        applied[2] = -float(np.sum(wheel_axle)) + push_force * push_height * np.cos(theta)
        applied[0] = push_force
        applied[1] = yaw_torque - p.scrub_torque * np.tanh(self.qd[1] / 0.1)
        if self.fallen:
            applied[0] -= 0.6 * p.total_mass * GRAVITY * np.tanh(self.qd[0] / 0.02)  # the chassis scraping along the floor
            applied[1] -= 0.4 * np.tanh(self.qd[1] / 0.05)
        for i, side in enumerate(SIDE):
            speed_i = self.qd[0] - side * (p.track / 2.0) * self.qd[1]
            rolling = -p.rolling_resistance * load[i] * np.tanh(speed_i / 0.02)  # a drag at the axle level
            applied[0] += rolling
            applied[1] += -side * (p.track / 2.0) * rolling

        matrix = self.mass_matrix(theta)
        rhs = applied + self.bias_forces(theta, self.qd)
        contact_speed = np.array([self.qd[0] - side * (p.track / 2.0) * self.qd[1] - p.wheel_radius * self.qd[3 + i] for i, side in enumerate(SIDE)])

        slipping: dict[int, float] = {}  # wheel -> the kinetic friction force it pushes with
        for i in range(2):
            if abs(contact_speed[i]) > SLIP_EPS:
                slipping[i] = -p.friction_kinetic * load[i] * float(np.sign(contact_speed[i]))
        use = np.zeros(2)
        for _ in range(3):
            stuck = [i for i in range(2) if i not in slipping]
            force_rhs = rhs.copy()
            for i, force in slipping.items():
                force_rhs = force_rhs + self.G[:, i] * force
            acceleration, grip = self._solve(matrix, force_rhs, stuck)
            changed = False
            for index, i in enumerate(stuck):
                use[i] = abs(grip[index]) / (p.friction_static * load[i])
                if use[i] > 1.0:
                    slipping[i] = float(np.sign(grip[index])) * p.friction_kinetic * load[i]
                    changed = True
            if not changed:
                break
        for i, force in slipping.items():
            use[i] = abs(force) / (p.friction_static * load[i])

        if held:
            acceleration = np.zeros(5)
            self.qd[:] = 0.0
        self.qdd = acceleration
        self.qd = self.qd + acceleration * dt
        # stuck tyres: remove the small velocity drift of the rolling constraint
        for i in range(2):
            if i not in slipping:
                self.qd[3 + i] = (self.qd[0] - SIDE[i] * (p.track / 2.0) * self.qd[1]) / p.wheel_radius
        self.q = self.q + self.qd * dt

        heading = self.q[1] - 0.5 * self.qd[1] * dt
        self.x += self.qd[0] * np.cos(heading) * dt
        self.z += self.qd[0] * np.sin(heading) * dt

        self._limit_pitch()
        if contacts:
            self._contacts(contacts, contact_height)
        self.telemetry.currents = self.currents.copy()
        self.telemetry.motor_torque = torque
        self.telemetry.traction_use = use
        self.telemetry.slipping = np.array([i in slipping for i in range(2)])
        self.time += dt

    def _limit_pitch(self) -> None:
        """The chassis lies on the floor at the fall angle: an inelastic stop."""
        limit = self.p.fall_angle
        if abs(self.q[2]) >= limit:
            self.q[2] = float(np.sign(self.q[2])) * limit
            if self.q[2] * self.qd[2] > 0.0:
                self.qd[2] = 0.0
            self.fallen = True

    def _contacts(self, contacts: list[Contact], height: float) -> None:
        """Push out of walls and remove the speed into them, through the constrained mass matrix so the body feels the bump."""
        p = self.p
        heading = np.array([np.cos(self.q[1]), np.sin(self.q[1])])
        for contact in contacts:
            normal = np.array(contact.normal)
            self.x += normal[0] * contact.depth
            self.z += normal[1] * contact.depth
            along = float(heading @ normal)
            if abs(along) < 1e-3:
                continue
            jacobian = np.zeros(5)
            jacobian[0] = along
            jacobian[2] = along * height * np.cos(self.q[2])
            approach = float(jacobian @ self.qd)
            if approach >= 0.0:
                continue
            unit, _ = self._solve(self.mass_matrix(float(self.q[2])), jacobian, [0, 1])  # velocity change per unit impulse
            effective = float(jacobian @ unit)
            impulse = -(1.0 + 0.15) * approach / max(effective, 1e-9)
            self.qd = self.qd + impulse * unit

    # --- what the sensors would feel ------------------------------------------------------------

    def imu_truth(self) -> tuple[np.ndarray, np.ndarray]:
        """(specific force, angular velocity) at the IMU in the body frame: x forward, y left, z up. Accelerations are from the last step."""
        p = self.p
        theta, s_dot, psi_dot, theta_dot = self.q[2], self.qd[0], self.qd[1], self.qd[2]
        s_ddot, psi_ddot, theta_ddot = self.qdd[0], self.qdd[1], self.qdd[2]
        h = p.imu_height
        sin, cos = np.sin(theta), np.cos(theta)
        v_h = s_dot + h * theta_dot * cos
        v_l = h * sin * psi_dot  # towards the robot's right
        a_h = s_ddot + h * (theta_ddot * cos - theta_dot**2 * sin) - psi_dot * v_l
        a_l = h * (theta_dot * cos * psi_dot + sin * psi_ddot) + psi_dot * v_h
        a_z = -h * (theta_ddot * sin + theta_dot**2 * cos)
        up = a_z + GRAVITY
        force = np.array([a_h * cos - up * sin, -a_l, a_h * sin + up * cos])
        rates = np.array([psi_dot * sin, theta_dot, -psi_dot * cos])  # about forward (roll), left (leaning forward is positive), up (counter-clockwise positive: the heading turns the other way)
        return force, rates
