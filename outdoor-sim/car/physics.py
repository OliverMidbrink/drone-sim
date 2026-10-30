"""The car: a rigid body on four wheels, each hung on a parallel-arm suspension and rolling on the ground's height function.

What is simulated, at 600 steps a second:

  Body        6 degrees of freedom (position, rotation matrix), inertia tensor, gravity, air drag, and 13
              points on the box of the body that collide with the ground (so it can bottom out, roll over).
  Suspension  each wheel hub slides on the arc of two parallel arms (the wheel stays upright; it moves
              in a little as it goes up and down). Its travel `s` is a degree of freedom with the hub's
              mass: coil spring with a fixed preload (so softer springs do not change the ride height),
              damper (stiffer to rebound than to compress), progressive bump stops, droop straps and an
              anti-roll bar. The force of the tyre on the hub is split exactly: the part along the arm's
              path moves the hub, the rest goes to the body, as does the reaction of the hub's accelerating.
  Wheel       a disc: its rim is sampled at 17 angles and 3 positions across the tread, and the deepest
              point under the ground is the contact, so a wheel climbs a bump or an edge the way a tyre does
              (not by a ray straight down). The tyre is a spring and damper along the ground's normal.
  Tyre        slip velocity of the tread against the ground, normalised by speed, goes through a
              magic-formula curve: grip rises to its peak at ~15 % slip and falls to 70 % when sliding, shared
              between along and across by a friction circle. That is what lets it drift. Wheel spin is
              integrated implicitly in the tyre's stiffness so it stays stable at 600 Hz.
  Driveline   motor torque split over the driven wheels (AWD / FWD / RWD), falling to zero at top speed;
              foot brake with ABS; handbrake locking the rear wheels; Ackermann steering.
"""

from __future__ import annotations

import numpy as np

from .params import G, CarParams, Controls, Telemetry, Tuning

STEP = 1.0 / 600.0
RIM_ANGLES = np.radians(np.linspace(-68.0, 68.0, 17))
MAGIC_B, MAGIC_C = 13.3, 1.41  # D(s) = sin(C atan(B s)) peaks (= 1) at s = 0.15 and settles at 0.8 when sliding
MIN_SLIP_SPEED = 1.0  # m/s: slip is normalised by at least this, so a stopped tyre acts like a damper
SIDE = np.array([-1.0, 1.0, -1.0, 1.0])
FRONT = np.array([True, True, False, False])
DRIVEN = {"awd": np.array([1.0, 1.0, 1.0, 1.0]), "fwd": np.array([1.0, 1.0, 0.0, 0.0]), "rwd": np.array([0.0, 0.0, 1.0, 1.0])}
BRAKE_SHARE = np.array([0.3, 0.3, 0.2, 0.2])


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.cross(a, b)


def _rodrigues(vector: np.ndarray) -> np.ndarray:
    """Rotation matrix of a rotation vector."""
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3)
    k = vector / angle
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


class CarSim:
    def __init__(self, ground, params: CarParams | None = None, x: float = 0.0, z: float = 0.0, heading: float = 0.0) -> None:
        self.ground = ground
        self.params = p = params or CarParams()
        self.tuning = Tuning()
        self.controls = Controls()
        self.telemetry = Telemetry()
        self.mid = p.travel / 2.0
        self._inertia = np.diag(p.inertia)
        self._inertia_inverse = np.diag(1.0 / np.array(p.inertia))
        half = p.body_half
        cy = p.body_centre_y
        corners = [(sx * half[0], cy + sy * half[1], sz * half[2]) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        middles = [(sx * half[0], cy - half[1], 0.0) for sx in (-1, 1)] + [(0.0, cy - half[1], sz * half[2]) for sz in (-1, 1)] + [(0.0, cy - half[1], 0.0)]
        self.body_points = np.array(corners + middles, float)
        self.place(x, z, heading)

    # --- state ---------------------------------------------------------------------------------

    def place(self, x: float, z: float, heading: float = 0.0) -> None:
        """Put the car at rest on the ground with its nose along `heading` (0 = +x, positive turns towards +z)."""
        p = self.params
        c, s = np.cos(heading), np.sin(heading)
        self.R = np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])
        ground = float(self.ground.sample(np.array([x]), np.array([z]))[0])
        self.pos = np.array([x, ground + p.com_height, z])
        self.vel = np.zeros(3)
        self.omega = np.zeros(3)
        self.s = np.full(4, self.mid)
        self.sd = np.zeros(4)
        self.spin = np.zeros(4)  # rad/s, positive rolling forward
        self.spin_angle = np.zeros(4)
        self.steer = 0.0  # the steering angle of the virtual centre wheel
        self.a_com = np.zeros(3)
        self.alpha = np.zeros(3)
        self.time = 0.0
        self.steps = 0
        self._owed = 0.0
        self.telemetry = Telemetry()

    @property
    def heading(self) -> float:
        return float(np.arctan2(self.R[2, 0], self.R[0, 0]))

    @property
    def forward(self) -> np.ndarray:
        return self.R[:, 0]

    @property
    def speed(self) -> float:
        """Signed speed along the car's nose, m/s."""
        return float(self.vel @ self.R[:, 0])

    @property
    def upside_down(self) -> bool:
        return float(self.R[1, 1]) < 0.2

    def hub_kinematics(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Hub positions and the direction each hub travels in as `s` grows, in the body frame, (4, 3) each."""
        p = self.params
        q = s - self.mid
        root = np.sqrt(p.arm_length**2 - q**2)
        inset = p.arm_length - root
        position = np.stack([p.wheel_x, p.hub_y + q, SIDE * (p.track / 2.0 - inset)], axis=1)
        tangent = np.stack([np.zeros(4), np.ones(4), -SIDE * q / root], axis=1)
        return position, tangent

    def steer_angles(self) -> np.ndarray:
        """Ackermann steering angle of each wheel."""
        p = self.params
        d = self.steer
        if abs(d) < 1e-6:
            return np.zeros(4)
        t = np.tan(d)
        left = np.arctan(p.wheelbase * t / (p.wheelbase + p.track / 2.0 * t))
        right = np.arctan(p.wheelbase * t / (p.wheelbase - p.track / 2.0 * t))
        return np.array([left, right, 0.0, 0.0])

    # --- one physics step ----------------------------------------------------------------------

    def advance(self, dt: float) -> None:
        """Run the physics for `dt` seconds in fixed steps (what is left over is carried to the next call)."""
        self._owed += dt
        steps = int(self._owed / STEP + 1e-9)
        self._owed -= steps * STEP
        for _ in range(steps):
            self.step(STEP)

    def step(self, dt: float) -> None:
        p, tune, ctl, R = self.params, self.tuning, self.controls, self.R
        mass = p.mass
        wm = p.wheel_mass
        gravity = np.array([0.0, -G, 0.0])
        radius = p.wheel_radius

        # steering: rate-limited towards the key, the more so the faster we go
        speed = self.speed
        limit = p.steer_max / (1.0 + (abs(speed) / p.steer_fade) ** 2)
        target = ctl.steer * limit
        if ctl.assist and abs(speed) > 3.0 and speed > 0.0:
            slide = self._front_slide()  # the wheels turn to where the front axle is going, as a driver counter-steers
            authority = float(np.clip(1.0 - abs(slide) / np.radians(p.assist_fade), 0.25, 1.0))  # in a big slide the keys only trim the drift angle
            target = float(np.clip(target * authority + p.assist_gain * slide, -p.steer_max, p.steer_max))
        turning_in = abs(target) > abs(self.steer) and target * self.steer >= 0.0
        rate = (1.0 if turning_in else 1.6) * p.steer_rate * dt
        self.steer = float(np.clip(self.steer + np.clip(target - self.steer, -rate, rate), -max(limit, p.steer_max * 0.6 if ctl.assist else 0.0), max(limit, p.steer_max * 0.6 if ctl.assist else 0.0)))
        delta = self.steer_angles()

        # hubs
        hub_local, tangent_local = self.hub_kinematics(self.s)
        hub_rel = hub_local @ R.T
        tangent = tangent_local @ R.T
        hub = self.pos + hub_rel
        hub_velocity = self.vel + _cross(self.omega, hub_rel) + self.sd[:, None] * tangent

        # wheel axes
        cd, sd = np.cos(delta), np.sin(delta)
        f = np.stack([cd, np.zeros(4), sd], axis=1) @ R.T  # rolling direction
        a = np.stack([-sd, np.zeros(4), cd], axis=1) @ R.T  # axle, to the right
        up = _cross(a, f)

        # contact: the deepest point of the rim under the ground
        offsets = radius * (np.sin(RIM_ANGLES)[None, :, None] * f[:, None, :] - np.cos(RIM_ANGLES)[None, :, None] * up[:, None, :])  # (4, K, 3)
        lateral = np.array([-0.45, 0.0, 0.45]) * p.wheel_width
        points = hub[:, None, None, :] + offsets[:, :, None, :] + lateral[None, None, :, None] * a[:, None, None, :]  # (4, K, 3, 3)
        flat = points.reshape(4, -1, 3)
        height = self.ground.sample(flat[:, :, 0], flat[:, :, 2])
        depth = height - flat[:, :, 1]
        best = np.argmax(depth, axis=1)
        rows = np.arange(4)
        depth_v = depth[rows, best]
        touching = depth_v > 0.0
        cp = flat[rows, best]
        normal = self.ground.normal(cp[:, 0], cp[:, 2])
        paved = self.ground.paved(cp[:, 0], cp[:, 2])

        # normal force
        depth_n = depth_v * normal[:, 1]
        v_in = np.einsum("ij,ij->i", hub_velocity, normal)
        load = np.where(touching, np.maximum(p.tyre_stiffness * depth_n - p.tyre_damping * v_in, 0.0), 0.0)

        # tyre slip
        r_c = cp - hub
        spin_vector = -self.spin[:, None] * a
        u = hub_velocity + _cross(self.omega, r_c) + _cross(spin_vector, r_c)
        u_t = u - np.einsum("ij,ij->i", u, normal)[:, None] * normal
        fg = f - np.einsum("ij,ij->i", f, normal)[:, None] * normal
        fg /= np.linalg.norm(fg, axis=1, keepdims=True)
        sg = _cross(fg, normal)
        ux, uy = np.einsum("ij,ij->i", u_t, fg), np.einsum("ij,ij->i", u_t, sg)
        v_t = hub_velocity - np.einsum("ij,ij->i", hub_velocity, normal)[:, None] * normal
        speed_t = np.linalg.norm(v_t, axis=1)
        ref_x = np.maximum(np.maximum(speed_t, np.abs(self.spin) * radius), MIN_SLIP_SPEED)  # slip ratio: against the faster of car and tread
        ref_y = np.maximum(speed_t, MIN_SLIP_SPEED)  # slip angle: against the car's speed, so wheelspin does not hide sideways grip
        sx, sy = ux / ref_x, uy / ref_y
        s = np.hypot(sx, sy)
        s_safe = np.maximum(s, 1e-9)
        grip = p.grip * tune.grip * (p.grass_grip + (1.0 - p.grass_grip) * paved)
        shape = np.sin(MAGIC_C * np.arctan(MAGIC_B * s))  # 0..1
        mu_n = grip * load * shape / s_safe  # friction force per unit of normalised slip
        secant_x, secant_y = mu_n / ref_x, mu_n / ref_y  # N per m/s of slip velocity
        f_tan = -(secant_x * ux)[:, None] * fg - (secant_y * uy)[:, None] * sg

        # driveline: torque on each wheel
        throttle = float(np.clip(ctl.throttle, -1.0, 1.0))
        if ctl.assist and throttle > 0.0 and speed > 3.0:  # the assist also eases the throttle in a big slide so the rear tyres get their side grip back: a held drift
            throttle *= float(np.clip(1.0 - (abs(self.slip_angle()) - p.assist_slide) / p.assist_slide_range, 0.15, 1.0))
        driven = DRIVEN[tune.drivetrain]
        thrust_torque = tune.power * p.thrust * mass * G * radius
        mean_speed = float((driven * self.spin).sum() / driven.sum() * radius)
        if throttle >= 0.0:
            taper = np.clip(1.0 - mean_speed / p.top_speed, 0.0, 1.0)
        else:
            taper = np.clip(1.0 + mean_speed / p.reverse_speed, 0.0, 1.0)
        drive = throttle * thrust_torque * taper * driven / driven.sum()
        roll = np.where(touching, (p.rolling_resistance + 0.05 * (1.0 - paved)) * load * radius, 0.0) * np.tanh(self.spin / 2.0)
        torque = drive - roll
        wheel_speed_roll = np.einsum("ij,ij->i", v_t, fg) / radius
        if ctl.brake > 0.0:
            brake_torque = ctl.brake * p.brake * mass * G * radius * BRAKE_SHARE
            locking = self.spin < 0.82 * wheel_speed_roll  # ABS: let go of a wheel that is slowing much faster than the car
            brake_torque = np.where(locking & (wheel_speed_roll > 2.0), 0.0, brake_torque)
            torque -= np.sign(self.spin) * np.minimum(brake_torque, p.wheel_inertia * np.abs(self.spin) / dt)
        if ctl.handbrake:
            torque[2:] -= np.sign(self.spin[2:]) * np.minimum(p.handbrake_torque, p.wheel_inertia * np.abs(self.spin[2:]) / dt)

        # wheel spin, implicit in the tyre's stiffness along the wheel
        torque_tyre = np.einsum("ij,ij->i", _cross(r_c, f_tan), -a)
        longitudinal_stiffness = secant_x * radius * radius  # N m per rad/s of wheel speed
        d_spin = dt * (torque + torque_tyre) / (p.wheel_inertia + dt * longitudinal_stiffness)
        f_tan = f_tan + (secant_x * radius * d_spin)[:, None] * fg
        self.spin = self.spin + d_spin
        self.spin_angle = (self.spin_angle + self.spin * dt) % (2.0 * np.pi)

        tyre_force = np.where(touching[:, None], load[:, None] * normal + f_tan, 0.0)

        # suspension forces along each hub's path (positive pushes the hub down)
        q = self.s
        k = p.spring_rate * tune.stiffness
        c = np.where(self.sd > 0.0, p.damping_compression, p.damping_rebound) * tune.damping
        spring = p.corner_load + k * (q - self.mid) + c * self.sd
        spring += p.anti_roll * (q - q[[1, 0, 3, 2]])
        over = np.maximum(q - p.travel, 0.0)
        under = np.maximum(-q, 0.0)
        spring += p.stop_stiffness * over + np.where(over > 0.0, p.stop_damping * np.maximum(self.sd, 0.0), 0.0)
        spring -= p.stop_stiffness * under + np.where(under > 0.0, p.stop_damping * np.maximum(-self.sd, 0.0), 0.0)

        # the hubs' accelerations relative to the body, using the body's acceleration of the last step
        a_hub = self.a_com + _cross(self.alpha, hub_rel) + _cross(self.omega, _cross(self.omega, hub_rel))
        along = np.einsum("ij,ij->i", tyre_force + wm * gravity, tangent) - spring
        s_dd = along / wm - np.einsum("ij,ij->i", a_hub, tangent)

        # forces on the body: what the hubs hand over, gravity, drag, and the body's own contact with the ground
        hub_force = tyre_force + wm * gravity - wm * s_dd[:, None] * tangent
        force = p.sprung_mass * gravity + hub_force.sum(axis=0)
        torque_body = _cross(hub_rel, hub_force).sum(axis=0)
        force += -0.5 * 1.2 * p.drag_area * np.linalg.norm(self.vel) * self.vel
        bf, bt = self._body_contact()
        force += bf
        torque_body += bt

        # integrate
        self.a_com = force / mass
        inertia = R @ self._inertia @ R.T
        inertia_inverse = R @ self._inertia_inverse @ R.T
        self.alpha = inertia_inverse @ (torque_body - _cross(self.omega, inertia @ self.omega))
        self.vel = self.vel + self.a_com * dt
        self.omega = self.omega + self.alpha * dt
        self.pos = self.pos + self.vel * dt
        self.R = _rodrigues(self.omega * dt) @ R
        self.sd = self.sd + s_dd * dt
        self.s = self.s + self.sd * dt
        self.steps += 1
        if self.steps % 64 == 0:
            u_, _, vt = np.linalg.svd(self.R)
            self.R = u_ @ vt
        self.time += dt

        t = self.telemetry
        t.travel, t.load, t.touching = self.s.copy(), load, touching
        t.slip = np.where(touching, s, 0.0)
        t.contact = cp + np.stack([np.zeros(4), depth_v, np.zeros(4)], axis=1)  # on the ground surface
        t.tread = t.contact - np.einsum("ij,ij->i", t.contact - hub, a)[:, None] * a  # in the middle of the tread
        t.axle, t.normal = a, normal
        t.body_acceleration = R.T @ (self.a_com - gravity)

    def _body_contact(self) -> tuple[np.ndarray, np.ndarray]:
        """Force and torque of the body box's corners and floor points against the ground."""
        R = self.R
        rel = self.body_points @ R.T
        world = self.pos + rel
        height = self.ground.sample(world[:, 0], world[:, 2])
        depth = height - world[:, 1]
        hit = depth > 0.0
        if not hit.any():
            return np.zeros(3), np.zeros(3)
        rel, world, depth = rel[hit], world[hit], depth[hit]
        normal = self.ground.normal(world[:, 0], world[:, 2])
        velocity = self.vel + _cross(self.omega, rel)
        v_n = np.einsum("ij,ij->i", velocity, normal)
        load = np.maximum(40e3 * depth - 1200.0 * v_n, 0.0)
        v_t = velocity - v_n[:, None] * normal
        speed = np.maximum(np.linalg.norm(v_t, axis=1), 1e-9)
        friction = -(0.5 * load * np.tanh(speed / 0.3) / speed)[:, None] * v_t
        forces = load[:, None] * normal + friction
        return forces.sum(axis=0), _cross(rel, forces).sum(axis=0)

    # --- derived numbers for the HUD and the model -----------------------------------------------

    def euler(self) -> tuple[float, float, float]:
        """Yaw, pitch and roll of the body in degrees (nose up and right side down positive)."""
        R = self.R
        pitch = float(np.degrees(np.arcsin(np.clip(R[1, 0], -1.0, 1.0))))
        roll = float(np.degrees(np.arctan2(-R[1, 2], R[1, 1])))
        return float(np.degrees(self.heading)), pitch, roll

    def _front_slide(self) -> float:
        """Angle (rad) between the car's nose and the direction the front axle moves in: what counter-steering has to cancel."""
        axle = self.R @ np.array([self.params.wheel_x[0], 0.0, 0.0])
        v = self.R.T @ (self.vel + _cross(self.omega, axle))
        return float(np.arctan2(v[2], max(v[0], 0.5)))

    def slip_angle(self) -> float:
        """Angle between where the car points and where it moves, degrees (0 when slow)."""
        v = np.array([self.vel[0], self.vel[2]])
        if np.linalg.norm(v) < 1.5:
            return 0.0
        h = np.array([self.R[0, 0], self.R[2, 0]])
        return float(np.degrees(np.arctan2(h[0] * v[1] - h[1] * v[0], h @ v)))

    def wheel_matrices(self) -> list[np.ndarray]:
        """4x4 matrices (wheel model frame: x rolling direction, y up, z axle) of each wheel, spin included."""
        R = self.R
        hub_local, _ = self.hub_kinematics(self.s)
        delta = self.steer_angles()
        out = []
        for i in range(4):
            cd, sd = np.cos(delta[i]), np.sin(delta[i])
            f = R @ np.array([cd, 0.0, sd])
            a = R @ np.array([-sd, 0.0, cd])
            up = np.cross(a, f)
            c, s = np.cos(self.spin_angle[i]), np.sin(self.spin_angle[i])
            m = np.eye(4)
            m[:3, 0], m[:3, 1], m[:3, 2] = f * c - up * s, f * s + up * c, a  # the rotation by the spin angle about the axle
            m[:3, 3] = self.pos + R @ hub_local[i]
            out.append(m)
        return out
