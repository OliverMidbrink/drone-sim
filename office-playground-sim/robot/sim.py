"""The robot as a running system: plant at 2 kHz, IMU at 1 kHz, firmware at 200 Hz, with the delays between them.

    plant --(IMU 1 kHz, encoders)--> firmware --(duty, one control period late)--> plant

The firmware sees the newest IMU sample and the encoder counts at the moment it runs, and what it
computes reaches the motors one control period later (the time the microcontroller takes to compute,
read the sensors and write the PWM registers). Walls and furniture are axis-aligned boxes in plan; the robot's
body is a rectangle that is checked against them.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .dynamics import Contact, TwipPlant
from .firmware import CONTROL_RATE, BalanceFirmware, Command
from .params import RobotParams
from .sensors import Encoders, Imu, ImuSample, ImuSpec

PHYSICS_RATE = 1000.0
HISTORY_SECONDS = 8.0


@dataclass
class Box:
    """An obstacle: an axis-aligned box in plan (x, z) with a top height (anything below the robot's body height blocks it)."""

    x_min: float
    z_min: float
    x_max: float
    z_max: float
    height: float = 2.5


def body_corners(x: float, z: float, heading: float, depth: float, width: float) -> np.ndarray:
    forward = np.array([np.cos(heading), np.sin(heading)])
    right = np.array([-np.sin(heading), np.cos(heading)])
    centre = np.array([x, z])
    return np.array([centre + sx * depth / 2.0 * forward + sz * width / 2.0 * right for sx, sz in ((1, 1), (1, -1), (-1, -1), (-1, 1))])


def collide(x: float, z: float, heading: float, depth: float, width: float, boxes: list[Box]) -> list[Contact]:
    """Separating-axis test of the robot's rectangle against each box; the contact normal points at the robot."""
    contacts: list[Contact] = []
    corners = body_corners(x, z, heading, depth, width)
    centre = np.array([x, z])
    axes_robot = [np.array([np.cos(heading), np.sin(heading)]), np.array([-np.sin(heading), np.cos(heading)])]
    for box in boxes:
        if abs(x - 0.5 * (box.x_min + box.x_max)) > 0.5 * (box.x_max - box.x_min) + width or abs(z - 0.5 * (box.z_min + box.z_max)) > 0.5 * (box.z_max - box.z_min) + width:
            continue
        box_corners = np.array([[box.x_min, box.z_min], [box.x_max, box.z_min], [box.x_max, box.z_max], [box.x_min, box.z_max]])
        box_centre = box_corners.mean(axis=0)
        best_depth, best_normal = np.inf, None
        for axis in [np.array([1.0, 0.0]), np.array([0.0, 1.0]), *axes_robot]:
            a, b = corners @ axis, box_corners @ axis
            overlap = min(a.max(), b.max()) - max(a.min(), b.min())
            if overlap <= 0.0:
                best_normal = None
                break
            if overlap < best_depth:
                best_depth = overlap
                best_normal = axis if (centre - box_centre) @ axis >= 0.0 else -axis
        if best_normal is not None:
            contacts.append(Contact((float(best_normal[0]), float(best_normal[1])), float(best_depth)))
    return contacts


@dataclass
class Sample:
    t: float
    pitch: float
    pitch_estimate: float
    speed: float
    speed_estimate: float
    current: float
    battery_voltage: float
    yaw_rate: float


class RobotSim:
    def __init__(
        self,
        x: float = 0.0,
        z: float = 0.0,
        heading: float = 0.0,
        seed: int = 0,
        nominal: RobotParams | None = None,
        true: RobotParams | None = None,
        imu_spec: ImuSpec | None = None,
        spread: float = 1.0,
        obstacles: list[Box] | None = None,
    ) -> None:
        self.nominal = nominal or RobotParams()
        # the real robot differs from the datasheet the firmware was designed with
        self.params = true or self.nominal.perturbed(np.random.default_rng(seed + 1000), spread)
        self.plant = TwipPlant(self.params, x, z, heading)
        self.imu = Imu(imu_spec, seed)
        self.encoders = Encoders(self.plant)
        self.firmware = BalanceFirmware(self.nominal)
        self.obstacles = obstacles or []
        self._bounds = np.array([[b.x_min, b.z_min, b.x_max, b.z_max, getattr(b, "y_min", 0.0), b.height] for b in self.obstacles], float).reshape(-1, 6)
        self._near: list = []
        self.command = Command()
        self.duty = np.zeros(2)  # being applied
        self.pending_duty = np.zeros(2)  # computed, applied after the next control tick
        self.imu_sample: ImuSample = self.imu.sample(self.plant)
        self.held = True  # a hand holds the robot upright until the firmware has calibrated
        self.push_force = 0.0
        self.push_until = 0.0
        self.yaw_kick = 0.0
        self.yaw_kick_until = 0.0
        self.history: deque[Sample] = deque(maxlen=int(HISTORY_SECONDS * CONTROL_RATE))
        self._physics_dt = 1.0 / PHYSICS_RATE
        self._imu_every = int(round(PHYSICS_RATE / self.imu.spec.rate))
        self._control_every = int(round(PHYSICS_RATE / CONTROL_RATE))
        self._tick = 0
        self._residual = 0.0
        self.collisions = 0

    # --- user actions -------------------------------------------------------------------------

    def push(self, force: float = 3.0, duration: float = 0.12) -> None:
        """A shove at the top of the body: `force` newtons forward (negative pushes backwards) for `duration` seconds."""
        self.push_force, self.push_until = force, self.plant.time + duration

    def twist(self, torque: float = 0.4, duration: float = 0.1) -> None:
        self.yaw_kick, self.yaw_kick_until = torque, self.plant.time + duration

    def stand_up(self) -> None:
        """Pick the robot up, stand it upright where it is, and hold it until the firmware is balancing again."""
        p = self.plant
        p.q = np.array([0.0, p.q[1], 0.0, 0.0, 0.0])
        p.qd[:] = 0.0
        p.qdd[:] = 0.0
        p.currents[:] = 0.0
        p.fallen = False
        self.encoders.zero()
        self.firmware.counts_previous = None
        self.firmware.state = "cutoff"
        self.firmware.upright_time = 0.0
        self.duty = np.zeros(2)
        self.pending_duty = np.zeros(2)
        self.held = True

    def place(self, x: float, z: float, heading: float) -> None:
        self.plant.x, self.plant.z = x, z
        self.plant.q[1] = heading
        self.stand_up()

    # --- stepping -----------------------------------------------------------------------------

    @property
    def balancing(self) -> bool:
        return self.firmware.state == "balancing" and not self.plant.fallen

    def step(self) -> None:
        plant, firmware = self.plant, self.firmware
        if self._tick % self._imu_every == 0:
            self.imu_sample = self.imu.sample(plant)
        if self._tick % self._control_every == 0:
            self.duty = self.pending_duty  # what was computed one period ago reaches the motors now
            self.pending_duty = firmware.update(self.imu_sample, self.encoders.counts(), self.command)
            if self.held and firmware.state == "balancing":
                self.held = False
            self._record()
            self._find_near()
        t = plant.time
        push = self.push_force if t < self.push_until else 0.0
        yaw = self.yaw_kick if t < self.yaw_kick_until else 0.0
        contacts = collide(plant.x, plant.z, plant.heading, self.params.body_size[0], self.params.body_size[1], self._near) if self._near else None
        if contacts:
            self.collisions += 1
        plant.step(self._physics_dt, self.duty, push_force=push, yaw_torque=yaw, contacts=contacts, held=self.held)
        self._tick += 1

    def advance(self, seconds: float) -> None:
        """Run the simulation for `seconds` of simulated time."""
        self._residual += seconds
        steps = int(self._residual / self._physics_dt)
        self._residual -= steps * self._physics_dt
        for _ in range(steps):
            self.step()

    def _find_near(self) -> None:
        """The obstacles within reach of the robot over the next control period, so the collision test only looks at a few boxes."""
        if not len(self._bounds):
            return
        x, z, reach = self.plant.x, self.plant.z, 0.45
        b = self._bounds
        tall = self.params.com_height + 0.5 * self.params.body_size[2] + 0.02  # a box only matters if it is in the way of the wheels or the body
        keep = np.nonzero((b[:, 0] - reach < x) & (x < b[:, 2] + reach) & (b[:, 1] - reach < z) & (z < b[:, 3] + reach) & (b[:, 4] < tall) & (b[:, 5] > 0.04))[0]
        self._near = [self.obstacles[i] for i in keep]

    def _record(self) -> None:
        p, f = self.plant, self.firmware
        self.history.append(Sample(p.time, p.pitch, f.pitch, p.speed, f.speed, float(np.mean(np.abs(p.currents))), p.telemetry.battery_voltage, p.yaw_rate))
