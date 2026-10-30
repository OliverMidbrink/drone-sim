"""A person walking about the office: gravity, jumping, walls and furniture that stop you, desks you can jump onto.

The walker is a vertical capsule seen from above as a circle. The furniture and walls come as
axis-aligned boxes (`layout.Box`, with an absolute top height). A box lower than a step above the
feet is something to stand on; a taller one is a wall.
"""

from __future__ import annotations

import numpy as np


class Walker:
    RADIUS = 0.25
    EYE = 1.65  # eye height above the feet
    HEAD = 1.75  # top of the head above the feet
    STEP = 0.30  # highest edge that is stepped up without a jump
    GRAVITY = 9.81
    JUMP_SPEED = 4.4  # about 1 m of height
    WALK = 3.0  # m/s
    SPRINT = 6.0
    CEILING = 3.0

    def __init__(self, boxes, x: float = 0.0, z: float = 0.0, yaw: float = 0.0) -> None:
        self.bounds = np.array([[b.x_min, b.z_min, b.x_max, b.z_max, b.height] for b in boxes], float).reshape(-1, 5)
        self.position = np.array([x, z], float)
        self.feet = 0.0
        self.velocity = np.zeros(2)
        self.vertical = 0.0
        self.grounded = True
        self.yaw = yaw  # camera yaw: 0 looks along -z, positive turns right
        self.pitch = 0.0

    @property
    def eye(self) -> np.ndarray:
        return np.array([self.position[0], self.feet + self.EYE, self.position[1]])

    def teleport(self, x: float, z: float, yaw: float | None = None) -> None:
        self.position = np.array([x, z], float)
        self.feet, self.vertical, self.velocity = 0.0, 0.0, np.zeros(2)
        if yaw is not None:
            self.yaw = yaw

    def look(self, delta_yaw: float, delta_pitch: float) -> None:
        self.yaw = (self.yaw + delta_yaw) % (2.0 * np.pi)
        self.pitch = float(np.clip(self.pitch + delta_pitch, -np.radians(89.0), np.radians(89.0)))

    def _near(self, reach: float) -> np.ndarray:
        b = self.bounds
        x, z = self.position
        keep = (b[:, 0] - reach < x) & (x < b[:, 2] + reach) & (b[:, 1] - reach < z) & (z < b[:, 3] + reach)
        return b[keep]

    def update(self, dt: float, move: np.ndarray, sprint: bool, jump: bool) -> None:
        """`move` is the wanted direction in plan (x, z) with length up to 1; steps of at most 8 ms keep fast motion from tunnelling."""
        steps = max(1, int(np.ceil(dt / 0.008)))
        for _ in range(steps):
            self._step(dt / steps, move, sprint, jump)
            jump = False

    def _step(self, dt: float, move: np.ndarray, sprint: bool, jump: bool) -> None:
        wanted = move * (self.SPRINT if sprint else self.WALK)
        grip = 14.0 if self.grounded else 1.5  # barely any steering in the air
        self.velocity += (wanted - self.velocity) * min(grip * dt, 1.0)
        if jump and self.grounded:
            self.vertical, self.grounded = self.JUMP_SPEED, False

        self.position = self.position + self.velocity * dt
        near = self._near(self.RADIUS + 0.1)
        for x0, z0, x1, z1, top in near:
            if top <= self.feet + self.STEP:
                continue  # low enough to step onto
            closest = np.clip(self.position, (x0, z0), (x1, z1))
            offset = self.position - closest
            distance = float(np.hypot(*offset))
            if distance >= self.RADIUS:
                continue
            if distance > 1e-9:
                normal = offset / distance
                self.position = closest + normal * self.RADIUS
            else:  # the centre is inside the box: leave by the nearest face
                gaps = np.array([self.position[0] - x0, x1 - self.position[0], self.position[1] - z0, z1 - self.position[1]])
                face = int(gaps.argmin())
                normal = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)][face], float)
                self.position = np.where(normal != 0, (x0, x1, z0, z1)[face] + normal * self.RADIUS, self.position)
            into = float(self.velocity @ normal)
            if into < 0.0:
                self.velocity -= into * normal  # slide along it

        # vertical: stand on the floor or on whatever is low enough under the circle
        support = 0.0
        for x0, z0, x1, z1, top in self._near(self.RADIUS):
            if top > support and top <= self.feet + self.STEP:
                closest = np.clip(self.position, (x0, z0), (x1, z1))
                if np.hypot(*(self.position - closest)) < self.RADIUS:
                    support = float(top)
        self.vertical -= self.GRAVITY * dt
        self.feet += self.vertical * dt
        if self.feet <= support and self.vertical <= 0.0:
            self.feet, self.vertical, self.grounded = support, 0.0, True
        else:
            self.grounded = self.feet <= support + 1e-3 and self.vertical <= 0.0
        head = self.CEILING - 0.05 - self.HEAD
        if self.feet > head:
            self.feet, self.vertical = head, min(self.vertical, 0.0)
