"""A free-flying camera: position plus yaw / pitch / roll, y up, looking down -z at yaw 0."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

UP = np.array([0.0, 1.0, 0.0])
MAX_PITCH = np.radians(89.0)


@dataclass
class Camera:
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw: float = 0.0  # radians, turning right is positive
    pitch: float = 0.0  # radians, looking up is positive
    roll: float = 0.0  # radians, leaning to the right is positive
    fov_deg: float = 90.0  # horizontal field of view
    basis: tuple | None = None  # (forward, right, up) that replaces yaw / pitch / roll, for cameras mounted on a body

    def axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(forward, right, up) unit vectors, roll included."""
        if self.basis is not None:
            return self.basis
        cos_pitch = np.cos(self.pitch)
        forward = np.array([np.sin(self.yaw) * cos_pitch, np.sin(self.pitch), -np.cos(self.yaw) * cos_pitch])
        right = np.cross(forward, UP)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        if self.roll != 0.0:
            cos, sin = np.cos(self.roll), np.sin(self.roll)
            right, up = cos * right - sin * up, cos * up + sin * right
        return forward, right, up

    def look(self, delta_yaw: float, delta_pitch: float) -> None:
        self.yaw = (self.yaw + delta_yaw) % (2.0 * np.pi)
        self.pitch = float(np.clip(self.pitch + delta_pitch, -MAX_PITCH, MAX_PITCH))

    def view_matrix(self) -> np.ndarray:
        forward, right, up = self.axes()
        rotation = np.eye(4)
        rotation[0, :3], rotation[1, :3], rotation[2, :3] = right, up, -forward
        translation = np.eye(4)
        translation[:3, 3] = -self.position
        return rotation @ translation

    def projection_matrix(self, aspect: float, near: float, far: float) -> np.ndarray:
        fov_y = 2.0 * np.arctan(np.tan(np.radians(self.fov_deg) / 2.0) / aspect)
        f = 1.0 / np.tan(fov_y / 2.0)
        mat = np.zeros((4, 4))
        mat[0, 0] = f / aspect
        mat[1, 1] = f
        mat[2, 2] = (far + near) / (near - far)
        mat[2, 3] = 2.0 * far * near / (near - far)
        mat[3, 2] = -1.0
        return mat
