"""RawSlam whose frame-to-frame rotation comes from the gyroscope.

A hovering drone mostly rotates (wobbles, pitches to accelerate) and hardly translates, and from a
handful of pixels of motion a camera alone cannot tell the two apart: the essential matrix mixes
them up. The gyro knows the rotation, so this subclass takes the pictures' rotational flow out
first ("subtract the drone's rotation from the camera's") and only asks the pictures for the
direction of translation. The rotation is then polished with the pictures, but held close to the gyro
by a prior, which is also how the camera ends up correcting the gyro's slow drift.

Inputs are exactly what a real drone has: pictures and gyro samples. No position, no map, no truth.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "3D-SLAM"))

from raw_slam import RawSlam  # noqa: E402

RANSAC_ITERATIONS = 60
INLIER_PX = 1.0
MIN_DEROTATED_PARALLAX_PX = 3.0  # translation-only motion before a pose is computed
GYRO_PRIOR_DEG = 0.4  # how far the pictures may pull the gyro's rotation over one keyframe interval
PIXEL_SIGMA = 0.7
# The step length the pictures give is held to what the accelerometer says the drone moved (see `_step_scale`).
SCALE_BAND_SIGMAS = 3.0
SCALE_BAND_FLOOR_M = 0.04  # error of the inertial displacement over one keyframe interval, metres
SCALE_BAND_RELATIVE = 0.2  # ... and the part of it that grows with the distance moved


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def rotation_vector_to_matrix(v: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.eye(3) + skew(v)
    k = skew(v / angle)
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def epipolar_pixels(rotation: np.ndarray, t: np.ndarray, x1: np.ndarray, x2: np.ndarray, focal: float) -> np.ndarray:
    """Distance (px, first-order) of each point from its epipolar line for the motion x2 ~ R x1 + t."""
    essential = skew(t) @ rotation
    line2 = x1 @ essential.T  # E x1
    line1 = x2 @ essential  # E^T x2
    numerator = np.sum(x2 * line2, axis=1)
    denominator = np.sqrt(line2[:, 0] ** 2 + line2[:, 1] ** 2 + line1[:, 0] ** 2 + line1[:, 1] ** 2)
    return focal * numerator / np.maximum(denominator, 1e-12)


def translation_with_known_rotation(
    rotation: np.ndarray, x1: np.ndarray, x2: np.ndarray, focal: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray] | None:
    """Unit translation t (x2 ~ R x1 + t, up to scale) and the inlier mask: two matches fix t, RANSAC picks the best."""
    count = len(x1)
    if count < 8:
        return None
    normals = np.cross(x1 @ rotation.T, x2)  # t . n = 0 for every match
    best, best_inliers = None, None
    for _ in range(RANSAC_ITERATIONS):
        a, b = rng.choice(count, 2, replace=False)
        t = np.cross(normals[a], normals[b])
        norm = np.linalg.norm(t)
        if norm < 1e-12:
            continue
        t /= norm
        inliers = np.abs(epipolar_pixels(rotation, t, x1, x2, focal)) < INLIER_PX
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best, best_inliers = t, inliers
    if best is None or best_inliers.sum() < 8:
        return None
    # refit on the inliers: the right singular vector of the smallest singular value
    _, _, vt = np.linalg.svd(normals[best_inliers], full_matrices=False)
    t = vt[-1]
    if np.dot(t, best) < 0:
        t = -t
    # the sign: points must lie in front of both cameras
    rotated = x1 @ rotation.T
    cross = np.cross(rotated, x2)
    depth1 = -np.sum(np.cross(t, x2) * cross, axis=1) / np.maximum(np.sum(cross * cross, axis=1), 1e-18)
    inliers = np.abs(epipolar_pixels(rotation, t, x1, x2, focal)) < INLIER_PX
    if np.sum(depth1[inliers] > 0) < 0.5 * inliers.sum():
        t = -t
    return t, inliers


def refine_motion(
    rotation: np.ndarray, t: np.ndarray, x1: np.ndarray, x2: np.ndarray, focal: float
) -> tuple[np.ndarray, np.ndarray]:
    """Levenberg-Marquardt over a small rotation correction (3) and the translation direction (2), gyro rotation as a prior."""
    basis_a = np.cross(t, [1.0, 0.0, 0.0] if abs(t[0]) < 0.9 else [0.0, 1.0, 0.0])
    basis_a /= np.linalg.norm(basis_a)
    basis_b = np.cross(t, basis_a)
    prior = np.radians(GYRO_PRIOR_DEG)

    def residual(p: np.ndarray) -> np.ndarray:
        r = rotation_vector_to_matrix(p[:3]) @ rotation
        direction = t + p[3] * basis_a + p[4] * basis_b
        direction = direction / np.linalg.norm(direction)
        image = epipolar_pixels(r, direction, x1, x2, focal) / PIXEL_SIGMA
        return np.concatenate([np.clip(image, -4.0, 4.0), p[:3] / prior])  # clipped: outliers stop pulling

    p = np.zeros(5)
    damping = 1e-2
    current = residual(p)
    cost = float(current @ current)
    for _ in range(6):
        jacobian = np.empty((len(current), 5))
        for k in range(5):
            step = np.zeros(5)
            step[k] = 1e-5
            jacobian[:, k] = (residual(p + step) - current) / 1e-5
        normal = jacobian.T @ jacobian
        gradient = jacobian.T @ current
        improved = False
        for _ in range(4):
            delta = -np.linalg.solve(normal + damping * np.diag(np.diag(normal) + 1e-9), gradient)
            trial = residual(p + delta)
            trial_cost = float(trial @ trial)
            if trial_cost < cost:
                p, current, cost, improved = p + delta, trial, trial_cost, True
                damping = max(damping * 0.3, 1e-6)
                break
            damping *= 10.0
        if not improved or np.linalg.norm(delta) < 1e-7:
            break
    refined = rotation_vector_to_matrix(p[:3]) @ rotation
    direction = t + p[3] * basis_a + p[4] * basis_b
    return refined, direction / np.linalg.norm(direction)


class ImuAidedSlam(RawSlam):
    def __init__(self, first_frame: np.ndarray, intrinsic: np.ndarray, exclude_mask: np.ndarray | None = None) -> None:
        self.gyro_rotation = np.eye(3)  # camera now -> camera at the first frame, integrated from the gyro
        self.key_gyro = np.eye(3)
        self.rng = np.random.default_rng(1)
        self.rotation_corrections = 0.0  # running sum of how far the pictures moved the gyro rotation (deg), for display
        self.use_hint = True  # hold the step length to the inertial displacement when the flight computer has a calibrated scale
        self.hint: tuple[np.ndarray, float, bool] | None = None  # (body position in metres, SLAM units per metre, scale calibrated)
        self.key_position: np.ndarray | None = None  # the hint's position when the keyframe was set
        super().__init__(first_frame, intrinsic, exclude_mask)
        self.min_parallax = MIN_DEROTATED_PARALLAX_PX

    def process(self, frame: np.ndarray, gyro_rotation: np.ndarray | None = None, motion_hint=None) -> bool:
        """One BGR frame, the gyro-integrated rotation of the camera since the first frame (camera-now -> camera-first), and
        optionally what the flight computer's inertial filter says: (position in metres, SLAM units per metre, scale calibrated)."""
        if gyro_rotation is not None:
            self.gyro_rotation = gyro_rotation
        self.set_motion_hint(motion_hint)
        return super().process(frame)

    def set_motion_hint(self, motion_hint, keyframe: bool = False) -> None:
        self.hint = None if motion_hint is None else (np.asarray(motion_hint[0], dtype=np.float64), float(motion_hint[1]), bool(motion_hint[2]))
        if keyframe and self.hint is not None:
            self.key_position = self.hint[0].copy()

    def _set_keyframe(self) -> None:
        super()._set_keyframe()
        self.key_gyro = self.gyro_rotation.copy()
        self.key_position = None if self.hint is None else self.hint[0].copy()

    def _step_scale(self, ids, current_px, rotation, direction):
        """The map's step length, kept within what the accelerometer says the drone moved since the keyframe.

        A monocular map has no memory of metres except its own landmarks, and in fast flight those are replaced within seconds:
        every step then re-measures the scale from freshly triangulated points and the unit random-walks upward (measured: map
        units per metre went from 4.4 to 130 in 20 s while the drone flew laps). The inertial filter knows the metric scale (learned
        while the unit was young) and the accelerometer gives the distance over a keyframe interval to a few centimetres, so the
        pictures only decide *inside* a band around it (clamped to its edge). In a hover the band is wide (the distance is below the
        inertial noise) and the pictures decide alone. The metric scale handed in should be the one from the camera's start-up
        calibration (fpv.py --scale-ref calibrated), not the filter's current one: the filter learns it from the SLAM, so a drifting
        SLAM would drag its own bound along. Tried and measured worse: replacing an out-of-band step by the inertial one instead of
        clamping it."""
        scale = super()._step_scale(ids, current_px, rotation, direction)
        if scale is None or not self.use_hint or self.hint is None or self.key_position is None or self.commits == 0:
            return scale
        position, per_metre, ready = self.hint
        if not ready:
            return scale
        expected = per_metre * float(np.linalg.norm(position - self.key_position))
        sigma = float(np.hypot(per_metre * SCALE_BAND_FLOOR_M, SCALE_BAND_RELATIVE * expected))
        held = float(np.clip(scale, max(expected - SCALE_BAND_SIGMAS * sigma, 1e-6), expected + SCALE_BAND_SIGMAS * sigma))
        self.telemetry.update(hint_expected=expected, hint_raw=float(scale), hint_held=float(held != scale))
        return held

    def _rays(self, pixels: np.ndarray) -> np.ndarray:
        return np.column_stack([pixels, np.ones(len(pixels))]) @ self.intrinsic_inverse.T

    def _gyro_rotation_since_keyframe(self) -> np.ndarray:
        """Rotation taking key-camera coordinates to current-camera coordinates."""
        return self.gyro_rotation.T @ self.key_gyro

    def _parallax(self, key_px: np.ndarray, current_px: np.ndarray) -> float:
        """Motion left after the gyro's rotation has been taken out of the keyframe's points."""
        rotated = self._rays(key_px) @ self._gyro_rotation_since_keyframe().T
        predicted = rotated[:, :2] / rotated[:, 2:3] * self.intrinsic[0, 0] + self.intrinsic[:2, 2]
        return float(np.median(np.linalg.norm(current_px - predicted, axis=1)))

    def _estimate_motion(self, key_px: np.ndarray, current_px: np.ndarray):
        rotation = self._gyro_rotation_since_keyframe()
        x1, x2 = self._rays(key_px), self._rays(current_px)
        found = translation_with_known_rotation(rotation, x1, x2, self.intrinsic[0, 0], self.rng)
        if found is None:
            return None
        t, inliers = found
        refined, t = refine_motion(rotation, t, x1[inliers], x2[inliers], self.intrinsic[0, 0])
        correction = np.degrees(np.arccos(np.clip((np.trace(refined @ rotation.T) - 1.0) / 2.0, -1.0, 1.0)))
        self.rotation_corrections += float(correction)
        return refined, t, inliers
