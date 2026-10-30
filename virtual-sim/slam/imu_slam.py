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

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "3D-SLAM"))

from raw_slam import MAX_HAMMING, MIN_PNP_INLIERS, RawSlam  # noqa: E402

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
        self.policy = None  # a slam_worker.FixPolicy: how anchor fixes are judged and applied (None = raw: every fix whole)
        self.refused_fixes = 0
        self.banned: set[int] = set()  # anchor rows that pulled suspicious fixes away from the inertial pose too often
        self.strikes: dict[int, int] = {}
        self.path_length = 0.0  # metres flown (inertial position, summed) since the last accepted fix
        self.hint_time: float | None = None
        self.last_fix_time: float | None = None
        self._last_hint_position: np.ndarray | None = None
        self.map_policy = None  # a slam_worker.MapPolicy: how anchors are made, merged, matched and culled (None = raw)
        self.fused = 0  # new anchors that turned out to be an existing anchor seen again (linked to it instead of duplicated)
        self.anchor_score: dict[int, list[int]] = {}  # anchor row -> [fixes it was an inlier of, fixes it was matched but an outlier]
        super().__init__(first_frame, intrinsic, exclude_mask)
        self.min_parallax = MIN_DEROTATED_PARALLAX_PX

    def process(self, frame: np.ndarray, gyro_rotation: np.ndarray | None = None, motion_hint=None) -> bool:
        """One BGR frame, the gyro-integrated rotation of the camera since the first frame (camera-now -> camera-first), and
        optionally what the flight computer's inertial filter says: (position in metres, SLAM units per metre, scale calibrated)."""
        if gyro_rotation is not None:
            self.gyro_rotation = gyro_rotation
        self.set_motion_hint(motion_hint)
        self.telemetry["fix_refused"] = 0.0
        m = self.map_policy
        self.carry_scale = not (m is not None and m.rigid_carry and self.hint is not None and self.hint[2])
        return super().process(frame)

    def set_motion_hint(self, motion_hint, keyframe: bool = False) -> None:
        self.hint = None if motion_hint is None else (np.asarray(motion_hint[0], dtype=np.float64), float(motion_hint[1]), bool(motion_hint[2]))
        if self.hint is not None:
            if len(motion_hint) > 3:
                self.hint_time = float(motion_hint[3])
                if self.last_fix_time is None:
                    self.last_fix_time = self.hint_time
            if self._last_hint_position is not None:
                self.path_length += float(np.linalg.norm(self.hint[0] - self._last_hint_position))
            self._last_hint_position = self.hint[0].copy()
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

    # ------------------------------------------------------------------ anchor fixes, judged against the inertial data

    def _fix_budget(self) -> tuple[float, float]:
        """(SLAM units per metre, metres): how far the current pose may plausibly be from the pose a good fix gives.

        Between two accepted fixes the pose is dead-reckoned: each step is held to the accelerometer's displacement within a band
        (`_step_scale`), so the error grows with the distance flown, and the accelerometer's own error adds 0.5*a*T^2 over the time T. The
        fix itself has a noise floor (frozen anchors carry the triangulation errors they were made with)."""
        p = self.policy
        elapsed = 0.0 if self.hint_time is None or self.last_fix_time is None else max(self.hint_time - self.last_fix_time, 0.0)
        return max(self.hint[1], 1e-3), p.sigma_m + p.drift_rel * self.path_length + 0.5 * p.drift_acc * elapsed * elapsed

    def _filter_matches(self, image, world, rows):
        p = self.policy
        if p is None or not p.prior_gate or self.hint is None or not self.hint[2]:
            return super()._filter_matches(image, world, rows)
        lam, budget = self._fix_budget()
        in_camera = (world - self.world_position) @ self.world_rotation  # camera axes
        depth = in_camera[:, 2]
        focal = self.intrinsic[0, 0]
        expected = in_camera[:, :2] / np.maximum(depth, 1e-6)[:, None] * focal + self.intrinsic[:2, 2]
        allowed = 6.0 + focal * (0.035 + p.z_max * budget * lam / np.maximum(depth, 1e-3))  # px: 2 degrees of attitude plus the position budget at the anchor's depth
        keep = (depth > 1e-3) & (np.linalg.norm(expected - image, axis=1) < allowed)
        self.telemetry["prior_kept"] = float(keep.mean()) if len(keep) else 1.0
        return keep

    def _apply_fix(self, fix: dict) -> None:
        p = self.policy
        if p is None or self.hint is None or not self.hint[2]:
            self.path_length, self.last_fix_time = 0.0, self.hint_time
            return super()._apply_fix(fix)
        lam, budget = self._fix_budget()
        move = float(np.linalg.norm(fix["position"] - self.world_position)) / lam
        z = move / budget
        self.telemetry.update(fix_z=z, fix_budget=budget, fix_refused=0.0, fix_flagged=0.0)
        if p.blame and z > p.z_blame:
            self.telemetry["fix_flagged"] = float(self._blame(fix, lam, budget))
        if p.guard and z > p.z_max:
            self.refused_fixes += 1
            self.telemetry["fix_refused"] = 1.0
            return
        if p.gain < 1.0:
            relative = cv2.Rodrigues(fix["rotation"] @ self.world_rotation.T)[0].ravel() * p.gain
            fix = dict(fix, position=self.world_position + p.gain * (fix["position"] - self.world_position), rotation=cv2.Rodrigues(relative)[0] @ self.world_rotation)
        self.path_length, self.last_fix_time = 0.0, self.hint_time
        super()._apply_fix(fix)

    def _blame(self, fix: dict, lam: float, budget: float) -> int:
        """Which anchors of this fix pull it away from the inertial pose? Strike them; ban the repeat offenders. Returns how many were flagged.

        The pose is re-fitted with the current (dead-reckoned) pose as a prior, its width the position budget, robust (Huber) to
        outliers. Anchors that agree with the inertial pose fit well; a few anchors that drag the fix elsewhere are left with
        large residuals. If every anchor has a similar residual the whole fix is simply a correction and nobody is flagged."""
        pre = self.world_rotation.T
        x_pre = np.concatenate([cv2.Rodrigues(pre)[0].ravel(), -pre @ self.world_position])
        found = fix["rotation"].T
        x = np.concatenate([cv2.Rodrigues(found)[0].ravel(), -found @ fix["position"]])
        prior = np.array([1.0 / np.radians(1.5)] * 3 + [1.0 / max(budget * lam, 1e-6)] * 3)
        world, image = fix["world"], fix["image"]
        for _ in range(8):
            projected, jacobian = cv2.projectPoints(world, x[:3], x[3:], self.intrinsic, None)
            residual = projected.reshape(-1, 2) - image
            size = np.linalg.norm(residual, axis=1)
            weight = np.repeat(np.where(size <= 2.0, 1.0, 2.0 / np.maximum(size, 1e-9)), 2)
            j = jacobian[:, :6]
            step = -np.linalg.solve(j.T @ (j * weight[:, None]) + np.diag(prior**2), j.T @ (weight * residual.ravel()) + prior**2 * (x - x_pre))
            x = x + step
            if np.linalg.norm(step) < 1e-7:
                break
        projected, _ = cv2.projectPoints(world, x[:3], x[3:], self.intrinsic, None)
        size = np.linalg.norm(projected.reshape(-1, 2) - image, axis=1)
        flagged = size > 2.0 * np.median(size) + 3.0
        for row in fix["rows"][flagged]:
            row = int(row)
            self.strikes[row] = self.strikes.get(row, 0) + 1
            if self.strikes[row] >= self.policy.strikes:
                self.banned.add(row)
        return int(flagged.sum())


    # ------------------------------------------------------------------ the anchor map (MapPolicy): what ORB-SLAM does for its map points
    def _anchor_ready(self, track: int) -> bool:
        """Anchors are frozen for good, so only well-triangulated landmarks become one (ORB-SLAM: a parallax check at creation and
        culling of points seen from too few keyframes). A landmark is anchored once it has been triangulated over enough steps and
        across a wide enough angle; before that its depth can be off by tens of percent."""
        m = self.map_policy
        if m is None:
            return True
        return self.landmark_views.get(track, 0) >= m.anchor_min_views and self.landmark_angle.get(track, 0.0) >= m.anchor_min_angle_deg

    def _project_points(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(pixels, depth) of world points under the current pose."""
        in_camera = (points - self.world_position) @ self.world_rotation
        depth = in_camera[:, 2]
        pixels = in_camera[:, :2] / np.maximum(depth, 1e-6)[:, None] * self.intrinsic[0, 0] + self.intrinsic[:2, 2]
        return pixels, depth

    def _fuse_anchors(self, keypoints, descriptors, rows, chosen, ids, positions):
        """A landmark about to be anchored may be a place already anchored (a second lap, a track lost and found again). If an
        existing anchor projects next to it, looks the same and sits at about the same depth, the track is linked to that anchor
        instead of making a second copy at a slightly different position (ORB-SLAM's map point fusion). Duplicates make the
        matcher's ratio test reject both copies, and copies made in different laps carry different drift and pull fixes apart."""
        m = self.map_policy
        if m is None or not m.fuse or len(self.anchor_points) == 0:
            return rows, chosen
        pixels, depth = self._project_points(self.anchor_points)
        usable = depth > 1e-3
        if self.banned:
            usable[np.fromiter(self.banned, np.int64)] = False
        candidates = np.flatnonzero(usable)
        if len(candidates) == 0:
            return rows, chosen
        _, live_depth = self._project_points(positions[chosen])
        keep_rows, keep_chosen = [], []
        for row, k, d_live in zip(rows, chosen, live_depth):
            near = candidates[np.linalg.norm(pixels[candidates] - np.asarray(keypoints[row].pt), axis=1) < m.fuse_px]
            if len(near):
                distance = hamming(descriptors[row], self.anchor_descriptors[near])
                best = int(np.argmin(distance))
                ratio = depth[near[best]] / max(float(d_live), 1e-6)
                if distance[best] <= m.fuse_hamming and abs(ratio - 1.0) < m.fuse_depth_tolerance:
                    self.anchor_row[ids[k]] = int(near[best])
                    self.fused += 1
                    continue
            keep_rows.append(row)
            keep_chosen.append(k)
        self.telemetry["anchors_fused"] = float(self.fused)
        return keep_rows, keep_chosen

    def _match_anchors(self, keypoints, descriptors, old):
        """Search by projection: each old anchor is only compared with the picture's features near where the current pose says it
        should appear (ORB-SLAM's SearchByProjection), the ratio test among those few. Matching against the whole map only when that
        finds too little (relocalisation)."""
        m = self.map_policy
        self.matched_by_projection = False
        if m is None or not m.guided or len(keypoints) == 0:
            return super()._match_anchors(keypoints, descriptors, old)
        pixels, depth = self._project_points(self.anchor_points[old])
        size = 2.0 * self.intrinsic[:2, 2]
        inside = (depth > 1e-3) & (pixels[:, 0] > -m.guided_px) & (pixels[:, 1] > -m.guided_px) & (pixels[:, 0] < size[0] + m.guided_px) & (pixels[:, 1] < size[1] + m.guided_px)
        candidates, pixels = old[inside], pixels[inside]
        points = np.array([k.pt for k in keypoints])
        if len(candidates):
            a, k = np.nonzero(np.linalg.norm(pixels[:, None, :] - points[None, :, :], axis=2) < m.guided_px)
        else:
            a = k = np.empty(0, np.int64)
        image, rows = np.empty((0, 2)), np.empty(0, np.int64)
        if len(a):
            distance = np.unpackbits(np.bitwise_xor(self.anchor_descriptors[candidates[a]], descriptors[k]), axis=1).sum(axis=1)
            order = np.lexsort((distance, a))
            a, k, distance = a[order], k[order], distance[order]
            first = np.r_[True, a[1:] != a[:-1]]  # each anchor's best keypoint
            second = np.r_[a[1:] == a[:-1], False] & first
            next_distance = np.r_[distance[1:], 0]
            ok = first & (distance <= MAX_HAMMING) & (~second | (distance < m.guided_ratio * next_distance))
            a, k, distance = a[ok], k[ok], distance[ok]
            order = np.lexsort((distance, k))  # one anchor per keypoint: the closest in appearance
            a, k = a[order], k[order]
            unique = np.r_[True, k[1:] != k[:-1]]
            image, rows = points[k[unique]], candidates[a[unique]]
        self.telemetry.update(guided_matches=float(len(rows)), guided_inview=float(len(candidates)))
        self.matched_by_projection = len(rows) >= MIN_PNP_INLIERS or not m.global_fallback
        if not self.matched_by_projection:
            return super()._match_anchors(keypoints, descriptors, old)
        return image, rows, len(rows)

    def _anchor_usable(self, rows: np.ndarray) -> np.ndarray:
        if not self.banned:
            return np.ones(len(rows), bool)
        return ~np.isin(rows, np.fromiter(self.banned, np.int64))

    def _solve_fix(self, keypoints, descriptors):
        """Also keeps score of every anchor a fix matched: inlier or outlier of the pose. Anchors that keep landing on the wrong side
        are culled (ORB-SLAM culls map points that are found in too few of the frames that should see them)."""
        fix = super()._solve_fix(keypoints, descriptors)
        m = self.map_policy
        if fix is not None and m is not None and m.cull and getattr(self, "matched_by_projection", False):  # a whole-map match is often simply a wrong match: no blame
            inliers = set(int(r) for r in fix["rows"])
            for row in fix["matched_rows"]:
                row = int(row)
                score = self.anchor_score.setdefault(row, [0, 0])
                score[0 if row in inliers else 1] += 1
                if score[1] >= m.cull_strikes and score[1] > 2 * score[0]:
                    self.banned.add(row)
            self.telemetry["anchors_banned"] = float(len(self.banned))
        return fix


def hamming(one: np.ndarray, many: np.ndarray) -> np.ndarray:
    """Bit distance between one ORB descriptor (32 bytes) and each row of many."""
    return np.unpackbits(np.bitwise_xor(many, one[None, :]), axis=1).sum(axis=1)
