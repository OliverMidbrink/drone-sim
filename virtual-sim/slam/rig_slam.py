"""One SLAM for the whole camera rig: the two fisheyes share a single map, a single set of anchors, a single scale and a single pose.

The two cameras are rigid on the drone (known mounting rotation and a known offset from the drone's centre: 6 cm to each side), so there is
nothing to estimate between them. What is estimated is the pose of the *body* in the SLAM frame S (the body at the first picture is the
origin and its axes are the axes), and every camera is just a window onto that one pose:

    camera axes -> S   R_sb @ R_bc           camera centre in S   p + R_sb @ (lambda * lever)

where R_bc is the mounting rotation, lever the camera's offset from the body centre in metres and lambda the SLAM units per metre (only
known once the flight computer's inertial filter has calibrated it; until then the lever arm is taken as zero, a 6 cm error on a map
whose first baseline defines the unit).

What is shared, compared with two independent SLAMs (`ImuAidedSlam`, one per camera):
    * Landmarks and anchors live in one frame: an anchor made by the left camera is matched by the right one and vice versa, so after the
      drone turns, the camera that now looks where the other one looked finds that side's map.
    * One keyframe for both cameras. A step is taken when either has enough motion, every camera that has contributes the direction
      of the translation (epipolar geometry with the gyro's rotation, as in `ImuAidedSlam`), the step length comes from the shared
      landmarks, and the body's displacement is the weighted mean of what each camera says. There is no second chain of poses that could
      drift away from the first.
    * An anchor fix solves ONE pose from the matches of both cameras at once (a generalised PnP: hypotheses from each camera's own
      RANSAC, the best one by inliers over both, then a joint refinement). Matches all around the drone pin the pose down much better
      than the 110 degrees of one camera, which is what keeps a position error and a rotation error from trading off against each other.

Inputs are what a real drone has: pictures, the gyro, and optionally the inertial filter's position and metric scale. Nothing else.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "3D-SLAM"))

import raw_slam as rs  # noqa: E402
from imu_slam import (  # noqa: E402
    GYRO_PRIOR_DEG,
    MIN_DEROTATED_PARALLAX_PX,
    PIXEL_SIGMA,
    SCALE_BAND_FLOOR_M,
    SCALE_BAND_RELATIVE,
    SCALE_BAND_SIGMAS,
    hamming,
    refine_motion,
    rotation_vector_to_matrix,
    skew,
    translation_with_known_rotation,
)

OPTS = set(filter(None, os.environ.get("RIG_OPTS", "").split(",")))
ID_BASE = 10_000_000  # track ids are camera_index * ID_BASE + running number: unique across the rig
HYPOTHESIS_MIN_MATCHES = 12  # a camera needs this many anchor matches to propose a pose of its own
MIN_STEP_PARALLAX_SHARE = 0.5  # a camera takes part in a step when its parallax is at least this share of the trigger


def sampson_px(rotation: np.ndarray, t: np.ndarray, b1: np.ndarray, b2: np.ndarray, focal: float) -> np.ndarray:
    """Distance (px, first order) of each match from its epipolar constraint for x2 ~ R x1 + t, for unit rays in any common axes."""
    essential = skew(t) @ rotation
    line2 = b1 @ essential.T  # E b1
    line1 = b2 @ essential  # E^T b2
    numerator = np.sum(b2 * line2, axis=1)
    line2 = line2 - b2 * np.sum(line2 * b2, axis=1, keepdims=True)  # only the part across each ray moves a pixel
    line1 = line1 - b1 * np.sum(line1 * b1, axis=1, keepdims=True)
    return focal * numerator / np.maximum(np.sqrt(np.sum(line2 * line2, axis=1) + np.sum(line1 * line1, axis=1)), 1e-12)


@dataclass
class RigCamera:
    name: str
    body_from_cv: np.ndarray  # OpenCV camera axes -> body axes
    lever_m: np.ndarray  # lens position in the body frame, metres


class CameraTrack:
    """Feature tracks of one camera (Lucas-Kanade), its keyframe, and where each track was born. The map itself is the rig's."""

    def __init__(self, index: int, spec: RigCamera, intrinsic: np.ndarray) -> None:
        self.index, self.spec = index, spec
        self.intrinsic = intrinsic
        self.intrinsic_inverse = np.linalg.inv(intrinsic)
        self.next_id = index * ID_BASE
        self.track_points = np.empty((0, 2), np.float32)
        self.track_ids = np.empty(0, np.int64)
        self.key_points = np.empty((0, 2), np.float32)
        self.key_ids = np.empty(0, np.int64)
        self.birth: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}  # track -> (pixel, camera->S rotation, camera centre in S) where it first appeared
        self.prev_gray: np.ndarray | None = None
        # this camera's keyframe: the body pose then (S), the gyro then and the inertial position then. Each camera keys on its own
        # schedule: a camera that sat out a step (too little parallax) keeps its keyframe and its baseline keeps growing.
        self.key_rotation = np.eye(3)
        self.key_position = np.zeros(3)
        self.key_gyro = np.eye(3)
        self.key_hint: np.ndarray | None = None
        self.orb = cv2.ORB_create(nfeatures=rs.ORB_FEATURES)

    def rays(self, pixels: np.ndarray) -> np.ndarray:
        return np.column_stack([pixels, np.ones(len(pixels))]) @ self.intrinsic_inverse.T

    def add_tracks(self, gray: np.ndarray) -> None:
        wanted = rs.MAX_FEATURES - len(self.track_points)
        if wanted <= 0:
            return
        mask = np.full(gray.shape, 255, np.uint8)
        for x, y in self.track_points:
            cv2.circle(mask, (int(x), int(y)), rs.FEATURE_MIN_DISTANCE, 0, -1)
        found = cv2.goodFeaturesToTrack(gray, wanted, rs.FEATURE_QUALITY, rs.FEATURE_MIN_DISTANCE, mask=mask)
        if found is None:
            return
        fresh = found.reshape(-1, 2).astype(np.float32)
        ids = np.arange(self.next_id, self.next_id + len(fresh), dtype=np.int64)
        self.next_id += len(fresh)
        self.track_points = np.vstack([self.track_points, fresh])
        self.track_ids = np.concatenate([self.track_ids, ids])

    def track(self, gray: np.ndarray) -> None:
        if len(self.track_points) >= 1:
            height, width = gray.shape
            moved, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.track_points.reshape(-1, 1, 2), None, **rs.LK_PARAMS)
            moved = moved.reshape(-1, 2)
            keep = (status.reshape(-1) == 1) & (moved[:, 0] >= 0) & (moved[:, 0] < width - 1) & (moved[:, 1] >= 0) & (moved[:, 1] < height - 1)
            self.track_points, self.track_ids = moved[keep], self.track_ids[keep]

    def set_keyframe(self, rotation_cs: np.ndarray, centre_s: np.ndarray) -> None:
        order = np.argsort(self.track_ids)
        self.key_ids = self.track_ids[order]
        self.key_points = self.track_points[order]
        alive = {int(i) for i in self.track_ids}
        for track, pixel in zip(self.track_ids, self.track_points):
            self.birth.setdefault(int(track), (pixel.astype(np.float64), rotation_cs.copy(), centre_s.copy()))
        self.birth = {i: v for i, v in self.birth.items() if i in alive}


class RigSlam:
    def __init__(self, cameras: list[RigCamera], intrinsic: np.ndarray, first_frames: dict[str, np.ndarray]) -> None:
        self.cams = [CameraTrack(i, spec, intrinsic) for i, spec in enumerate(cameras)]
        self.by_name = {c.spec.name: c for c in self.cams}
        self.intrinsic = intrinsic
        self.rotation = np.eye(3)  # body -> S
        self.position = np.zeros(3)  # body centre in S
        self.gyro_body = np.eye(3)  # body now -> body at the first picture, integrated from the gyro
        self.pose_gyro = np.eye(3)  # gyro_body when self.rotation was last set: the rotation since is the gyro's
        self.rng = np.random.default_rng(1)
        self.hint: tuple[np.ndarray, float, bool] | None = None
        self.use_hint = True
        self.own_keys = "shared-keys" not in OPTS
        self.policy = None  # a slam_worker.FixPolicy (None = every fix whole)
        self.map_policy = None  # a slam_worker.MapPolicy (None = the raw anchor map)
        self.refused_fixes = 0
        self.side_picks = 0
        self.banned: set[int] = set()
        self.anchor_score: dict[int, list[int]] = {}
        self.fused = 0
        self.landmark_views: dict[int, int] = {}
        self.landmark_angle: dict[int, float] = {}
        self.path_length = 0.0  # metres flown (inertial) since the last accepted fix
        self.hint_time: float | None = None
        self.last_fix_time: float | None = None
        self._last_hint_position: np.ndarray | None = None
        self.commits = 0
        self.fixes = 0
        self.make_anchors = True
        self.rotation_corrections = 0.0
        self.skipped_no_scale = 0
        self.restarts = 0
        self.failed_fits = 0
        self.rekeys_without_step = 0
        self.telemetry: dict[str, float] = {}
        self.landmarks: dict[int, np.ndarray] = {}  # track id -> [sum of weight * position (3), sum of weights], in S
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.anchor_descriptors = np.empty((0, 32), np.uint8)
        self.anchor_points = np.empty((0, 3))
        self.anchor_step = np.empty(0, np.int64)
        self.anchor_row: dict[int, int] = {}
        for cam in self.cams:
            gray = cv2.cvtColor(first_frames[cam.spec.name], cv2.COLOR_BGR2GRAY)
            cam.add_tracks(gray)
            cam.prev_gray = gray
        self._set_keyframe()

    # ------------------------------------------------------------------ geometry
    def set_hint(self, hint) -> None:
        self.hint = None if hint is None else (np.asarray(hint[0], dtype=np.float64), float(hint[1]), bool(hint[2]))
        if self.hint is not None:
            if len(hint) > 3:
                self.hint_time = float(hint[3])
                if self.last_fix_time is None:
                    self.last_fix_time = self.hint_time
            if self._last_hint_position is not None:
                self.path_length += float(np.linalg.norm(self.hint[0] - self._last_hint_position))
            self._last_hint_position = self.hint[0].copy()

    def _lever(self, cam: CameraTrack) -> np.ndarray:
        """The lens offset in S units, with the filter's scale from the first picture on (its estimate before calibration, which only
        drifts slowly). Switching the offset on at calibration moved every camera centre by 6 cm * scale at once, under landmarks and
        birth views computed without it. `lever-at-cal` restores that; `no-lever` treats the rig as a point."""
        if self.hint is None or "no-lever" in OPTS or ("lever-at-cal" in OPTS and not self.hint[2]):
            return np.zeros(3)
        return cam.spec.lever_m * float(np.clip(self.hint[1], 0.1, 100.0))

    def camera_pose(self, cam: CameraTrack, rotation: np.ndarray | None = None, position: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(camera -> S rotation, camera centre in S) for a body pose (default: the current one)."""
        r = self.rotation if rotation is None else rotation
        p = self.position if position is None else position
        return r @ cam.spec.body_from_cv, p + r @ self._lever(cam)

    def _predicted_rotation(self) -> np.ndarray:
        """The body's rotation now: the last pose's, carried on by the gyro."""
        return self.rotation @ self.pose_gyro.T @ self.gyro_body

    def _set_keyframe(self, cams: list | None = None) -> None:
        """Key the given cameras (default all) at the current pose. The rotation is first brought up to date with the gyro: a keyframe
        set between steps would otherwise start from the last step's rotation and lose the turn since (it was, before)."""
        self.rotation = self._predicted_rotation()
        self.pose_gyro = self.gyro_body.copy()
        for cam in self.cams if cams is None else cams:
            cam.set_keyframe(*self.camera_pose(cam))
            cam.key_rotation, cam.key_position = self.rotation.copy(), self.position.copy()
            cam.key_gyro = self.gyro_body.copy()
            cam.key_hint = None if self.hint is None else self.hint[0].copy()
        alive = set().union(*[set(map(int, c.track_ids)) for c in self.cams])
        self.landmarks = {i: v for i, v in self.landmarks.items() if i in alive}
        self.landmark_views = {i: v for i, v in self.landmark_views.items() if i in self.landmarks}
        self.landmark_angle = {i: v for i, v in self.landmark_angle.items() if i in self.landmarks}

    def _camera_gyro_since_key(self, cam: CameraTrack) -> np.ndarray:
        """Rotation taking key-camera coordinates to current-camera coordinates, from the gyro."""
        rbc = cam.spec.body_from_cv
        return rbc.T @ (self.gyro_body.T @ cam.key_gyro) @ rbc

    # ------------------------------------------------------------------ frames
    def process(self, frames: dict[str, np.ndarray], gyro_body: np.ndarray | None = None, motion_hint=None) -> bool:
        """One picture per camera (BGR, taken at the same moment), the gyro-integrated body rotation since the first picture, and the inertial hint."""
        self.telemetry["fix_now"] = 0.0
        if gyro_body is not None:
            self.gyro_body = gyro_body
        self.set_hint(motion_hint)
        grays = {}
        for cam in self.cams:
            if cam.spec.name in frames:
                grays[cam.spec.name] = cv2.cvtColor(frames[cam.spec.name], cv2.COLOR_BGR2GRAY)
                cam.track(grays[cam.spec.name])
        changed, rekey = self._try_step(grays)
        for cam in self.cams:
            if cam.spec.name in grays and len(cam.track_points) < rs.REFILL_BELOW:
                cam.add_tracks(grays[cam.spec.name])
        if rekey:
            if not changed:
                self.rekeys_without_step += 1
            self._set_keyframe(rekey)
        for cam in self.cams:
            if cam.spec.name in grays:
                cam.prev_gray = grays[cam.spec.name]
        self.telemetry["rekeys_no_step"] = float(self.rekeys_without_step)
        return changed

    def _try_step(self, grays: dict[str, np.ndarray]) -> tuple[bool, list]:
        """(a new pose?, the cameras to key afresh)."""
        usable = []  # (camera, key pixels, current pixels, ids, derotated parallax)
        lost = []  # cameras with nothing left to step from: keyed afresh
        for cam in self.cams:
            if cam.spec.name not in grays:
                continue
            if len(cam.track_points) < 8 or len(cam.key_ids) == 0:
                lost.append(cam)
                continue
            position = np.minimum(np.searchsorted(cam.key_ids, cam.track_ids), len(cam.key_ids) - 1)
            matched = cam.key_ids[position] == cam.track_ids
            if matched.sum() < rs.KEYFRAME_MIN_MATCHES:
                lost.append(cam)
                continue
            key_px, current_px = cam.key_points[position[matched]], cam.track_points[matched]
            rotated = cam.rays(key_px) @ self._camera_gyro_since_key(cam).T
            predicted = rotated[:, :2] / rotated[:, 2:3] * self.intrinsic[0, 0] + self.intrinsic[:2, 2]
            usable.append((cam, key_px, current_px, cam.track_ids[matched], float(np.median(np.linalg.norm(current_px - predicted, axis=1)))))
        everyone = [c for c in self.cams if c.spec.name in grays]
        if not usable:
            return False, everyone  # no camera has a keyframe worth keeping
        trigger = max(u[4] for u in usable)
        if trigger < MIN_DEROTATED_PARALLAX_PX:
            return False, lost if self.own_keys else (everyone if lost else [])
        if self.own_keys:
            fits, new_rotation = self._motion_own_keys(usable)
        elif "separate-motion" in OPTS:
            fits, body_since_key = self._motion_per_camera(usable, self.gyro_body.T @ usable[0][0].key_gyro)
            new_rotation = usable[0][0].key_rotation @ body_since_key.T
        else:
            fits, body_since_key = self._motion_joint(usable, self.gyro_body.T @ usable[0][0].key_gyro)
            new_rotation = usable[0][0].key_rotation @ body_since_key.T
        if not fits:
            self.failed_fits += 1
            if trigger > 4.0 * MIN_DEROTATED_PARALLAX_PX:
                return False, ([u[0] for u in usable if u[4] > 4.0 * MIN_DEROTATED_PARALLAX_PX] + lost) if self.own_keys else everyone
            return False, lost if self.own_keys else []

        # --- the step length from the shared map, per camera, then the body's displacement
        restarted = False
        if self.commits > 0 and len(self.landmarks) < rs.MIN_SCALE_LANDMARKS:
            self.restarts += 1
            self.landmarks = {}
            restarted = True  # a new unit, as for the very first step
        positions, step_weights, directions, stepped = [], [], [], []
        for cam, ids, current_px, refined, direction, _, count, parallax in fits:
            direction_s = cam.key_rotation @ cam.spec.body_from_cv @ direction
            if self.commits == 0 or restarted:
                scale = 1.0
            else:
                scale = self._step_scale(cam, ids, current_px, cam.key_rotation, cam.key_position, new_rotation, direction)
                if scale is None:
                    continue
                scale = self._hold_to_inertial(scale, cam.key_hint)
            lever = self._lever(cam)
            positions.append(cam.key_position + direction_s * scale - (new_rotation - cam.key_rotation) @ lever)  # the lens swings about the body centre when the body turns
            step_weights.append(count * parallax)
            directions.append(direction_s)
            stepped.append(cam)
        if not positions:
            self.skipped_no_scale += 1
            return False, lost if self.own_keys else []
        self.telemetry["cam_disagree_deg"] = float(np.degrees(np.arccos(np.clip(directions[0] @ directions[1], -1.0, 1.0)))) if len(directions) == 2 else float("nan")
        self.telemetry["cam_disagree_pos"] = float(np.linalg.norm(positions[0] - positions[1])) if len(positions) == 2 else float("nan")
        self.telemetry["step_cams"] = float(len(directions))
        self.position = np.average(np.array(positions), axis=0, weights=step_weights)
        self.rotation = new_rotation
        self.pose_gyro = self.gyro_body.copy()
        self.commits += 1
        for cam, ids, current_px, *_ in fits:
            self._observe(cam, ids, current_px)
        self._anchors_and_fix(grays)
        if not self.own_keys:
            return True, everyone
        return True, stepped + [c for c in lost if c not in stepped]

    def _motion_own_keys(self, usable: list) -> tuple[list, np.ndarray]:
        """Each taking-part camera fits its own motion since its own keyframe (with the gyro's rotation); the body's rotation now is the
        gyro's, corrected by the inlier-weighted mean of what the cameras say."""
        predicted = self._predicted_rotation()
        fits, corrections, weights = [], [], []
        for cam, key_px, current_px, ids, parallax in usable:
            if parallax < MIN_STEP_PARALLAX_SHARE * MIN_DEROTATED_PARALLAX_PX:
                continue
            rotation = self._camera_gyro_since_key(cam)
            x1, x2 = cam.rays(key_px), cam.rays(current_px)
            found = translation_with_known_rotation(rotation, x1, x2, self.intrinsic[0, 0], self.rng)
            if found is None:
                continue
            t, inliers = found
            refined, t = refine_motion(rotation, t, x1[inliers], x2[inliers], self.intrinsic[0, 0])
            rbc = cam.spec.body_from_cv
            says = cam.key_rotation @ (rbc @ refined @ rbc.T).T  # the body's rotation now (body -> S) according to this camera
            corrections.append(cv2.Rodrigues(says @ predicted.T)[0].ravel())
            weights.append(int(inliers.sum()))
            fits.append((cam, ids[inliers], current_px[inliers], None, -(refined.T @ t), None, int(inliers.sum()), parallax))
        if not fits:
            return [], predicted
        mean = np.average(np.array(corrections), axis=0, weights=weights) if "gyro-rot" not in OPTS else np.zeros(3)
        self.rotation_corrections += float(np.degrees(np.linalg.norm(mean)))
        return fits, rotation_vector_to_matrix(mean) @ predicted

    # ------------------------------------------------------------------ the body's motion over a keyframe interval
    def _motion_per_camera(self, usable: list, body_gyro: np.ndarray) -> tuple[list, np.ndarray]:
        """The first version: each camera fits its own motion; rotation = weighted mean of the corrections, directions are kept per camera."""
        fits = []  # (camera, ids, current pixels, unit direction of the camera's displacement in its key axes, inliers, parallax)
        corrections, weights = [], []
        for cam, key_px, current_px, ids, parallax in usable:
            if parallax < MIN_STEP_PARALLAX_SHARE * MIN_DEROTATED_PARALLAX_PX:
                continue
            rotation = self._camera_gyro_since_key(cam)
            x1, x2 = cam.rays(key_px), cam.rays(current_px)
            found = translation_with_known_rotation(rotation, x1, x2, self.intrinsic[0, 0], self.rng)
            if found is None:
                continue
            t, inliers = found
            refined, t = refine_motion(rotation, t, x1[inliers], x2[inliers], self.intrinsic[0, 0])
            fits.append((cam, ids[inliers], current_px[inliers], None, -(refined.T @ t), None, int(inliers.sum()), parallax))
            correction = cv2.Rodrigues(refined @ rotation.T)[0].ravel()
            corrections.append(cam.spec.body_from_cv @ correction)
            weights.append(int(inliers.sum()))
        if not fits:
            return [], body_gyro
        mean = np.average(np.array(corrections), axis=0, weights=weights) if "gyro-rot" not in OPTS else np.zeros(3)
        self.rotation_corrections += float(np.degrees(np.linalg.norm(mean)))
        return fits, rotation_vector_to_matrix(mean) @ body_gyro

    def _body_rays(self, cam: CameraTrack, pixels: np.ndarray) -> np.ndarray:
        rays = cam.rays(pixels) @ cam.spec.body_from_cv.T  # camera axes -> body axes
        return rays / np.linalg.norm(rays, axis=1, keepdims=True)

    def _motion_joint(self, usable: list, body_gyro: np.ndarray) -> tuple[list, np.ndarray]:
        """ONE motion for the rig: all matches of both cameras, as unit rays in body axes, constrain one rotation and one translation direction.

        A single camera cannot tell a small turn from a sideways move (the flow is nearly the same). Two cameras looking opposite ways
        can: the flow a turn makes is the same on both sides, while a move makes opposite flow on them. That is the one thing a
        rig has over two independent cameras, so the rotation and the direction are solved together, not averaged afterwards."""
        focal = self.intrinsic[0, 0]
        b1, b2, owner, columns = [], [], [], []
        for cam, key_px, current_px, ids, parallax in usable:
            b1.append(self._body_rays(cam, key_px))
            b2.append(self._body_rays(cam, current_px))
            owner += [cam.index] * len(key_px)
            columns.append((cam, len(key_px), current_px, ids, parallax))
        b1, b2, owner = np.vstack(b1), np.vstack(b2), np.array(owner)
        if len(b1) < 8:
            return [], body_gyro
        # translation direction (x2 ~ R x1 + t, in current-body axes) with the gyro's rotation: two matches fix it, RANSAC picks
        normals = np.cross(b1 @ body_gyro.T, b2)
        best, best_inliers = None, None
        for _ in range(2 * 60):
            a, b = self.rng.choice(len(b1), 2, replace=False)
            t = np.cross(normals[a], normals[b])
            norm = np.linalg.norm(t)
            if norm < 1e-12:
                continue
            t /= norm
            inliers = np.abs(sampson_px(body_gyro, t, b1, b2, focal)) < 1.0
            if best_inliers is None or inliers.sum() > best_inliers.sum():
                best, best_inliers = t, inliers
        if best is None or best_inliers.sum() < 8:
            return [], body_gyro
        t = np.linalg.svd(normals[best_inliers], full_matrices=False)[2][-1]
        if np.dot(t, best) < 0:
            t = -t
        cross = np.cross(b1 @ body_gyro.T, b2)
        depth1 = -np.sum(np.cross(t, b2) * cross, axis=1) / np.maximum(np.sum(cross * cross, axis=1), 1e-18)
        inliers = np.abs(sampson_px(body_gyro, t, b1, b2, focal)) < 1.0
        if np.sum(depth1[inliers] > 0) < 0.5 * inliers.sum():
            t = -t
        rotation, t = self._refine_body_motion(body_gyro, t, b1[inliers], b2[inliers], focal)
        if "gyro-rot" in OPTS:
            rotation = body_gyro
        inliers = np.abs(sampson_px(rotation, t, b1, b2, focal)) < 1.0
        direction_key = -(rotation.T @ t)  # the body's displacement, unit, in key-body axes
        self.rotation_corrections += float(np.degrees(np.linalg.norm(cv2.Rodrigues(rotation @ body_gyro.T)[0])))
        fits, start = [], 0
        for cam, n, current_px, ids, parallax in columns:
            mine = inliers[start : start + n]
            start += n
            if parallax < MIN_STEP_PARALLAX_SHARE * MIN_DEROTATED_PARALLAX_PX or mine.sum() < 8:
                continue
            fits.append((cam, ids[mine], current_px[mine], None, cam.spec.body_from_cv.T @ direction_key, None, int(mine.sum()), parallax))
        return fits, rotation

    def _refine_body_motion(self, rotation: np.ndarray, t: np.ndarray, b1: np.ndarray, b2: np.ndarray, focal: float) -> tuple[np.ndarray, np.ndarray]:
        """Levenberg-Marquardt over a small rotation correction (3) and the translation direction (2); the gyro's rotation is a prior."""
        basis_a = np.cross(t, [1.0, 0.0, 0.0] if abs(t[0]) < 0.9 else [0.0, 1.0, 0.0])
        basis_a /= np.linalg.norm(basis_a)
        basis_b = np.cross(t, basis_a)
        prior = np.radians(GYRO_PRIOR_DEG)

        def parts(p: np.ndarray):
            direction = t + p[3] * basis_a + p[4] * basis_b
            return rotation_vector_to_matrix(p[:3]) @ rotation, direction / np.linalg.norm(direction)

        def residual(p: np.ndarray) -> np.ndarray:
            r, direction = parts(p)
            return np.concatenate([np.clip(sampson_px(r, direction, b1, b2, focal) / PIXEL_SIGMA, -4.0, 4.0), p[:3] / prior])

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
            normal, gradient = jacobian.T @ jacobian, jacobian.T @ current
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
        return parts(p)

    # ------------------------------------------------------------------ step length
    def _hold_to_inertial(self, scale: float, key_hint: np.ndarray | None) -> float:
        """Keep the step within what the accelerometer says the drone moved since the camera's keyframe (see ImuAidedSlam._step_scale)."""
        if not self.use_hint or self.hint is None or key_hint is None:
            return scale
        position, per_metre, ready = self.hint
        if not ready:
            return scale
        expected = per_metre * float(np.linalg.norm(position - key_hint))
        sigma = float(np.hypot(per_metre * SCALE_BAND_FLOOR_M, SCALE_BAND_RELATIVE * expected))
        held = float(np.clip(scale, max(expected - SCALE_BAND_SIGMAS * sigma, 1e-6), expected + SCALE_BAND_SIGMAS * sigma))
        self.telemetry.update(hint_expected=expected, hint_raw=float(scale), hint_held=float(held != scale))
        return held

    def _step_scale(self, cam, ids, current_px, key_rotation, key_position, new_rotation, direction) -> float | None:
        """Step length that puts every known landmark seen by this camera on the ray it is seen along now (see RawSlam._step_scale)."""
        rows = [i for i, track in enumerate(ids) if int(track) in self.landmarks]
        if len(rows) < rs.MIN_SCALE_LANDMARKS:
            return None
        entries = np.array([self.landmarks[int(ids[i])] for i in rows])
        reliability = entries[:, 3]
        known = entries[:, :3] / reliability[:, None]
        r_key_cam = key_rotation @ cam.spec.body_from_cv
        r_cur_cam = new_rotation @ cam.spec.body_from_cv
        centre = key_position + key_rotation @ self._lever(cam)
        a = (known - centre) @ r_cur_cam
        b = r_cur_cam.T @ (r_key_cam @ direction)
        pixels = cam.rays(current_px[rows])
        rays = pixels / np.linalg.norm(pixels, axis=1, keepdims=True)
        along_a, along_b = np.cross(rays, a), np.cross(rays, b[None, :])
        sensitivity = np.sum(along_b * along_b, axis=1)
        estimate = np.sum(along_a * along_b, axis=1) / np.maximum(sensitivity, 1e-18)
        weight = sensitivity / np.maximum(np.sum(a * a, axis=1), 1e-12) * reliability
        ok = np.isfinite(estimate) & (sensitivity > 1e-8) & (estimate > 0.0) & (weight > 0.0)
        if ok.sum() < rs.MIN_SCALE_LANDMARKS:
            return None
        value = rs.weighted_median(estimate[ok], weight[ok])
        self.telemetry.update(step_scale=float(value), step_landmarks=float(ok.sum()), depth_med=float(np.median(a[ok, 2])), step_spread=float(np.percentile(estimate[ok], 75) / max(np.percentile(estimate[ok], 25), 1e-12)))
        return value

    def _observe(self, cam: CameraTrack, ids: np.ndarray, current_px: np.ndarray) -> None:
        """Triangulate each point from the view it was born in to the current view and average it into its landmark."""
        rows = [i for i, track in enumerate(ids) if int(track) in cam.birth]
        if not rows:
            return
        births = [cam.birth[int(ids[i])] for i in rows]
        birth_px = np.array([b[0] for b in births])
        birth_rotation = np.array([b[1] for b in births])
        birth_centre = np.array([b[2] for b in births])
        rotation_now, centre_now = self.camera_pose(cam)
        first = np.einsum("nij,nj->ni", birth_rotation, cam.rays(birth_px))
        second = cam.rays(current_px[rows]) @ rotation_now.T
        first /= np.linalg.norm(first, axis=1, keepdims=True)
        second /= np.linalg.norm(second, axis=1, keepdims=True)
        offset = birth_centre - centre_now
        cos = np.sum(first * second, axis=1)
        d, e = np.sum(first * offset, axis=1), np.sum(second * offset, axis=1)
        denominator = 1.0 - cos * cos
        valid = denominator > 1e-8
        denominator = np.where(valid, denominator, 1.0)
        along_first = (cos * e - d) / denominator
        along_second = (e - cos * d) / denominator
        point_first = birth_centre + along_first[:, None] * first
        point_second = centre_now + along_second[:, None] * second
        gap = np.linalg.norm(point_first - point_second, axis=1)
        valid &= (along_first > 0) & (along_second > 0) & (gap < rs.MAX_RAY_GAP * along_second)
        angle_deg = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
        weight = angle_deg**2
        if valid.any():
            ordered = np.sort(weight[valid])
            self.telemetry.update(tri_valid=float(valid.mean()), tri_angle_med=float(np.median(angle_deg[valid])), tri_top_share=float(ordered[-max(1, len(ordered) // 10):].sum() / max(ordered.sum(), 1e-12)))
        midpoint = 0.5 * (point_first + point_second)
        for index in np.flatnonzero(valid):
            track = int(ids[rows[index]])
            entry = self.landmarks.setdefault(track, np.zeros(4))
            entry[:3] += weight[index] * midpoint[index]
            entry[3] += weight[index]
            self.landmark_views[track] = self.landmark_views.get(track, 0) + 1
            self.landmark_angle[track] = max(self.landmark_angle.get(track, 0.0), float(angle_deg[index]))
        for track in [t for t, v in self.landmarks.items() if v[3] <= 0.0]:
            del self.landmarks[track]

    # ------------------------------------------------------------------ anchors
    def _anchors_and_fix(self, grays: dict[str, np.ndarray]) -> None:
        features = {}
        for cam in self.cams:
            if cam.spec.name in grays:
                keypoints, descriptors = cam.orb.detectAndCompute(grays[cam.spec.name], None)
                if descriptors is not None and len(keypoints):
                    features[cam.spec.name] = (keypoints, descriptors)
        if self.commits % rs.LOCALIZE_EVERY == 0:
            fix = self._solve_fix(features)
            if fix is not None:
                self._apply_fix(fix)
        for cam in self.cams:
            if cam.spec.name in features:
                self._update_anchors(cam, *features[cam.spec.name])

    def _update_anchors(self, cam: CameraTrack, keypoints: list, descriptors: np.ndarray) -> None:
        tracked = [i for i, track in enumerate(cam.track_ids) if int(track) in self.landmarks]
        if not tracked or not self.make_anchors:
            return
        ids = [int(cam.track_ids[i]) for i in tracked]
        positions = np.array([self.landmarks[t][:3] / self.landmarks[t][3] for t in ids])
        fresh = [k for k, t in enumerate(ids) if t not in self.anchor_row and self._anchor_ready(t)]
        if not fresh:
            return
        keypoint_xy = np.array([k.pt for k in keypoints], dtype=np.float32)
        distance = np.linalg.norm(keypoint_xy[:, None, :] - cam.track_points[[tracked[k] for k in fresh]][None, :, :], axis=2)
        nearest = distance.argmin(axis=0)
        close = distance[nearest, np.arange(len(fresh))] <= rs.ANCHOR_LANDMARK_PX
        taken: set[int] = set()
        rows, chosen = [], []
        for column in np.flatnonzero(close):
            keypoint = int(nearest[column])
            if keypoint in taken:
                continue
            taken.add(keypoint)
            rows.append(keypoint)
            chosen.append(fresh[column])
        if rows:
            rows, chosen = self._fuse_anchors(cam, keypoints, descriptors, rows, chosen, ids, positions)
        if not rows:
            return
        first_row = len(self.anchor_points)
        self.anchor_descriptors = np.vstack([self.anchor_descriptors, descriptors[rows]])
        self.anchor_points = np.vstack([self.anchor_points, positions[chosen]])
        self.anchor_step = np.concatenate([self.anchor_step, np.full(len(rows), self.commits, np.int64)])
        for offset, k in enumerate(chosen):
            self.anchor_row[ids[k]] = first_row + offset

    def _anchor_ready(self, track: int) -> bool:
        """With a MapPolicy only well-triangulated landmarks become anchors (enough steps, a wide enough angle; see ImuAidedSlam)."""
        m = self.map_policy
        if m is None:
            return True
        return self.landmark_views.get(track, 0) >= m.anchor_min_views and self.landmark_angle.get(track, 0.0) >= m.anchor_min_angle_deg

    def _project_into(self, cam: CameraTrack, points: np.ndarray, rotation: np.ndarray | None = None, position: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(pixels, depth) of points in S seen by one camera, for a body pose (default: the current one)."""
        r_cam, centre = self.camera_pose(cam, rotation, position)
        in_camera = (points - centre) @ r_cam
        depth = in_camera[:, 2]
        return in_camera[:, :2] / np.maximum(depth, 1e-6)[:, None] * self.intrinsic[0, 0] + self.intrinsic[:2, 2], depth

    def _fuse_anchors(self, cam, keypoints, descriptors, rows, chosen, ids, positions):
        """A landmark about to become an anchor that is an existing anchor seen again (by either camera) is linked to it, not copied."""
        m = self.map_policy
        if m is None or not m.fuse or len(self.anchor_points) == 0:
            return rows, chosen
        pixels, depth = self._project_into(cam, self.anchor_points)
        usable = depth > 1e-3
        if self.banned:
            usable[np.fromiter(self.banned, np.int64)] = False
        candidates = np.flatnonzero(usable)
        if len(candidates) == 0:
            return rows, chosen
        _, live_depth = self._project_into(cam, positions[chosen])
        keep_rows, keep_chosen = [], []
        for row, k, d_live in zip(rows, chosen, live_depth):
            near = candidates[np.linalg.norm(pixels[candidates] - np.asarray(keypoints[row].pt), axis=1) < m.fuse_px]
            if len(near):
                distance = hamming(descriptors[row], self.anchor_descriptors[near])
                best = int(np.argmin(distance))
                if distance[best] <= m.fuse_hamming and abs(depth[near[best]] / max(float(d_live), 1e-6) - 1.0) < m.fuse_depth_tolerance:
                    self.anchor_row[ids[k]] = int(near[best])
                    self.fused += 1
                    continue
            keep_rows.append(row)
            keep_chosen.append(k)
        self.telemetry["anchors_fused"] = float(self.fused)
        return keep_rows, keep_chosen

    def _match_global(self, keypoints, descriptors, old: np.ndarray) -> tuple[list, list]:
        pairs = self.matcher.knnMatch(descriptors, self.anchor_descriptors[old], k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance <= rs.MAX_HAMMING and p[0].distance < rs.MATCH_RATIO * p[1].distance]
        return [keypoints[g.queryIdx].pt for g in good], [int(old[g.trainIdx]) for g in good]

    def _match_guided(self, cam: CameraTrack, keypoints, descriptors, old: np.ndarray) -> tuple[list, list]:
        """Search by projection: each anchor only against this camera's features near where the current pose puts it."""
        m = self.map_policy
        pixels, depth = self._project_into(cam, self.anchor_points[old])
        size = 2.0 * self.intrinsic[:2, 2]
        inside = (depth > 1e-3) & (pixels[:, 0] > -m.guided_px) & (pixels[:, 1] > -m.guided_px) & (pixels[:, 0] < size[0] + m.guided_px) & (pixels[:, 1] < size[1] + m.guided_px)
        candidates, pixels = old[inside], pixels[inside]
        points = np.array([k.pt for k in keypoints])
        if len(candidates) == 0 or len(points) == 0:
            return [], []
        a, k = np.nonzero(np.linalg.norm(pixels[:, None, :] - points[None, :, :], axis=2) < m.guided_px)
        if len(a) == 0:
            return [], []
        distance = np.unpackbits(np.bitwise_xor(self.anchor_descriptors[candidates[a]], descriptors[k]), axis=1).sum(axis=1)
        order = np.lexsort((distance, a))
        a, k, distance = a[order], k[order], distance[order]
        first = np.r_[True, a[1:] != a[:-1]]
        second = np.r_[a[1:] == a[:-1], False] & first
        next_distance = np.r_[distance[1:], 0]
        ok = first & (distance <= rs.MAX_HAMMING) & (~second | (distance < m.guided_ratio * next_distance))
        a, k, distance = a[ok], k[ok], distance[ok]
        order = np.lexsort((distance, k))
        a, k = a[order], k[order]
        unique = np.r_[True, k[1:] != k[:-1]]
        return [tuple(p) for p in points[k[unique]]], [int(r) for r in candidates[a[unique]]]

    def _fix_budget(self) -> tuple[float, float]:
        """(SLAM units per metre, metres): how far the pose may plausibly be from a good fix (see ImuAidedSlam._fix_budget)."""
        p = self.policy
        elapsed = 0.0 if self.hint_time is None or self.last_fix_time is None else max(self.hint_time - self.last_fix_time, 0.0)
        return max(self.hint[1], 1e-3), p.sigma_m + p.drift_rel * self.path_length + 0.5 * p.drift_acc * elapsed * elapsed

    def _project(self, rotation: np.ndarray, position: np.ndarray, world: np.ndarray, image: np.ndarray, cam_index: np.ndarray) -> np.ndarray:
        """Pixel residual (n, 2) of anchors seen by the cameras `cam_index`, for a body pose; 1e3 for a point behind its camera."""
        residual = np.zeros((len(world), 2))
        focal, centre_px = self.intrinsic[0, 0], self.intrinsic[:2, 2]
        for cam in self.cams:
            m = cam_index == cam.index
            if not m.any():
                continue
            r_cam, centre = self.camera_pose(cam, rotation, position)
            in_camera = (world[m] - centre) @ r_cam
            z = in_camera[:, 2]
            safe = np.where(z > 1e-6, z, 1.0)
            projected = in_camera[:, :2] / safe[:, None] * focal + centre_px
            residual[m] = np.where((z > 1e-6)[:, None], projected - image[m], 1e3)
        return residual

    def _solve_fix(self, features: dict) -> dict | None:
        """The one body pose that all the old anchors seen by both cameras agree on, or None."""
        old = np.flatnonzero(self.anchor_step <= self.commits - rs.MIN_ANCHOR_AGE)
        if self.banned:
            old = old[~np.isin(old, np.fromiter(self.banned, np.int64))]
        if len(old) < rs.MIN_PNP_INLIERS:
            return None
        m, p = self.map_policy, self.policy
        found = {}
        guided = m is not None and m.guided
        if guided:
            found = {cam.index: self._match_guided(cam, *features[cam.spec.name], old) for cam in self.cams if cam.spec.name in features}
            if sum(len(r) for _, r in found.values()) < rs.MIN_PNP_INLIERS and m.global_fallback:
                guided = False
        if not guided:
            found = {cam.index: self._match_global(*features[cam.spec.name], old) for cam in self.cams if cam.spec.name in features}
        images, rows, cams = [], [], []
        for index, (pts, rws) in found.items():
            images += pts
            rows += rws
            cams += [index] * len(rws)
        if len(rows) < rs.MIN_PNP_INLIERS:
            return None
        image, rows, cams = np.array(images, dtype=np.float64).reshape(-1, 2), np.array(rows), np.array(cams)
        world = self.anchor_points[rows]
        if p is not None and p.prior_gate and self.hint is not None and self.hint[2]:
            lam, budget = self._fix_budget()
            keep = np.ones(len(rows), bool)
            for cam in self.cams:
                mine = cams == cam.index
                if mine.any():
                    expected, depth = self._project_into(cam, world[mine])
                    allowed = 6.0 + self.intrinsic[0, 0] * (0.035 + p.z_max * budget * lam / np.maximum(depth, 1e-3))
                    keep[mine] = (depth > 1e-3) & (np.linalg.norm(expected - image[mine], axis=1) < allowed)
            self.telemetry["prior_kept"] = float(keep.mean())
            image, rows, cams, world = image[keep], rows[keep], cams[keep], world[keep]
            if len(rows) < rs.MIN_PNP_INLIERS:
                return None
        matched_rows = rows.copy()

        best, best_count = None, 0  # hypotheses: each camera's own RANSAC, judged on the matches of both cameras
        for cam in self.cams:
            mine = cams == cam.index
            if mine.sum() < HYPOTHESIS_MIN_MATCHES:
                continue
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(world[mine], image[mine], self.intrinsic, None, iterationsCount=rs.PNP_ITERATIONS, reprojectionError=rs.PNP_PX, flags=cv2.SOLVEPNP_AP3P)
            if not ok or inliers is None or len(inliers) < HYPOTHESIS_MIN_MATCHES:
                continue
            r_wc = cv2.Rodrigues(rvec)[0]
            r_cam, centre = r_wc.T, (-r_wc.T @ tvec).ravel()
            rotation = r_cam @ cam.spec.body_from_cv.T
            position = centre - rotation @ self._lever(cam)
            count = int((np.linalg.norm(self._project(rotation, position, world, image, cams), axis=1) < rs.PNP_PX).sum())
            if count > best_count:
                best, best_count = (rotation, position), count
        if best is None or best_count < rs.MIN_PNP_INLIERS:
            return None
        rotation, position = best
        for _ in range(2):  # joint refinement over the matches of both cameras, then re-pick the inliers
            inlier = np.linalg.norm(self._project(rotation, position, world, image, cams), axis=1) < rs.PNP_PX
            if inlier.sum() < rs.MIN_PNP_INLIERS:
                return None
            rotation, position = self._refine_pose(rotation, position, world[inlier], image[inlier], cams[inlier])
        inlier = np.linalg.norm(self._project(rotation, position, world, image, cams), axis=1) < rs.PNP_PX
        if inlier.sum() < rs.MIN_PNP_INLIERS:
            return None
        rotation, position, inlier = self._side_check(rotation, position, world, image, cams, inlier)
        if inlier is None:
            return None
        if m is not None and m.cull and guided:  # a whole-map match is often simply a wrong match: no blame
            for row, good in zip(matched_rows, inlier):
                score = self.anchor_score.setdefault(int(row), [0, 0])
                score[0 if good else 1] += 1
                if score[1] >= m.cull_strikes and score[1] > 2 * score[0]:
                    self.banned.add(int(row))
            self.telemetry["anchors_banned"] = float(len(self.banned))
        return dict(rotation=rotation, position=position, image=image[inlier], world=world[inlier], rows=rows[inlier], cams=cams[inlier], matches=len(rows), candidates=len(old))

    def _side_check(self, rotation, position, world, image, cams, inlier):
        """Do the two sides agree? Each camera's anchors were made by its own monocular steps, so the two halves of the map can drift
        to slightly different scales, and then no single body pose fits both: the joint pose lands between them, right for neither.
        Each side with enough inliers is refined on its own; if the two disagree by more than the policy allows, the fix is the side
        with more inliers alone."""
        p = self.policy
        if p is None or not p.side_check or "no-side-check" in OPTS:
            return rotation, position, inlier
        sides = {}
        for cam in self.cams:
            mine = inlier & (cams == cam.index)
            if mine.sum() >= rs.MIN_PNP_INLIERS:
                sides[cam.index] = (*self._refine_pose(rotation, position, world[mine], image[mine], cams[mine]), int(mine.sum()))
        if len(sides) < 2:
            return rotation, position, inlier
        (r0, p0, n0), (r1, p1, n1) = sides.values()
        lam = self.hint[1] if self.hint is not None and self.hint[2] else None
        gap_deg = float(np.degrees(np.arccos(np.clip((np.trace(r0.T @ r1) - 1.0) / 2.0, -1.0, 1.0))))
        gap_m = float(np.linalg.norm(p0 - p1)) / lam if lam else float("nan")
        self.telemetry.update(fix_side_gap_deg=gap_deg, fix_side_gap_m=gap_m, fix_side_pick=0.0)
        if gap_deg <= p.side_gap_deg and not (lam and gap_m > p.side_gap_m):
            return rotation, position, inlier
        index = max(sides, key=lambda i: sides[i][2])
        r, pos, _ = sides[index]
        inlier = (np.linalg.norm(self._project(r, pos, world, image, cams), axis=1) < rs.PNP_PX) & (cams == index)
        if inlier.sum() < rs.MIN_PNP_INLIERS:
            return rotation, position, None
        self.side_picks += 1
        self.telemetry["fix_side_pick"] = 1.0
        return r, pos, inlier

    def _refine_pose(self, rotation: np.ndarray, position: np.ndarray, world: np.ndarray, image: np.ndarray, cams: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Gauss-Newton on the body pose (3 rotation, 3 position), Huber-weighted pixel residuals of both cameras' matches."""
        x = np.zeros(6)

        def pose(v: np.ndarray):
            return rotation_vector_to_matrix(v[:3]) @ rotation, position + v[3:]

        for _ in range(8):
            r, p = pose(x)
            residual = self._project(r, p, world, image, cams).ravel()
            jacobian = np.empty((len(residual), 6))
            for k in range(6):
                step = np.zeros(6)
                step[k] = 1e-5 if k < 3 else 1e-5 * max(float(np.linalg.norm(position)), 1.0)
                jacobian[:, k] = (self._project(*pose(x + step), world, image, cams).ravel() - residual) / step[k]
            size = np.linalg.norm(residual.reshape(-1, 2), axis=1)
            weight = np.repeat(np.where(size <= 2.0, 1.0, 2.0 / np.maximum(size, 1e-9)), 2)
            delta = -np.linalg.solve(jacobian.T @ (jacobian * weight[:, None]) + 1e-6 * np.eye(6), jacobian.T @ (weight * residual))
            x = x + delta
            if np.linalg.norm(delta) < 1e-7:
                break
        return pose(x)

    def _apply_fix(self, fix: dict) -> None:
        p = self.policy
        self.telemetry["fix_refused"] = 0.0
        if p is not None and self.hint is not None and self.hint[2]:
            lam, budget = self._fix_budget()
            z = float(np.linalg.norm(fix["position"] - self.position)) / lam / budget
            self.telemetry.update(fix_z=z, fix_budget=budget)
            if p.guard and z > p.z_max:
                self.refused_fixes += 1
                self.telemetry["fix_refused"] = 1.0
                return
            if p.gain < 1.0:
                relative = cv2.Rodrigues(fix["rotation"] @ self.rotation.T)[0].ravel() * p.gain
                fix = dict(fix, position=self.position + p.gain * (fix["position"] - self.position), rotation=rotation_vector_to_matrix(relative) @ self.rotation)
        self.path_length, self.last_fix_time = 0.0, self.hint_time
        new_rotation, new_position = fix["rotation"], fix["position"]
        rows_used = fix["rows"]
        alive_rows = {self.anchor_row[int(t)] for cam in self.cams for t in cam.track_ids if int(t) in self.anchor_row}
        age = self.commits - self.anchor_step[rows_used]
        self.telemetry.update(
            fix_matches=float(fix["matches"]), fix_inliers=float(len(rows_used)), fix_candidates=float(fix["candidates"]), fix_age_med=float(np.median(age)), fix_age_min=float(age.min()), fix_age_max=float(age.max()),
            fix_live=float(np.mean([r in alive_rows for r in rows_used])),
            fix_rot_deg=float(np.degrees(np.arccos(np.clip((np.trace(new_rotation.T @ self.rotation) - 1.0) / 2.0, -1.0, 1.0)))),
            fix_move=float(np.linalg.norm(new_position - self.position)), fix_now=1.0, fix_cam_share=float(np.mean(fix["cams"] == 0)),
        )
        self._carry_live_map(fix)
        self.rotation, self.position = new_rotation, new_position
        self.pose_gyro = self.gyro_body.copy()
        for cam in self.cams:  # the keyframes move with the map: a step measured from them must land in the snapped frame
            relative = new_rotation @ fix["old_rotation"].T
            cam.key_position = new_position + fix["carry_scale"] * (relative @ (cam.key_position - fix["old_position"]))
            cam.key_rotation = relative @ cam.key_rotation
        self.fixes += 1

    def _carry_live_map(self, fix: dict) -> None:
        """Move the landmarks and birth views of the live tracks along with the pose snap (a similarity: see RawSlam._carry_live_map)."""
        old_rotation, old_position = self.rotation, self.position
        new_rotation, new_position = fix["rotation"], fix["position"]
        relative = new_rotation @ old_rotation.T
        ratios = []
        for cam in self.cams:
            m = fix["cams"] == cam.index
            tracked = [i for i, track in enumerate(cam.track_ids) if int(track) in self.landmarks]
            if not m.any() or not tracked:
                continue
            live = np.array([self.landmarks[int(cam.track_ids[i])][:3] / self.landmarks[int(cam.track_ids[i])][3] for i in tracked])
            distance = np.linalg.norm(fix["image"][m][:, None, :] - cam.track_points[tracked][None, :, :], axis=2)
            nearest = distance.argmin(axis=1)
            close = distance[np.arange(int(m.sum())), nearest] <= rs.ANCHOR_LANDMARK_PX
            r_old, c_old = self.camera_pose(cam)
            r_new, c_new = self.camera_pose(cam, new_rotation, new_position)
            depth_live = ((live[nearest[close]] - c_old) @ r_old)[:, 2]
            depth_anchor = ((fix["world"][m][close] - c_new) @ r_new)[:, 2]
            good = (depth_live > 1e-6) & (depth_anchor > 1e-6)
            ratios += list(depth_anchor[good] / depth_live[good])
        scale = float(np.median(ratios)) if len(ratios) >= rs.MIN_SCALE_LANDMARKS else 1.0
        self.telemetry.update(fix_pairs=float(len(ratios)), fix_scale=scale)
        m = self.map_policy
        if m is not None and m.rigid_carry and self.hint is not None and self.hint[2]:
            scale = 1.0  # once the metric scale is calibrated the steps are held to it: the map's unit is right, a fix only moves it
        fix.update(old_rotation=old_rotation, old_position=old_position, carry_scale=scale)

        def carry(point: np.ndarray) -> np.ndarray:
            return new_position + scale * (relative @ (point - old_position))

        for entry in self.landmarks.values():
            entry[:3] = entry[3] * carry(entry[:3] / entry[3])
        for cam in self.cams:
            cam.birth = {track: (pixel, relative @ rotation, carry(centre)) for track, (pixel, rotation, centre) in cam.birth.items()}

    def describe(self) -> str:
        return (f"steps {self.commits}  fixes {self.fixes} (refused {self.refused_fixes}, one side {self.side_picks})  landmarks {len(self.landmarks)}  anchors {len(self.anchor_points)} "
                f"(fused {self.fused}, banned {len(self.banned)})  no-scale skips {self.skipped_no_scale}  restarts {self.restarts}  failed fits {self.failed_fits}")
