"""Raw monocular SLAM: live feature tracking plus global ORB anchors. Nothing else.

What it does
  1. Tracks corners frame to frame (Lucas-Kanade).
  2. Once the tracks have moved 8 px against the last keyframe, fits the essential matrix
     (RANSAC) for the rotation and the direction of translation.
  3. Takes the length of that translation from the map: every tracked point that already has
     a landmark must end up where it is seen now, which fixes the step length (one linear
     equation per point, combined by a weighted median). Only the very first step has no map;
     it defines the map's unit (one baseline = 1).
  4. Landmarks are triangulated from the view where a track was born to the current view, so
     their baseline keeps growing, and every new observation is averaged in weighted by its
     triangulation angle squared. Depth therefore gets better the longer a point is tracked.
  5. A landmark with an ORB feature on top of it becomes an anchor (descriptor + 3D position,
     kept up to date while the point is tracked). Every few steps the whole frame's ORB
     features are matched against all old anchors with no pose prior, and a PnP fit snaps the
     pose onto the map.

What it deliberately does not have: no assumed or carried-over speed, no clamping of the step,
no scene-depth scale, no rotation repair, no sign tie-breaking, no gating of fixes, no
smoothing of the trajectory, no track churn / thinning / pruning. When no scale can be taken
from the map the step is skipped and counted (`skipped_no_scale`).
"""

from __future__ import annotations

import cv2
import numpy as np

# tracking
MAX_FEATURES = 500
REFILL_BELOW = 400  # detect new corners when fewer tracks than this remain
FEATURE_QUALITY = 0.01
FEATURE_MIN_DISTANCE = 10
LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)
# pose
KEYFRAME_MIN_PARALLAX_PX = 8.0  # median motion against the keyframe before a pose is computed
KEYFRAME_MIN_MATCHES = 60  # fewer surviving tracks than this: start over from a new keyframe
ESSENTIAL_RANSAC_PX = 1.0
MIN_SCALE_LANDMARKS = 12  # known landmarks needed to take the step length from the map
MAX_RAY_GAP = 0.1  # two rays that miss each other by more than this share of the depth do not triangulate
# anchors
ORB_FEATURES = 1500
ANCHOR_LANDMARK_PX = 2.0  # an ORB feature this close to a tracked landmark becomes its anchor
LOCALIZE_EVERY = 3  # steps between global anchor localizations
MIN_ANCHOR_AGE = 15  # steps; younger anchors would only confirm what we just did
MAX_HAMMING = 64
MATCH_RATIO = 0.85
PNP_ITERATIONS = 1000
PNP_PX = 4.0
MIN_PNP_INLIERS = 30


def pinhole(width: int, height: int, fov_deg: float) -> np.ndarray:
    focal = width / (2.0 * np.tan(np.radians(fov_deg) / 2.0))
    return np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def estimate_motion(
    previous: np.ndarray, current: np.ndarray, intrinsic: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """(rotation, unit translation, inlier mask) taking previous-camera coordinates to current-camera coordinates."""
    if len(previous) < 8:
        return None
    essential, mask = cv2.findEssentialMat(
        previous, current, intrinsic, method=cv2.RANSAC, prob=0.999, threshold=ESSENTIAL_RANSAC_PX
    )
    if essential is None or mask is None or essential.shape != (3, 3):
        return None
    _, rotation, translation, pose_mask = cv2.recoverPose(essential, previous, current, intrinsic, mask=mask)
    inliers = pose_mask.ravel() > 0
    if inliers.sum() < 8:
        return None
    return rotation, translation.reshape(3), inliers


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, 0.5 * cumulative[-1])])


class RawSlam:
    def __init__(self, first_frame: np.ndarray, intrinsic: np.ndarray, exclude_mask: np.ndarray | None = None) -> None:
        """`exclude_mask`: uint8 image, 0 where the frame must be ignored (overlays, the drone's own body)."""
        self.exclude_mask = exclude_mask
        self.intrinsic = intrinsic
        self.intrinsic_inverse = np.linalg.inv(intrinsic)
        self.frame_number = 0
        self.world_rotation = np.eye(3)  # camera -> world
        self.world_position = np.zeros(3)  # camera centre in the world
        self.commits = 0  # pose steps taken
        self.fixes = 0  # anchor localizations applied
        self.skipped_no_scale = 0  # steps skipped because the map gave no scale
        self.restarts = 0  # steps that began a new unit because the map had too few landmarks left
        self.failed_fits = 0  # parallax was there but the essential matrix failed

        # track id -> (pixel, camera rotation, camera centre) in the keyframe where the track first appeared
        self.birth: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        # track id -> [sum of weight * position (3), sum of weights]: the landmark is their ratio
        self.landmarks: dict[int, np.ndarray] = {}
        self.orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.anchor_descriptors = np.empty((0, 32), np.uint8)
        self.anchor_points = np.empty((0, 3))
        self.anchor_step = np.empty(0, np.int64)  # commit count when the anchor was made
        self.anchor_row: dict[int, int] = {}  # track id -> its row in the anchor arrays

        gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        self.next_id = 0
        self.track_points = np.empty((0, 2), np.float32)
        self.track_ids = np.empty(0, np.int64)
        self._add_tracks(gray)
        self.key_points = self.track_points.copy()
        self.key_ids = self.track_ids.copy()
        self._set_keyframe()
        self.prev_gray = gray

    # ------------------------------------------------------------------ tracking
    def _add_tracks(self, gray: np.ndarray) -> None:
        wanted = MAX_FEATURES - len(self.track_points)
        if wanted <= 0:
            return
        mask = np.full(gray.shape, 255, np.uint8) if self.exclude_mask is None else self.exclude_mask.copy()
        for x, y in self.track_points:
            cv2.circle(mask, (int(x), int(y)), FEATURE_MIN_DISTANCE, 0, -1)
        found = cv2.goodFeaturesToTrack(gray, wanted, FEATURE_QUALITY, FEATURE_MIN_DISTANCE, mask=mask)
        if found is None:
            return
        fresh = found.reshape(-1, 2).astype(np.float32)
        ids = np.arange(self.next_id, self.next_id + len(fresh), dtype=np.int64)
        self.next_id += len(fresh)
        self.track_points = np.vstack([self.track_points, fresh])
        self.track_ids = np.concatenate([self.track_ids, ids])

    def _set_keyframe(self) -> None:
        """The current frame, with the current pose, becomes the keyframe; new tracks are born here."""
        order = np.argsort(self.track_ids)
        self.key_ids = self.track_ids[order]
        self.key_points = self.track_points[order]
        pose = (self.world_rotation.copy(), self.world_position.copy())
        alive = {int(i) for i in self.track_ids}
        for track, pixel in zip(self.track_ids, self.track_points):
            self.birth.setdefault(int(track), (pixel.astype(np.float64), *pose))
        self.birth = {i: v for i, v in self.birth.items() if i in alive}
        self.landmarks = {i: v for i, v in self.landmarks.items() if i in alive}

    def process(self, frame: np.ndarray) -> bool:
        """Advance by one BGR frame. True when the pose changed (a step and/or an anchor fix)."""
        self.frame_number += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        if len(self.track_points) >= 1:
            moved, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.track_points.reshape(-1, 1, 2), None, **LK_PARAMS
            )
            moved = moved.reshape(-1, 2)
            keep = (
                (status.reshape(-1) == 1)
                & (moved[:, 0] >= 0) & (moved[:, 0] < width - 1)
                & (moved[:, 1] >= 0) & (moved[:, 1] < height - 1)
            )
            if self.exclude_mask is not None and keep.any():
                column = np.clip(np.round(moved[:, 0]).astype(int), 0, width - 1)
                row = np.clip(np.round(moved[:, 1]).astype(int), 0, height - 1)
                keep &= self.exclude_mask[row, column] > 0
            self.track_points, self.track_ids = moved[keep], self.track_ids[keep]

        changed, rekey = False, False
        if len(self.track_points) < 8:
            rekey = True
        else:
            changed, rekey = self._try_step(gray)
        if len(self.track_points) < REFILL_BELOW:
            self._add_tracks(gray)
        if rekey:
            self._set_keyframe()
        self.prev_gray = gray
        return changed

    # ------------------------------------------------------------------ pose step
    def _try_step(self, gray: np.ndarray) -> tuple[bool, bool]:
        """(pose changed, start a new keyframe)"""
        if len(self.key_ids) == 0:
            return False, True
        position = np.minimum(np.searchsorted(self.key_ids, self.track_ids), len(self.key_ids) - 1)
        matched = self.key_ids[position] == self.track_ids
        if matched.sum() < KEYFRAME_MIN_MATCHES:
            return False, True
        key_px = self.key_points[position[matched]]
        current_px = self.track_points[matched]
        ids = self.track_ids[matched]
        parallax = float(np.median(np.linalg.norm(current_px - key_px, axis=1)))
        if parallax < KEYFRAME_MIN_PARALLAX_PX:
            return False, False
        motion = estimate_motion(key_px, current_px, self.intrinsic)
        if motion is None:
            self.failed_fits += 1
            return False, parallax > 4.0 * KEYFRAME_MIN_PARALLAX_PX
        rotation, translation, inliers = motion
        current_px, ids = current_px[inliers], ids[inliers]

        direction = -(rotation.T @ translation)  # unit direction the camera moved, in key-camera coordinates
        scale = self._step_scale(ids, current_px, rotation, direction)
        if scale is None:
            self.skipped_no_scale += 1
            return False, False

        key_rotation = self.world_rotation.copy()
        self.world_position = self.world_position + key_rotation @ (direction * scale)
        self.world_rotation = key_rotation @ rotation.T
        self.commits += 1

        self._observe(ids, current_px)
        self._anchors_and_fix(gray)
        return True, True

    def _step_scale(
        self, ids: np.ndarray, current_px: np.ndarray, rotation: np.ndarray, direction: np.ndarray
    ) -> float | None:
        """Step length that puts every known landmark on the ray it is seen along now.

        A landmark X seen along ray u from a camera moved by s*d satisfies u x (a - s*b) = 0 with
        a = R^T (X - C) and b = R^T d, so each landmark gives s = (u x a).(u x b) / |u x b|^2. The
        estimates are combined by a median weighted by how much the landmark can tell (|u x b|^2
        over its squared distance) and by how well the landmark itself is known.
        """
        if self.commits == 0:
            return 1.0  # the very first step defines the unit of the map
        if len(self.landmarks) < MIN_SCALE_LANDMARKS:
            # The map holds too few points to ever give a scale again, and landmarks are only made
            # by steps, so waiting would wait forever (all points lost, e.g. after fast motion).
            # Start a new unit exactly like the very first step; anchor fixes later bring the
            # live map back onto the old one, scale included.
            self.restarts += 1
            self.landmarks = {}
            return 1.0
        rows = [i for i, track in enumerate(ids) if int(track) in self.landmarks]
        if len(rows) < MIN_SCALE_LANDMARKS:
            return None
        entries = np.array([self.landmarks[int(ids[i])] for i in rows])
        reliability = entries[:, 3]
        known = entries[:, :3] / reliability[:, None]
        current_rotation = self.world_rotation @ rotation.T
        a = (known - self.world_position) @ current_rotation
        b = current_rotation.T @ (self.world_rotation @ direction)
        pixels = np.column_stack([current_px[rows], np.ones(len(rows))]) @ self.intrinsic_inverse.T
        rays = pixels / np.linalg.norm(pixels, axis=1, keepdims=True)
        along_a, along_b = np.cross(rays, a), np.cross(rays, b[None, :])
        sensitivity = np.sum(along_b * along_b, axis=1)
        estimate = np.sum(along_a * along_b, axis=1) / np.maximum(sensitivity, 1e-18)
        weight = sensitivity / np.maximum(np.sum(a * a, axis=1), 1e-12) * reliability
        usable = np.isfinite(estimate) & (sensitivity > 1e-8) & (estimate > 0.0) & (weight > 0.0)
        if usable.sum() < MIN_SCALE_LANDMARKS:
            return None
        return weighted_median(estimate[usable], weight[usable])

    def _observe(self, ids: np.ndarray, current_px: np.ndarray) -> None:
        """Triangulate each point from the view it was born in to the current view and average it into its landmark."""
        rows = [i for i, track in enumerate(ids) if int(track) in self.birth]
        if not rows:
            return
        births = [self.birth[int(ids[i])] for i in rows]
        birth_px = np.array([b[0] for b in births])
        birth_rotation = np.array([b[1] for b in births])
        birth_centre = np.array([b[2] for b in births])

        def homogeneous(pixels: np.ndarray) -> np.ndarray:
            return np.column_stack([pixels, np.ones(len(pixels))]) @ self.intrinsic_inverse.T

        first = np.einsum("nij,nj->ni", birth_rotation, homogeneous(birth_px))
        second = homogeneous(current_px[rows]) @ self.world_rotation.T
        first /= np.linalg.norm(first, axis=1, keepdims=True)
        second /= np.linalg.norm(second, axis=1, keepdims=True)
        offset = birth_centre - self.world_position
        cos = np.sum(first * second, axis=1)
        d, e = np.sum(first * offset, axis=1), np.sum(second * offset, axis=1)
        denominator = 1.0 - cos * cos
        valid = denominator > 1e-8
        denominator = np.where(valid, denominator, 1.0)
        along_first = (cos * e - d) / denominator
        along_second = (e - cos * d) / denominator
        point_first = birth_centre + along_first[:, None] * first
        point_second = self.world_position + along_second[:, None] * second
        gap = np.linalg.norm(point_first - point_second, axis=1)
        valid &= (along_first > 0) & (along_second > 0) & (gap < MAX_RAY_GAP * along_second)
        weight = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))) ** 2
        midpoint = 0.5 * (point_first + point_second)
        for index in np.flatnonzero(valid):
            entry = self.landmarks.setdefault(int(ids[rows[index]]), np.zeros(4))
            entry[:3] += weight[index] * midpoint[index]
            entry[3] += weight[index]
        # a landmark whose weights summed to nothing has no position yet
        for track in [t for t, v in self.landmarks.items() if v[3] <= 0.0]:
            del self.landmarks[track]

    # ------------------------------------------------------------------ anchors
    def _anchors_and_fix(self, gray: np.ndarray) -> None:
        keypoints, descriptors = self.orb.detectAndCompute(gray, self.exclude_mask)
        if descriptors is None or len(keypoints) == 0:
            return
        if self.commits % LOCALIZE_EVERY == 0:
            self._localize(keypoints, descriptors)
        self._update_anchors(keypoints, descriptors)

    def _update_anchors(self, keypoints: list, descriptors: np.ndarray) -> None:
        """Refresh the anchors of tracked landmarks and make anchors of tracked landmarks that have none yet."""
        tracked = [i for i, track in enumerate(self.track_ids) if int(track) in self.landmarks]
        if not tracked:
            return
        ids = [int(self.track_ids[i]) for i in tracked]
        positions = np.array([self.landmarks[t][:3] / self.landmarks[t][3] for t in ids])
        have = [(k, self.anchor_row[t]) for k, t in enumerate(ids) if t in self.anchor_row]
        if have:
            self.anchor_points[[row for _, row in have]] = positions[[k for k, _ in have]]
        fresh = [k for k, t in enumerate(ids) if t not in self.anchor_row]
        if not fresh:
            return
        keypoint_xy = np.array([k.pt for k in keypoints], dtype=np.float32)
        distance = np.linalg.norm(keypoint_xy[:, None, :] - self.track_points[[tracked[k] for k in fresh]][None, :, :], axis=2)
        nearest = distance.argmin(axis=0)  # per fresh track, its closest keypoint
        close = distance[nearest, np.arange(len(fresh))] <= ANCHOR_LANDMARK_PX
        taken: set[int] = set()
        rows, chosen = [], []
        for column in np.flatnonzero(close):
            keypoint = int(nearest[column])
            if keypoint in taken:
                continue
            taken.add(keypoint)
            rows.append(keypoint)
            chosen.append(fresh[column])
        if not rows:
            return
        first_row = len(self.anchor_points)
        self.anchor_descriptors = np.vstack([self.anchor_descriptors, descriptors[rows]])
        self.anchor_points = np.vstack([self.anchor_points, positions[chosen]])
        self.anchor_step = np.concatenate([self.anchor_step, np.full(len(rows), self.commits, np.int64)])
        for offset, k in enumerate(chosen):
            self.anchor_row[ids[k]] = first_row + offset

    def _localize(self, keypoints: list, descriptors: np.ndarray) -> None:
        """Match the whole frame against every old anchor (no pose prior) and snap the pose with PnP."""
        old = np.flatnonzero(self.anchor_step <= self.commits - MIN_ANCHOR_AGE)
        if len(old) < MIN_PNP_INLIERS:
            return
        pairs = self.matcher.knnMatch(descriptors, self.anchor_descriptors[old], k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance <= MAX_HAMMING and p[0].distance < MATCH_RATIO * p[1].distance]
        if len(good) < MIN_PNP_INLIERS:
            return
        image = np.array([keypoints[m.queryIdx].pt for m in good], dtype=np.float64)
        world = self.anchor_points[old[[m.trainIdx for m in good]]]
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            world, image, self.intrinsic, None,
            iterationsCount=PNP_ITERATIONS, reprojectionError=PNP_PX, flags=cv2.SOLVEPNP_AP3P,
        )
        if not ok or inliers is None or len(inliers) < MIN_PNP_INLIERS:
            return
        inliers = inliers.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(world[inliers], image[inliers], self.intrinsic, None, rvec, tvec)
        rotation_world_to_camera = cv2.Rodrigues(rvec)[0]
        new_rotation = rotation_world_to_camera.T
        new_position = (-rotation_world_to_camera.T @ tvec).ravel()
        self._carry_live_map(new_rotation, new_position, image[inliers], world[inliers])
        self.world_rotation, self.world_position = new_rotation, new_position
        self.fixes += 1

    def _carry_live_map(
        self, new_rotation: np.ndarray, new_position: np.ndarray, anchor_pixels: np.ndarray, anchor_world: np.ndarray
    ) -> None:
        """Move the landmarks and birth views of the tracks still alive along with the pose snap.

        The snap puts the camera into the anchors' frame, which after a stretch of drift (above all
        after a lap) differs from the frame the live landmarks were built in, in position, rotation
        and scale. Left alone, the next step would be measured against landmarks that no longer fit
        the pose. So every live landmark and birth view is mapped by the same similarity transform
        that takes the old camera pose to the new one. The scale of that transform is the median
        ratio of depths of the same physical points: anchor depth from the new pose over live
        landmark depth from the old pose (1 if fewer than MIN_SCALE_LANDMARKS points pair up).
        """
        old_rotation, old_position = self.world_rotation, self.world_position
        relative = new_rotation @ old_rotation.T
        scale = 1.0
        tracked = [i for i, track in enumerate(self.track_ids) if int(track) in self.landmarks]
        if tracked:
            live = np.array([self.landmarks[int(self.track_ids[i])][:3] / self.landmarks[int(self.track_ids[i])][3] for i in tracked])
            distance = np.linalg.norm(anchor_pixels[:, None, :] - self.track_points[tracked][None, :, :], axis=2)
            nearest = distance.argmin(axis=1)
            close = distance[np.arange(len(anchor_pixels)), nearest] <= ANCHOR_LANDMARK_PX
            depth_live = ((live[nearest[close]] - old_position) @ old_rotation)[:, 2]
            depth_anchor = ((anchor_world[close] - new_position) @ new_rotation)[:, 2]
            usable = (depth_live > 1e-6) & (depth_anchor > 1e-6)
            if usable.sum() >= MIN_SCALE_LANDMARKS:
                scale = float(np.median(depth_anchor[usable] / depth_live[usable]))

        def carry(point: np.ndarray) -> np.ndarray:
            return new_position + scale * (relative @ (point - old_position))

        for entry in self.landmarks.values():
            entry[:3] = entry[3] * carry(entry[:3] / entry[3])  # the weights stay: angles do not change
        self.birth = {
            track: (pixel, relative @ rotation, carry(centre)) for track, (pixel, rotation, centre) in self.birth.items()
        }

    def describe(self) -> str:
        return (
            f"steps {self.commits}  fixes {self.fixes}  landmarks {len(self.landmarks)}  anchors {len(self.anchor_points)}  "
            f"no-scale skips {self.skipped_no_scale}  restarts {self.restarts}  failed fits {self.failed_fits}"
        )
