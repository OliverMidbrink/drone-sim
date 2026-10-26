"""Global anchor features: relocalise against places seen before and close loops.

Tracked points (optical flow) are cheap and local. Anchors are a much smaller set
of ORB features with a 3D world position and the direction they were seen from.
Matching the current frame against them (and solving PnP) gives a camera pose in
*map* coordinates, independent of how far dead-reckoning has drifted. The
difference between the two is a loop-closure correction, spread back over the
path travelled since the last fix.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MIN_AGE_STEPS = 15  # anchors younger than this just echo dead-reckoning
MIN_PNP_MATCHES = 30
MIN_PNP_INLIERS = 30
MIN_INLIER_SPREAD = 0.25  # inlier bounding box, as a fraction of the image
RANSAC_PX = 6.0  # anchor depths are approximate, so allow a loose first pass
FINAL_PX = 4.0
MAX_HAMMING = 64
MATCH_RATIO = 0.85
SAME_POINT_TOL = 0.05  # anchors this close (fraction of depth) are the same point
NEW_VIEW_DEG = 35.0  # a new view of a known point is stored past this angle
MAX_VIEW_DEG = 60.0  # refine the pose with anchors seen from within this angle


@dataclass
class Correction:
    """Rigid world->world transform x' = R x + t, applied by fraction along the path."""

    rotation: np.ndarray
    translation: np.ndarray

    def apply(self, points: np.ndarray, fraction: np.ndarray | float) -> np.ndarray:
        points = np.atleast_2d(points)
        fraction = np.broadcast_to(np.asarray(fraction, dtype=np.float64), (len(points),))
        rotvec, _ = cv2.Rodrigues(self.rotation)
        rotvec = rotvec.reshape(3)
        angle = float(np.linalg.norm(rotvec))
        moved = points + fraction[:, None] * self.translation
        if angle < 1e-9:
            return moved
        axis = rotvec / angle
        phi = (fraction * angle)[:, None]
        cos, sin = np.cos(phi), np.sin(phi)
        rotated = (
            points * cos
            + np.cross(axis, points) * sin
            + axis * (points @ axis)[:, None] * (1.0 - cos)
        )
        return rotated + fraction[:, None] * self.translation

    @property
    def magnitude(self) -> tuple[float, float]:
        rotvec, _ = cv2.Rodrigues(self.rotation)
        return float(np.linalg.norm(self.translation)), float(
            np.degrees(np.linalg.norm(rotvec))
        )


@dataclass
class PoseFix:
    rotation: np.ndarray  # camera->world
    position: np.ndarray
    inliers: int
    matches: int


def keypoint_depths(
    keypoints: np.ndarray,
    track_pixels: np.ndarray,
    track_depths: np.ndarray,
    shape: tuple[int, int],
    radius: int = 8,
    max_spread: float = 0.06,
) -> np.ndarray:
    """Depth at arbitrary pixels from nearby tracked landmarks (NaN if unreliable)."""
    height, width = shape
    out = np.full(len(keypoints), np.nan)
    if len(track_pixels) < 4 or len(keypoints) == 0:
        return out
    xi = np.clip(np.round(track_pixels[:, 0]).astype(int), 0, width - 1)
    yi = np.clip(np.round(track_pixels[:, 1]).astype(int), 0, height - 1)
    depth = track_depths.astype(np.float32)

    occupancy = np.zeros((height, width), np.float32)
    total = np.zeros((height, width), np.float32)
    top = np.zeros((height, width), np.float32)
    bottom = np.full((height, width), 1e9, np.float32)
    occupancy[yi, xi] = 1.0
    total[yi, xi] = depth
    top[yi, xi] = depth
    bottom[yi, xi] = depth

    size = 2 * radius + 1
    kernel = np.ones((size, size), np.uint8)
    count = cv2.boxFilter(occupancy, -1, (size, size), normalize=False)
    mean = cv2.boxFilter(total, -1, (size, size), normalize=False) / np.maximum(count, 1.0)
    spread = (cv2.dilate(top, kernel) - cv2.erode(bottom, kernel)) / np.maximum(mean, 1e-6)

    kx = np.clip(np.round(keypoints[:, 0]).astype(int), 0, width - 1)
    ky = np.clip(np.round(keypoints[:, 1]).astype(int), 0, height - 1)
    good = (count[ky, kx] >= 2) & (spread[ky, kx] < max_spread)
    out[good] = mean[ky, kx][good]
    return out


class AnchorMap:
    def __init__(self, max_anchors: int = 60_000, n_features: int = 1200) -> None:
        self.max_anchors = max_anchors
        self._orb = cv2.ORB_create(
            nfeatures=n_features,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=15,
            fastThreshold=10,
        )
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.descriptors = np.empty((0, 32), dtype=np.uint8)
        self.positions = np.empty((0, 3), dtype=np.float64)
        self.views = np.empty((0, 3), dtype=np.float64)  # unit vector point -> camera
        self.steps = np.empty(0, dtype=np.int32)
        self.last_fix_step = 0

    def __len__(self) -> int:
        return len(self.positions)

    def detect(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        keypoints, descriptors = self._orb.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) == 0:
            return (
                np.empty((0, 2)),
                np.empty((0, 32), np.uint8),
                np.empty(0),
            )
        pixels = np.array([k.pt for k in keypoints], dtype=np.float64)
        response = np.array([k.response for k in keypoints], dtype=np.float64)
        return pixels, descriptors, response

    def match(
        self, descriptors: np.ndarray, camera_position: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Query->anchor matches (ratio test; ambiguity between views of one point is fine)."""
        if len(self) < 2 or len(descriptors) == 0:
            return np.empty(0, dtype=int), np.empty(0, dtype=int)
        pairs = self._matcher.knnMatch(descriptors, self.descriptors, k=2)
        query, anchor = [], []
        for pair in pairs:
            if len(pair) < 2:
                continue
            first, second = pair
            if first.distance > MAX_HAMMING:
                continue
            ok = first.distance < MATCH_RATIO * second.distance
            if not ok:
                # Two stored views of the same physical point are not ambiguous.
                gap = np.linalg.norm(self.positions[first.trainIdx] - self.positions[second.trainIdx])
                depth = np.linalg.norm(self.positions[first.trainIdx] - camera_position)
                ok = gap < SAME_POINT_TOL * max(depth, 1e-6)
            if ok:
                query.append(first.queryIdx)
                anchor.append(first.trainIdx)
        return np.array(query, dtype=int), np.array(anchor, dtype=int)

    def localize(
        self,
        pixels: np.ndarray,
        anchor_idx: np.ndarray,
        intrinsic: np.ndarray,
    ) -> PoseFix | None:
        if len(anchor_idx) < MIN_PNP_MATCHES:
            return None
        obj = self.positions[anchor_idx].reshape(-1, 1, 3)
        img = pixels.reshape(-1, 1, 2)
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj,
            img,
            intrinsic,
            None,
            iterationsCount=400,
            reprojectionError=RANSAC_PX,
            confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not ok or inliers is None or len(inliers) < MIN_PNP_INLIERS:
            return None

        def residuals(rv: np.ndarray, tv: np.ndarray) -> np.ndarray:
            projected, _ = cv2.projectPoints(obj, rv, tv, intrinsic, None)
            return np.linalg.norm(projected.reshape(-1, 2) - pixels, axis=1)

        chosen = inliers.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(obj[chosen], img[chosen], intrinsic, None, rvec, tvec)
        keep = residuals(rvec, tvec) < FINAL_PX
        if keep.sum() < MIN_PNP_INLIERS:
            return None

        # Anchors seen from near the current viewing direction localise best.
        rotation_wc = cv2.Rodrigues(rvec)[0].T
        position = (-rotation_wc @ tvec).reshape(3)
        to_camera = position - self.positions[anchor_idx]
        to_camera /= np.maximum(np.linalg.norm(to_camera, axis=1, keepdims=True), 1e-9)
        similar = (
            np.einsum("ij,ij->i", to_camera, self.views[anchor_idx])
            > np.cos(np.radians(MAX_VIEW_DEG))
        )
        best = keep & similar
        if best.sum() >= MIN_PNP_INLIERS:
            rvec, tvec = cv2.solvePnPRefineLM(
                obj[best], img[best], intrinsic, None, rvec, tvec
            )
            keep = residuals(rvec, tvec) < FINAL_PX
            if keep.sum() < MIN_PNP_INLIERS:
                return None

        inlier_px = pixels[keep]
        spread_x = np.ptp(inlier_px[:, 0]) / (2.0 * intrinsic[0, 2])
        spread_y = np.ptp(inlier_px[:, 1]) / (2.0 * intrinsic[1, 2])
        if spread_x < MIN_INLIER_SPREAD or spread_y < MIN_INLIER_SPREAD * 0.5:
            return None

        rotation_wc = cv2.Rodrigues(rvec)[0].T
        position = (-rotation_wc @ tvec).reshape(3)
        return PoseFix(rotation_wc, position, int(keep.sum()), len(anchor_idx))

    def add(
        self,
        descriptors: np.ndarray,
        positions: np.ndarray,
        views: np.ndarray,
        step: int,
    ) -> None:
        if len(positions) == 0:
            return
        self.descriptors = np.vstack([self.descriptors, descriptors])
        self.positions = np.vstack([self.positions, positions])
        self.views = np.vstack([self.views, views])
        self.steps = np.concatenate([self.steps, np.full(len(positions), step, dtype=np.int32)])
        excess = len(self.positions) - self.max_anchors
        if excess > 0:
            self.descriptors = self.descriptors[excess:]
            self.positions = self.positions[excess:]
            self.views = self.views[excess:]
            self.steps = self.steps[excess:]

    def warp(self, correction: Correction, start_step: int, end_step: int) -> None:
        if len(self.positions) == 0:
            return
        span = max(end_step - start_step, 1)
        fraction = np.clip((self.steps - start_step) / span, 0.0, 1.0)
        moving = fraction > 0.0
        if moving.any():
            self.positions[moving] = correction.apply(self.positions[moving], fraction[moving])
            # views rotate with the correction too
            self.views[moving] = correction.apply(self.views[moving], fraction[moving]) - (
                fraction[moving, None] * correction.translation
            )


def correction_between(
    estimated_rotation: np.ndarray,
    estimated_position: np.ndarray,
    fixed_rotation: np.ndarray,
    fixed_position: np.ndarray,
) -> Correction:
    """World->world transform that carries the estimated pose onto the fixed one."""
    rotation = fixed_rotation @ estimated_rotation.T
    translation = fixed_position - rotation @ estimated_position
    return Correction(rotation, translation)
