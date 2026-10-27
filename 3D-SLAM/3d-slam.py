#!/usr/bin/env python3
"""Monocular video SLAM: sparse map + camera trajectory (metric scale arbitrary)."""

import argparse
import sys
from collections import deque
from enum import IntEnum
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from anchor_map import (
    GUIDED_MIN_INLIERS,
    GUIDED_MIN_MATCHES,
    GUIDED_RADIUS_FRACTION,
    MIN_AGE_STEPS,
    NEW_VIEW_DEG,
    SAME_POINT_TOL,
    AnchorMap,
    PoseFix,
    correction_between,
    keypoint_depths,
)
from camera_calibration import Undistorter, load_rays
from map_renderer import MAP_WINDOW, MapRenderer
from motion_deviation import DeviationMeter, MotionSample
from map_store import MapStore
from mesh_builder import MeshBuilder, write_ply, write_stl
from saturation import SaturationMap

SAMPLES_DIR = Path(__file__).resolve().parent / "video-samples"
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
DEVIATION_WINDOW = "Motion deviation"
DEVIATION_WIDTH = 720  # displayed width of the deviation window
DEFAULT_VIDEO = SAMPLES_DIR / "forest-walking.mp4"

# Image regions SLAM must ignore for a given sample (overlays, watermarks, the drone's
# own body). Boxes are fractions of the frame: (x0, y0, x1, y1), origin top-left.
# Looked up by file name; videos not listed here are used whole.
# Tracking profiles. "legacy" is the original behaviour (kept for the forest video, where
# the tuned profile felt unstable); "tuned" was tuned on the drone clip with forward/reverse
# benchmarks. Videos not listed in VIDEO_PROFILE use "legacy".
#   guided         pose-guided anchor matching on every pose update
#   robust_anchors AP3P pose solver + trusting strong fixes + skipping trivial ones
#   scale_window   recent steps whose median bounds the next step's scale (1 = last step only)
#   scale_ratio    allowed factor above/below that reference
#   robust_pose    repair essential-matrix poses whose rotation contradicts the observed flow
#   map_scale      scale each step from landmark depths (False: from that step's scene depth,
#                  which cannot feed back into itself and so does not run away over time)
#   carry_scale    without a usable map scale, reuse the recent speed (True) or take the
#                  scene depth of this step alone (False); optional, default True
# "drone" is the experimental profile for the FPV clip: "tuned" plus whatever is being
# explored for it. "legacy" and "tuned" are never changed by that work.
# "pure" has no clamping and no carried-over speed: each step is scaled by the map's landmark
# depths when enough are in view, otherwise by that step's own scene depth.
PROFILES: dict[str, dict[str, float | bool | int]] = {
    "pure": {"guided": True, "robust_anchors": True, "scale_window": 1, "scale_ratio": float("inf"), "robust_pose": True, "map_scale": True, "carry_scale": False},
    "legacy": {"guided": False, "robust_anchors": False, "scale_window": 1, "scale_ratio": 4.0, "robust_pose": False, "map_scale": True},
    "tuned": {"guided": True, "robust_anchors": True, "scale_window": 12, "scale_ratio": 1.1, "robust_pose": False, "map_scale": True},
    "drone": {"guided": True, "robust_anchors": True, "scale_window": 12, "scale_ratio": 1.1, "robust_pose": True, "map_scale": False},
}
VIDEO_PROFILE: dict[str, str] = {
    "drone-flying-industrial-cut.mp4": "drone",
    "forest-walking.mp4": "legacy",
}

VIDEO_EXCLUDE_REGIONS: dict[str, list[tuple[float, float, float, float]]] = {
    # the controller-cam overlay measures ~26% wide x ~35% tall
    "drone-flying-industrial.mp4": [(0.0, 0.64, 0.27, 1.0)],
    "drone-flying-industrial-cut.mp4": [(0.0, 0.64, 0.27, 1.0)],
}
# repair_motion: redo a pose when the rotation-free flow exceeds REPAIR_RATIO * raw flow + REPAIR_MIN_PX
REPAIR_RATIO = 2.0
REPAIR_MIN_PX = 4.0
ROTATION_RANSAC_ITERS = 150
SIGN_CONFIDENT_FRONT = 0.8  # share of points in front of both cameras that settles the sign of t
SIGN_TIE_MARGIN = 0.3  # +t and -t this close count as a tie
DEFAULT_MAP_CACHE = Path(__file__).resolve().parent / ".map-cache"
DEFAULT_EXPORT_DIR = Path(__file__).resolve().parent / "exports"

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

TRACK_WINDOW = "SLAM tracks"
TRANSPORT_BAR_HEIGHT = 48


class PlaybackMode(IntEnum):
    PAUSED = 0
    FORWARD = 1
    REVERSE = 2


VIEW_TOGGLE = "view-toggle"  # transport-bar region key for the Points/Mesh button

BarRegion = tuple["PlaybackMode | str", int, int, int, int]


def draw_transport_bar(
    width: int,
    mode: PlaybackMode,
    view_mode: str | None = None,
) -> tuple[np.ndarray, list[BarRegion]]:
    bar = np.zeros((TRANSPORT_BAR_HEIGHT, width, 3), dtype=np.uint8)
    bar[:] = (42, 42, 48)

    labels = (
        (PlaybackMode.REVERSE, "Rewind"),
        (PlaybackMode.PAUSED, "Pause"),
        (PlaybackMode.FORWARD, "Play"),
    )
    button_w = min(120, max(72, width // 4))
    gap = 10
    total_w = len(labels) * button_w + (len(labels) - 1) * gap
    x = max(8, (width - total_w) // 2)
    regions: list[BarRegion] = []

    for play_mode, text in labels:
        x1 = x + button_w
        y1 = TRANSPORT_BAR_HEIGHT - 8
        active = mode == play_mode
        fill = (70, 130, 210) if active else (58, 58, 64)
        cv2.rectangle(bar, (x, 6), (x1, y1), fill, -1)
        cv2.rectangle(bar, (x, 6), (x1, y1), (110, 110, 120), 1)
        cv2.putText(
            bar,
            text,
            (x + 12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 245),
            1,
            cv2.LINE_AA,
        )
        regions.append((play_mode, x, 6, x1, y1))
        x = x1 + gap

    if view_mode is not None:
        toggle_w = 132
        tx0 = width - toggle_w - 8
        ty1 = TRANSPORT_BAR_HEIGHT - 8
        if tx0 > x:  # only when it does not collide with the playback buttons
            mesh = view_mode == "mesh"
            cv2.rectangle(bar, (tx0, 6), (tx0 + toggle_w, ty1), (80, 150, 90) if mesh else (58, 58, 64), -1)
            cv2.rectangle(bar, (tx0, 6), (tx0 + toggle_w, ty1), (110, 110, 120), 1)
            cv2.putText(
                bar,
                "View: Mesh" if mesh else "View: Points",
                (tx0 + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (240, 240, 245),
                1,
                cv2.LINE_AA,
            )
            regions.append((VIEW_TOGGLE, tx0, 6, tx0 + toggle_w, ty1))

    return bar, regions


def compose_track_display(
    vis: np.ndarray,
    hud_lines: list[str],
    mode: PlaybackMode,
    view_mode: str | None = None,
) -> tuple[np.ndarray, list[BarRegion], int]:
    overlay = vis.copy()
    y = 24
    for line in hud_lines:
        cv2.putText(
            overlay,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        y += 26
    bar, regions = draw_transport_bar(overlay.shape[1], mode, view_mode)
    return np.vstack([overlay, bar]), regions, overlay.shape[0]


def transport_hit(
    x: int,
    y: int,
    video_height: int,
    regions: list[BarRegion],
) -> "PlaybackMode | str | None":
    if y < video_height:
        return None
    y_in_bar = y - video_height
    for play_mode, x0, y0, x1, y1 in regions:
        if x0 <= x <= x1 and y0 <= y_in_bar <= y1:
            return play_mode
    return None


def camera_matrix(width: int, height: int, fov_deg: float | None = None) -> np.ndarray:
    if fov_deg is None:
        focal = 0.9 * width
    else:
        focal = width / (2.0 * np.tan(np.radians(fov_deg) / 2.0))
    return np.array(
        [
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


TRACK_GRID_COLS = 16
TRACK_GRID_ROWS = 9
TRACK_REFILL_FRACTION = 0.7  # refill a cell once it drops below this share of target
TRACK_CELL_CAP_FACTOR = 2  # max tracks per cell, as a multiple of the per-cell target
KEYFRAME_MIN_PARALLAX_PX = 8.0  # median pixel motion vs keyframe before updating pose
KEYFRAME_MIN_MATCHES = 60  # re-anchor the keyframe if fewer tracks survive


SATURATED_TRACK_DROP = 0.75  # a fully complete cell keeps this much less of its budget
MAX_CELL_BOOST = 2.5  # an unmapped cell may take up to this multiple of the base budget
MIN_CELL_TRACKS = 4  # complete cells still keep a few tracks for pose estimation
MIN_CELL_OBSERVATIONS = 5  # observations before a cell's saturation is trusted
SATURATION_INTERVAL_SECONDS = 3.0  # how often finished map regions are thinned to anchors

# Track churn ("scramble"): once the pose is held by plenty of tracks, retire a share of
# the mature ones and re-seed into places that were not tracked before.
CHURN_EVERY_COMMITS = 6  # pose updates (each means real camera motion) between churns
CHURN_MIN_INLIERS = 300  # pose must be this well supported to churn
CHURN_FRACTION = 0.25  # share of all tracks retired per churn
CHURN_MIN_WEIGHT = 20.0  # only tracks whose landmark has matured are retired
CHURN_CLEARANCE_PX = 15  # new corners keep this far from the retired ones


def per_cell_target(target_total: int) -> int:
    return max(3, int(np.ceil(target_total / (TRACK_GRID_COLS * TRACK_GRID_ROWS))))


def allocate_cell_targets(target_total: int, saturation: np.ndarray) -> np.ndarray:
    """Per-cell track budget: cells showing finished map give theirs to unmapped cells.

    `saturation` is the share (0..1) of each image cell already covered by settled
    anchors. The total stays near `target_total`, so cost is unchanged.
    """
    base = target_total / (TRACK_GRID_COLS * TRACK_GRID_ROWS)
    weight = 1.0 - SATURATED_TRACK_DROP * np.clip(saturation, 0.0, 1.0)
    targets = base * weight / weight.mean()
    return np.clip(targets, MIN_CELL_TRACKS, MAX_CELL_BOOST * base)


def prune_crowded(
    points: np.ndarray,
    ids: np.ndarray,
    width: int,
    height: int,
    target_total: int,
    targets: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Cap tracks per grid cell (oldest kept) so crowded areas never starve the rest."""
    if len(ids) == 0:
        return points, ids
    if targets is None:
        targets = np.full((TRACK_GRID_ROWS, TRACK_GRID_COLS), per_cell_target(target_total))
    flat = points.reshape(-1, 2)
    cx = np.clip((flat[:, 0] * TRACK_GRID_COLS / width).astype(int), 0, TRACK_GRID_COLS - 1)
    cy = np.clip((flat[:, 1] * TRACK_GRID_ROWS / height).astype(int), 0, TRACK_GRID_ROWS - 1)
    cell = cy * TRACK_GRID_COLS + cx

    order = np.lexsort((ids, cell))  # by cell, then oldest id first
    sorted_cell = cell[order]
    group_starts = np.r_[0, np.flatnonzero(np.diff(sorted_cell)) + 1]
    group_sizes = np.diff(np.r_[group_starts, len(order)])
    rank = np.arange(len(order)) - np.repeat(group_starts, group_sizes)

    caps = np.ceil(TRACK_CELL_CAP_FACTOR * targets[cy, cx]).astype(int)
    keep = np.zeros(len(ids), dtype=bool)
    keep[order[rank < caps[order]]] = True
    return points[keep], ids[keep]


def detect_balanced(
    gray: np.ndarray,
    existing: np.ndarray | None,
    target_total: int,
    targets: np.ndarray | None = None,
    avoid: np.ndarray | None = None,
    refill_fraction: float = TRACK_REFILL_FRACTION,
    exclude_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Detect corners so every grid cell of the frame keeps a share of tracks.

    `avoid` lists pixels (e.g. just-retired tracks) that new corners must keep clear of.
    """
    height, width = gray.shape[:2]
    cols, rows = TRACK_GRID_COLS, TRACK_GRID_ROWS
    if targets is None:
        targets = np.full((rows, cols), per_cell_target(target_total))

    pts = (
        np.empty((0, 2), dtype=np.float32)
        if existing is None
        else existing.reshape(-1, 2)
    )

    counts = np.zeros((rows, cols), dtype=np.int32)
    free = np.full((height, width), 255, np.uint8) if exclude_mask is None else exclude_mask.copy()
    if len(pts):
        cx = np.clip((pts[:, 0] * cols / width).astype(int), 0, cols - 1)
        cy = np.clip((pts[:, 1] * rows / height).astype(int), 0, rows - 1)
        np.add.at(counts, (cy, cx), 1)

        xi = np.clip(np.round(pts[:, 0]).astype(int), 0, width - 1)
        yi = np.clip(np.round(pts[:, 1]).astype(int), 0, height - 1)
        free[yi, xi] = 0
        free = cv2.erode(free, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    if avoid is not None and len(avoid):
        apts = avoid.reshape(-1, 2)
        keep_clear = np.full((height, width), 255, np.uint8)
        keep_clear[
            np.clip(np.round(apts[:, 1]).astype(int), 0, height - 1),
            np.clip(np.round(apts[:, 0]).astype(int), 0, width - 1),
        ] = 0
        keep_clear = cv2.erode(
            keep_clear, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CHURN_CLEARANCE_PX, CHURN_CLEARANCE_PX))
        )
        free = cv2.bitwise_and(free, keep_clear)

    pad = 4
    found: list[np.ndarray] = []
    for r in range(rows):
        y0, y1 = r * height // rows, (r + 1) * height // rows
        for c in range(cols):
            have = int(counts[r, c])
            per_cell = int(round(targets[r, c]))
            if have >= per_cell * refill_fraction:
                continue
            x0, x1 = c * width // cols, (c + 1) * width // cols
            ya, yb = max(0, y0 - pad), min(height, y1 + pad)
            xa, xb = max(0, x0 - pad), min(width, x1 + pad)

            roi_mask = np.zeros((yb - ya, xb - xa), np.uint8)
            roi_mask[y0 - ya : y1 - ya, x0 - xa : x1 - xa] = free[y0:y1, x0:x1]
            corners = cv2.goodFeaturesToTrack(
                np.ascontiguousarray(gray[ya:yb, xa:xb]),
                maxCorners=per_cell - have,
                qualityLevel=0.004,
                minDistance=4,
                mask=roi_mask,
            )
            if corners is None:
                continue
            corners = corners.reshape(-1, 2)
            corners[:, 0] += xa
            corners[:, 1] += ya
            found.append(corners)

    if not found:
        return np.empty((0, 1, 2), dtype=np.float32)
    return np.vstack(found).astype(np.float32).reshape(-1, 1, 2)


def track_features(
    prev_gray: np.ndarray,
    gray: np.ndarray,
    points: np.ndarray,
    fb_max_px: float = 0.0,
    exclude_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """LK tracking; a point is kept while LK finds it and it stays in the frame.

    With fb_max_px > 0 an extra backward pass also drops points that do not
    return to where they started (stricter, off by default).
    """
    height, width = gray.shape[:2]
    forward, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, points, None, **LK_PARAMS
    )
    flat = forward.reshape(-1, 2)
    keep = (
        (status.reshape(-1) == 1)
        & (flat[:, 0] >= 0)
        & (flat[:, 0] < width - 1)
        & (flat[:, 1] >= 0)
        & (flat[:, 1] < height - 1)
    )
    if exclude_mask is not None and keep.any():
        xi = np.clip(np.round(flat[:, 0]).astype(int), 0, width - 1)
        yi = np.clip(np.round(flat[:, 1]).astype(int), 0, height - 1)
        keep &= exclude_mask[yi, xi] > 0
    if fb_max_px > 0 and keep.any():
        backward, status_b, _ = cv2.calcOpticalFlowPyrLK(
            gray, prev_gray, forward, None, **LK_PARAMS
        )
        fb_error = np.linalg.norm((points - backward).reshape(-1, 2), axis=1)
        keep &= (status_b.reshape(-1) == 1) & (fb_error < fb_max_px)
    return forward, keep


def estimate_motion(
    previous: np.ndarray,
    current: np.ndarray,
    intrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if len(previous) < 8 or len(current) < 8:
        return None

    essential, mask = cv2.findEssentialMat(
        previous,
        current,
        intrinsic,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0,
    )
    if essential is None or mask is None:
        return None

    _, rotation, translation, pose_mask = cv2.recoverPose(
        essential,
        previous,
        current,
        intrinsic,
        mask=mask,
    )
    inliers = pose_mask.ravel() > 0
    if inliers.sum() < 8:
        return None

    return rotation, translation, inliers


def _rays(points: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points, np.ones(len(points))]) @ np.linalg.inv(intrinsic).T
    return homogeneous / np.linalg.norm(homogeneous, axis=1, keepdims=True)


def _kabsch(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotation R minimising |target - R source| for matched unit vectors (rows)."""
    u, _, vt = np.linalg.svd(source.T @ target)
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ np.diag([1.0, 1.0, sign]) @ u.T


def _estimate_rotation(
    previous: np.ndarray, current: np.ndarray, intrinsic: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Rotation-only RANSAC; the flow of one commit is small, so this stays near identity."""
    rays_a, rays_b = _rays(previous, intrinsic), _rays(current, intrinsic)
    best, best_count = None, -1
    for _ in range(ROTATION_RANSAC_ITERS):
        idx = rng.choice(len(rays_a), 3, replace=False)
        rotation = _kabsch(rays_a[idx], rays_b[idx])
        moved = (rotation @ rays_a.T).T @ intrinsic.T
        moved = moved[:, :2] / np.maximum(moved[:, 2:3], 1e-9)
        inliers = np.linalg.norm(moved - current, axis=1) < 2.0
        if inliers.sum() > best_count:
            best, best_count = inliers, int(inliers.sum())
    return _kabsch(rays_a[best], rays_b[best])


def _translation_given_rotation(
    previous: np.ndarray, current: np.ndarray, intrinsic: np.ndarray, rotation: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray] | None:
    """With R known, each match gives one linear constraint n . t = 0 (n = (R x1) x x2)."""
    rays_a, rays_b = _rays(previous, intrinsic), _rays(current, intrinsic)
    normals = np.cross((rotation @ rays_a.T).T, rays_b)
    focal = float(intrinsic[0, 0])
    best, best_count = None, -1
    for _ in range(ROTATION_RANSAC_ITERS):
        idx = rng.choice(len(normals), 2, replace=False)
        t = np.cross(normals[idx[0]], normals[idx[1]])
        norm = np.linalg.norm(t)
        if norm < 1e-9:
            continue
        inliers = np.abs(normals @ (t / norm)) * focal < 1.0
        if inliers.sum() > best_count:
            best, best_count = inliers, int(inliers.sum())
    if best is None or best_count < 15:
        return None
    t = np.linalg.svd(normals[best])[2][-1]
    # Cheirality: the sign of t is the one that puts most points in front of both cameras.
    cloud = cv2.triangulatePoints(
        intrinsic @ np.hstack([np.eye(3), np.zeros((3, 1))]),
        intrinsic @ np.hstack([rotation, t.reshape(3, 1)]),
        previous[best].T.astype(np.float64),
        current[best].T.astype(np.float64),
    )
    points = (cloud[:3] / cloud[3]).T
    front = ((points[:, 2] > 0) & ((points @ rotation.T + t)[:, 2] > 0)).mean()
    if front < 0.5:
        t = -t
    return t.reshape(3, 1), best


def repair_motion(
    motion: tuple[np.ndarray, np.ndarray, np.ndarray],
    previous: np.ndarray,
    current: np.ndarray,
    intrinsic: np.ndarray,
    stats: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Reject essential-matrix poses whose rotation contradicts the measured flow.

    One commit spans only ~10-40 px of flow, so a genuine rotation cannot be several
    degrees. When the flow left over after removing the estimated rotation is much bigger
    than the raw flow, the 5-point fit has traded translation against rotation. The fix is
    to estimate rotation on its own (RANSAC, three DoF), then solve translation linearly.
    """
    rotation, _, inliers = motion
    raw = float(np.median(np.linalg.norm(current[inliers] - previous[inliers], axis=1)))
    left_over = translation_parallax(rotation, intrinsic, previous[inliers], current[inliers])
    if left_over <= REPAIR_RATIO * raw + REPAIR_MIN_PX:
        return motion
    rng = np.random.default_rng(len(previous))
    rotation = _estimate_rotation(previous, current, intrinsic, rng)
    solved = _translation_given_rotation(previous, current, intrinsic, rotation, rng)
    if solved is None:
        stats["pose_rejected"] = stats.get("pose_rejected", 0) + 1
        return None
    translation, inliers = solved
    stats["pose_repaired"] = stats.get("pose_repaired", 0) + 1
    return rotation, translation, inliers


def triangulation_scale(points_cam: np.ndarray) -> float:
    depths = points_cam[:, 2]
    positive = depths > 0.1
    if positive.sum() < 8:
        return 1.0
    median_depth = float(np.median(depths[positive]))
    if median_depth < 1e-6:
        return 1.0
    return 1.0 / median_depth


def triangulate_pair(
    intrinsic: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    previous_px: np.ndarray,
    current_px: np.ndarray,
) -> np.ndarray:
    projection_previous = intrinsic @ np.hstack([np.eye(3), np.zeros((3, 1))])
    projection_current = intrinsic @ np.hstack([rotation, translation.reshape(3, 1)])
    homogeneous = cv2.triangulatePoints(
        projection_previous,
        projection_current,
        previous_px.T,
        current_px.T,
    )
    points = (homogeneous[:3] / homogeneous[3]).T
    return points


MIN_TRIANGULATION_DEG = 1.0  # ignore observations with less ray separation than this
MAX_RAY_GAP = 0.1  # reject if the two rays miss by more than this fraction of depth
MAX_OBSERVATION_WEIGHT = 100.0
MIN_SCALE_MATCHES = 12  # landmarks needed to anchor the step scale to the map
ANCHOR_INTERVAL = 3  # add anchors / run the global (no-prior) match every Nth pose update
STRONG_FIX_INLIERS = 60  # a fix this well supported is trusted even if it looks like a big jump
STRONG_FIX_RATIO = 0.3  # ...provided this share of its matches agree
TRIVIAL_FIX_STEPS = 0.25  # skip fixes smaller than this many typical steps (and TRIVIAL_FIX_DEG)
TRIVIAL_FIX_DEG = 0.3
PRUNE_INTERVAL_SECONDS = 4.0  # how often isolated landmarks are checked for
MAX_NEW_ANCHORS = 300  # per anchor step
MIN_ANCHOR_LANDMARK_WEIGHT = 2.0  # only well-triangulated landmarks seed anchors


def pixel_rays(
    pixels: np.ndarray,
    intrinsic: np.ndarray,
    cam_to_world: np.ndarray,
) -> np.ndarray:
    """Unit world-space viewing rays; cam_to_world is (3,3) or per-pixel (N,3,3)."""
    homogeneous = np.column_stack([pixels, np.ones(len(pixels))])
    rays_cam = homogeneous @ np.linalg.inv(intrinsic).T
    if cam_to_world.ndim == 2:
        rays = rays_cam @ cam_to_world.T
    else:
        rays = np.einsum("nij,nj->ni", cam_to_world, rays_cam)
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def triangulate_midpoint(
    origin_a: np.ndarray,
    dir_a: np.ndarray,
    origin_b: np.ndarray,
    dir_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closest point between two ray sets: (points, angle in degrees, valid mask)."""
    cos_ab = np.einsum("ij,ij->i", dir_a, dir_b)
    offset = origin_a - origin_b
    d = np.einsum("ij,ij->i", dir_a, offset)
    e = np.einsum("ij,ij->i", dir_b, offset)
    denom = np.maximum(1.0 - cos_ab**2, 1e-12)
    s = (cos_ab * e - d) / denom
    t = (e - cos_ab * d) / denom

    point_a = origin_a + s[:, None] * dir_a
    point_b = origin_b + t[:, None] * dir_b
    gap = np.linalg.norm(point_a - point_b, axis=1)
    angle = np.degrees(np.arccos(np.clip(cos_ab, -1.0, 1.0)))

    valid = (
        (s > 0.0)
        & (t > 0.0)
        & (angle >= MIN_TRIANGULATION_DEG)
        & (gap <= MAX_RAY_GAP * np.maximum(s, 1e-9))
    )
    return 0.5 * (point_a + point_b), angle, valid


def transform_points_to_world(
    points_cam: np.ndarray,
    world_rotation: np.ndarray,
    world_position: np.ndarray,
) -> np.ndarray:
    return (world_rotation @ points_cam.T).T + world_position


def sample_colors(image: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    xi = np.clip(np.round(pixels[:, 0]).astype(int), 0, width - 1)
    yi = np.clip(np.round(pixels[:, 1]).astype(int), 0, height - 1)
    bgr = image[yi, xi]
    return bgr[:, ::-1].astype(np.float64) / 255.0


def build_exclude_mask(
    shape: tuple[int, int],
    regions: list[tuple[float, float, float, float]] | None,
) -> np.ndarray | None:
    """uint8 mask, 255 = usable and 0 = ignored, from fractional (x0, y0, x1, y1) boxes."""
    if not regions:
        return None
    height, width = shape
    mask = np.full((height, width), 255, np.uint8)
    for x0, y0, x1, y1 in regions:
        mask[int(round(y0 * height)) : int(round(y1 * height)), int(round(x0 * width)) : int(round(x1 * width))] = 0
    return mask


def resolve_exclusions(args: argparse.Namespace) -> None:
    """Fill args.exclude_regions and the tracking profile for the chosen video."""
    name = "legacy" if args.profile is None else args.profile
    if args.profile is None and args.video is not None:
        name = VIDEO_PROFILE.get(Path(args.video).name, "legacy")
    profile = PROFILES[name]
    args.profile_name = name
    args.guided = bool(profile["guided"]) and not args.no_guided
    args.robust_anchors = bool(profile["robust_anchors"])
    args.scale_window = int(profile["scale_window"])
    args.robust_pose = bool(profile["robust_pose"])
    args.no_map_scale = args.no_map_scale or not bool(profile["map_scale"])
    args.scale_ratio = float(profile["scale_ratio"])
    args.carry_scale = bool(profile.get("carry_scale", True))
    regions: list[tuple[float, float, float, float]] = []
    if args.video is not None and not args.no_default_exclude:
        regions.extend(VIDEO_EXCLUDE_REGIONS.get(Path(args.video).name, []))
    regions.extend(args.exclude or [])
    args.exclude_regions = regions


def translation_parallax(
    rotation: np.ndarray, intrinsic: np.ndarray, key_px: np.ndarray, current_px: np.ndarray
) -> float:
    """Median pixel motion left over once the pure-rotation part is removed.

    Rotation alone moves points by H = K R K^-1 regardless of depth. What is left is
    caused by translation; near zero means the camera only turned (or barely moved), so
    the translation direction from the essential matrix is just noise.
    """
    homography = intrinsic @ rotation @ np.linalg.inv(intrinsic)
    moved = np.column_stack([key_px, np.ones(len(key_px))]) @ homography.T
    moved = moved[:, :2] / np.maximum(moved[:, 2:3], 1e-9)
    return float(np.median(np.linalg.norm(moved - current_px, axis=1)))


def draw_tracks(
    image: np.ndarray, points: np.ndarray, exclude_mask: np.ndarray | None = None
) -> np.ndarray:
    vis = image.copy()
    if exclude_mask is not None:
        ignored = exclude_mask == 0
        vis[ignored] = (vis[ignored] * 0.35).astype(np.uint8)
    for x, y in points.reshape(-1, 2):
        cv2.circle(vis, (int(round(x)), int(round(y))), 2, (0, 255, 0), -1)
    return vis


def resize_frame(frame: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return frame
    return cv2.resize(
        frame,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )


class SlamSession:
    """Headless SLAM: frames in, pose / map out. The interactive app and benchmark share this."""

    def __init__(
        self,
        first_frame: np.ndarray,
        args: argparse.Namespace,
        *,
        fps: float = 30.0,
        intrinsic: np.ndarray | None = None,
    ) -> None:
        """`intrinsic` overrides the pinhole matrix derived from args.fov_deg (frames must then match it)."""
        self.args = args
        self.fps = float(fps) or 30.0
        self.frame_number = 0  # 0 = first frame, +1 per process() call
        self.key_frame_number = 0  # frame number of the current keyframe image
        self.motion_sample: MotionSample | None = None  # latest camera-motion estimate and the points behind it
        self.height, self.width = first_frame.shape[:2]
        if intrinsic is None:
            self.intrinsic = camera_matrix(self.width, self.height, args.fov_deg)
        else:
            self.intrinsic = np.asarray(intrinsic, dtype=np.float64)
        self.exclude_mask = build_exclude_mask(
            (self.height, self.width), getattr(args, "exclude_regions", None)
        )

        gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        self.prev_gray = gray
        self.track_points = detect_balanced(
            gray, None, args.max_features, exclude_mask=self.exclude_mask
        )
        self.track_ids = np.arange(len(self.track_points), dtype=np.int64)
        self.next_track_id = len(self.track_points)

        self.world_rotation = np.eye(3, dtype=np.float64)
        self.world_position = np.zeros(3, dtype=np.float64)
        self.trajectory = [self.world_position.copy()]
        self.last_step_length: float | None = None

        self.key_ids = np.empty(0, dtype=np.int64)
        self.key_pts = np.empty((0, 2), dtype=np.float64)
        self.key_birth_px = np.empty((0, 2), dtype=np.float64)
        self.key_birth_R = np.empty((0, 3, 3), dtype=np.float64)
        self.key_birth_C = np.empty((0, 3), dtype=np.float64)
        self.key_gray: np.ndarray | None = None
        self.set_keyframe(self.track_points, self.track_ids, gray)

        cache_dir = None if args.no_disk_cache else args.map_cache.expanduser().resolve()
        self.map_store = MapStore(
            cache_dir=cache_dir,
            ram_archive_limit=args.ram_archive_limit,
            max_live_points=args.max_live_points,
        )
        self.mesh_builder = None if args.no_mesh else MeshBuilder(self.map_store)
        self.anchors = None if args.no_anchors else AnchorMap(max_anchors=args.max_anchors, robust=args.robust_anchors)
        self.stats = {
            "commits": 0, "fixes": 0, "fix_rejected": 0, "guided_tries": 0, "guided_fixes": 0, "associated": 0, "pruned": 0, "thinned": 0,
            "redundant": 0, "churned": 0, "churns": 0, "mesh_tris": 0,
        }
        self.churn_state = {"commits": 0, "inliers": 0}
        self.prune_state = {"last": 0.0, "saturate": 0.0}
        self.saturation_map = None if args.no_saturation else SaturationMap()
        self.cell_targets = np.full(
            (TRACK_GRID_ROWS, TRACK_GRID_COLS),
            float(per_cell_target(args.max_features)),
        )
        self.processed = 0
        # one row per pose event: (kind, step, inliers, parallax_px, rot_free_px, turn_deg, length)
        self.commit_log: list[tuple] = []
        self._last_delta: np.ndarray | None = None
        # Monocular scale has no absolute reference, so the step length is carried over as a speed
        # (distance per frame): commits span a varying number of frames, a per-commit length would not.
        self.last_frame_speed: float | None = None
        self.recent_steps: deque[float] = deque(maxlen=args.scale_window)  # recent speeds (distance per frame)

    def video_clock(self) -> float:
        return self.processed * max(self.args.stride, 1) / max(self.fps, 1e-6)

    def set_keyframe(self, points: np.ndarray, ids: np.ndarray, image: np.ndarray) -> None:
        self.key_gray = image
        self.key_frame_number = self.frame_number
        order = np.argsort(ids)
        new_ids = ids[order]
        new_pts = points.reshape(-1, 2)[order].astype(np.float64)
        count = len(new_ids)
        birth_px = new_pts.copy()
        birth_R = np.broadcast_to(self.world_rotation, (count, 3, 3)).copy()
        birth_C = np.broadcast_to(self.world_position, (count, 3)).copy()
        if len(self.key_ids) > 0 and count > 0:
            old = np.minimum(np.searchsorted(self.key_ids, new_ids), len(self.key_ids) - 1)
            known = self.key_ids[old] == new_ids
            birth_px[known] = self.key_birth_px[old[known]]
            birth_R[known] = self.key_birth_R[old[known]]
            birth_C[known] = self.key_birth_C[old[known]]
        self.key_ids, self.key_pts = new_ids, new_pts.astype(np.float32)
        self.key_birth_px, self.key_birth_R, self.key_birth_C = birth_px, birth_R, birth_C

    def _update_cell_targets(self, pixels: np.ndarray, settled: np.ndarray) -> None:
        cx = np.clip((pixels[:, 0] * TRACK_GRID_COLS / self.width).astype(int), 0, TRACK_GRID_COLS - 1)
        cy = np.clip((pixels[:, 1] * TRACK_GRID_ROWS / self.height).astype(int), 0, TRACK_GRID_ROWS - 1)
        total = np.zeros((TRACK_GRID_ROWS, TRACK_GRID_COLS))
        done = np.zeros_like(total)
        np.add.at(total, (cy, cx), 1.0)
        np.add.at(done, (cy, cx), settled.astype(float))
        fraction = np.where(total >= MIN_CELL_OBSERVATIONS, done / np.maximum(total, 1.0), 0.0)
        wanted = allocate_cell_targets(self.args.max_features, fraction)
        self.cell_targets = 0.6 * self.cell_targets + 0.4 * wanted

    def _apply_fix(self, fix: PoseFix, step: int) -> bool:
        """Snap the pose to an anchor fix and spread the correction over the path since the last one.

        Returns True when the fix agrees with (or replaces) the live pose, False if it was
        rejected as implausible.
        """
        anchors = self.anchors
        start = anchors.last_fix_step
        correction = correction_between(
            self.world_rotation, self.world_position, fix.rotation, fix.position
        )
        shift, turn = correction.magnitude
        stats = self.stats
        robust = self.args.robust_anchors
        if robust and shift < TRIVIAL_FIX_STEPS * (self.last_step_length or 0.0) and turn < TRIVIAL_FIX_DEG:
            return True  # live pose already agrees with the map; nothing to correct

        path = np.asarray(self.trajectory[start:])
        path_len = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()) if len(path) > 1 else 0.0
        # A fix backed by many consistent inliers beats dead-reckoning outright: after a
        # fast stretch the live pose can be arbitrarily far off, and no gate based on how
        # far we *think* we travelled can tell. Weaker fixes still have to look plausible.
        strong = robust and fix.inliers >= STRONG_FIX_INLIERS and fix.inliers >= STRONG_FIX_RATIO * fix.matches
        plausible = strong or (
            turn < 30.0 and shift < 0.6 * path_len + 5.0 * (self.last_step_length or 0.0)
        )
        if not plausible:
            stats["fix_rejected"] += 1
            return False

        poses = np.asarray(self.trajectory)
        fraction = np.clip((np.arange(len(poses)) - start) / max(step - start, 1), 0.0, 1.0)
        moving = fraction > 0.0
        poses[moving] = correction.apply(poses[moving], fraction[moving])
        self.trajectory[:] = [row.copy() for row in poses]
        self.map_store.warp(correction, start, step)
        if self.mesh_builder is not None:
            self.mesh_builder.version += 1
        anchors.warp(correction, start, step)
        anchors.last_fix_step = step
        if self.saturation_map is not None:
            self.saturation_map.mark_dirty()
        self.world_rotation, self.world_position = fix.rotation, fix.position.copy()
        self.key_ids = np.empty(0, dtype=np.int64)
        stats["fixes"] += 1
        return True

    def _run_anchor_step(
        self,
        gray: np.ndarray,
        track_ids_now: np.ndarray,
        track_px: np.ndarray,
        periodic: bool,
    ) -> None:
        anchors = self.anchors
        if anchors is None:
            return
        keypoints, descriptors, response = anchors.detect(gray, self.exclude_mask)
        if len(keypoints) == 0:
            return
        step = len(self.trajectory) - 1
        stale = step - MIN_AGE_STEPS

        # Tier 1: guided matching around where the live pose expects each anchor. Cheap and
        # clean, so it runs on every pose update and pulls the pose onto the map quickly.
        query, matched = np.empty(0, dtype=int), np.empty(0, dtype=int)
        settled = False
        if self.args.guided:
            query, matched = anchors.match_guided(
                keypoints, descriptors, self.world_rotation, self.world_position,
                self.intrinsic, (self.height, self.width), GUIDED_RADIUS_FRACTION * self.width,
            )
            old = anchors.steps[matched] <= stale
            if old.sum() >= GUIDED_MIN_MATCHES:
                self.stats["guided_tries"] += 1
                fix = anchors.localize(
                    keypoints[query[old]], matched[old], self.intrinsic,
                    min_matches=GUIDED_MIN_MATCHES, min_inliers=GUIDED_MIN_INLIERS,
                )
                if fix is not None:
                    settled = self._apply_fix(fix, step)
                    self.stats["guided_fixes"] += int(settled)

        # Tier 2: match against every anchor (no pose prior) to recover from large drift.
        # Runs when the guided tier found nothing, and cannot be fooled by a bad live pose.
        if not settled and (periodic or not self.args.guided):
            query, matched = anchors.match(descriptors, self.world_position)
            if len(query):
                old = anchors.steps[matched] <= stale
                fix = anchors.localize(keypoints[query[old]], matched[old], self.intrinsic)
                if fix is not None:
                    self._apply_fix(fix, step)
        if not periodic:
            return

        found, landmarks = self.map_store.lookup(track_ids_now)
        if found.sum() < 20:
            return
        weights = self.map_store.lookup_weights(track_ids_now)[found]
        landmarks = landmarks[weights >= MIN_ANCHOR_LANDMARK_WEIGHT]
        pixels = track_px[found][weights >= MIN_ANCHOR_LANDMARK_WEIGHT]
        camera = (landmarks - self.world_position) @ self.world_rotation
        ahead = camera[:, 2] > 0.2
        depth = keypoint_depths(keypoints, pixels[ahead], camera[ahead, 2], (self.height, self.width))
        valid = np.isfinite(depth)
        if not valid.any():
            return

        normalized = np.column_stack([keypoints, np.ones(len(keypoints))]) @ np.linalg.inv(self.intrinsic).T
        points = (self.world_rotation @ (normalized * depth[:, None]).T).T + self.world_position
        to_camera = self.world_position - points
        to_camera /= np.maximum(np.linalg.norm(to_camera, axis=1, keepdims=True), 1e-9)

        wanted = valid.copy()
        if len(query):
            gap = np.linalg.norm(points[query] - anchors.positions[matched], axis=1)
            same = gap < SAME_POINT_TOL * np.nan_to_num(depth[query], nan=1e9)
            cos_view = np.einsum("ij,ij->i", to_camera[query], anchors.views[matched])
            fresh_view = cos_view < np.cos(np.radians(NEW_VIEW_DEG))
            wanted[query] = valid[query] & same & fresh_view
        candidates = np.flatnonzero(wanted)
        if len(candidates) > MAX_NEW_ANCHORS:
            candidates = candidates[np.argsort(-response[candidates])[:MAX_NEW_ANCHORS]]
        anchors.add(descriptors[candidates], points[candidates], to_camera[candidates], step)

    def process(self, frame: np.ndarray) -> None:
        """Advance tracking, pose and map by one BGR frame."""
        args = self.args
        self.frame_number += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if len(self.track_points) < 8:
            self.track_points = detect_balanced(
                gray, None, args.max_features, exclude_mask=self.exclude_mask
            )
            self.track_ids = np.arange(
                self.next_track_id, self.next_track_id + len(self.track_points), dtype=np.int64
            )
            self.next_track_id += len(self.track_points)
            self.set_keyframe(self.track_points, self.track_ids, gray)
            self.prev_gray = gray
            self.processed += 1
            return

        next_points, keep = track_features(
            self.prev_gray, gray, self.track_points, args.fb_check, self.exclude_mask
        )
        good_current = next_points[keep]
        good_ids = self.track_ids[keep]

        rekey = False
        committed = False
        pos = np.minimum(np.searchsorted(self.key_ids, good_ids), len(self.key_ids) - 1)
        matched = self.key_ids[pos] == good_ids
        if matched.sum() < KEYFRAME_MIN_MATCHES:
            rekey = True
        else:
            key_matched = self.key_pts[pos[matched]]
            current_matched = good_current[matched].reshape(-1, 2)
            ids_matched = good_ids[matched]
            parallax = float(np.median(np.linalg.norm(current_matched - key_matched, axis=1)))
            if parallax >= KEYFRAME_MIN_PARALLAX_PX:
                motion = estimate_motion(key_matched, current_matched, self.intrinsic)
                if motion is not None and args.robust_pose:
                    motion = repair_motion(motion, key_matched, current_matched, self.intrinsic, self.stats)
                if motion is None:
                    rekey = parallax > 4.0 * KEYFRAME_MIN_PARALLAX_PX
                else:
                    self.motion_sample = MotionSample(
                        key_matched, current_matched, motion[0], motion[1], self.key_frame_number, self.frame_number
                    )
                    self._commit_pose(
                        frame, gray, motion, key_matched, current_matched, ids_matched, pos[matched]
                    )
                    committed = True
                    rekey = True

        self.track_points = good_current.reshape(-1, 1, 2)
        self.track_ids = good_ids
        flat_tracks, self.track_ids = prune_crowded(
            self.track_points, self.track_ids, self.width, self.height, args.max_features, self.cell_targets
        )
        self.track_points = flat_tracks.reshape(-1, 1, 2)

        retired_px = None
        if (
            committed
            and not args.no_churn
            and self.churn_state["commits"] >= CHURN_EVERY_COMMITS
            and self.churn_state["inliers"] >= CHURN_MIN_INLIERS
        ):
            weight_now = self.map_store.lookup_weights(self.track_ids)
            mature = np.flatnonzero(weight_now >= CHURN_MIN_WEIGHT)
            n_retire = min(len(mature), int(CHURN_FRACTION * len(self.track_ids)))
            if n_retire >= 20:
                retire = mature[np.argsort(self.track_ids[mature])[:n_retire]]
                keep_track = np.ones(len(self.track_ids), dtype=bool)
                keep_track[retire] = False
                retired_px = self.track_points.reshape(-1, 2)[retire]
                self.track_points, self.track_ids = self.track_points[keep_track], self.track_ids[keep_track]
                self.stats["churned"] += n_retire
                self.stats["churns"] += 1
                self.churn_state["commits"] = 0
        fresh = detect_balanced(
            gray,
            self.track_points,
            args.max_features,
            self.cell_targets,
            avoid=retired_px,
            refill_fraction=1.0 if retired_px is not None else TRACK_REFILL_FRACTION,
            exclude_mask=self.exclude_mask,
        )
        if len(fresh):
            fresh_ids = np.arange(self.next_track_id, self.next_track_id + len(fresh), dtype=np.int64)
            self.next_track_id += len(fresh)
            self.track_points = np.vstack([self.track_points, fresh])
            self.track_ids = np.concatenate([self.track_ids, fresh_ids])

        if rekey:
            self.set_keyframe(self.track_points, self.track_ids, gray)
        self.prev_gray = gray
        self.processed += 1

    def _log_commit(
        self, kind: str, inliers: int, parallax: float, rot_free: float, delta: np.ndarray, rot_deg: float = 0.0
    ) -> None:
        length = float(np.linalg.norm(delta))
        turn = 0.0
        if self._last_delta is not None and length > 1e-12:
            cos = float(delta @ self._last_delta) / (length * float(np.linalg.norm(self._last_delta)) + 1e-12)
            turn = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
        if length > 1e-12:
            self._last_delta = delta.copy()
        self.commit_log.append((kind, len(self.trajectory), inliers, parallax, rot_free, turn, length, rot_deg))

    def _settle_direction(
        self,
        rotation: np.ndarray,
        translation: np.ndarray,
        unit_points: np.ndarray,
        key_inlier: np.ndarray,
        current_inlier: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Break translation-sign ties with the previous direction of travel.

        When the matches barely constrain depth (near-pure rotation), +t and -t put about
        as many points in front of the cameras, so the sign is a coin flip and the camera
        appears to reverse. If the data cannot tell the signs apart, keep the old heading.
        """
        def in_front(points: np.ndarray, t: np.ndarray) -> float:
            return float(((points[:, 2] > 0) & ((points @ rotation.T + t.reshape(3))[:, 2] > 0)).mean())

        front = in_front(unit_points, translation)
        if front >= SIGN_CONFIDENT_FRONT:
            return translation, unit_points
        flipped = -translation
        flipped_points = triangulate_pair(self.intrinsic, rotation, flipped, key_inlier, current_inlier)
        if front - in_front(flipped_points, flipped) > SIGN_TIE_MARGIN:
            return translation, unit_points
        heading = self.world_rotation @ (-(rotation.T @ translation.reshape(3)))
        if float(heading @ self._last_delta) >= 0.0:
            return translation, unit_points
        self.stats["sign_flips"] = self.stats.get("sign_flips", 0) + 1
        return flipped, flipped_points

    def _commit_pose(
        self,
        frame: np.ndarray,
        gray: np.ndarray,
        motion: tuple[np.ndarray, np.ndarray, np.ndarray],
        key_matched: np.ndarray,
        current_matched: np.ndarray,
        ids_matched: np.ndarray,
        match_rows: np.ndarray,
    ) -> None:
        args = self.args
        rotation, translation, inliers = motion
        key_inlier = key_matched[inliers]
        current_inlier = current_matched[inliers]
        inlier_ids = ids_matched[inliers]
        inlier_rows = match_rows[inliers]
        good_ids = self.track_ids

        unit_points = triangulate_pair(
            self.intrinsic, rotation, translation, key_inlier, current_inlier
        )
        if args.robust_pose and self._last_delta is not None:
            translation, unit_points = self._settle_direction(
                rotation, translation, unit_points, key_inlier, current_inlier
            )
        scale = None
        self.map_store.reactivate_in_view(
            self.world_rotation, self.world_position, self.intrinsic, self.width, self.height
        )
        found, landmarks = self.map_store.lookup(inlier_ids)
        if found.sum() >= MIN_SCALE_MATCHES and not args.no_map_scale:
            depth_unit = unit_points[found, 2]
            depth_map = ((landmarks - self.world_position) @ self.world_rotation)[:, 2]
            usable = (depth_unit > 1e-3) & (depth_map > 1e-3)
            if usable.sum() >= MIN_SCALE_MATCHES:
                scale = float(np.median(depth_map[usable] / depth_unit[usable]))
        frames_spanned = max(self.frame_number - self.key_frame_number, 1)
        if scale is None:
            scale = (
                self.last_frame_speed * frames_spanned
                if args.carry_scale and self.last_frame_speed is not None
                else triangulation_scale(unit_points)
            )
        if self.recent_steps and np.isfinite(args.scale_ratio):
            # Monocular scale has no absolute reference, so a few bad depth matches can make
            # it jump or snowball. Keep it within a band around the recent median speed
            # times the frames this commit spans.
            reference = float(np.median(self.recent_steps)) * frames_spanned
            scale = float(np.clip(scale, reference / args.scale_ratio, reference * args.scale_ratio))

        camera_delta = -(rotation.T @ (translation.reshape(3) * scale))
        world_delta = self.world_rotation @ camera_delta
        self._log_commit(
            "step", int(len(inlier_ids)), float(np.median(np.linalg.norm(current_inlier - key_inlier, axis=1))),
            translation_parallax(rotation, self.intrinsic, key_inlier, current_inlier), world_delta,
            float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))),
        )
        self.world_position = self.world_position + world_delta
        self.world_rotation = self.world_rotation @ rotation.T
        self.last_step_length = float(np.linalg.norm(camera_delta))
        self.last_frame_speed = self.last_step_length / frames_spanned
        self.recent_steps.append(self.last_frame_speed)
        self.trajectory.append(self.world_position.copy())

        rays_birth = pixel_rays(self.key_birth_px[inlier_rows], self.intrinsic, self.key_birth_R[inlier_rows])
        rays_now = pixel_rays(current_inlier, self.intrinsic, self.world_rotation)
        points_world, angle_deg, usable = triangulate_midpoint(
            self.key_birth_C[inlier_rows],
            rays_birth,
            np.broadcast_to(self.world_position, (len(rays_now), 3)),
            rays_now,
        )
        step = len(self.trajectory) - 1
        if usable.any():
            weights = np.clip(angle_deg[usable] ** 2, 0.01, MAX_OBSERVATION_WEIGHT)
            colors = sample_colors(frame, current_inlier[usable])
            obs_ids = inlier_ids[usable]
            obs_points = points_world[usable]
            obs_pixels = current_inlier[usable]
            known, _ = self.map_store.lookup(obs_ids)
            fresh = np.flatnonzero(~known)
            if len(fresh):
                hit, target = self.map_store.find_existing(
                    obs_points[fresh],
                    obs_pixels[fresh],
                    self.world_rotation,
                    self.world_position,
                    self.intrinsic,
                    self.width,
                    self.height,
                    exclude_ids=good_ids,
                )
                if hit.any():
                    source = obs_ids[fresh][hit]
                    obs_ids = remap_ids(obs_ids, source, target[hit])
                    inlier_ids = remap_ids(inlier_ids, source, target[hit])
                    self.stats["associated"] += int(hit.sum())
            if self.saturation_map is not None and len(obs_points):
                distance = np.linalg.norm(obs_points - self.world_position, axis=1)
                settled = self.saturation_map.settled(self.map_store, obs_points, distance)
                self._update_cell_targets(obs_pixels, settled)
                known_now, _ = self.map_store.lookup(obs_ids)
                redundant = settled & ~known_now
                if redundant.any():
                    self.stats["redundant"] += int(redundant.sum())
                    fresh_only = ~redundant
                    obs_points, colors = obs_points[fresh_only], colors[fresh_only]
                    obs_ids, weights = obs_ids[fresh_only], weights[fresh_only]
            self.map_store.add(obs_points, colors, obs_ids, weights, step, now=self.video_clock())

        self.stats["commits"] += 1
        self.churn_state["commits"] += 1
        self.churn_state["inliers"] = len(inlier_ids)
        if self.mesh_builder is not None:
            found_m, pos_m, weight_m = self.map_store.lookup_all(inlier_ids)
            if found_m.sum() >= 12:
                self.stats["mesh_tris"] = self.mesh_builder.update(
                    rotation=self.world_rotation,
                    position=self.world_position,
                    ids=inlier_ids[found_m],
                    pixels=current_inlier[found_m],
                    points=pos_m,
                    weights=weight_m,
                    gray=gray,
                    key_gray=self.key_gray,
                    key_ids=self.key_ids,
                    key_pts=self.key_pts,
                )
            self.mesh_builder.purge(self.video_clock())
        if self.anchors is not None:
            periodic = self.stats["commits"] % ANCHOR_INTERVAL == 0
            if periodic or args.guided:
                self._run_anchor_step(gray, inlier_ids, current_inlier, periodic)

        clock = self.video_clock()
        if args.prune_age > 0 and clock - self.prune_state["last"] >= PRUNE_INTERVAL_SECONDS:
            self.prune_state["last"] = clock
            self.stats["pruned"] += self.map_store.prune_isolated(
                clock,
                np.asarray(self.trajectory),
                args.prune_age,
                args.prune_isolation,
                check_window=2.0 * PRUNE_INTERVAL_SECONDS,
                exempt_ids=None if self.saturation_map is None else self.saturation_map.anchor_ids,
            )
        if self.saturation_map is not None and clock - self.prune_state["saturate"] >= SATURATION_INTERVAL_SECONDS:
            self.prune_state["saturate"] = clock
            _, thinned = self.saturation_map.consolidate(self.map_store, clock, np.asarray(self.trajectory))
            self.stats["thinned"] += thinned
        self.map_store.archive_out_of_walker_view(
            self.world_rotation, self.world_position, self.intrinsic, self.width, self.height
        )

    def finalize(self) -> None:
        self.map_store.finalize()


def remap_ids(ids: np.ndarray, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if len(source) == 0:
        return ids
    order = np.argsort(source)
    source, target = source[order], target[order]
    pos = np.minimum(np.searchsorted(source, ids), len(source) - 1)
    hit = source[pos] == ids
    out = ids.copy()
    out[hit] = target[pos[hit]]
    return out


def layout_windows(video_width: int, video_height: int, map_height: int) -> None:
    margin_x = 40
    margin_y = 40
    title_bar = 32
    cv2.namedWindow(TRACK_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(TRACK_WINDOW, video_width, video_height)
    cv2.moveWindow(TRACK_WINDOW, margin_x, margin_y + map_height + title_bar)


def parse_box(text: str) -> tuple[float, float, float, float]:
    try:
        x0, y0, x1, y1 = (float(v) for v in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("expected X0,Y0,X1,Y1 as fractions, e.g. 0,0.75,0.25,1") from None
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise argparse.ArgumentTypeError("box must satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1")
    return x0, y0, x1, y1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "video",
        nargs="?",
        type=Path,
        default=None,
        help="path to input video (default: choose from 3D-SLAM/video-samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="resize factor for processing (default: 0.5)",
    )
    parser.add_argument(
        "--fov-deg",
        type=float,
        default=50.0,
        metavar="DEG",
        help="horizontal field of view in degrees (default: 50)",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="stop video after this many seconds (default: 0 = full video)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="use every Nth frame (default: 1)",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=4000,
        help="target tracked features per frame (default: 4000)",
    )
    parser.add_argument(
        "--fb-check",
        type=float,
        default=0.0,
        metavar="PX",
        help="drop tracks that fail a backward-flow check by more than PX pixels "
        "(default: 0 = off, always keep tracking)",
    )
    parser.add_argument(
        "--max-live-points",
        type=int,
        default=100_000,
        help="max landmarks kept in active RAM (default: 100000)",
    )
    parser.add_argument(
        "--ram-archive-limit",
        type=int,
        default=120_000,
        help="max archived landmarks in RAM before disk flush (default: 120000)",
    )
    parser.add_argument(
        "--map-cache",
        type=Path,
        default=DEFAULT_MAP_CACHE,
        help="directory for archived landmark chunks (default: 3D-SLAM/.map-cache)",
    )
    parser.add_argument(
        "--no-disk-cache",
        action="store_true",
        help="keep archive in RAM only (trim when full)",
    )
    parser.add_argument(
        "--no-anchors",
        action="store_true",
        help="disable global anchor features (relocalisation / loop correction)",
    )
    parser.add_argument(
        "--max-anchors",
        type=int,
        default=60_000,
        help="max global anchor features kept (default: 60000)",
    )
    parser.add_argument(
        "--prune-age",
        type=float,
        default=10.0,
        metavar="SEC",
        help="delete landmarks still isolated after this many video seconds "
        "(default: 10, 0 = never delete)",
    )
    parser.add_argument(
        "--prune-isolation",
        type=float,
        default=0.12,
        metavar="FRAC",
        help="isolated = nearest neighbour farther than FRAC x the point's distance "
        "from the camera path (default: 0.12; typical spacing is ~0.03)",
    )
    parser.add_argument(
        "--no-saturation",
        action="store_true",
        help="keep mapping everywhere at full resolution (no thinning of finished "
        "regions, fixed tracking budget per image cell)",
    )
    parser.add_argument(
        "--no-mesh",
        action="store_true",
        help="do not build the coloured surface mesh (point cloud only)",
    )
    parser.add_argument(
        "--start-mesh-view",
        action="store_true",
        help="open the 3D view in mesh mode (default: point mode; toggle with the View button)",
    )
    parser.add_argument(
        "--export-mesh",
        type=Path,
        default=None,
        metavar="BASE",
        help="on exit write BASE.ply (coloured) and BASE.stl (geometry only)",
    )
    parser.add_argument(
        "--no-churn",
        action="store_true",
        help="keep tracking the same points instead of periodically re-seeding into "
        "untracked regions",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default=None,
        help="tracking profile (default: per video, see VIDEO_PROFILE; unlisted videos use legacy)",
    )
    parser.add_argument(
        "--no-map-scale",
        action="store_true",
        help="experimental: take each step's scale from the scene depth of that step alone "
        "instead of from the map's landmark depths (avoids the map<->scale feedback loop)",
    )
    parser.add_argument(
        "--no-guided",
        action="store_true",
        help="disable the pose-guided anchor matching tier (global matching only)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        type=parse_box,
        default=None,
        metavar="X0,Y0,X1,Y1",
        help="ignore this image box (fractions of the frame, origin top-left); repeatable. "
        "Added to the per-video defaults in VIDEO_EXCLUDE_REGIONS",
    )
    parser.add_argument(
        "--no-default-exclude",
        action="store_true",
        help="do not apply the built-in per-video exclusion boxes",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        metavar="NPZ",
        help="pixel-ray camera calibration (.npz from virtual/calibrate_camera.py); "
        "default: <video name>.calibration.npz next to the video, if it exists",
    )
    parser.add_argument(
        "--no-deviation-window",
        action="store_true",
        help="do not open the window showing which screen sections disagree with the camera motion",
    )
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="ignore any camera calibration and assume an ideal pinhole from --fov-deg",
    )
    parser.add_argument(
        "--plot-every",
        type=int,
        default=3,
        help="refresh 3D map every N processed frames (default: 3)",
    )
    args = parser.parse_args(argv)
    resolve_exclusions(args)
    return args


def choose_video() -> Path:
    """List the files in video-samples/ and ask which one to run."""
    videos = sorted(
        p
        for p in SAMPLES_DIR.glob("*")
        if p.suffix.lower() in VIDEO_SUFFIXES and p.is_file()
    )
    if not videos:
        raise SystemExit(f"No videos found in {SAMPLES_DIR}; pass a path instead.")
    default_index = next(
        (i for i, p in enumerate(videos) if p.resolve() == DEFAULT_VIDEO.resolve()), 0
    )
    if not sys.stdin.isatty():
        return videos[default_index]

    for i, path in enumerate(videos, start=1):
        size_mb = path.stat().st_size / 1e6
        tag = "  (default)" if i - 1 == default_index else ""
        print(f"  {i}) {path.name}  [{size_mb:.0f} MB]{tag}")
    prompt = f"Choose a video [1-{len(videos)}, Enter={default_index + 1}, q=quit]: "
    while True:
        try:
            answer = input(prompt).strip().lower()
        except EOFError:
            return videos[default_index]
        if answer in ("q", "quit"):
            raise SystemExit(0)
        if not answer:
            return videos[default_index]
        if answer.isdigit() and 1 <= int(answer) <= len(videos):
            return videos[int(answer) - 1]
        print("Invalid choice.")


def load_undistorter(args: argparse.Namespace, video_path: Path, size: tuple[int, int]) -> Undistorter | None:
    """Calibration for this camera, or None (no file, switched off, or it does not fit the frame size)."""
    if args.no_calibration:
        return None
    path = args.calibration
    if path is None:
        path = video_path.with_name(f"{video_path.stem}.calibration.npz")
        if not path.is_file():
            return None
    path = path.expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"Calibration file not found: {path}")
    try:
        undistorter = Undistorter(load_rays(path), size)
    except ValueError as error:
        raise SystemExit(f"Cannot use calibration {path.name}: {error}")
    print(f"Calibration: {path.name}, {undistorter.describe()}")
    return undistorter


def main() -> None:
    args = parse_args()
    if args.video is None:
        args.video = choose_video()
        resolve_exclusions(args)
    video_path = args.video.expanduser().resolve()
    if not video_path.is_file():
        raise SystemExit(f"Video not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    ok, frame = capture.read()
    if not ok:
        raise SystemExit("Video has no readable frames")

    frame = resize_frame(frame, args.scale)
    undistorter = load_undistorter(args, video_path, (frame.shape[1], frame.shape[0]))

    def prepare_frame(raw: np.ndarray) -> np.ndarray:
        """Resize to the processing size, then correct the lens with the calibration if there is one."""
        resized = resize_frame(raw, args.scale)
        return resized if undistorter is None else undistorter.apply(resized)

    frame = frame if undistorter is None else undistorter.apply(frame)
    video_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    session = SlamSession(
        frame, args, fps=video_fps, intrinsic=None if undistorter is None else undistorter.intrinsic
    )
    height, width = session.height, session.width
    trajectory = session.trajectory
    map_store = session.map_store
    mesh_builder = session.mesh_builder
    anchors = session.anchors
    stats = session.stats
    saturation_map = session.saturation_map
    cache_dir = None if args.no_disk_cache else args.map_cache.expanduser().resolve()
    fx = session.intrinsic[0, 0]
    print(
        f"Profile: {args.profile_name} (guided={args.guided}, robust anchors={args.robust_anchors}, "
        f"scale window={args.scale_window}, ratio={args.scale_ratio:g}, robust pose={args.robust_pose}, map scale={not args.no_map_scale}); "
        f"excluded regions: {len(args.exclude_regions)}"
    )
    if undistorter is None:
        print(f"Camera intrinsics: horizontal FOV {args.fov_deg:g}° -> fx={fx:.1f}px")
    else:
        print(f"Camera intrinsics: from calibration, fx={fx:.1f}px (--fov-deg not used)")

    map_height = max(320, height // 2)
    map_view = MapRenderer(width, map_height)
    map_view.setup_window()
    map_view.set_mesh(mesh_builder)
    if args.start_mesh_view and mesh_builder is not None:
        map_view.view_mode = "mesh"
    layout_windows(width, height, map_height)
    map_view.show(trajectory, map_store)

    frame_index = 0
    playback_mode = PlaybackMode.FORWARD
    last_vis: np.ndarray | None = None
    play_delay_ms = max(1, int(round(1000.0 / video_fps)))
    button_regions: list[BarRegion] = []
    video_panel_height = height
    video_active = True
    map_finalized = False

    def hud_lines() -> list[str]:
        t_sec = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        mode_label = playback_mode.name.lower() if video_active else "map only"
        return [
            (
                f"tracks: {len(session.track_points)}  poses: {len(trajectory)}  "
                f"live: {map_store.live_count}  stored: {map_store.archive_count}  "
                f"anchors: {len(anchors) if anchors is not None else 0}  "
                f"fixes: {stats['fixes']}  pruned: {stats['pruned']}  "
                f"settled: {len(saturation_map) if saturation_map is not None else 0}  "
                f"faces: {len(mesh_builder) if mesh_builder is not None else 0}  "
                f"churned: {stats['churned']}"
            ),
            f"time: {t_sec:.1f}s  frame: {int(capture.get(cv2.CAP_PROP_POS_FRAMES))}  ({mode_label})",
        ]

    def read_frame_forward() -> tuple[bool, np.ndarray | None]:
        ok, raw = capture.read()
        if not ok:
            return False, None
        return True, prepare_frame(raw)

    def read_frame_backward() -> tuple[bool, np.ndarray | None]:
        pos = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
        if pos <= 1:
            return False, None
        capture.set(cv2.CAP_PROP_POS_FRAMES, pos - 2)
        ok, raw = capture.read()
        if not ok:
            return False, None
        return True, prepare_frame(raw)

    def stop_video_playback() -> None:
        nonlocal video_active, map_finalized, playback_mode
        if not video_active:
            return
        video_active = False
        playback_mode = PlaybackMode.PAUSED
        if capture.isOpened():
            capture.release()
        cv2.destroyAllWindows()
        map_store.finalize()
        map_finalized = True
        map_view.show(trajectory, map_store)
        print(
            "Video stopped. Explore the 3D map "
            "(close the map window when finished)."
        )

    def show_vis(vis: np.ndarray) -> None:
        nonlocal last_vis, button_regions, video_panel_height
        last_vis = vis
        display, button_regions, video_panel_height = compose_track_display(
            vis,
            hud_lines(),
            playback_mode,
            map_view.view_mode if mesh_builder is not None else None,
        )
        cv2.imshow(TRACK_WINDOW, display)

    def on_track_mouse(event: int, x: int, y: int, _flags: int, _param) -> None:
        nonlocal playback_mode
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        choice = transport_hit(x, y, video_panel_height, button_regions)
        if choice == VIEW_TOGGLE:
            map_view.toggle_view_mode()
            if last_vis is not None:
                show_vis(last_vis)
        elif choice is not None:
            playback_mode = choice
            if last_vis is not None:
                show_vis(last_vis)

    def export_mesh(base: Path) -> None:
        if mesh_builder is None:
            return
        mesh_builder.purge(0.0, force=True)
        vertices, colors, faces = mesh_builder.indexed_mesh()
        if len(faces) == 0:
            print("[mesh] nothing to export yet (no faces passed validation)")
            return
        base = base.expanduser().resolve()
        base.parent.mkdir(parents=True, exist_ok=True)
        write_ply(base.with_suffix(".ply"), vertices, colors, faces)
        write_stl(base.with_suffix(".stl"), vertices, faces)
        print(f"[mesh] exported {len(faces)} faces / {len(vertices)} vertices to {base}.ply / .stl")

    deviation_meter = (
        None
        if args.no_deviation_window
        else DeviationMeter((width, height), session.intrinsic, "vs the SLAM's estimated motion")
    )
    last_sample: MotionSample | None = None

    def handle_frame(frame: np.ndarray) -> None:
        nonlocal frame_index, last_sample
        frame_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
        session.process(frame)
        map_view.camera_rotation = session.world_rotation
        if session.processed % args.plot_every == 0:
            map_view.show(trajectory, map_store)
        show_vis(draw_tracks(frame, session.track_points, session.exclude_mask))
        if deviation_meter is not None:
            sample = session.motion_sample
            if sample is not None and sample is not last_sample:
                last_sample = sample
                deviation_meter.add(sample)
            cv2.imshow(DEVIATION_WINDOW, deviation_meter.render(frame, DEVIATION_WIDTH))

    print(f"Running SLAM on {video_path}")
    if args.max_seconds > 0:
        print(f"Video stops after {args.max_seconds:g}s (--max-seconds 0 for full video)")
    print(
        "Video: Rewind / Pause / Play buttons, space = pause/play, q = quit, "
        "m = toggle point/mesh view (or the View button), e = export mesh to 3D-SLAM/exports/"
    )
    print(
        "Map window: LMB look around, MMB orbit, RMB pan, W/A/S/D fly, Space up, Ctrl down, "
        "Shift = 3x speed, scroll forward/back, Reset button top-left"
    )
    print(
        f"Map: up to {args.max_features} tracks, live cap {args.max_live_points}, "
        f"archive flush to {cache_dir or 'RAM trim only'}"
    )

    cv2.setMouseCallback(TRACK_WINDOW, on_track_mouse)
    show_vis(draw_tracks(frame, session.track_points, session.exclude_mask))

    try:
        while True:
            if not map_view.poll():
                break

            if not video_active:
                map_view.draw()
                continue

            delay = play_delay_ms if playback_mode != PlaybackMode.PAUSED else 1
            key = cv2.waitKey(delay) & 0xFF
            if key == ord("q"):
                break
            if key == ord("m") and mesh_builder is not None:
                map_view.toggle_view_mode()
                if last_vis is not None:
                    show_vis(last_vis)
                continue
            if key == ord("e"):
                export_mesh(DEFAULT_EXPORT_DIR / datetime.now().strftime("mesh-%Y%m%d-%H%M%S"))
                continue
            if key == ord(" "):
                playback_mode = (
                    PlaybackMode.FORWARD
                    if playback_mode == PlaybackMode.PAUSED
                    else PlaybackMode.PAUSED
                )
                if last_vis is not None:
                    show_vis(last_vis)
                continue

            if playback_mode == PlaybackMode.PAUSED:
                if last_vis is not None:
                    show_vis(last_vis)
                continue

            if playback_mode == PlaybackMode.FORWARD:
                ok, frame = read_frame_forward()
                if not ok:
                    print("End of video.")
                    stop_video_playback()
                    continue
                frame_index += 1
                if args.max_seconds > 0:
                    t_sec = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    if t_sec > args.max_seconds:
                        print(f"Reached --max-seconds {args.max_seconds:g}.")
                        stop_video_playback()
                        continue
            else:
                ok, frame = read_frame_backward()
                if not ok:
                    playback_mode = PlaybackMode.PAUSED
                    if last_vis is not None:
                        show_vis(last_vis)
                    continue
                frame_index = max(0, frame_index - 1)

            if frame_index % args.stride != 0:
                continue

            handle_frame(frame)
    finally:
        if capture.isOpened():
            capture.release()
        cv2.destroyAllWindows()
        if not map_finalized:
            map_store.finalize()
        if args.export_mesh is not None:
            export_mesh(args.export_mesh)
        if mesh_builder is not None:
            mesh_builder.purge(0.0, force=True)
        print(
            f"[map] {len(map_store)} landmarks | settled anchors "
            f"{len(saturation_map) if saturation_map is not None else 0}, "
            f"thinned {stats['thinned']}, redundant skipped {stats['redundant']} | "
            f"isolated pruned {stats['pruned']} | loop fixes {stats['fixes']} | "
            f"mesh faces {len(mesh_builder) if mesh_builder is not None else 0} | "
            f"churned {stats['churned']} in {stats['churns']} rounds"
        )
        map_view.show(trajectory, map_store)
        map_view.close()


if __name__ == "__main__":
    main()
