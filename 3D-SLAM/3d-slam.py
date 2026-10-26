#!/usr/bin/env python3
"""Monocular video SLAM: sparse map + camera trajectory (metric scale arbitrary)."""

import argparse
from enum import IntEnum
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from anchor_map import (
    MIN_AGE_STEPS,
    NEW_VIEW_DEG,
    SAME_POINT_TOL,
    AnchorMap,
    correction_between,
    keypoint_depths,
)
from map_renderer import MAP_WINDOW, MapRenderer
from map_store import MapStore
from mesh_builder import MeshBuilder, write_ply, write_stl
from saturation import SaturationMap

DEFAULT_VIDEO = (
    Path(__file__).resolve().parent / "video-samples" / "forest-walking.mp4"
)
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
    free = np.full((height, width), 255, np.uint8)
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
ANCHOR_INTERVAL = 3  # look for / add anchors every Nth committed pose update
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


def draw_tracks(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    vis = image.copy()
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


def layout_windows(video_width: int, video_height: int, map_height: int) -> None:
    margin_x = 40
    margin_y = 40
    title_bar = 32
    cv2.namedWindow(TRACK_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(TRACK_WINDOW, video_width, video_height)
    cv2.moveWindow(TRACK_WINDOW, margin_x, margin_y + map_height + title_bar)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "video",
        nargs="?",
        type=Path,
        default=DEFAULT_VIDEO,
        help=f"path to input video (default: {DEFAULT_VIDEO})",
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
        "--plot-every",
        type=int,
        default=3,
        help="refresh 3D map every N processed frames (default: 3)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    if not video_path.is_file():
        raise SystemExit(f"Video not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    ok, frame = capture.read()
    if not ok:
        raise SystemExit("Video has no readable frames")

    if args.scale != 1.0:
        frame = cv2.resize(
            frame,
            None,
            fx=args.scale,
            fy=args.scale,
            interpolation=cv2.INTER_AREA,
        )

    height, width = frame.shape[:2]
    intrinsic = camera_matrix(width, height, args.fov_deg)
    fx = intrinsic[0, 0]
    print(f"Camera intrinsics: horizontal FOV {args.fov_deg:g}° -> fx={fx:.1f}px")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = gray
    track_points = detect_balanced(gray, None, args.max_features)
    track_ids = np.arange(len(track_points), dtype=np.int64)
    next_track_id = len(track_points)

    world_rotation = np.eye(3, dtype=np.float64)
    world_position = np.zeros(3, dtype=np.float64)
    trajectory = [world_position.copy()]
    last_step_length: float | None = None

    # Keyframe: the tracked points (sorted by id) at the last pose update, plus each
    # track's *first* observation and the pose it was seen from, so depth can be
    # triangulated over the longest baseline available.
    key_ids = np.empty(0, dtype=np.int64)
    key_pts = np.empty((0, 2), dtype=np.float64)
    key_birth_px = np.empty((0, 2), dtype=np.float64)
    key_birth_R = np.empty((0, 3, 3), dtype=np.float64)
    key_birth_C = np.empty((0, 3), dtype=np.float64)
    key_gray: np.ndarray | None = None

    def set_keyframe(points: np.ndarray, ids: np.ndarray, image: np.ndarray) -> None:
        nonlocal key_ids, key_pts, key_birth_px, key_birth_R, key_birth_C, key_gray
        key_gray = image
        order = np.argsort(ids)
        new_ids = ids[order]
        new_pts = points.reshape(-1, 2)[order].astype(np.float64)
        count = len(new_ids)

        birth_px = new_pts.copy()
        birth_R = np.broadcast_to(world_rotation, (count, 3, 3)).copy()
        birth_C = np.broadcast_to(world_position, (count, 3)).copy()
        if len(key_ids) > 0 and count > 0:
            old = np.minimum(np.searchsorted(key_ids, new_ids), len(key_ids) - 1)
            known = key_ids[old] == new_ids
            birth_px[known] = key_birth_px[old[known]]
            birth_R[known] = key_birth_R[old[known]]
            birth_C[known] = key_birth_C[old[known]]

        key_ids, key_pts = new_ids, new_pts.astype(np.float32)
        key_birth_px, key_birth_R, key_birth_C = birth_px, birth_R, birth_C

    set_keyframe(track_points, track_ids, gray)

    cache_dir = None if args.no_disk_cache else args.map_cache.expanduser().resolve()
    map_store = MapStore(
        cache_dir=cache_dir,
        ram_archive_limit=args.ram_archive_limit,
        max_live_points=args.max_live_points,
    )

    mesh_builder = None if args.no_mesh else MeshBuilder(map_store)
    anchors = None if args.no_anchors else AnchorMap(max_anchors=args.max_anchors)
    stats = {
        "commits": 0, "fixes": 0, "associated": 0, "pruned": 0, "thinned": 0, "redundant": 0,
        "churned": 0, "churns": 0, "mesh_tris": 0,
    }
    churn_state = {"commits": 0, "inliers": 0}

    prune_state = {"last": 0.0, "saturate": 0.0}
    saturation_map = None if args.no_saturation else SaturationMap()
    cell_targets = np.full(
        (TRACK_GRID_ROWS, TRACK_GRID_COLS),
        float(per_cell_target(args.max_features)),
    )

    map_height = max(320, height // 2)
    map_view = MapRenderer(width, map_height)
    map_view.setup_window()
    map_view.set_mesh(mesh_builder)
    if args.start_mesh_view and mesh_builder is not None:
        map_view.view_mode = "mesh"
    layout_windows(width, height, map_height)
    map_view.show(trajectory, map_store)

    frame_index = 0
    processed = 0
    playback_mode = PlaybackMode.FORWARD
    last_vis: np.ndarray | None = None
    video_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
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
                f"tracks: {len(track_points)}  poses: {len(trajectory)}  "
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
        return True, resize_frame(raw, args.scale)

    def read_frame_backward() -> tuple[bool, np.ndarray | None]:
        pos = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
        if pos <= 1:
            return False, None
        capture.set(cv2.CAP_PROP_POS_FRAMES, pos - 2)
        ok, raw = capture.read()
        if not ok:
            return False, None
        return True, resize_frame(raw, args.scale)

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

    def update_cell_targets(pixels: np.ndarray, settled: np.ndarray) -> None:
        """Shift the tracking budget from image cells that show finished map to the rest."""
        nonlocal cell_targets
        cx = np.clip((pixels[:, 0] * TRACK_GRID_COLS / width).astype(int), 0, TRACK_GRID_COLS - 1)
        cy = np.clip((pixels[:, 1] * TRACK_GRID_ROWS / height).astype(int), 0, TRACK_GRID_ROWS - 1)
        total = np.zeros((TRACK_GRID_ROWS, TRACK_GRID_COLS))
        done = np.zeros_like(total)
        np.add.at(total, (cy, cx), 1.0)
        np.add.at(done, (cy, cx), settled.astype(float))
        fraction = np.where(total >= MIN_CELL_OBSERVATIONS, done / np.maximum(total, 1.0), 0.0)
        wanted = allocate_cell_targets(args.max_features, fraction)
        cell_targets = 0.6 * cell_targets + 0.4 * wanted

    def video_clock() -> float:
        """Seconds of video processed so far (the age of landmarks is measured in this)."""
        return processed * max(args.stride, 1) / max(video_fps, 1e-6)

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

    def run_anchor_step(gray: np.ndarray, track_ids_now: np.ndarray, track_px: np.ndarray) -> None:
        """Relocalise against old anchors (closing loops), then add new anchors."""
        nonlocal world_rotation, world_position, key_ids
        keypoints, descriptors, response = anchors.detect(gray)
        if len(keypoints) == 0:
            return
        step = len(trajectory) - 1
        query, matched = anchors.match(descriptors, world_position)

        if len(query):
            old = anchors.steps[matched] <= step - MIN_AGE_STEPS
            fix = anchors.localize(keypoints[query[old]], matched[old], intrinsic)
            if fix is not None:
                start = anchors.last_fix_step
                correction = correction_between(
                    world_rotation, world_position, fix.rotation, fix.position
                )
                shift, turn = correction.magnitude
                path = np.asarray(trajectory[start:])
                path_len = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()) if len(path) > 1 else 0.0
                plausible = turn < 30.0 and shift < 0.6 * path_len + 5.0 * (last_step_length or 0.0)
                if plausible:
                    poses = np.asarray(trajectory)
                    fraction = np.clip(
                        (np.arange(len(poses)) - start) / max(step - start, 1), 0.0, 1.0
                    )
                    moving = fraction > 0.0
                    poses[moving] = correction.apply(poses[moving], fraction[moving])
                    trajectory[:] = [row.copy() for row in poses]
                    map_store.warp(correction, start, step)
                    if mesh_builder is not None:
                        mesh_builder.version += 1  # corners moved: rebuild the view
                    anchors.warp(correction, start, step)
                    anchors.last_fix_step = step
                    if saturation_map is not None:
                        saturation_map.mark_dirty()
                    world_rotation, world_position = fix.rotation, fix.position.copy()
                    key_ids = np.empty(0, dtype=np.int64)  # births were in the old frame
                    stats["fixes"] += 1
                    print(
                        f"[anchors] fix @ step {step}: {fix.inliers}/{fix.matches} inliers, "
                        f"shift {shift:.3f} ({shift / max(path_len, 1e-9) * 100:.0f}% of path), "
                        f"turn {turn:.2f}°"
                    )

        # New anchors: keypoints with a trustworthy depth from nearby tracked landmarks.
        found, landmarks = map_store.lookup(track_ids_now)
        if found.sum() < 20:
            return
        weights = map_store.lookup_weights(track_ids_now)[found]
        landmarks = landmarks[weights >= MIN_ANCHOR_LANDMARK_WEIGHT]
        pixels = track_px[found][weights >= MIN_ANCHOR_LANDMARK_WEIGHT]
        camera = (landmarks - world_position) @ world_rotation
        ahead = camera[:, 2] > 0.2
        depth = keypoint_depths(keypoints, pixels[ahead], camera[ahead, 2], (height, width))
        valid = np.isfinite(depth)
        if not valid.any():
            return

        normalized = np.column_stack([keypoints, np.ones(len(keypoints))]) @ np.linalg.inv(intrinsic).T
        points = (world_rotation @ (normalized * depth[:, None]).T).T + world_position
        to_camera = world_position - points
        to_camera /= np.maximum(np.linalg.norm(to_camera, axis=1, keepdims=True), 1e-9)

        wanted = valid.copy()
        if len(query):
            gap = np.linalg.norm(points[query] - anchors.positions[matched], axis=1)
            same = gap < SAME_POINT_TOL * np.nan_to_num(depth[query], nan=1e9)
            cos_view = np.einsum("ij,ij->i", to_camera[query], anchors.views[matched])
            fresh_view = cos_view < np.cos(np.radians(NEW_VIEW_DEG))
            wanted[query] = valid[query] & same & fresh_view  # known point: only new views
        candidates = np.flatnonzero(wanted)
        if len(candidates) > MAX_NEW_ANCHORS:
            candidates = candidates[np.argsort(-response[candidates])[:MAX_NEW_ANCHORS]]
        anchors.add(descriptors[candidates], points[candidates], to_camera[candidates], step)

    def process_slam_frame(frame: np.ndarray, gray: np.ndarray) -> np.ndarray:
        nonlocal prev_gray, track_points, track_ids, next_track_id, processed
        nonlocal world_rotation, world_position, trajectory, last_step_length

        if len(track_points) < 8:
            track_points = detect_balanced(gray, None, args.max_features)
            track_ids = np.arange(
                next_track_id, next_track_id + len(track_points), dtype=np.int64
            )
            next_track_id += len(track_points)
            set_keyframe(track_points, track_ids, gray)
            prev_gray = gray
            processed += 1
            if processed % args.plot_every == 0:
                map_view.show(trajectory, map_store)
            return draw_tracks(frame, track_points)

        next_points, keep = track_features(
            prev_gray, gray, track_points, args.fb_check
        )
        good_current = next_points[keep]
        good_ids = track_ids[keep]

        # Estimate motion against the keyframe (not the previous frame) so there is
        # enough parallax; nothing is committed until the camera has actually moved.
        rekey = False
        committed = False
        pos = np.minimum(np.searchsorted(key_ids, good_ids), len(key_ids) - 1)
        matched = key_ids[pos] == good_ids
        if matched.sum() < KEYFRAME_MIN_MATCHES:
            rekey = True  # keyframe tracks lost: re-anchor, pose unchanged
        else:
            key_matched = key_pts[pos[matched]]
            current_matched = good_current[matched].reshape(-1, 2)
            ids_matched = good_ids[matched]
            parallax = float(
                np.median(np.linalg.norm(current_matched - key_matched, axis=1))
            )
            if parallax >= KEYFRAME_MIN_PARALLAX_PX:
                motion = estimate_motion(key_matched, current_matched, intrinsic)
                if motion is None:
                    rekey = parallax > 4.0 * KEYFRAME_MIN_PARALLAX_PX
                else:
                    rotation, translation, inliers = motion
                    key_inlier = key_matched[inliers]
                    current_inlier = current_matched[inliers]
                    inlier_ids = ids_matched[inliers]
                    inlier_rows = pos[matched][inliers]

                    # Step length: the essential matrix only gives direction, so anchor
                    # the scale to landmarks already in the map. This keeps the scale
                    # consistent from step to step instead of resetting it every time.
                    unit_points = triangulate_pair(
                        intrinsic, rotation, translation, key_inlier, current_inlier
                    )
                    scale = None
                    map_store.reactivate_in_view(
                        world_rotation, world_position, intrinsic, width, height
                    )
                    found, landmarks = map_store.lookup(inlier_ids)
                    if found.sum() >= MIN_SCALE_MATCHES:
                        depth_unit = unit_points[found, 2]
                        depth_map = ((landmarks - world_position) @ world_rotation)[:, 2]
                        usable = (depth_unit > 1e-3) & (depth_map > 1e-3)
                        if usable.sum() >= MIN_SCALE_MATCHES:
                            scale = float(np.median(depth_map[usable] / depth_unit[usable]))
                    if scale is None:
                        scale = (
                            last_step_length
                            if last_step_length is not None
                            else triangulation_scale(unit_points)
                        )
                    if last_step_length is not None:
                        scale = float(
                            np.clip(scale, last_step_length / 4.0, last_step_length * 4.0)
                        )

                    camera_delta = -(rotation.T @ (translation.reshape(3) * scale))
                    world_position = world_position + world_rotation @ camera_delta
                    world_rotation = world_rotation @ rotation.T
                    last_step_length = float(np.linalg.norm(camera_delta))
                    trajectory.append(world_position.copy())

                    # Triangulate from each track's first sighting to now: the baseline
                    # keeps growing as the camera moves, so distant and straight-ahead
                    # points sharpen over time instead of staying noisy.
                    rays_birth = pixel_rays(
                        key_birth_px[inlier_rows], intrinsic, key_birth_R[inlier_rows]
                    )
                    rays_now = pixel_rays(current_inlier, intrinsic, world_rotation)
                    points_world, angle_deg, usable = triangulate_midpoint(
                        key_birth_C[inlier_rows],
                        rays_birth,
                        np.broadcast_to(world_position, (len(rays_now), 3)),
                        rays_now,
                    )
                    step = len(trajectory) - 1
                    if usable.any():
                        weights = np.clip(angle_deg[usable] ** 2, 0.01, MAX_OBSERVATION_WEIGHT)
                        colors = sample_colors(frame, current_inlier[usable])
                        obs_ids = inlier_ids[usable]
                        obs_points = points_world[usable]
                        obs_pixels = current_inlier[usable]

                        # Tracks that are new but sit on a landmark we already have
                        # (e.g. returning to an area) adopt that landmark's id, so
                        # their observations refine it rather than duplicating it.
                        known, _ = map_store.lookup(obs_ids)
                        fresh = np.flatnonzero(~known)
                        if len(fresh):
                            hit, target = map_store.find_existing(
                                obs_points[fresh],
                                obs_pixels[fresh],
                                world_rotation,
                                world_position,
                                intrinsic,
                                width,
                                height,
                                exclude_ids=good_ids,
                            )
                            if hit.any():
                                source = obs_ids[fresh][hit]
                                obs_ids = remap_ids(obs_ids, source, target[hit])
                                inlier_ids = remap_ids(inlier_ids, source, target[hit])
                                good_ids = remap_ids(good_ids, source, target[hit])
                                stats["associated"] += int(hit.sum())
                        if saturation_map is not None and len(obs_points):
                            distance = np.linalg.norm(obs_points - world_position, axis=1)
                            settled = saturation_map.settled(map_store, obs_points, distance)
                            update_cell_targets(obs_pixels, settled)
                            known_now, _ = map_store.lookup(obs_ids)
                            redundant = settled & ~known_now  # complete region: nothing new to add
                            if redundant.any():
                                stats["redundant"] += int(redundant.sum())
                                fresh_only = ~redundant
                                obs_points, colors = obs_points[fresh_only], colors[fresh_only]
                                obs_ids, weights = obs_ids[fresh_only], weights[fresh_only]
                        map_store.add(obs_points, colors, obs_ids, weights, step, now=video_clock())

                    stats["commits"] += 1
                    churn_state["commits"] += 1
                    churn_state["inliers"] = len(inlier_ids)
                    committed = True
                    if mesh_builder is not None:
                        found_m, pos_m, weight_m = map_store.lookup_all(inlier_ids)
                        if found_m.sum() >= 12:
                            stats["mesh_tris"] = mesh_builder.update(
                                rotation=world_rotation,
                                position=world_position,
                                ids=inlier_ids[found_m],
                                pixels=current_inlier[found_m],
                                points=pos_m,
                                weights=weight_m,
                                gray=gray,
                                key_gray=key_gray,
                                key_ids=key_ids,
                                key_pts=key_pts,
                            )
                        mesh_builder.purge(video_clock())
                    if anchors is not None and stats["commits"] % ANCHOR_INTERVAL == 0:
                        run_anchor_step(gray, inlier_ids, current_inlier)

                    clock = video_clock()
                    if args.prune_age > 0 and clock - prune_state["last"] >= PRUNE_INTERVAL_SECONDS:
                        prune_state["last"] = clock
                        stats["pruned"] += map_store.prune_isolated(
                            clock,
                            np.asarray(trajectory),
                            args.prune_age,
                            args.prune_isolation,
                            check_window=2.0 * PRUNE_INTERVAL_SECONDS,
                            exempt_ids=None if saturation_map is None else saturation_map.anchor_ids,
                        )
                    if (
                        saturation_map is not None
                        and clock - prune_state["saturate"] >= SATURATION_INTERVAL_SECONDS
                    ):
                        prune_state["saturate"] = clock
                        _, thinned = saturation_map.consolidate(
                            map_store, clock, np.asarray(trajectory)
                        )
                        stats["thinned"] += thinned

                    map_store.archive_out_of_walker_view(
                        world_rotation,
                        world_position,
                        intrinsic,
                        width,
                        height,
                    )
                    rekey = True

        track_points = good_current.reshape(-1, 1, 2)
        track_ids = good_ids

        # Thin out crowded cells, then refill every sparse cell (including newly
        # revealed edges) so the whole image stays tracked in either direction.
        flat_tracks, track_ids = prune_crowded(
            track_points, track_ids, width, height, args.max_features, cell_targets
        )
        track_points = flat_tracks.reshape(-1, 1, 2)

        # Churn: with the pose firmly held, retire the oldest mature tracks and re-seed
        # elsewhere (keeping clear of where they were) so new ground gets covered.
        retired_px = None
        if (
            committed
            and not args.no_churn
            and churn_state["commits"] >= CHURN_EVERY_COMMITS
            and churn_state["inliers"] >= CHURN_MIN_INLIERS
        ):
            weight_now = map_store.lookup_weights(track_ids)
            mature = np.flatnonzero(weight_now >= CHURN_MIN_WEIGHT)
            n_retire = min(len(mature), int(CHURN_FRACTION * len(track_ids)))
            if n_retire >= 20:
                retire = mature[np.argsort(track_ids[mature])[:n_retire]]
                keep_track = np.ones(len(track_ids), dtype=bool)
                keep_track[retire] = False
                retired_px = track_points.reshape(-1, 2)[retire]
                track_points, track_ids = track_points[keep_track], track_ids[keep_track]
                stats["churned"] += n_retire
                stats["churns"] += 1
                churn_state["commits"] = 0
        fresh = detect_balanced(
            gray,
            track_points,
            args.max_features,
            cell_targets,
            avoid=retired_px,
            refill_fraction=1.0 if retired_px is not None else TRACK_REFILL_FRACTION,
        )
        if len(fresh):
            fresh_ids = np.arange(
                next_track_id, next_track_id + len(fresh), dtype=np.int64
            )
            next_track_id += len(fresh)
            track_points = np.vstack([track_points, fresh])
            track_ids = np.concatenate([track_ids, fresh_ids])

        if rekey:
            set_keyframe(track_points, track_ids, gray)

        prev_gray = gray
        processed += 1
        if processed % args.plot_every == 0:
            map_view.show(trajectory, map_store)
        return draw_tracks(frame, track_points)

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

    def handle_frame(frame: np.ndarray) -> None:
        nonlocal frame_index
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
        show_vis(process_slam_frame(frame, gray))

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
    show_vis(draw_tracks(frame, track_points))

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
