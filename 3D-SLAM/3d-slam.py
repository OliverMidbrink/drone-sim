#!/usr/bin/env python3
"""Monocular video SLAM: sparse map + camera trajectory (metric scale arbitrary)."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from map_renderer import MAP_WINDOW, MapRenderer
from map_store import MapStore

DEFAULT_VIDEO = (
    Path(__file__).resolve().parent / "video-samples" / "forest-walking.mp4"
)
DEFAULT_MAP_CACHE = Path(__file__).resolve().parent / ".map-cache"

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

TRACK_WINDOW = "SLAM tracks"


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


def detect_points(
    gray: np.ndarray,
    existing: np.ndarray | None,
    max_corners: int,
) -> np.ndarray:
    mask = np.full(gray.shape, 255, np.uint8)
    if existing is not None and len(existing):
        for x, y in existing.reshape(-1, 2).astype(int):
            cv2.circle(mask, (int(x), int(y)), 5, 0, -1)
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=0.008,
        minDistance=4,
        mask=mask,
    )
    if corners is None:
        return np.empty((0, 1, 2), dtype=np.float32)
    return corners.astype(np.float32)


def merge_points(tracked: np.ndarray, fresh: np.ndarray) -> np.ndarray:
    if len(tracked) == 0:
        return fresh
    if len(fresh) == 0:
        return tracked
    return np.vstack([tracked, fresh])


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
        default=2000,
        help="target tracked features per frame (default: 2000)",
    )
    parser.add_argument(
        "--max-live-points",
        type=int,
        default=50_000,
        help="max landmarks kept in active RAM (default: 50000)",
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
    track_points = detect_points(gray, None, args.max_features)

    world_rotation = np.eye(3, dtype=np.float64)
    world_position = np.zeros(3, dtype=np.float64)
    trajectory = [world_position.copy()]

    cache_dir = None if args.no_disk_cache else args.map_cache.expanduser().resolve()
    map_store = MapStore(
        cache_dir=cache_dir,
        ram_archive_limit=args.ram_archive_limit,
        max_live_points=args.max_live_points,
    )

    map_height = max(320, height // 2)
    map_view = MapRenderer(width, map_height)
    map_view.setup_window()
    layout_windows(width, height, map_height)
    map_view.show(trajectory, map_store)

    frame_index = 0
    processed = 0
    paused = False
    video_active = True
    map_finalized = False

    print(f"Running SLAM on {video_path}")
    if args.max_seconds > 0:
        print(f"Video stops after {args.max_seconds:g}s (--max-seconds 0 for full video)")
    print("Video: q = quit, space = pause/resume")
    print(f"{MAP_WINDOW}: LMB orbit, RMB pan, scroll dolly, Reset button top-left")
    print(
        f"Map: up to {args.max_features} tracks, live cap {args.max_live_points}, "
        f"archive flush to {cache_dir or 'RAM trim only'}"
    )

    try:
        while True:
            if not map_view.poll():
                break

            if not video_active:
                map_view.draw()
                continue

            if not paused:
                ok, frame = capture.read()
                if not ok:
                    print("End of video.")
                    video_active = False
                elif args.max_seconds > 0:
                    t_sec = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    if t_sec > args.max_seconds:
                        print(f"Reached --max-seconds {args.max_seconds:g}.")
                        video_active = False

                if not video_active:
                    capture.release()
                    cv2.destroyAllWindows()
                    map_store.finalize()
                    map_finalized = True
                    map_view.show(trajectory, map_store)
                    print(
                        "Video stopped. Explore the 3D map "
                        "(close the map window when finished)."
                    )
                    continue

                frame_index += 1
                if frame_index % args.stride != 0:
                    continue

                if args.scale != 1.0:
                    frame = cv2.resize(
                        frame,
                        None,
                        fx=args.scale,
                        fy=args.scale,
                        interpolation=cv2.INTER_AREA,
                    )

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if len(track_points) < 8:
                    track_points = detect_points(gray, None, args.max_features)
                    prev_gray = gray
                    processed += 1
                    cv2.imshow(TRACK_WINDOW, draw_tracks(frame, track_points))
                    if processed % args.plot_every == 0:
                        map_view.show(trajectory, map_store)
                    continue

                next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray,
                    gray,
                    track_points,
                    None,
                    **LK_PARAMS,
                )
                status = status.reshape(-1)
                good_previous = track_points[status == 1]
                good_current = next_points[status == 1]

                motion = estimate_motion(good_previous, good_current, intrinsic)
                if motion is not None:
                    rotation, translation, inliers = motion
                    previous_inlier = good_previous[inliers].reshape(-1, 2)
                    current_inlier = good_current[inliers].reshape(-1, 2)

                    points_cam = triangulate_pair(
                        intrinsic,
                        rotation,
                        translation,
                        previous_inlier,
                        current_inlier,
                    )
                    valid = points_cam[:, 2] > 0.1
                    points_cam = points_cam[valid]
                    current_inlier = current_inlier[valid]

                    scale = triangulation_scale(points_cam)
                    translation_scaled = translation.reshape(3) * scale
                    points_cam *= scale

                    if len(points_cam) > 0:
                        world_points = transform_points_to_world(
                            points_cam,
                            world_rotation,
                            world_position,
                        )
                        colors = sample_colors(frame, current_inlier)
                        map_store.add(world_points, colors)

                    # recoverPose: p_cam2 = R @ p_cam1 + t; camera center in cam1 is -R.T @ t
                    camera_delta = -(rotation.T @ translation_scaled)
                    world_position = world_position + world_rotation @ camera_delta
                    world_rotation = world_rotation @ rotation.T
                    trajectory.append(world_position.copy())

                    map_store.archive_out_of_walker_view(
                        world_rotation,
                        world_position,
                        intrinsic,
                        width,
                        height,
                    )

                track_points = good_current.reshape(-1, 1, 2)
                if len(track_points) < args.max_features // 2:
                    fresh = detect_points(
                        gray,
                        track_points.reshape(-1, 2),
                        args.max_features - len(track_points),
                    )
                    track_points = merge_points(track_points, fresh)

                prev_gray = gray
                processed += 1

                vis = draw_tracks(frame, track_points)
                cv2.putText(
                    vis,
                    (
                        f"poses: {len(trajectory)}  "
                        f"live: {map_store.live_count}  "
                        f"stored: {map_store.archive_count}  "
                        f"total: {len(map_store)}"
                    ),
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow(TRACK_WINDOW, vis)

                if processed % args.plot_every == 0:
                    map_view.show(trajectory, map_store)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused
    finally:
        if capture.isOpened():
            capture.release()
        cv2.destroyAllWindows()
        if not map_finalized:
            map_store.finalize()
        map_view.show(trajectory, map_store)
        map_view.close()


if __name__ == "__main__":
    main()
