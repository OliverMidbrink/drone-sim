#!/usr/bin/env python3
"""Monocular video SLAM: sparse map + camera trajectory (metric scale arbitrary)."""

import argparse
from enum import IntEnum
from pathlib import Path

import cv2
import numpy as np

from map_renderer import MAP_WINDOW, MapRenderer

DEFAULT_VIDEO = (
    Path(__file__).resolve().parent / "video-samples" / "forest-walking.mp4"
)

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


def draw_transport_bar(
    width: int,
    mode: PlaybackMode,
) -> tuple[np.ndarray, list[tuple[PlaybackMode, int, int, int, int]]]:
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
    regions: list[tuple[PlaybackMode, int, int, int, int]] = []

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

    return bar, regions


def compose_track_display(
    vis: np.ndarray,
    hud_lines: list[str],
    mode: PlaybackMode,
) -> tuple[np.ndarray, list[tuple[PlaybackMode, int, int, int, int]], int]:
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
    bar, regions = draw_transport_bar(overlay.shape[1], mode)
    return np.vstack([overlay, bar]), regions, overlay.shape[0]


def transport_hit(
    x: int,
    y: int,
    video_height: int,
    regions: list[tuple[PlaybackMode, int, int, int, int]],
) -> PlaybackMode | None:
    if y < video_height:
        return None
    y_in_bar = y - video_height
    for play_mode, x0, y0, x1, y1 in regions:
        if x0 <= x <= x1 and y0 <= y_in_bar <= y1:
            return play_mode
    return None


def camera_matrix(width: int, height: int) -> np.ndarray:
    focal = 0.9 * width
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
            cv2.circle(mask, (int(x), int(y)), 7, 0, -1)
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=0.01,
        minDistance=7,
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
    colors = []
    for x, y in pixels:
        xi = int(np.clip(round(x), 0, width - 1))
        yi = int(np.clip(round(y), 0, height - 1))
        bgr = image[yi, xi]
        colors.append((bgr[2], bgr[1], bgr[0]))
    return np.array(colors, dtype=np.float64) / 255.0


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
        "--stride",
        type=int,
        default=1,
        help="use every Nth frame (default: 1)",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=700,
        help="target number of tracked features (default: 700)",
    )
    parser.add_argument(
        "--max-map-points",
        type=int,
        default=25000,
        help="cap on stored map points (default: 25000)",
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
    intrinsic = camera_matrix(width, height)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = gray
    track_points = detect_points(gray, None, args.max_features)

    world_rotation = np.eye(3, dtype=np.float64)
    world_position = np.zeros(3, dtype=np.float64)
    trajectory = [world_position.copy()]

    map_points = np.empty((0, 3), dtype=np.float64)
    map_colors = np.empty((0, 3), dtype=np.float64)

    map_height = max(320, height // 2)
    map_view = MapRenderer(width, map_height)
    map_view.setup_window()
    layout_windows(width, height, map_height)
    map_view.show(trajectory, map_points, map_colors)

    frame_index = 0
    processed = 0
    playback_mode = PlaybackMode.FORWARD
    last_vis: np.ndarray | None = None
    video_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    play_delay_ms = max(1, int(round(1000.0 / video_fps)))
    button_regions: list[tuple[PlaybackMode, int, int, int, int]] = []
    video_panel_height = height

    def hud_lines() -> list[str]:
        t_sec = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        mode_label = playback_mode.name.lower()
        return [
            f"poses: {len(trajectory)}  map: {len(map_points)}",
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

    def process_slam_frame(frame: np.ndarray, gray: np.ndarray) -> np.ndarray:
        nonlocal prev_gray, track_points, processed
        nonlocal world_rotation, world_position, trajectory, map_points, map_colors

        if len(track_points) < 8:
            track_points = detect_points(gray, None, args.max_features)
            prev_gray = gray
            processed += 1
            if processed % args.plot_every == 0:
                map_view.show(trajectory, map_points, map_colors)
            return draw_tracks(frame, track_points)

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
            translation = translation.reshape(3) * scale
            points_cam *= scale

            if len(points_cam) > 0:
                world_points = transform_points_to_world(
                    points_cam,
                    world_rotation,
                    world_position,
                )
                colors = sample_colors(frame, current_inlier)
                map_points = np.vstack([map_points, world_points])
                map_colors = np.vstack([map_colors, colors])

                if len(map_points) > args.max_map_points:
                    keep = np.random.choice(
                        len(map_points),
                        args.max_map_points,
                        replace=False,
                    )
                    map_points = map_points[keep]
                    map_colors = map_colors[keep]

            world_position = world_position + world_rotation @ translation
            world_rotation = world_rotation @ rotation
            trajectory.append(world_position.copy())

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
        if processed % args.plot_every == 0:
            map_view.show(trajectory, map_points, map_colors)
        return draw_tracks(frame, track_points)

    def show_vis(vis: np.ndarray) -> None:
        nonlocal last_vis, button_regions, video_panel_height
        last_vis = vis
        display, button_regions, video_panel_height = compose_track_display(
            vis,
            hud_lines(),
            playback_mode,
        )
        cv2.imshow(TRACK_WINDOW, display)

    def on_track_mouse(event: int, x: int, y: int, _flags: int, _param) -> None:
        nonlocal playback_mode
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        choice = transport_hit(x, y, video_panel_height, button_regions)
        if choice is not None:
            playback_mode = choice
            if last_vis is not None:
                show_vis(last_vis)

    def handle_frame(frame: np.ndarray) -> None:
        nonlocal frame_index
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
        show_vis(process_slam_frame(frame, gray))

    print(f"Running SLAM on {video_path}")
    print("Video: Rewind / Pause / Play buttons, space = pause, q = quit")
    print(f"{MAP_WINDOW}: LMB orbit, RMB pan, scroll dolly, Reset button top-left")

    cv2.setMouseCallback(TRACK_WINDOW, on_track_mouse)
    show_vis(draw_tracks(frame, track_points))

    while True:
        if not map_view.poll():
            break

        delay = play_delay_ms if playback_mode != PlaybackMode.PAUSED else 1
        key = cv2.waitKey(delay) & 0xFF
        if key == ord("q"):
            break
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
                playback_mode = PlaybackMode.PAUSED
                continue
            frame_index += 1
        else:
            ok, frame = read_frame_backward()
            if not ok:
                playback_mode = PlaybackMode.PAUSED
                continue
            frame_index = max(0, frame_index - 1)

        if frame_index % args.stride != 0:
            continue

        handle_frame(frame)

    capture.release()
    map_view.show(trajectory, map_points, map_colors)
    print("Finished. Press a key in a video window to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    map_view.close()


if __name__ == "__main__":
    main()
