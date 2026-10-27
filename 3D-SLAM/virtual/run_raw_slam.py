#!/usr/bin/env python3
"""Run the raw SLAM (3D-SLAM/raw_slam.py) alone on the virtual figure-8 flight and score its trajectory.

Frames are rendered off-screen along the flight, one per 1 / fps seconds of flight time, and fed
into RawSlam (live feature tracking + global ORB anchors, nothing else). Two windows:
  - "Raw SLAM tracks": what the camera sees, with the tracked points
  - "Trajectory vs truth": top-down SLAM path coloured by error, plus error over time
Q or Esc in the tracks window stops it. On exit a per-segment table of the lap is printed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

import renderer
import world

SLAM_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(SLAM_DIR))
import camera_calibration  # noqa: E402
import raw_slam  # noqa: E402
from trajectory_check import WINDOW as TRAJECTORY_WINDOW, TrajectoryCheck  # noqa: E402

TRACKS_WINDOW = "Raw SLAM tracks"
DEFAULT_CALIBRATION = Path(__file__).resolve().parent / "out" / "camera_rays.npz"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fov-deg", type=float, default=world.DEFAULT_FOV_DEG, help="horizontal field of view (default: %(default)g)")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=float, default=30.0, help="frames per second of flight time (default: %(default)g)")
    parser.add_argument("--seed", type=int, default=world.DEFAULT_SEED)
    parser.add_argument("--seconds", type=float, default=0.0, help="stop after this much flight time (default: until you quit)")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION, help="pixel-ray calibration used if it exists (default: %(default)s)")
    parser.add_argument("--no-calibration", action="store_true", help="ignore the calibration, ideal pinhole from --fov-deg")
    return parser.parse_args(argv)


def load_undistorter(args: argparse.Namespace, size: tuple[int, int]):
    """The calibration's Undistorter, or None when there is none, it is switched off, or it does not fit this camera."""
    if args.no_calibration or not args.calibration.is_file():
        print("Calibration: none, ideal pinhole from --fov-deg")
        return None
    with np.load(args.calibration) as data:
        calibrated_fov = float(data["fov_deg"]) if "fov_deg" in data else None
    if calibrated_fov is not None and abs(calibrated_fov - args.fov_deg) > 1e-6:
        print(f"Calibration: ignored, it is for a {calibrated_fov:g} deg camera but this run uses {args.fov_deg:g} deg")
        return None
    try:
        undistorter = camera_calibration.Undistorter(camera_calibration.load_rays(args.calibration), size)
    except ValueError as error:
        print(f"Calibration: ignored, {error}")
        return None
    print(f"Calibration: using {args.calibration.name}, {undistorter.describe()}")
    return undistorter


def main() -> None:
    args = parse_args()
    size = (args.width, args.height)
    boxes = world.generate_boxes(args.seed)
    path = world.FigureEightPath(world.flight_altitude(boxes))
    print("\n".join(world.describe(boxes, path, args.fov_deg, args.width, args.height)))

    window, ctx = renderer.create_hidden_context()  # the only OpenGL context
    scene = renderer.SceneRenderer(ctx, boxes, args.seed)
    target = renderer.OffscreenTarget(ctx, size)
    undistorter = load_undistorter(args, size)

    def render(frame_index: int) -> np.ndarray:
        position, forward = path.pose(frame_index / args.fps)
        scene.draw(target.msaa, size, position, forward, args.fov_deg)
        image = target.read_bgr()
        return image if undistorter is None else undistorter.apply(image)

    def true_pose(number: int) -> tuple[np.ndarray, np.ndarray]:
        """Exact (world -> camera rotation, camera position) of a frame; camera axes x right, y down, z forward."""
        position, forward = path.pose(number / args.fps)
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, renderer.UP)
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        return np.stack([right, down, forward]), position

    intrinsic = raw_slam.pinhole(args.width, args.height, args.fov_deg) if undistorter is None else undistorter.intrinsic
    frame = render(0)
    slam = raw_slam.RawSlam(frame, intrinsic)
    check = TrajectoryCheck(true_pose, args.fps, path.lap_seconds)
    check.add(0, slam.world_position, slam.world_rotation)
    print(f"Raw SLAM, {args.width}x{args.height}, fx={intrinsic[0, 0]:.1f}px, {args.fps:g} fps -> {world.SPEED_MS / args.fps * 100:.1f} cm per frame")

    def show_tracks(image: np.ndarray, frame_index: int) -> None:
        vis = image.copy()
        for x, y in slam.track_points:
            cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
        text = f"frame {frame_index}  t={frame_index / args.fps:.1f}s  tracks {len(slam.track_points)}  {slam.describe()}"
        cv2.putText(vis, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(TRACKS_WINDOW, vis)

    show_tracks(frame, 0)
    cv2.imshow(TRAJECTORY_WINDOW, check.render())
    frame_index = 0
    try:
        while True:
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            if args.seconds > 0 and frame_index / args.fps >= args.seconds:
                break
            frame_index += 1
            frame = render(frame_index)
            if slam.process(frame):
                check.add(frame_index, slam.world_position, slam.world_rotation)
                cv2.imshow(TRAJECTORY_WINDOW, check.render())
            show_tracks(frame, frame_index)
    finally:
        print(f"\nRaw SLAM after {frame_index} frames: {slam.describe()}")
        check.print_summary()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
