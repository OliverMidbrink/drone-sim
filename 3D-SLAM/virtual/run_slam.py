#!/usr/bin/env python3
"""Run our SLAM (3d-slam.py's SlamSession and 3D map view) on the virtual drone camera.

No video file: frames are rendered off-screen along the figure-8 flight, one frame per
1 / fps seconds of flight (so the result does not depend on how fast your machine is), and
fed straight into the SLAM. Windows, like 3d-slam.py plus two checks against the truth:
  - "SLAM tracks": what the virtual camera sees, with the tracked points
  - the 3D map (mouse / WASD to fly around; camera axes: red x, green y, blue z)
  - "Motion deviation": which screen sections disagree with the camera motion
  - "Trajectory vs truth": top-down SLAM path coloured by error, plus error over time
    (a per-segment table of the lap is printed on exit)
Q or Esc in the tracks window, or closing the map window, stops it.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np

import renderer
import world

SLAM_DIR = Path(__file__).resolve().parent.parent


def load_slam():
    sys.path.append(str(SLAM_DIR))  # 3d-slam.py imports its sibling modules by name
    path = SLAM_DIR / "3d-slam.py"
    spec = importlib.util.spec_from_file_location("slam_app", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["slam_app"] = module
    spec.loader.exec_module(module)
    return module


slam = load_slam()
import camera_calibration  # noqa: E402  (lives in 3D-SLAM/, on sys.path after load_slam)
from motion_deviation import DeviationMeter, epipolar_pull  # noqa: E402
from trajectory_check import WINDOW as TRAJECTORY_WINDOW, TrajectoryCheck  # noqa: E402

DEVIATION_WINDOW = "Motion deviation"
DEVIATION_PANEL_WIDTH = 640  # each of the two panels

DEFAULT_CALIBRATION = Path(__file__).resolve().parent / "out" / "camera_rays.npz"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fov-deg", type=float, default=world.DEFAULT_FOV_DEG, help="horizontal field of view of the camera AND told to the SLAM (default: %(default)g)")
    parser.add_argument("--width", type=int, default=960, help="rendered frame width (default: %(default)d)")
    parser.add_argument("--height", type=int, default=540, help="rendered frame height (default: %(default)d)")
    parser.add_argument("--fps", type=float, default=30.0, help="frames per second of flight time (default: %(default)g)")
    parser.add_argument("--seed", type=int, default=world.DEFAULT_SEED, help="seed for box layout and textures (default: %(default)d)")
    parser.add_argument("--profile", choices=sorted(slam.PROFILES), default="drone", help="SLAM tracking profile (default: %(default)s)")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION, help="pixel-ray calibration from calibrate_camera.py, used if it exists (default: %(default)s)")
    parser.add_argument("--no-calibration", action="store_true", help="ignore the calibration and use an ideal pinhole from --fov-deg")
    parser.add_argument("--log-motion", action="store_true", help="print, per camera-motion estimate, rotation and direction of travel against the truth, plus a summary at the end")
    return parser.parse_args(argv)


def load_undistorter(args: argparse.Namespace, size: tuple[int, int]):
    """The calibration's Undistorter, or None when there is none, it is switched off, or it does not fit this camera."""
    if args.no_calibration:
        print("Calibration: switched off, ideal pinhole from --fov-deg")
        return None
    if not args.calibration.is_file():
        print(f"Calibration: none found at {args.calibration} (run calibrate_camera.py), ideal pinhole from --fov-deg")
        return None
    with np.load(args.calibration) as data:
        calibrated_fov = float(data["fov_deg"]) if "fov_deg" in data else None
    if calibrated_fov is not None and abs(calibrated_fov - args.fov_deg) > 1e-6:
        print(
            f"Calibration: ignored, it is for a {calibrated_fov:g} deg camera but this run uses {args.fov_deg:g} deg "
            f"(rerun: python calibrate_camera.py --fov-deg {args.fov_deg:g})"
        )
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
    if not boxes:
        raise SystemExit("No boxes could be placed")
    path = world.FigureEightPath(world.flight_altitude(boxes))
    print("\n".join(world.describe(boxes, path, args.fov_deg, args.width, args.height)))

    # One OpenGL context for everything: the map window's. The camera view is rendered into an
    # off-screen framebuffer of that same context (a second context is not detected on every setup).
    map_height = max(320, args.height // 2)
    map_view = slam.MapRenderer(args.width, map_height)
    map_view.setup_window()
    ctx = map_view.context
    scene = renderer.SceneRenderer(ctx, boxes, args.seed)
    target = renderer.OffscreenTarget(ctx, size)

    undistorter = load_undistorter(args, size)

    def render(frame_index: int):
        """The camera picture, corrected with the calibration when there is one (what the SLAM gets to see)."""
        position, forward = path.pose(frame_index / args.fps)
        scene.draw(target.msaa, size, position, forward, args.fov_deg)
        image = target.read_bgr()
        ctx.screen.use()  # hand the window back to the map view
        return image if undistorter is None else undistorter.apply(image)

    slam_args = slam.parse_args(["--profile", args.profile, "--fov-deg", str(args.fov_deg), "--no-disk-cache"])
    frame = render(0)
    session = slam.SlamSession(
        frame, slam_args, fps=args.fps, intrinsic=None if undistorter is None else undistorter.intrinsic
    )
    print(
        f"SLAM profile {slam_args.profile_name}, {session.width}x{session.height}, "
        f"fx={session.intrinsic[0, 0]:.1f}px, {args.fps:g} fps -> {world.SPEED_MS / args.fps * 100:.1f} cm per frame"
    )

    map_view.set_mesh(session.mesh_builder)
    slam.layout_windows(session.width, session.height, map_height)
    map_view.camera_rotation = session.world_rotation
    map_view.show(session.trajectory, session.map_store)

    def show_tracks(image, frame_index: int) -> None:
        vis = slam.draw_tracks(image, session.track_points, session.exclude_mask)
        text = (
            f"frame {frame_index}  t={frame_index / args.fps:.1f}s  tracks {len(session.track_points)}  "
            f"poses {len(session.trajectory)}  fixes {session.stats['fixes']}"
        )
        cv2.putText(vis, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(slam.TRACK_WINDOW, vis)

    # Motion deviation: the same point pairs judged against the SLAM's estimated motion and against the TRUE motion.
    # Large against the truth too -> tracking or camera model (calibration) problem; small against the truth
    # but large against the estimate -> the SLAM's motion estimate is the problem.
    meter_estimate = DeviationMeter(size, session.intrinsic, "vs SLAM's estimated motion")
    meter_truth = DeviationMeter(size, session.intrinsic, "vs TRUE motion")
    last_sample = None

    def true_pose(number: int) -> tuple[np.ndarray, np.ndarray]:
        """Exact (world -> camera rotation, camera position) of a frame; camera axes x right, y down, z forward."""
        position, forward = path.pose(number / args.fps)
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, renderer.UP)
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        return np.stack([right, down, forward]), position

    def true_motion(key_frame: int, frame_number: int) -> tuple[np.ndarray, np.ndarray]:
        """Exact (rotation, translation) taking key-frame camera coordinates to current-frame camera coordinates."""
        rotation_key, position_key = true_pose(key_frame)
        rotation_now, position_now = true_pose(frame_number)
        return rotation_now @ rotation_key.T, rotation_now @ (position_key - position_now)

    # Where in the figure 8 does the SLAM drift? Every pose commit is compared with the true pose of the same frame.
    trajectory_check = TrajectoryCheck(true_pose, args.fps, path.lap_seconds)
    trajectory_check.add(0, session.world_position, session.world_rotation)

    def rotation_angle_deg(rotation: np.ndarray) -> float:
        return float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))))

    def unit(vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float64).reshape(3)
        return vector / np.linalg.norm(vector)

    def misfit_px(sample, rotation: np.ndarray, translation: np.ndarray) -> float:
        """Median distance (px) of the tracked points from the epipolar lines of the given motion."""
        pull = epipolar_pull(sample.key_px, sample.current_px, rotation, translation, session.intrinsic)
        return float(np.nanmedian(np.linalg.norm(pull, axis=1)))

    motion_log: list[tuple[float, ...]] = []

    def log_motion(sample) -> None:
        """Rotation and direction of travel of one estimate against the truth, and how well each explains the flow."""
        true_rotation, true_translation = true_motion(sample.key_frame, sample.frame)
        estimated_direction, true_direction = unit(sample.translation), unit(true_translation)
        row = (
            rotation_angle_deg(true_rotation),
            rotation_angle_deg(sample.rotation),
            rotation_angle_deg(sample.rotation @ true_rotation.T),
            float(np.degrees(np.arccos(np.clip(estimated_direction @ true_direction, -1.0, 1.0)))),
            misfit_px(sample, sample.rotation, estimated_direction),
            misfit_px(sample, true_rotation, true_direction),
        )
        motion_log.append(row)
        print(
            f"frames {sample.key_frame:>4}->{sample.frame:<4} rotation: true {row[0]:5.2f} deg, SLAM {row[1]:5.2f} deg, "
            f"off by {row[2]:5.2f} | direction of travel off by {row[3]:5.1f} deg | "
            f"flow misfit: SLAM's motion {row[4]:.2f}px, true motion {row[5]:.2f}px"
        )

    def print_motion_summary(last_frame: int) -> None:
        if not motion_log:
            return
        rows = np.array(motion_log)
        print(f"\nMotion summary over {len(rows)} estimates")
        print(
            f"  rotation per estimate: true {rows[:, 0].mean():.2f} deg, SLAM {rows[:, 1].mean():.2f} deg "
            f"(the SLAM turns {100 * rows[:, 1].sum() / max(rows[:, 0].sum(), 1e-9):.0f}% of the true amount), "
            f"typical error {np.median(rows[:, 2]):.2f} deg"
        )
        print(f"  direction of travel: typical error {np.median(rows[:, 3]):.1f} deg, worst {rows[:, 3].max():.1f} deg")
        print(
            f"  flow misfit: SLAM's motion {np.median(rows[:, 4]):.2f}px, true motion {np.median(rows[:, 5]):.2f}px; "
            f"the SLAM's motion explains the flow at least as well as the truth in {100 * np.mean(rows[:, 4] <= rows[:, 5]):.0f}% of estimates"
        )
        total_true, _ = true_motion(0, last_frame)  # first camera -> last camera
        print(
            f"  total rotation since the start: true {rotation_angle_deg(total_true):.1f} deg, "
            f"SLAM {rotation_angle_deg(session.world_rotation):.1f} deg, "
            f"off by {rotation_angle_deg(session.world_rotation @ total_true):.1f} deg"
        )

    def show_deviation(image) -> None:
        panels = [meter.render(image, DEVIATION_PANEL_WIDTH) for meter in (meter_estimate, meter_truth)]
        cv2.imshow(DEVIATION_WINDOW, np.hstack(panels))

    show_tracks(frame, 0)
    show_deviation(frame)
    frame_index = 0
    try:
        while True:
            if not map_view.poll():
                break  # map window closed
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            frame_index += 1
            frame = render(frame_index)
            session.process(frame)
            map_view.camera_rotation = session.world_rotation
            if session.processed % slam_args.plot_every == 0:
                map_view.show(session.trajectory, session.map_store)
            show_tracks(frame, frame_index)
            sample = session.motion_sample
            if sample is not None and sample is not last_sample:
                last_sample = sample
                meter_estimate.add(sample)
                meter_truth.add(sample, *true_motion(sample.key_frame, sample.frame))
                trajectory_check.add(sample.frame, session.world_position, session.world_rotation)
                cv2.imshow(TRAJECTORY_WINDOW, trajectory_check.render())
                if args.log_motion:
                    log_motion(sample)
            show_deviation(frame)
    finally:
        print_motion_summary(frame_index)
        trajectory_check.print_summary()
        cv2.destroyAllWindows()
        session.finalize()
        map_view.close()


if __name__ == "__main__":
    main()
