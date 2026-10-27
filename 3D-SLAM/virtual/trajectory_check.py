"""Where along the figure 8 does the SLAM go wrong?

Every camera-pose commit of the SLAM is recorded next to the true camera pose of the same frame.
The SLAM's frame and scale are arbitrary (monocular), so the whole recorded trajectory is first fitted
onto the true one with one similarity transform (rotation, translation, scale; Umeyama). What is left
over is the drift. The window shows it top-down, coloured by position error, with an error-over-time
strip underneath; at the end a table breaks the flight into segments of the lap.
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

WINDOW = "Trajectory vs truth"
MAP_SIZE = (780, 500)  # top-down panel (width, height)
STRIP_HEIGHT = 190  # error-over-time panel below it
PIXELS_PER_METRE = 30.0
ERROR_FULL_M = 1.0  # position error drawn fully red
MIN_POSES = 8
MIN_SPREAD_RATIO = 0.15  # the fit needs a trajectory that is not (nearly) a straight line
TICK_SECONDS = 2.0
SEGMENTS_PER_LAP = 8

TRUE_PATH_BGR = (120, 120, 120)
POSITION_LINE_BGR = (0, 165, 255)
ORIENTATION_LINE_BGR = (255, 200, 0)


def _angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle between matching rows of a and b."""
    cos = np.sum(a * b, axis=1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _rotation_angle_deg(rotations: np.ndarray) -> np.ndarray:
    traces = np.trace(rotations, axis1=1, axis2=2)
    return np.degrees(np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0)))


def _error_color(error_m: float) -> tuple[int, int, int]:
    """Green (no error) to red (ERROR_FULL_M or more)."""
    fraction = float(np.clip(error_m / ERROR_FULL_M, 0.0, 1.0))
    hsv = np.uint8([[[int(60 * (1.0 - fraction)), 255, 255]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


class TrajectoryCheck:
    def __init__(
        self,
        true_pose: Callable[[int], tuple[np.ndarray, np.ndarray]],
        fps: float,
        lap_seconds: float,
    ) -> None:
        """`true_pose(frame)` gives the true (world -> camera rotation, camera position)."""
        self.true_pose = true_pose
        self.fps = fps
        self.lap_seconds = lap_seconds
        self.frames: list[int] = []
        self._slam_positions: list[np.ndarray] = []
        self._slam_rotations: list[np.ndarray] = []  # camera -> SLAM world
        self._true_positions: list[np.ndarray] = []
        self._true_rotations: list[np.ndarray] = []  # camera -> true world
        self._cache: tuple[int, dict | None] = (-1, None)
        lap_frames = np.arange(int(lap_seconds * fps) + 1)
        self._lap_positions = np.array([true_pose(int(f))[1] for f in lap_frames])

    def add(self, frame: int, slam_position: np.ndarray, slam_rotation: np.ndarray) -> None:
        rotation_world_to_camera, position = self.true_pose(frame)
        self.frames.append(frame)
        self._slam_positions.append(np.array(slam_position, dtype=np.float64).reshape(3))
        self._slam_rotations.append(np.array(slam_rotation, dtype=np.float64))
        self._true_positions.append(position)
        self._true_rotations.append(rotation_world_to_camera.T)

    # ------------------------------------------------------------------ analysis
    def analysis(self) -> dict | None:
        """Fit and per-pose errors, or None while the trajectory is too short or too straight to fit."""
        count = len(self.frames)
        if self._cache[0] == count:
            return self._cache[1]
        result = self._analyse() if count >= MIN_POSES else None
        self._cache = (count, result)
        return result

    def _analyse(self) -> dict | None:
        slam = np.array(self._slam_positions)
        true = np.array(self._true_positions)
        slam_centre, true_centre = slam.mean(axis=0), true.mean(axis=0)
        slam_c, true_c = slam - slam_centre, true - true_centre
        spread = np.linalg.svd(slam_c, compute_uv=False)
        if spread[0] < 1e-9 or spread[1] < MIN_SPREAD_RATIO * spread[0]:
            return None
        u, singular, vt = np.linalg.svd(true_c.T @ slam_c / len(slam))
        flip = np.diag([1.0, 1.0, np.sign(np.linalg.det(u) * np.linalg.det(vt))])
        rotation = u @ flip @ vt
        scale = float(np.trace(np.diag(singular) @ flip) / (np.sum(slam_c**2) / len(slam)))
        shift = true_centre - scale * rotation @ slam_centre
        aligned = scale * slam @ rotation.T + shift

        position_error = np.linalg.norm(aligned - true, axis=1)
        aligned_rotations = rotation @ np.array(self._slam_rotations)
        relative = np.transpose(aligned_rotations, (0, 2, 1)) @ np.array(self._true_rotations)
        orientation_error = _rotation_angle_deg(relative)

        # Local errors between consecutive commits, independent of how the drift accumulated.
        step_direction = np.full(len(slam), np.nan)
        step_ratio = np.full(len(slam), np.nan)
        slam_steps, true_steps = np.diff(aligned, axis=0), np.diff(true, axis=0)
        moving = np.linalg.norm(true_steps, axis=1) > 1e-6
        step_direction[1:][moving] = _angle_deg(slam_steps[moving], true_steps[moving])
        step_ratio[1:][moving] = np.linalg.norm(slam_steps[moving], axis=1) / np.linalg.norm(true_steps[moving], axis=1)
        return {
            "scale": scale,
            "aligned": aligned,
            "true": true,
            "position_error": position_error,
            "orientation_error": orientation_error,
            "step_direction": step_direction,
            "step_ratio": step_ratio,
        }

    # ------------------------------------------------------------------ window
    @staticmethod
    def _to_px(x: float, z: float) -> tuple[int, int]:
        return int(MAP_SIZE[0] / 2 + x * PIXELS_PER_METRE), int(MAP_SIZE[1] / 2 + z * PIXELS_PER_METRE)

    def render(self) -> np.ndarray:
        map_w, map_h = MAP_SIZE
        image = np.full((map_h + STRIP_HEIGHT, map_w, 3), 24, np.uint8)
        for x in range(-10, 11, 5):
            cv2.line(image, self._to_px(x, -8), self._to_px(x, 8), (45, 45, 45), 1)
        for z in range(-5, 6, 5):
            cv2.line(image, self._to_px(-12.5, z), self._to_px(12.5, z), (45, 45, 45), 1)
        lap = np.array([self._to_px(p[0], p[2]) for p in self._lap_positions], dtype=np.int32)
        cv2.polylines(image, [lap], False, TRUE_PATH_BGR, 2, cv2.LINE_AA)
        for second in np.arange(0.0, self.lap_seconds, TICK_SECONDS):
            x, y = self._to_px(*self._lap_positions[int(second * self.fps)][[0, 2]])
            cv2.circle(image, (x, y), 3, (200, 200, 200), -1, cv2.LINE_AA)
            cv2.putText(image, f"{second:g}s", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        result = self.analysis()
        elapsed = self.frames[-1] / self.fps if self.frames else 0.0
        if result is None:
            header = f"{len(self.frames)} poses - collecting (needs {MIN_POSES} poses along a curve to fit the truth)"
        else:
            aligned, error = result["aligned"], result["position_error"]
            for index in range(1, len(aligned)):
                a, b = self._to_px(aligned[index - 1, 0], aligned[index - 1, 2]), self._to_px(aligned[index, 0], aligned[index, 2])
                cv2.line(image, a, b, _error_color(float(error[index])), 3, cv2.LINE_AA)
            tx, tz = result["true"][-1][[0, 2]]
            cv2.circle(image, self._to_px(tx, tz), 7, (255, 255, 255), 1, cv2.LINE_AA)
            header = (
                f"t={elapsed:.1f}s  poses {len(self.frames)}  position error {error[-1]:.2f}m  "
                f"heading error {result['orientation_error'][-1]:.1f}deg  fitted scale {result['scale']:.2f}"
            )
        cv2.putText(image, header, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, "grey = true path, line = SLAM (green no error, red >= 1 m)", (10, map_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA)
        self._draw_strip(image, result, elapsed)
        return image

    def _draw_strip(self, image: np.ndarray, result: dict | None, elapsed: float) -> None:
        map_w, map_h = MAP_SIZE
        left, right = 45, map_w - 15
        top, bottom = map_h + 30, map_h + STRIP_HEIGHT - 22
        cv2.line(image, (0, map_h), (map_w, map_h), (90, 90, 90), 1)
        span = max(self.lap_seconds, elapsed)
        for second in np.arange(0.0, span + 1e-9, TICK_SECONDS):
            x = int(left + second / span * (right - left))
            cv2.line(image, (x, top), (x, bottom), (45, 45, 45), 1)
            cv2.putText(image, f"{second:g}", (x - 6, bottom + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(image, "time (s)", (right - 50, map_h + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
        if result is None:
            return
        times = np.array(self.frames) / self.fps
        xs = (left + times / span * (right - left)).astype(int)
        for key, color, floor, label, row in (
            ("position_error", POSITION_LINE_BGR, ERROR_FULL_M, "position error (m)", 0),
            ("orientation_error", ORIENTATION_LINE_BGR, 5.0, "heading error (deg)", 1),
        ):
            values = result[key]
            top_value = max(floor, float(values.max()))
            ys = (bottom - values / top_value * (bottom - top)).astype(int)
            cv2.polylines(image, [np.stack([xs, ys], axis=1).astype(np.int32)], False, color, 2, cv2.LINE_AA)
            cv2.putText(image, f"{label}: top = {top_value:.1f}", (left + 8 + row * 250, map_h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------ report
    def print_summary(self) -> None:
        result = self.analysis()
        if result is None:
            print("\nTrajectory check: too little motion recorded to fit the SLAM onto the true path")
            return
        times = np.array(self.frames) / self.fps
        segment = self.lap_seconds / SEGMENTS_PER_LAP
        print(f"\nTrajectory check over {len(times)} poses, {times[-1]:.1f}s (fitted scale {result['scale']:.3f})")
        print(
            "  segment      where (x, z)    turn deg/s | position err m: mean   max | heading err deg | "
            "step direction err deg | step length ratio"
        )
        for index in range(int(np.ceil(times[-1] / segment))):
            start, end = index * segment, (index + 1) * segment
            chosen = (times >= start) & (times < end)
            if not chosen.any():
                continue
            frame_start, frame_end = int(round(start * self.fps)), int(round(min(end, times[-1]) * self.fps))
            middle = self.true_pose(int(round((frame_start + frame_end) / 2)))[1]
            turn = float(
                _rotation_angle_deg((self.true_pose(frame_end)[0] @ self.true_pose(frame_start)[0].T)[None])[0]
            ) / max((frame_end - frame_start) / self.fps, 1e-9)
            error = result["position_error"][chosen]
            print(
                f"  {start:5.1f}-{end:5.1f}s  ({middle[0]:6.1f},{middle[2]:6.1f})   {turn:8.1f}    |"
                f"            {error.mean():6.2f} {error.max():5.2f} |"
                f"       {result['orientation_error'][chosen].mean():6.1f}     |"
                f"          {np.nanmedian(result['step_direction'][chosen]) if np.isfinite(result['step_direction'][chosen]).any() else float('nan'):6.1f}         |"
                f"      {np.nanmedian(result['step_ratio'][chosen]) if np.isfinite(result['step_ratio'][chosen]).any() else float('nan'):6.2f}"
            )
