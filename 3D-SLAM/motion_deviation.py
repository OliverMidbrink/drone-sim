"""Where on the screen do tracked points disagree with the camera motion everyone else agrees on?

Given the camera motion between two frames (rotation R, translation direction t), every
tracked point has to land on its epipolar line in the new frame. A point that lands off the
line "wants" the camera to have moved differently. Its pull is the shortest shift that puts it
back on the line (pixels, perpendicular to the line; motion along the line is not measurable
without depth).

Pulls are collected per screen section over the last few motion estimates. Per section the
median pull minus the median pull of all points is the section's deviation, i.e. how differently
this part of the screen wants to move the camera compared with everyone else. Random tracking
noise cancels in the median; a lens distortion, a wrong focal length or a wrong rotation shows
up as a pattern across the sections.
"""

from __future__ import annotations

from collections import deque
from typing import NamedTuple

import cv2
import numpy as np

GRID_COLS, GRID_ROWS = 8, 6
WINDOW_COMMITS = 30  # motion estimates pooled in the picture
MIN_POINTS = 8  # a section needs this many pulls to be shown
FULL_SCALE_PX = 1.0  # deviation that is drawn fully red
ARROW_GAIN = 40.0  # arrow length in screen px per px of deviation (at the original image size)


class MotionSample(NamedTuple):
    """One camera-motion estimate with the point pairs it was made from (key frame -> frame)."""

    key_px: np.ndarray  # (N, 2) points in the key frame
    current_px: np.ndarray  # (N, 2) the same points in the current frame
    rotation: np.ndarray  # (3, 3) X_current = R X_key + t
    translation: np.ndarray  # (3,) or (3, 1), only its direction is used
    key_frame: int  # frame numbers of the two images (0 = first frame)
    frame: int


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def epipolar_pull(
    key_px: np.ndarray,
    current_px: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    """(N, 2) shift in pixels that moves each current point onto the epipolar line of its key-frame point."""
    t = np.asarray(translation, dtype=np.float64).reshape(3)
    length = float(np.linalg.norm(t))
    if length < 1e-12:
        return np.full((len(key_px), 2), np.nan)
    essential = _skew(t / length) @ np.asarray(rotation, dtype=np.float64)
    inverse = np.linalg.inv(intrinsic)
    fundamental = inverse.T @ essential @ inverse
    first = np.column_stack([np.asarray(key_px, dtype=np.float64).reshape(-1, 2), np.ones(len(key_px))])
    second = np.column_stack([np.asarray(current_px, dtype=np.float64).reshape(-1, 2), np.ones(len(current_px))])
    lines = first @ fundamental.T  # epipolar line a x + b y + c = 0 in the current frame, one per point
    off_line = np.einsum("ij,ij->i", lines, second)
    denominator = lines[:, 0] ** 2 + lines[:, 1] ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        return -(off_line / denominator)[:, None] * lines[:, :2]


class DeviationMeter:
    """Pools pulls per screen section and draws them over the camera picture."""

    def __init__(self, size: tuple[int, int], intrinsic: np.ndarray, title: str) -> None:
        self.size = size  # (width, height) of the camera frames
        self.intrinsic = np.asarray(intrinsic, dtype=np.float64)
        self.title = title
        self._pools: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=WINDOW_COMMITS)

    def add(
        self,
        sample: MotionSample,
        rotation: np.ndarray | None = None,
        translation: np.ndarray | None = None,
    ) -> None:
        """Add a motion estimate's pulls. Give rotation/translation to test another motion (e.g. the true one)."""
        pull = epipolar_pull(
            sample.key_px,
            sample.current_px,
            sample.rotation if rotation is None else rotation,
            sample.translation if translation is None else translation,
            self.intrinsic,
        )
        points = np.asarray(sample.current_px, dtype=np.float64).reshape(-1, 2)
        good = np.isfinite(pull).all(axis=1)
        width, height = self.size
        column = np.clip((points[:, 0] / width * GRID_COLS).astype(int), 0, GRID_COLS - 1)
        row = np.clip((points[:, 1] / height * GRID_ROWS).astype(int), 0, GRID_ROWS - 1)
        self._pools.append(((row * GRID_COLS + column)[good], pull[good]))

    def sections(self) -> tuple[np.ndarray, np.ndarray, int, float, float]:
        """(deviation (rows, cols, 2) with NaN where too few points, point count per section,
        total points, typical pull length of all points, spread of the section deviations)."""
        deviation = np.full((GRID_ROWS * GRID_COLS, 2), np.nan)
        counts = np.zeros(GRID_ROWS * GRID_COLS, dtype=int)
        if not self._pools:
            return deviation.reshape(GRID_ROWS, GRID_COLS, 2), counts.reshape(GRID_ROWS, GRID_COLS), 0, 0.0, 0.0
        cells = np.concatenate([c for c, _ in self._pools])
        pulls = np.concatenate([p for _, p in self._pools])
        if len(pulls) == 0:
            return deviation.reshape(GRID_ROWS, GRID_COLS, 2), counts.reshape(GRID_ROWS, GRID_COLS), 0, 0.0, 0.0
        consensus = np.median(pulls, axis=0)
        for cell in range(GRID_ROWS * GRID_COLS):
            chosen = pulls[cells == cell]
            counts[cell] = len(chosen)
            if len(chosen) >= MIN_POINTS:
                deviation[cell] = np.median(chosen, axis=0) - consensus
        lengths = np.linalg.norm(deviation, axis=1)
        shown = lengths[np.isfinite(lengths)]
        spread = float(np.sqrt(np.mean(shown**2))) if len(shown) else 0.0
        typical = float(np.median(np.linalg.norm(pulls, axis=1)))
        return (
            deviation.reshape(GRID_ROWS, GRID_COLS, 2),
            counts.reshape(GRID_ROWS, GRID_COLS),
            len(pulls),
            typical,
            spread,
        )

    def render(self, frame: np.ndarray, display_width: int | None = None) -> np.ndarray:
        """The camera picture dimmed, with one coloured cell + arrow per screen section."""
        width, height = self.size
        scale = 1.0 if display_width is None else display_width / width
        shown_w, shown_h = int(round(width * scale)), int(round(height * scale))
        base = frame if scale == 1.0 else cv2.resize(frame, (shown_w, shown_h), interpolation=cv2.INTER_AREA)
        image = (base * 0.45).astype(np.uint8)
        deviation, counts, total, typical, spread = self.sections()

        cell_w, cell_h = shown_w / GRID_COLS, shown_h / GRID_ROWS
        overlay = image.copy()
        lengths = np.linalg.norm(deviation, axis=2)
        for row in range(GRID_ROWS):
            for column in range(GRID_COLS):
                if np.isfinite(lengths[row, column]):
                    heat = float(np.clip(lengths[row, column] / FULL_SCALE_PX, 0.0, 1.0))
                    colour = (0, int(255 * (1.0 - heat)), int(255 * heat))  # green -> red (BGR)
                    p0 = (int(column * cell_w), int(row * cell_h))
                    p1 = (int((column + 1) * cell_w), int((row + 1) * cell_h))
                    cv2.rectangle(overlay, p0, p1, colour, -1)
        image = cv2.addWeighted(overlay, 0.45, image, 0.55, 0)

        for column in range(1, GRID_COLS):
            x = int(column * cell_w)
            cv2.line(image, (x, 0), (x, shown_h), (90, 90, 90), 1)
        for row in range(1, GRID_ROWS):
            y = int(row * cell_h)
            cv2.line(image, (0, y), (shown_w, y), (90, 90, 90), 1)

        longest_arrow = 0.42 * min(cell_w, cell_h)
        for row in range(GRID_ROWS):
            for column in range(GRID_COLS):
                x0, y0 = column * cell_w, row * cell_h
                if not np.isfinite(lengths[row, column]):
                    _text(image, "-", (int(x0 + 4), int(y0 + 14)), 0.45, (160, 160, 160))
                    continue
                centre = np.array([x0 + cell_w / 2, y0 + cell_h / 2])
                shift = deviation[row, column] * ARROW_GAIN * scale
                size = float(np.linalg.norm(shift))
                if size > longest_arrow:
                    shift *= longest_arrow / size
                if size > 1.0:
                    start = centre - shift / 2
                    end = centre + shift / 2
                    cv2.arrowedLine(
                        image,
                        (int(start[0]), int(start[1])),
                        (int(end[0]), int(end[1])),
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                        tipLength=0.35,
                    )
                _text(image, f"{lengths[row, column]:.2f}", (int(x0 + 4), int(y0 + 14)), 0.45, (255, 255, 255))

        _text(image, f"{self.title}", (8, 20), 0.6, (255, 255, 255))
        if total:
            summary = f"{total} point pulls | typical pull {typical:.2f}px | sections differ by {spread:.2f}px (rms)"
        else:
            summary = "waiting for the first camera-motion estimate"
        _text(image, summary, (8, 40), 0.5, (255, 255, 255))
        _text(
            image,
            f"arrow: where a section wants the camera moved, relative to everyone else (x{ARROW_GAIN:g}); "
            f"number px; red = {FULL_SCALE_PX:g}px",
            (8, shown_h - 8),
            0.42,
            (220, 220, 220),
        )
        return image


def _text(image: np.ndarray, text: str, origin: tuple[int, int], scale: float, colour: tuple[int, int, int]) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)
