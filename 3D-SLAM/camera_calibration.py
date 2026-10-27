"""Use a pixel-level camera calibration with the SLAM, which itself assumes an ideal pinhole camera.

A calibration file is an .npz (written by virtual/calibrate_camera.py) holding
    rays   (H, W, 3)   unit viewing ray of every pixel, x right, y down, z forward
The `Undistorter` remaps each camera frame to the ideal pinhole picture that has those same ray
directions, so the SLAM can keep its pinhole maths; `intrinsic` is the matrix of that picture.
For an already-pinhole camera the remap is the identity and frames pass through untouched.

If the corrected picture would need pixels from outside the camera image (e.g. fisheye corners),
it is zoomed in (cropped) just enough that every output pixel has a real source pixel.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

NEWTON_STEPS = 12
ZOOM_STEP = 0.02
MAX_ZOOM = 3.0
EDGE_TOLERANCE_PX = 0.01
IDENTITY_TOLERANCE_PX = 0.01


def _sample(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Exact bilinear lookup at pixel-index coordinates (u, v); outside the image the edge value is used.

    (cv2.remap quantises coordinates to 1/32 px, which is too coarse for inverting the ray table.)
    """
    height, width = image.shape
    u = np.clip(u, 0.0, width - 1.0)
    v = np.clip(v, 0.0, height - 1.0)
    x0 = np.minimum(u.astype(np.int64), width - 2)
    y0 = np.minimum(v.astype(np.int64), height - 2)
    wx = u - x0
    wy = v - y0
    top = image[y0, x0] * (1.0 - wx) + image[y0, x0 + 1] * wx
    bottom = image[y0 + 1, x0] * (1.0 - wx) + image[y0 + 1, x0 + 1] * wx
    return top * (1.0 - wy) + bottom * wy


class Undistorter:
    """Remaps camera frames of `size` = (width, height) to an ideal pinhole image."""

    def __init__(self, rays: np.ndarray, size: tuple[int, int]) -> None:
        width, height = size
        rays = np.asarray(rays, dtype=np.float64)
        if rays.ndim != 3 or rays.shape[2] != 3:
            raise ValueError("calibration 'rays' must have shape (H, W, 3)")
        calibrated_height, calibrated_width = rays.shape[:2]
        if abs(calibrated_width / calibrated_height - width / height) > 0.01 * width / height:
            raise ValueError(
                f"calibration is for {calibrated_width}x{calibrated_height}, frames are {width}x{height} (different aspect ratio)"
            )
        if rays[..., 2].min() < 0.05:
            raise ValueError("calibration has rays more than ~87 degrees off the optical axis; a pinhole picture cannot hold them")

        self.size = size
        self.calibrated_size = (calibrated_width, calibrated_height)
        slope_x = rays[..., 0] / rays[..., 2]  # tan of the horizontal angle at every pixel
        slope_y = rays[..., 1] / rays[..., 2]
        if (calibrated_width, calibrated_height) != (width, height):
            slope_x = cv2.resize(slope_x, (width, height), interpolation=cv2.INTER_LINEAR)  # pixel-centre aligned
            slope_y = cv2.resize(slope_y, (width, height), interpolation=cv2.INTER_LINEAR)
        self._slope_x, self._slope_y = slope_x, slope_y

        u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))  # pixel index coordinates
        self._fx, self._cx = np.polyfit(slope_x.ravel(), u.ravel(), 1)
        self._fy, self._cy = np.polyfit(slope_y.ravel(), v.ravel(), 1)

        self.zoom = self._find_zoom()
        focal = 0.5 * (self._fx + self._fy) * self.zoom
        centre_x, centre_y = (width - 1) / 2.0, (height - 1) / 2.0
        self.intrinsic = np.array([[focal, 0.0, centre_x], [0.0, focal, centre_y], [0.0, 0.0, 1.0]], dtype=np.float64)

        source_x, source_y = self._source_coordinates((u - centre_x) / focal, (v - centre_y) / focal)
        self.max_shift_px = float(np.hypot(source_x - u, source_y - v).max())
        self.identity = self.max_shift_px < IDENTITY_TOLERANCE_PX
        self._map_x = source_x.astype(np.float32)
        self._map_y = source_y.astype(np.float32)

    def _source_coordinates(self, target_x: np.ndarray, target_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Camera pixel (index coordinates) whose viewing ray has slope (target_x, target_y); inverts the ray table."""
        u = self._fx * target_x + self._cx  # start from the pinhole fit, then refine
        v = self._fy * target_y + self._cy
        for _ in range(NEWTON_STEPS):
            u = u + (target_x - _sample(self._slope_x, u, v)) * self._fx
            v = v + (target_y - _sample(self._slope_y, u, v)) * self._fy
        return u, v

    def _find_zoom(self) -> float:
        """Smallest zoom >= 1 whose whole output picture is covered by real camera pixels."""
        width, height = self.size
        xs = np.arange(width, dtype=np.float64)
        ys = np.arange(height, dtype=np.float64)
        edge_u = np.concatenate([xs, xs, np.zeros(height), np.full(height, width - 1.0)])
        edge_v = np.concatenate([np.zeros(width), np.full(width, height - 1.0), ys, ys])
        centre_x, centre_y = (width - 1) / 2.0, (height - 1) / 2.0
        focal0 = 0.5 * (self._fx + self._fy)
        zoom = 1.0
        while zoom <= MAX_ZOOM:
            focal = focal0 * zoom
            source_x, source_y = self._source_coordinates((edge_u - centre_x) / focal, (edge_v - centre_y) / focal)
            inside = (
                (source_x >= -EDGE_TOLERANCE_PX)
                & (source_x <= width - 1 + EDGE_TOLERANCE_PX)
                & (source_y >= -EDGE_TOLERANCE_PX)
                & (source_y <= height - 1 + EDGE_TOLERANCE_PX)
            )
            if inside.all():
                return zoom
            zoom += ZOOM_STEP
        raise ValueError(f"calibration needs more than {MAX_ZOOM:g}x zoom to fill the picture; check the calibration")

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape[1::-1] != self.size:
            raise ValueError(f"frame is {frame.shape[1]}x{frame.shape[0]}, calibration was prepared for {self.size[0]}x{self.size[1]}")
        if self.identity:
            return frame
        return cv2.remap(frame, self._map_x, self._map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def describe(self) -> str:
        kind = "already a pinhole, frames pass through" if self.identity else f"frames are remapped (largest shift {self.max_shift_px:.1f}px)"
        return (
            f"calibration {self.calibrated_size[0]}x{self.calibrated_size[1]}: {kind}; "
            f"zoom x{self.zoom:.2f}, fx={self.intrinsic[0, 0]:.1f}px"
        )


def load_rays(path: Path) -> np.ndarray:
    with np.load(path) as data:
        if "rays" not in data:
            raise ValueError(f"{path} has no 'rays' array (expected a file written by virtual/calibrate_camera.py)")
        return np.asarray(data["rays"], dtype=np.float64)
