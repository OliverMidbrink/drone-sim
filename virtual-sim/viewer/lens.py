"""Lens geometry shared by the renderer and by anything that reads the pictures (no OpenGL in here).

The side cameras are 360-camera style: one circular equidistant fisheye per side, square picture,
the image circle touching all four edges, 180 degrees from edge to edge in every direction through
the centre. A point at angle theta from the optical axis lands at distance r = theta / 90 degrees *
(width / 2) from the picture centre. (Real dual-lens 360 cameras make the circle a little bigger than
180 degrees so the two halves overlap for stitching; the drone has no use for that.)

`rectify_maps` is how a program that wants an ordinary pinhole picture (a SLAM, say) gets one out of
the fisheye picture: for every pixel of a virtual pinhole camera sharing the lens' optical centre it
says where to look in the fisheye picture (feed it to cv2.remap).
"""

from __future__ import annotations

import numpy as np

# Camera frames used in this project:
#   "GL"  camera: x right, y up, -z forward (the renderer and the drone body share this convention)
#   "CV"  camera: x right, y down, +z forward (OpenCV, and the SLAM)
GL_TO_CV = np.diag([1.0, -1.0, -1.0])


def mount_rotation(yaw_degrees: float, pitch_degrees: float = 0.0) -> np.ndarray:
    """GL-camera-to-body rotation for a camera looking `yaw_degrees` to the right of the drone's nose (negative = left)
    and `pitch_degrees` above the horizon (-90 = straight down; the top of a downward picture then points along the yaw direction)."""
    yaw, pitch = np.radians(yaw_degrees), np.radians(pitch_degrees)
    heading = np.array([np.sin(yaw), 0.0, -np.cos(yaw)])
    forward = np.cos(pitch) * heading + np.array([0.0, np.sin(pitch), 0.0])
    reference = np.array([0.0, 1.0, 0.0]) if abs(np.sin(pitch)) < 0.99 else heading  # "up" in the picture
    up = reference - (reference @ forward) * forward
    up /= np.linalg.norm(up)
    right = np.cross(forward, up)
    return np.stack([right, up, -forward], axis=1)  # columns: camera x, y, z in body coordinates


LENS_FOV = 160.0  # degrees across the image circle of the side cameras (180 = a full hemisphere)


def pinhole_matrix(size: int, fov_degrees: float) -> np.ndarray:
    """Intrinsics of a square pinhole picture `size` pixels wide with this field of view across its width."""
    focal = size / (2.0 * np.tan(np.radians(fov_degrees) / 2.0))
    return np.array([[focal, 0.0, size / 2.0], [0.0, focal, size / 2.0], [0.0, 0.0, 1.0]])


def rectify_maps(fisheye_size: int, pinhole_size: int, fov_degrees: float, lens_fov_degrees: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """cv2.remap maps (x, y) that cut a square pinhole picture out of the middle of a square fisheye picture.

    The pinhole picture has the same optical axis and centre of projection as the fisheye one, so a
    camera pose found in it is the lens' pose.
    """
    intrinsic = pinhole_matrix(pinhole_size, fov_degrees)
    u, v = np.meshgrid(np.arange(pinhole_size, dtype=np.float64), np.arange(pinhole_size, dtype=np.float64))
    x = (u - intrinsic[0, 2]) / intrinsic[0, 0]  # CV camera: right
    y = (v - intrinsic[1, 2]) / intrinsic[1, 1]  # CV camera: down
    theta = np.arctan2(np.hypot(x, y), 1.0)  # angle from the optical axis
    radius = theta / np.radians((lens_fov_degrees or LENS_FOV) / 2.0) * (fisheye_size / 2.0)
    azimuth = np.arctan2(y, x)
    centre = fisheye_size / 2.0 - 0.5
    map_x = centre + radius * np.cos(azimuth)
    map_y = centre + radius * np.sin(azimuth)  # picture rows run downward like the CV camera's y
    return map_x.astype(np.float32), map_y.astype(np.float32)
