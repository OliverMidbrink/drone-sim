#!/usr/bin/env python3
"""3D map from image 1 versus image N only, for N = 2 .. 10; browse N to watch the map update.

Standalone: plain OpenCV, nothing from the SLAM. The only thing reused is the chained dense
optical flow of flow_arrows.py, which says where each point of image 1 is in image N.

For every N, only image 1 and image N are used:
  correspondences (100 x 100 grid of image 1  ->  where each is in image N)
  -> essential matrix (RANSAC) -> relative pose -> triangulation.
Intermediate images only matter through the chained flow that links 1 to N.

Scale is arbitrary in monocular geometry. For every N the map is scaled so its median depth
is 1, which keeps the maps comparable while browsing; camera N then sits at the (relative)
baseline that image N implies.

Browse: Left / Right (or A / D) or the slider. Drag to rotate. Q / Esc closes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from flow_arrows import DEFAULT_VIDEO, accumulate_grid, load_frames


@dataclass
class TwoViewMap:
    image_number: int  # N
    points: np.ndarray  # (M, 3) in image 1's camera frame: x right, y down, z forward; median depth 1
    camera: np.ndarray  # (3,) position of camera N in that frame
    correspondences: int
    ransac_inliers: int


def camera_matrix(width: int, height: int, fov_deg: float) -> np.ndarray:
    focal = width / (2.0 * np.tan(np.radians(fov_deg) / 2.0))
    return np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def two_view_map(
    number: int, base: np.ndarray, moved: np.ndarray, intrinsic: np.ndarray, width: int, height: int
) -> TwoViewMap | None:
    """Map from image 1 and image `number` alone. `moved` is where each `base` point is in image `number`."""
    inside = (moved[:, 0] >= 0) & (moved[:, 0] < width - 1) & (moved[:, 1] >= 0) & (moved[:, 1] < height - 1)
    first = base[inside].astype(np.float64)
    other = moved[inside].astype(np.float64)
    if len(first) < 8:
        return None

    essential, mask = cv2.findEssentialMat(first, other, intrinsic, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if essential is None or mask is None:
        return None
    essential = essential[:3]  # several solutions can come back stacked; take the first
    _, rotation, translation, pose_mask = cv2.recoverPose(essential, first, other, intrinsic, mask=mask)
    good = pose_mask.ravel() > 0
    if good.sum() < 8:
        return None

    projection_first = intrinsic @ np.hstack([np.eye(3), np.zeros((3, 1))])
    projection_other = intrinsic @ np.hstack([rotation, translation.reshape(3, 1)])
    homogeneous = cv2.triangulatePoints(projection_first, projection_other, first[good].T, other[good].T)
    points = (homogeneous[:3] / homogeneous[3]).T
    depth_other = (points @ rotation.T + translation.reshape(3))[:, 2]
    usable = np.isfinite(points).all(axis=1) & (points[:, 2] > 0) & (depth_other > 0)
    points = points[usable]
    if len(points) < 8:
        return None

    scale = 1.0 / float(np.median(points[:, 2]))
    camera = -(rotation.T @ translation.reshape(3)) * scale
    return TwoViewMap(number, points * scale, camera, int(len(first)), int(good.sum()))


def shared_limits(maps: list[TwoViewMap]) -> tuple[np.ndarray, np.ndarray]:
    """Axis limits shared by all maps (robust percentiles, widened to include every camera)."""
    everything = np.vstack([m.points for m in maps])
    low = np.percentile(everything, 2, axis=0)
    high = np.percentile(everything, 98, axis=0)
    low[2] = 0.0
    for m in maps:
        low = np.minimum(low, m.camera)
        high = np.maximum(high, m.camera)
    low = np.minimum(low, 0.0)
    high = np.maximum(high, 0.0)
    return low, high


class MapBrowser:
    def __init__(self, maps: list[TwoViewMap], low: np.ndarray, high: np.ndarray, plt) -> None:
        from matplotlib.widgets import Slider

        self.maps, self.low, self.high, self.plt = maps, low, high, plt
        self.position = 0
        self.view = (20.0, -70.0)

        self.fig = plt.figure(figsize=(12, 9))
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.1)
        self.ax = self.fig.add_axes((0.02, 0.1, 0.96, 0.83), projection="3d")
        slider_ax = self.fig.add_axes((0.2, 0.03, 0.6, 0.03))
        self.slider = Slider(
            slider_ax, "image N", maps[0].image_number, maps[-1].image_number, valinit=maps[0].image_number, valstep=1, valfmt="%d"
        )
        by_number = {m.image_number: i for i, m in enumerate(maps)}  # an N whose pose failed has no map
        self.slider.on_changed(lambda value: self.show(by_number[int(value)]) if int(value) in by_number else None)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _on_key(self, event) -> None:
        if event.key in ("right", "d"):
            self.step(1)
        elif event.key in ("left", "a"):
            self.step(-1)
        elif event.key in ("q", "escape"):
            self.plt.close(self.fig)

    def step(self, delta: int) -> None:
        target = int(np.clip(self.position + delta, 0, len(self.maps) - 1))
        self.slider.set_val(self.maps[target].image_number)  # triggers show()

    def show(self, position: int) -> None:
        position = int(np.clip(position, 0, len(self.maps) - 1))
        ax = self.ax
        if ax.elev is not None:
            self.view = (ax.elev, ax.azim)  # keep the user's rotation across images
        self.position = position
        m = self.maps[position]
        low, high = self.low, self.high

        shown = np.all((m.points >= low) & (m.points <= high), axis=1)
        p = m.points[shown]
        far = float(high[2])
        colors = self.plt.get_cmap("turbo_r")(np.clip(p[:, 2] / max(far, 1e-9), 0.0, 1.0))[:, :3]

        ax.clear()
        # plot x right, z (depth) away from the viewer, -y up
        ax.scatter(p[:, 0], p[:, 2], -p[:, 1], c=colors, s=2, linewidths=0, depthshade=False)
        ax.scatter([0], [0], [0], c="red", marker="^", s=90, label="camera 1")
        ax.scatter([m.camera[0]], [m.camera[2]], [-m.camera[1]], c="lime", marker="^", s=90, label=f"camera {m.image_number}")
        ax.plot([0, m.camera[0]], [0, m.camera[2]], [0, -m.camera[1]], c="gray", lw=1)
        ax.set_xlim(low[0], high[0])
        ax.set_ylim(0.0, far)
        ax.set_zlim(-high[1], -low[1])
        ax.set_box_aspect(tuple(max(float(v), 1e-6) for v in (high[0] - low[0], far, high[1] - low[1])))
        ax.set_xlabel("x (right)")
        ax.set_ylabel("depth z")
        ax.set_zlabel("up")
        ax.view_init(elev=self.view[0], azim=self.view[1])
        ax.legend(loc="upper left")
        ax.set_title(
            f"image 1 vs image {m.image_number}:  {len(m.points)} points from {m.correspondences} correspondences "
            f"({m.ransac_inliers} fit the pose),  {len(m.points) - int(shown.sum())} off-chart\n"
            f"median depth = 1,  baseline to camera {m.image_number} = {np.linalg.norm(m.camera):.3f}",
            fontsize=10,
        )
        self.fig.suptitle("[Left/Right: browse   drag: rotate]", fontsize=9)
        self.fig.canvas.draw_idle()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", nargs="?", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--frames", type=int, default=10, help="number of images (default: 10)")
    parser.add_argument("--scale", type=float, default=0.5, help="resize factor before computing flow (default: 0.5)")
    parser.add_argument("--fov-deg", type=float, default=120.0, help="horizontal field of view in degrees (default: 120)")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    video = args.video.expanduser().resolve()
    frames = load_frames(video, args.frames, args.scale)
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    height, width = grays[0].shape
    intrinsic = camera_matrix(width, height, args.fov_deg)

    base, positions = accumulate_grid(grays)
    maps: list[TwoViewMap] = []
    for k in range(1, len(frames)):
        number = k + 1
        result = two_view_map(number, base, positions[k], intrinsic, width, height)
        if result is None:
            print(f"image 1 vs image {number}: no pose could be estimated, skipped")
            continue
        maps.append(result)
        print(
            f"image 1 vs image {number}: {result.correspondences} correspondences, {result.ransac_inliers} fit the pose, "
            f"{len(result.points)} 3D points, baseline {np.linalg.norm(result.camera):.3f}"
        )
    if not maps:
        raise SystemExit("No map could be built")

    import matplotlib

    matplotlib.use("QtAgg")
    import matplotlib.pyplot as plt

    low, high = shared_limits(maps)
    browser = MapBrowser(maps, low, high, plt)
    browser.show(0)
    plt.show()


if __name__ == "__main__":
    main()
