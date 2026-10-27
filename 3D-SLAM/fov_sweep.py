#!/usr/bin/env python3
"""Which field of view gives the best 3D map? Browse random scenes x FOVs.

Picks 5 random start frames (the same 5 for every FOV). For each start frame S and each FOV
(80, 90, ... 150 degrees horizontal) it builds a 3D map from image S versus image S + 15,
using the same two-view method as two_view_map.py (chained dense flow -> essential matrix ->
triangulation). The flow does not depend on the FOV, so it is computed once per start frame;
only the camera matrix changes between FOVs.

Up / Down : previous / next start frame
Left / Right : previous / next FOV
Drag to rotate (the view is kept while you browse). Q closes.

Each map is scaled to median depth 1 (monocular scale is arbitrary); axis limits are shared
by all FOVs of one start frame so differences between FOVs are visible.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from flow_arrows import DEFAULT_VIDEO, accumulate_grid
from two_view_map import TwoViewMap, camera_matrix, shared_limits
from two_view_map import two_view_map as build_map

FOVS = tuple(range(80, 151, 10))  # horizontal degrees
SCENES = 5
FRAMES_FORWARD = 15


def read_window(capture: cv2.VideoCapture, start: int, count: int, scale: float) -> list[np.ndarray]:
    """`count` consecutive BGR frames beginning at frame `start`."""
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    while len(frames) < count:
        ok, raw = capture.read()
        if not ok:
            break
        if scale != 1.0:
            raw = cv2.resize(raw, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        frames.append(raw)
    return frames


class SweepBrowser:
    def __init__(
        self,
        starts: list[int],
        images: list[tuple[np.ndarray, np.ndarray]],
        maps: list[list[TwoViewMap | None]],
        limits: list[tuple[np.ndarray, np.ndarray]],
        plt,
    ) -> None:
        self.starts, self.images, self.maps, self.limits, self.plt = starts, images, maps, limits, plt
        self.row = 0
        self.col = 0
        self.view = (20.0, -70.0)

        # arrow keys must reach us, not matplotlib's back / forward navigation
        plt.rcParams["keymap.back"] = []
        plt.rcParams["keymap.forward"] = []

        self.fig = plt.figure(figsize=(16, 8.5))
        grid = self.fig.add_gridspec(2, 2, width_ratios=[1, 2.2], hspace=0.12, wspace=0.03, left=0.02, right=0.99, top=0.9, bottom=0.04)
        self.ax_first = self.fig.add_subplot(grid[0, 0])
        self.ax_last = self.fig.add_subplot(grid[1, 0])
        self.ax = self.fig.add_subplot(grid[:, 1], projection="3d")
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _on_key(self, event) -> None:
        if event.key == "up":
            self.row = max(self.row - 1, 0)
        elif event.key == "down":
            self.row = min(self.row + 1, len(self.starts) - 1)
        elif event.key == "left":
            self.col = max(self.col - 1, 0)
        elif event.key == "right":
            self.col = min(self.col + 1, len(FOVS) - 1)
        else:
            return
        self.show()

    def show(self) -> None:
        row, col = self.row, self.col
        first, last = self.images[row]
        m = self.maps[row][col]
        low, high = self.limits[row]

        for ax, image, label in (
            (self.ax_first, first, f"frame {self.starts[row]}"),
            (self.ax_last, last, f"frame {self.starts[row] + FRAMES_FORWARD}"),
        ):
            ax.clear()
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(label, fontsize=9)

        ax = self.ax
        if ax.elev is not None:
            self.view = (ax.elev, ax.azim)
        ax.clear()
        far = float(high[2])
        if m is None:
            ax.text2D(0.5, 0.5, "no pose could be estimated", transform=ax.transAxes, ha="center")
            summary = "no map"
        else:
            shown = np.all((m.points >= low) & (m.points <= high), axis=1)
            p = m.points[shown]
            colors = self.plt.get_cmap("turbo_r")(np.clip(p[:, 2] / max(far, 1e-9), 0.0, 1.0))[:, :3]
            # plot x right, z (depth) away from the viewer, -y up
            ax.scatter(p[:, 0], p[:, 2], -p[:, 1], c=colors, s=2, linewidths=0, depthshade=False)
            ax.scatter([0], [0], [0], c="red", marker="^", s=90)
            ax.scatter([m.camera[0]], [m.camera[2]], [-m.camera[1]], c="lime", marker="^", s=90)
            ax.plot([0, m.camera[0]], [0, m.camera[2]], [0, -m.camera[1]], c="gray", lw=1)
            summary = (
                f"{len(m.points)} points, {m.ransac_inliers} of {m.correspondences} correspondences fit the pose, "
                f"baseline {np.linalg.norm(m.camera):.2f}, {len(m.points) - int(shown.sum())} off-chart"
            )
        ax.set_xlim(low[0], high[0])
        ax.set_ylim(0.0, far)
        ax.set_zlim(-high[1], -low[1])
        ax.set_box_aspect(tuple(max(float(v), 1e-6) for v in (high[0] - low[0], far, high[1] - low[1])))
        ax.set_xlabel("x (right)")
        ax.set_ylabel("depth z")
        ax.set_zlabel("up")
        ax.view_init(elev=self.view[0], azim=self.view[1])
        self.fig.suptitle(
            f"FOV {FOVS[col]}°  ({col + 1}/{len(FOVS)})     start frame {self.starts[row]}  ({row + 1}/{len(self.starts)})\n"
            f"{summary}\n[Up/Down: start frame   Left/Right: FOV   drag: rotate]",
            fontsize=10,
        )
        self.fig.canvas.draw_idle()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", nargs="?", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--seed", type=int, default=None, help="random seed for the start frames (default: new each run, printed)")
    parser.add_argument("--scale", type=float, default=0.5, help="resize factor before computing flow (default: 0.5)")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    video = args.video.expanduser().resolve()
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {video}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < SCENES * (FRAMES_FORWARD + 1):
        raise SystemExit("Video is too short")

    seed = args.seed if args.seed is not None else random.randrange(1 << 30)
    starts = sorted(random.Random(seed).sample(range(0, total - FRAMES_FORWARD), SCENES))
    print(f"seed {seed}, start frames {starts} (the same for every FOV)")

    images: list[tuple[np.ndarray, np.ndarray]] = []
    maps: list[list[TwoViewMap | None]] = []
    limits: list[tuple[np.ndarray, np.ndarray]] = []
    for start in starts:
        frames = read_window(capture, start, FRAMES_FORWARD + 1, args.scale)
        if len(frames) < FRAMES_FORWARD + 1:
            raise SystemExit(f"Could not read {FRAMES_FORWARD + 1} frames from frame {start}")
        grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
        height, width = grays[0].shape
        base, positions = accumulate_grid(grays)  # chained flow, independent of the FOV

        row = [
            build_map(FRAMES_FORWARD + 1, base, positions[-1], camera_matrix(width, height, fov), width, height)
            for fov in FOVS
        ]
        built = [m for m in row if m is not None]
        if not built:
            raise SystemExit(f"No FOV produced a map for start frame {start}")
        maps.append(row)
        limits.append(shared_limits(built))
        images.append((cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB), cv2.cvtColor(frames[-1], cv2.COLOR_BGR2RGB)))
        print(f"frame {start:>6}: " + "  ".join(f"{fov}°:{'-' if m is None else len(m.points)}" for fov, m in zip(FOVS, row)) + "   (3D points)")
    capture.release()

    import matplotlib

    matplotlib.use("QtAgg")
    import matplotlib.pyplot as plt

    browser = SweepBrowser(starts, images, maps, limits, plt)
    browser.show()
    plt.show()


if __name__ == "__main__":
    main()
