#!/usr/bin/env python3
"""Raw-feed experiment: one 3D point cloud per frame for the first few frames of a video.

Runs the real SLAM (``SlamSession`` from 3d-slam.py) on the first frames only and, for each
frame, keeps just the points the SLAM triangulated *live* on that frame (the output of its
own ``triangulate_midpoint``). Nothing is accumulated into a map, so every cloud shows what
the feed alone gives you at that instant, expressed in that frame's camera coordinates.

A cloud needs motion, and frame 0 has no predecessor, so frame 0 is the reference and
clouds 1..N come from frames 1..N.

Browse with Left/Right (or the slider). "c" toggles RGB / depth colouring.
Use --save DIR to write PNGs instead of opening a window.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_VIDEO = HERE / "video-samples" / "drone-flying-industrial-cut.mp4"


def _load_slam():
    path = HERE / "3d-slam.py"
    spec = importlib.util.spec_from_file_location("slam_app", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["slam_app"] = module
    spec.loader.exec_module(module)
    return module


slam = _load_slam()

# The SLAM calls triangulate_midpoint (via its module globals) once per pose update. Wrap it
# to keep that live result; the computation itself is untouched.
_captured: dict[str, tuple] = {}
_original_midpoint = slam.triangulate_midpoint


def _recording_midpoint(*args, **kwargs):
    result = _original_midpoint(*args, **kwargs)
    _captured["last"] = result
    return result


slam.triangulate_midpoint = _recording_midpoint


@dataclass
class FrameCloud:
    frame_index: int
    image_rgb: np.ndarray  # processed frame, RGB
    points: np.ndarray  # (N, 3) camera frame: x right, y down, z forward (SLAM units)
    colors: np.ndarray  # (N, 3) RGB 0..1
    pixels: np.ndarray  # (N, 2) where each point was seen in this frame
    angle_deg: np.ndarray  # (N,) triangulation angle of each point
    candidates: int  # pose-inlier matches before the SLAM's own triangulation validity filter
    pose_fixed: bool  # an anchor fix moved the pose during this frame (points then unreliable)


class RecordingSession(slam.SlamSession):
    """SlamSession that remembers, per pose update, the points it triangulated live."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.clouds: list[FrameCloud] = []
        self.current_index = 0

    def _commit_pose(self, frame, gray, motion, key_matched, current_matched, ids_matched, match_rows):
        fixes_before = self.stats["fixes"]
        _captured.pop("last", None)
        super()._commit_pose(frame, gray, motion, key_matched, current_matched, ids_matched, match_rows)
        if "last" not in _captured:
            return
        points_world, angle_deg, usable = _captured.pop("last")
        pixels = current_matched[motion[2]]
        camera = (points_world[usable] - self.world_position) @ self.world_rotation
        self.clouds.append(
            FrameCloud(
                frame_index=self.current_index,
                image_rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                points=camera,
                colors=slam.sample_colors(frame, pixels[usable]),
                pixels=pixels[usable],
                angle_deg=angle_deg[usable],
                candidates=int(len(points_world)),
                pose_fixed=self.stats["fixes"] != fixes_before,
            )
        )


def run_experiment(args: argparse.Namespace) -> list[FrameCloud]:
    slam_args = slam.parse_args([str(args.video), "--no-disk-cache", "--scale", str(args.scale)])
    slam.resolve_exclusions(slam_args)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frames = []
    while len(frames) < args.clouds + 1:
        ok, raw = capture.read()
        if not ok:
            break
        frames.append(slam.resize_frame(raw, slam_args.scale))
    capture.release()
    if len(frames) < 2:
        raise SystemExit("Not enough frames")

    session = RecordingSession(frames[0], slam_args, fps=fps)
    print(
        f"Profile {slam_args.profile_name}, {session.width}x{session.height}, "
        f"fx={session.intrinsic[0, 0]:.0f}px, {len(session.track_points)} initial tracks"
    )
    for i, frame in enumerate(frames[1:], start=1):
        session.current_index = i
        before = len(session.clouds)
        session.process(frame)
        if len(session.clouds) == before:
            print(f"  frame {i}: SLAM made no pose update, no cloud")
    return session.clouds


def depth_limits(clouds: list[FrameCloud]) -> tuple[np.ndarray, np.ndarray]:
    """Shared axis limits (robust percentiles) so clouds stay comparable while browsing."""
    allp = np.vstack([c.points for c in clouds if len(c.points)])
    allp = allp[allp[:, 2] > 0]
    lo = np.percentile(allp, 2, axis=0)
    hi = np.percentile(allp, 98, axis=0)
    lo[2] = 0.0
    return lo, hi


def summarize(clouds: list[FrameCloud], lo: np.ndarray, hi: np.ndarray) -> None:
    print(f"\n{'cloud':>5} {'points':>7} {'in view':>8} {'behind':>7} {'median z':>9} {'z p10-p90':>16} {'angle med':>10}")
    for n, c in enumerate(clouds, start=1):
        z = c.points[:, 2]
        inside = np.all((c.points >= lo) & (c.points <= hi), axis=1)
        p10, p90 = np.percentile(z, [10, 90]) if len(z) else (np.nan, np.nan)
        print(
            f"{n:>5} {len(c.points):>7} {inside.sum():>8} {(z <= 0).sum():>7} "
            f"{np.median(z):>9.3f} {p10:>7.2f}-{p90:<8.2f} {np.median(c.angle_deg):>9.2f}°"
        )


class CloudBrowser:
    def __init__(self, clouds: list[FrameCloud], lo: np.ndarray, hi: np.ndarray, plt) -> None:
        self.clouds, self.lo, self.hi, self.plt = clouds, lo, hi, plt
        self.index = 0
        self.depth_colors = True
        self.slider = None
        self._updating = False

        self.fig = plt.figure(figsize=(17, 8.5))
        self.fig.subplots_adjust(left=0.02, right=0.99, top=0.88, bottom=0.07)
        grid = self.fig.add_gridspec(2, 2, width_ratios=[1, 1.5], height_ratios=[1, 0.05], hspace=0.08, wspace=0.03)
        self.ax_img = self.fig.add_subplot(grid[0, 0])
        self.ax3d = self.fig.add_subplot(grid[0, 1], projection="3d")
        self.ax_slider = self.fig.add_subplot(grid[1, :])
        self.view = (20.0, -70.0)

    def attach(self, interactive: bool) -> None:
        if not interactive:
            self.ax_slider.axis("off")
            return
        from matplotlib.widgets import Slider

        self.slider = Slider(
            self.ax_slider, "cloud", 1, len(self.clouds), valinit=1, valstep=1, valfmt="%d"
        )
        self.slider.on_changed(lambda v: self.show(int(v) - 1))
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _on_key(self, event) -> None:
        if event.key in ("right", "d"):
            self.set_index(self.index + 1)
        elif event.key in ("left", "a"):
            self.set_index(self.index - 1)
        elif event.key == "c":
            self.depth_colors = not self.depth_colors
            self.show(self.index)

    def set_index(self, index: int) -> None:
        index = int(np.clip(index, 0, len(self.clouds) - 1))
        if self.slider is not None and int(self.slider.val) != index + 1:
            self.slider.set_val(index + 1)  # triggers show()
        else:
            self.show(index)

    def show(self, index: int) -> None:
        if self.ax3d.elev is not None:
            self.view = (self.ax3d.elev, self.ax3d.azim)  # keep the user's rotation across clouds
        self.index = index
        c = self.clouds[index]
        lo, hi = self.lo, self.hi
        z = c.points[:, 2]
        near, far = 0.0, float(hi[2])
        inside = np.all((c.points >= lo) & (c.points <= hi), axis=1)
        depth_norm = np.clip((z - near) / max(far - near, 1e-9), 0.0, 1.0)

        # --- left: the frame, with where each point was seen, coloured by depth
        ax = self.ax_img
        ax.clear()
        ax.imshow(c.image_rgb)
        ax.scatter(c.pixels[:, 0], c.pixels[:, 1], c=depth_norm, cmap="turbo_r", s=6, linewidths=0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"video frame {c.frame_index}   (dots: red = near, blue = far)", fontsize=9)

        # --- right: the cloud in this frame's camera coordinates
        ax3 = self.ax3d
        ax3.clear()
        p = c.points[inside]
        colors = self.plt.get_cmap("turbo_r")(depth_norm[inside])[:, :3] if self.depth_colors else c.colors[inside]
        # plot x right, z (depth) away from the camera, -y up
        ax3.scatter(p[:, 0], p[:, 2], -p[:, 1], c=colors, s=7, linewidths=0, depthshade=False)
        ax3.scatter([0], [0], [0], c="red", marker="^", s=80, label="camera")
        ax3.set_xlim(lo[0], hi[0])
        ax3.set_ylim(near, far)
        ax3.set_zlim(-hi[1], -lo[1])
        ax3.set_box_aspect((hi[0] - lo[0], far - near, hi[1] - lo[1]))
        ax3.set_xlabel("x (right)")
        ax3.set_ylabel("depth z")
        ax3.set_zlabel("up")
        ax3.view_init(elev=self.view[0], azim=self.view[1])
        offchart = len(c.points) - int(inside.sum())
        ax3.set_title(
            f"cloud {index + 1}/{len(self.clouds)}: {len(c.points)} of {c.candidates} matches kept, {offchart} off-chart "
            f"(median depth {np.median(z):.2f}, median triangulation angle {np.median(c.angle_deg):.1f}°)"
            + ("  [ANCHOR FIX MOVED POSE]" if c.pose_fixed else ""),
            fontsize=9,
        )
        self.fig.suptitle(
            "Raw-feed point cloud, live SLAM depth only, no accumulation   "
            "[Left/Right: browse   c: depth/RGB colours   drag: rotate]",
            fontsize=10,
        )
        self.fig.canvas.draw_idle()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", nargs="?", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--clouds", type=int, default=10, help="number of clouds / frames after the reference (default: 10)")
    parser.add_argument("--scale", type=float, default=0.5, help="processing resize factor (default: 0.5, as the app)")
    parser.add_argument("--save", type=Path, default=None, metavar="DIR", help="write cloud-NN.png files instead of opening a window")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    args.video = args.video.expanduser().resolve()
    clouds = run_experiment(args)
    if not clouds:
        raise SystemExit("The SLAM produced no clouds")
    lo, hi = depth_limits(clouds)
    summarize(clouds, lo, hi)

    import matplotlib

    matplotlib.use("Agg" if args.save else "QtAgg")
    import matplotlib.pyplot as plt

    browser = CloudBrowser(clouds, lo, hi, plt)
    if args.save:
        browser.attach(interactive=False)
        args.save.mkdir(parents=True, exist_ok=True)
        for i in range(len(clouds)):
            browser.show(i)
            out = args.save / f"cloud-{i + 1:02d}.png"
            browser.fig.savefig(out, dpi=90)
            print(f"wrote {out}")
        return
    browser.attach(interactive=True)
    browser.show(0)
    plt.show()


if __name__ == "__main__":
    main()
