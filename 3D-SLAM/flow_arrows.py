#!/usr/bin/env python3
"""Dense optical flow, accumulated from image 1, shown as a 100 x 100 grid of arrows.

Standalone: plain OpenCV (Farneback dense flow), nothing from the SLAM, no feature tracking.

Image 1 is divided into a 100 x 100 grid. The dense flow between each pair of consecutive
images is chained, so for every grid cell we know where that spot of image 1 is in image k.
You browse the images; on each one, every arrow has its base at the cell centre in image 1
and its tip where that spot is in the image on screen (true pixel scale). Every cell always
gets an arrow, nothing is filtered out.

Browse: Left / Right (or A / D), or the Prev / Next buttons. Q or Esc quits.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_VIDEO = HERE / "video-samples" / "drone-flying-industrial-cut.mp4"

GRID_COLS, GRID_ROWS = 100, 100
# Farneback: pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags
FARNEBACK = (0.5, 5, 21, 3, 7, 1.5, 0)

WINDOW = "Accumulated optical flow"
BAR_HEIGHT = 48
LEFT_KEYS = {ord("a"), 81, 65361, 2424832}
RIGHT_KEYS = {ord("d"), 83, 65363, 2555904}


def load_frames(path: Path, count: int, scale: float) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {path}")
    frames = []
    while len(frames) < count:
        ok, raw = capture.read()
        if not ok:
            break
        if scale != 1.0:
            raw = cv2.resize(raw, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        frames.append(raw)
    capture.release()
    if len(frames) < 2:
        raise SystemExit("Need at least 2 frames")
    return frames


def accumulate_grid(grays: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Follow every grid cell centre of image 1 through the dense flow of each consecutive pair.

    Returns (base, positions): base is (10000, 2) cell centres in image 1 (row-major, i.e. the
    100 x 100 matrix reshaped to a list); positions[k] is (10000, 2), where each of them is in
    image k + 1.
    """
    height, width = grays[0].shape
    xs = (np.arange(GRID_COLS) + 0.5) * width / GRID_COLS
    ys = (np.arange(GRID_ROWS) + 0.5) * height / GRID_ROWS
    grid_x, grid_y = np.meshgrid(xs, ys)  # (rows, cols)
    base = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2).astype(np.float32)

    current = base.copy()
    positions = [current.copy()]
    for previous, following in zip(grays, grays[1:]):
        flow = cv2.calcOpticalFlowFarneback(previous, following, None, *FARNEBACK)
        map_x = current[:, 0].reshape(1, -1).astype(np.float32)
        map_y = current[:, 1].reshape(1, -1).astype(np.float32)
        # the flow field read (bilinearly) at where each spot currently is
        flow_x = cv2.remap(np.ascontiguousarray(flow[..., 0]), map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        flow_y = cv2.remap(np.ascontiguousarray(flow[..., 1]), map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        current = current + np.stack([flow_x[0], flow_y[0]], axis=1)
        positions.append(current.copy())
    return base, positions


def put_text(image: np.ndarray, text: str, origin: tuple[int, int], scale: float = 0.6) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def draw_flow(image: np.ndarray, base: np.ndarray, position: np.ndarray, number: int, total: int) -> np.ndarray:
    """The current image with one arrow per grid cell: base at the image-1 cell centre, tip where it is now."""
    vis = image.copy()
    for origin_f, tip_f in zip(base, position):
        origin = (int(round(float(origin_f[0]))), int(round(float(origin_f[1]))))
        tip = (int(round(float(tip_f[0]))), int(round(float(tip_f[1]))))
        # thin and outline-free: with 10,000 cells anything heavier hides the image
        if np.hypot(tip[0] - origin[0], tip[1] - origin[1]) >= 1.0:
            cv2.arrowedLine(vis, origin, tip, (0, 255, 255), 1, cv2.LINE_AA, tipLength=0.3)
        vis[min(max(origin[1], 0), vis.shape[0] - 1), min(max(origin[0], 0), vis.shape[1] - 1)] = (0, 0, 255)

    put_text(vis, f"image {number} of {total}", (10, 24))
    put_text(vis, "arrow base = where the cell was in image 1, tip = where it is now", (10, 48), 0.5)
    return vis


def draw_bar(width: int, number: int, total: int) -> tuple[np.ndarray, dict[str, tuple[int, int, int, int]]]:
    bar = np.full((BAR_HEIGHT, width, 3), (42, 42, 48), np.uint8)
    button_w, gap = 130, 10
    label_w = 90
    x = (width - (2 * button_w + gap * 2 + label_w)) // 2
    regions: dict[str, tuple[int, int, int, int]] = {}
    for name, text in (("prev", "< Prev"), ("next", "Next >")):
        box = (x, 6, x + button_w, BAR_HEIGHT - 8)
        cv2.rectangle(bar, box[:2], box[2:], (58, 58, 64), -1)
        cv2.rectangle(bar, box[:2], box[2:], (110, 110, 120), 1)
        cv2.putText(bar, text, (x + 28, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 245), 1, cv2.LINE_AA)
        regions[name] = box
        x += button_w + gap
        if name == "prev":
            cv2.putText(bar, f"{number} / {total}", (x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 245), 1, cv2.LINE_AA)
            x += label_w + gap
    return bar, regions


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", nargs="?", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--frames", type=int, default=10, help="number of images to browse (default: 10)")
    parser.add_argument("--scale", type=float, default=0.5, help="resize factor before computing flow (default: 0.5)")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    video = args.video.expanduser().resolve()
    frames = load_frames(video, args.frames, args.scale)
    total = len(frames)
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    height, width = grays[0].shape

    base, positions = accumulate_grid(grays)
    print(f"{total} images, dense flow chained from image 1 for a {GRID_COLS} x {GRID_ROWS} grid")

    state = {"index": 0, "regions": {}}

    def render() -> None:
        i = state["index"]
        vis = draw_flow(frames[i], base, positions[i], i + 1, total)
        bar, state["regions"] = draw_bar(width, i + 1, total)
        cv2.imshow(WINDOW, np.vstack([vis, bar]))

    def step(delta: int) -> None:
        state["index"] = int(np.clip(state["index"] + delta, 0, total - 1))
        render()

    def on_mouse(event: int, x: int, y: int, _flags: int, _param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        y_in_bar = y - height
        for name, (x0, y0, x1, y1) in state["regions"].items():
            if x0 <= x <= x1 and y0 <= y_in_bar <= y1:
                step(-1 if name == "prev" else 1)

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)
    render()
    while True:
        key = cv2.waitKeyEx(30)
        if key >= 0:
            letter = key & 0xFF
            if letter in (ord("q"), 27):
                break
            if key in LEFT_KEYS or letter == ord("a"):
                step(-1)
            elif key in RIGHT_KEYS or letter == ord("d"):
                step(1)
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
