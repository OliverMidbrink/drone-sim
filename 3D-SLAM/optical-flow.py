#!/usr/bin/env python3
"""Play a video with dense optical flow (Farneback) visualized alongside."""

import argparse
from pathlib import Path

import cv2
import numpy as np

DEFAULT_VIDEO = (
    Path(__file__).resolve().parent / "video-samples" / "forest-walking.mp4"
)


def flow_to_bgr(flow: np.ndarray) -> np.ndarray:
    """Encode flow direction as hue and magnitude as brightness."""
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = np.uint8(angle * 180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def draw_flow_arrows(image: np.ndarray, flow: np.ndarray, step: int = 16) -> np.ndarray:
    vis = image.copy()
    h, w = flow.shape[:2]
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            fx, fy = flow[y, x]
            x2 = int(round(x + fx))
            y2 = int(round(y + fy))
            cv2.arrowedLine(vis, (x, y), (x2, y2), (0, 255, 0), 1, tipLength=0.25)
    return vis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "video",
        nargs="?",
        type=Path,
        default=DEFAULT_VIDEO,
        help=f"path to input video (default: {DEFAULT_VIDEO})",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="resize factor for processing and display (default: 0.5)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=16,
        help="arrow grid spacing in pixels (default: 16)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    if not video_path.is_file():
        raise SystemExit(f"Video not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    delay = max(1, int(round(1000 / fps)))

    prev_gray = None
    paused = False
    window = "Optical flow (video | flow map) — q quit, space pause"

    print(f"Playing {video_path}")
    print("Controls: q = quit, space = pause/resume")

    while True:
        if not paused:
            ok, frame = capture.read()
            if not ok:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                prev_gray = None
                continue

            if args.scale != 1.0:
                frame = cv2.resize(
                    frame,
                    None,
                    fx=args.scale,
                    fy=args.scale,
                    interpolation=cv2.INTER_AREA,
                )

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is None:
                prev_gray = gray
                flow_vis = np.zeros_like(frame)
                overlay = frame.copy()
            else:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray,
                    gray,
                    None,
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0,
                )
                prev_gray = gray
                flow_vis = flow_to_bgr(flow)
                overlay = draw_flow_arrows(frame, flow, step=args.step)

            combined = np.hstack([overlay, flow_vis])
            cv2.imshow(window, combined)

        key = cv2.waitKey(delay if not paused else 30) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
