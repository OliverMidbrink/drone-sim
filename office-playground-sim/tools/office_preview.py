"""Screenshots of the office: a plan view with the ceiling off and a few views from inside.

    python office-playground-sim/tools/office_preview.py [--seed 0]      # out/office_*.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import glfw
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths  # noqa: E402,F401

from camera import Camera  # noqa: E402
from office.layout import OfficeLayout  # noqa: E402
from office.renderer import OfficeRenderer  # noqa: E402
from render import OffscreenTarget, create_hidden_context  # noqa: E402

VIEWS = {
    "plan": ((9.0, 20.5, 6.0), 0.0, -89.0, 55.0),
    "open_office": ((0.7, 1.5, 4.5), 90.0, -12.0, 80.0),
    "desk": ((3.6, 1.3, 3.9), 20.0, -20.0, 70.0),
    "corridor": ((0.7, 1.2, 10.5), 90.0, -4.0, 80.0),
    "conference": ((12.6, 1.6, 0.6), 130.0, -15.0, 80.0),
    "lounge": ((12.4, 1.6, 9.4), 130.0, -12.0, 80.0),
    "server": ((12.4, 1.5, 6.3), 130.0, -10.0, 80.0),
    "manager": ((9.3, 1.6, 0.4), 130.0, -18.0, 80.0),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, nargs=2, default=(1280, 800))
    parser.add_argument("--views", nargs="*", default=list(VIEWS))
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "out")
    args = parser.parse_args()
    _window, ctx = create_hidden_context()
    layout = OfficeLayout(args.seed)
    renderer = OfficeRenderer(ctx)
    renderer.set_static(layout.building, "building")
    renderer.set_static(layout.ceiling, "ceiling")
    renderer.set_lights(layout.lights)
    print(f"{layout.items} placed models, {layout.building.triangles} triangles, {len(layout.boxes)} collision boxes, {len(layout.lights)} lights")
    target = OffscreenTarget(ctx, tuple(args.size))
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.views:
        position, heading, pitch, fov = VIEWS[name]
        renderer.hidden_layers = {"ceiling"} if name == "plan" else set()
        yaw = np.radians(heading)  # camera yaw: 0 looks along -z (up the plan), 90 along +x
        camera = Camera(np.array(position), yaw=yaw, pitch=np.radians(pitch), fov_deg=fov)
        renderer.draw(target.framebuffer, target.size, camera)
        path = args.out / f"office_{name}.png"
        cv2.imwrite(str(path), target.read_bgr())
        print(f"Wrote {path}")
    glfw.terminate()


if __name__ == "__main__":
    main()
