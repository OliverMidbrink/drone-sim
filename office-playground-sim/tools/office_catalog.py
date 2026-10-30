"""Contact sheets of the office asset packs: every model in its own tile with its name, as authored.

    python office-playground-sim/tools/office_catalog.py                  # out/office_catalog_*.png
    python office-playground-sim/tools/office_catalog.py --only Sofa Table_Type_1 lp/bookshelf
    python office-playground-sim/tools/office_catalog.py --yaw 215        # look at the models from the other side

Each model is framed from a three-quarter view next to a red / green / blue marker (x / y / z axes
at the model's origin, a fifth of its size long) so its facing and pivot can be read off. This is how
`layout.Placer` learned which way each model faces. Each tile's caption gives the model's size in metres
(x y z) as authored.
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
from office import geometry  # noqa: E402
from office.assets import AssetLibrary  # noqa: E402
from office.renderer import Light, MeshBuilder, OfficeRenderer  # noqa: E402
from render import OffscreenTarget, create_hidden_context  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", default=None, help="model names to show (default: all)")
    parser.add_argument("--columns", type=int, default=6)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--tile", type=int, default=300, help="tile size in pixels")
    parser.add_argument("--yaw", type=float, default=35.0, help="degrees the camera looks around from straight down -z")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "out")
    args = parser.parse_args()

    _window, ctx = create_hidden_context()
    library = AssetLibrary()
    names = args.only if args.only else library.names()
    renderer = OfficeRenderer(ctx, exposure=1.0, ambient_sky=(0.30, 0.30, 0.32), ambient_ground=(0.20, 0.19, 0.18))
    tile = args.tile
    target = OffscreenTarget(ctx, (tile, tile))
    args.out.mkdir(parents=True, exist_ok=True)
    per_sheet = args.columns * args.rows

    for sheet, start in enumerate(range(0, len(names), per_sheet)):
        image = np.full((args.rows * tile, args.columns * tile, 3), 40, np.uint8)
        for i, name in enumerate(names[start : start + per_sheet]):
            asset = library.get(name)
            builder = MeshBuilder()
            for primitive in asset.primitives:
                builder.add(primitive, np.eye(4))
            size = asset.hi - asset.lo
            radius = max(float(np.linalg.norm(size)) / 2.0, 0.02)
            centre = (asset.lo + asset.hi) / 2.0
            marker = radius * 0.5
            for axis, colour in enumerate(((0.9, 0.05, 0.05), (0.05, 0.8, 0.05), (0.1, 0.2, 1.0))):
                thin = np.where(np.eye(3)[axis] > 0, marker, marker * 0.06)
                builder.add(geometry.box(np.eye(3)[axis] * marker / 2.0, thin, colour), np.eye(4))
            floor = max(radius * 3.0, 0.5)
            y = asset.lo[1] - 0.001
            builder.add(geometry.quad([(-floor, y, -floor), (-floor, y, floor), (floor, y, floor), (floor, y, -floor)], geometry.srgb(120, 120, 124)), np.eye(4))
            renderer.set_static(builder)
            renderer.set_lights([Light((centre[0] + radius, centre[1] + radius * 4, centre[2] + radius * 2), (1.4, 1.4, 1.4), radius * 6)])
            yaw, pitch = np.radians(args.yaw), np.radians(-28.0)
            forward = np.array([np.sin(yaw) * np.cos(pitch), np.sin(pitch), -np.cos(yaw) * np.cos(pitch)])
            camera = Camera(centre - forward * radius * 2.6, yaw=yaw, pitch=pitch, fov_deg=45.0)
            renderer.draw(target.framebuffer, target.size, camera)
            row, column = divmod(i, args.columns)
            image[row * tile : (row + 1) * tile, column * tile : (column + 1) * tile] = target.read_bgr()
            for text, offset, scale, colour in ((name, 18, 0.5, (255, 255, 255)), (f"{size[0]:.2f} {size[1]:.2f} {size[2]:.2f}", tile - 8, 0.42, (160, 220, 255))):
                origin = (column * tile + 6, row * tile + offset)
                cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)
        path = args.out / f"office_catalog_{sheet}.png"
        cv2.imwrite(str(path), image)
        print(f"Wrote {path}")
    glfw.terminate()


if __name__ == "__main__":
    main()
