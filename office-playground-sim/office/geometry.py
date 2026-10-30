"""Small mesh builders (boxes, cylinders, quads) that return `gltf.Primitive`s, the same type the asset packs load as.

Colours are linear base-colour factors, like glTF's. All shapes are y up, and a primitive carries one
tint, so a model with several colours is a list of primitives.
"""

from __future__ import annotations

import numpy as np

from gltf import Primitive


def _primitive(positions, normals, indices, color) -> Primitive:
    positions = np.asarray(positions, np.float32)
    return Primitive(
        positions,
        np.asarray(normals, np.float32),
        np.zeros((len(positions), 2), np.float32),
        np.asarray(indices, np.uint32).reshape(-1),
        None,
        (*[float(c) for c in color[:3]], 1.0),
        False,
        True,
    )


def srgb(r: int, g: int, b: int) -> tuple[float, float, float]:
    """A colour picked in 0-255 sRGB as the linear factor the renderer wants."""
    return tuple((c / 255.0) ** 2.2 for c in (r, g, b))


_FACES = (  # (normal, u axis, v axis) of a unit cube's faces, counter-clockwise seen from outside
    ((1, 0, 0), (0, 1, 0), (0, 0, -1)),
    ((-1, 0, 0), (0, 1, 0), (0, 0, 1)),
    ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
    ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
    ((0, 0, 1), (0, 1, 0), (1, 0, 0)),
    ((0, 0, -1), (1, 0, 0), (0, 1, 0)),
)


def box(center, size, color) -> Primitive:
    """An axis-aligned box: `center` and `size` in metres."""
    positions, normals, indices = [], [], []
    half = np.asarray(size, float) / 2.0
    for normal, u, v in _FACES:
        n, u, v = (np.array(a, float) for a in (normal, u, v))
        base = len(positions)
        for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            positions.append(np.asarray(center, float) + (n + su * u + sv * v) * half)
            normals.append(n)
        indices += [base, base + 1, base + 2, base, base + 2, base + 3]
    return _primitive(positions, normals, indices, color)


def cylinder(center, radius: float, length: float, color, axis: int = 1, segments: int = 24, caps: bool = True) -> Primitive:
    """A cylinder whose axis runs along x (0), y (1) or z (2) through `center`."""
    along = np.eye(3)[axis]
    a, b = np.eye(3)[(axis + 1) % 3], np.eye(3)[(axis + 2) % 3]
    centre = np.asarray(center, float)
    positions, normals, indices = [], [], []
    for i in range(segments):
        angle = 2.0 * np.pi * i / segments
        radial = np.cos(angle) * a + np.sin(angle) * b
        for end in (-1.0, 1.0):
            positions.append(centre + radial * radius + along * end * length / 2.0)
            normals.append(radial)
    for i in range(segments):
        j = (i + 1) % segments
        indices += [2 * i, 2 * j, 2 * j + 1, 2 * i, 2 * j + 1, 2 * i + 1]
    if caps:
        for end in (-1.0, 1.0):
            first = len(positions)
            positions.append(centre + along * end * length / 2.0)
            normals.append(along * end)
            for i in range(segments):
                angle = 2.0 * np.pi * i / segments
                positions.append(centre + (np.cos(angle) * a + np.sin(angle) * b) * radius + along * end * length / 2.0)
                normals.append(along * end)
            for i in range(segments):
                j = (i + 1) % segments
                indices += [first, first + 1 + i, first + 1 + j] if end > 0 else [first, first + 1 + j, first + 1 + i]
    return _primitive(positions, normals, indices, color)


def quad(corners, color) -> Primitive:
    """One flat quad from four corners in counter-clockwise order (the renderer draws both sides)."""
    corners = np.asarray(corners, float)
    normal = np.cross(corners[1] - corners[0], corners[2] - corners[0])
    normal /= np.linalg.norm(normal)
    return _primitive(corners, np.tile(normal, (4, 1)), [0, 1, 2, 0, 2, 3], color)

