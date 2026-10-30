"""Where to start the SLAM drone: a small clearing with forest around it."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from collision import CollisionWorld
from drone import QuadParams
from terrain import Terrain

FREE_RADIUS = 7.0  # m: no tree trunk closer than this to the spawn point
NEAR, FAR = 7.0, 35.0  # m: trees in this ring around the spawn are what the cameras get to look at
HEIGHT_ABOVE_GROUND = 2.0  # m


@dataclass
class Spawn:
    position: np.ndarray  # x, y (altitude), z
    nearest_tree: float  # m
    trees_in_view_range: int


def find_spawn(terrain: Terrain, collisions: CollisionWorld, search_radius: float = 150.0, step: float = 4.0) -> Spawn:
    """Flat ground, at least FREE_RADIUS from any tree, with as many trees as possible between NEAR and FAR metres away.

    A clearing with nothing around it would leave a camera-based position estimate nothing to look at.
    """
    trunks = collisions.trunks[:, :2]
    axis = np.arange(-search_radius, search_radius + step, step)
    gx, gz = np.meshgrid(axis, axis)
    points = np.stack([gx.ravel(), gz.ravel()], axis=1)
    inside = np.abs(points).max(axis=1) < terrain.half - 40.0
    points = points[inside & (np.asarray(terrain.flatness_at(points[:, 0], points[:, 1])) > 0.97)]
    if len(trunks) == 0 or len(points) == 0:
        x, z = 0.0, 0.0
        return Spawn(np.array([x, float(terrain.height_at(x, z)) + HEIGHT_ABOVE_GROUND, z]), np.inf, 0)
    nearest = np.empty(len(points))
    in_range = np.zeros(len(points), int)
    for start in range(0, len(points), 400):
        chunk = points[start : start + 400]
        distance = np.linalg.norm(chunk[:, None, :] - trunks[None, :, :], axis=2)
        nearest[start : start + 400] = distance.min(axis=1)
        in_range[start : start + 400] = ((distance > NEAR) & (distance < FAR)).sum(axis=1)
    free = nearest >= FREE_RADIUS
    if not free.any():
        free = nearest >= nearest.max() - 1e-6
    score = np.where(free, in_range - 2.0 * np.hypot(points[:, 0], points[:, 1]) / search_radius, -np.inf)
    best = int(np.argmax(score))
    x, z = float(points[best, 0]), float(points[best, 1])
    return Spawn(np.array([x, float(terrain.height_at(x, z)) + HEIGHT_ABOVE_GROUND, z]), float(nearest[best]), int(in_range[best]))
