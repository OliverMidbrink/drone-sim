"""Puts the code folders on sys.path so modules keep importing each other by plain name.

    virtual-sim/
      fpv.py, simulator.py     entry points (the FPV / SLAM drone app; the plain world viewer)
      camera, render, terrain, trees, gltf, world      the world and its renderer
      flight/    drone physics, collision, spawn, IMU/barometer model, flight controller, route planner, visual-inertial filter
      slam/      the gyro-aided SLAM and the worker processes that run it
      viewer/    fisheye lens + camera, HUD, 3D map view, the drone's 3D body
      tools/     experiments and diagnostics (not part of the sim)
    3D-SLAM/     the raw monocular SLAM this builds on (raw_slam.py)

Import this module first in any entry point: `import paths  # noqa: F401`.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for folder in (ROOT, ROOT / "flight", ROOT / "slam", ROOT / "viewer", ROOT.parent / "3D-SLAM"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
