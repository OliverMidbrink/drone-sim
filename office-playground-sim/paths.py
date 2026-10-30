"""Puts this folder and the shared rendering code of virtual-sim on sys.path.

    office-playground-sim/
      playground.py     entry point: walk around the office (WASD, mouse, jump) with the balancing robot
      office/           the office: asset pack loading, indoor renderer, the building layout
      robot/            the two-wheeled self-balancing robot: physics, sensors, firmware, 3D model
      tools/            screenshots and the asset catalog
      tests/            robot tests
    virtual-sim/        camera, gltf loader, texture helpers and the HUD overlay are borrowed from here

Import this module first in any entry point: `import paths  # noqa: F401`.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VIRTUAL_SIM = ROOT.parent / "virtual-sim"
for folder in (ROOT, VIRTUAL_SIM, VIRTUAL_SIM / "viewer"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
