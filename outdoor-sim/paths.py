"""Puts this folder and the shared code of the other sims on sys.path.

    outdoor-sim/
      drive.py          entry point: drive the RC car round the proving ground (arrow keys, tuning keys, HUD)
      ground.py         the proving ground: virtual-sim's terrain flattened round a skid pad and a test road with bumps
      track.py          the RC race track (centre-line spline, kerbs, height table) and the lap timer
      view.py           the renderer: virtual-sim's sky, grass terrain and trees plus road, car parts and skid marks
      panel.py          the on-screen panel (speed, tuning keys and values, tyre slip) and suspension charts
      car/              the half-scale RC car: parameters, physics (suspension, wheels, tyres, drivetrain), 3D model
      tests/            car and ground tests
    virtual-sim/        terrain generator, tree scattering, renderer, camera, HUD overlay are reused from here
    office-playground-sim/office/geometry.py    box / cylinder meshes

Import this module first in any entry point: `import paths  # noqa: F401`.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VIRTUAL_SIM = ROOT.parent / "virtual-sim"
OFFICE_SIM = ROOT.parent / "office-playground-sim"
for folder in (ROOT, VIRTUAL_SIM, VIRTUAL_SIM / "viewer", OFFICE_SIM):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
