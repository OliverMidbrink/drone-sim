"""The car drawn from boxes and cylinders: a red buggy body, rear wing, four wheels and, per wheel, two parallel A-arms with a coil-over damper.

Everything is posed from the simulation's state, so the arms swing, the springs shorten and the
wheels turn and steer exactly as the physics has them.
"""

from __future__ import annotations

import numpy as np

from office import geometry
from office.geometry import srgb

from .params import CarParams
from .physics import SIDE, CarSim

PAINT = srgb(200, 28, 30)
PAINT_DARK = srgb(120, 16, 20)
WHITE = srgb(235, 235, 235)
GLASS = srgb(18, 24, 32)
BLACK = srgb(24, 24, 26)
TYRE = srgb(22, 22, 24)
RIM = srgb(190, 192, 198)
ARM = srgb(70, 74, 84)
SILVER = srgb(205, 208, 214)
SPRING = srgb(250, 190, 20)
LIGHT = srgb(255, 244, 200)
TAIL = srgb(255, 30, 20)
MARK = srgb(250, 190, 20)

UPPER_ARM, LOWER_ARM = 0.075, -0.045  # heights of the two arms relative to the hub at the middle of its travel
ARM_SPREAD = 0.08  # fore-aft spacing of the two legs of an A-arm at the body


def bar_matrix(a: np.ndarray, b: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Matrix that maps a unit bar along x (length 1, centred) to run from a to b."""
    d = b - a
    length = float(np.linalg.norm(d))
    x = d / max(length, 1e-9)
    z = np.cross(x, up)
    z /= max(float(np.linalg.norm(z)), 1e-9)
    y = np.cross(z, x)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x * length, y, z, (a + b) / 2.0
    return m


class CarModel:
    def __init__(self, renderer, params: CarParams) -> None:
        self.p = params
        p = params
        r, w = p.wheel_radius, p.wheel_width
        body = [
            geometry.box((0.0, -0.17, 0.0), (1.85, 0.025, 0.46), BLACK),  # chassis plate
            geometry.box((-0.02, -0.065, 0.0), (1.70, 0.18, 0.50), PAINT),  # tub
            geometry.box((0.82, -0.085, 0.0), (0.34, 0.12, 0.46), PAINT_DARK),  # nose
            geometry.box((0.90, -0.025, 0.0), (0.18, 0.05, 0.44), PAINT),  # hood front
            geometry.box((0.97, -0.13, 0.0), (0.06, 0.07, 0.62), BLACK),  # front bumper
            geometry.box((-0.97, -0.115, 0.0), (0.06, 0.08, 0.62), BLACK),  # rear bumper
            geometry.box((-0.18, 0.115, 0.0), (0.80, 0.18, 0.44), PAINT),  # cabin
            geometry.box((0.215, 0.115, 0.0), (0.02, 0.15, 0.38), GLASS),  # windscreen
            geometry.box((-0.18, 0.115, 0.226), (0.55, 0.12, 0.02), GLASS),  # side windows
            geometry.box((-0.18, 0.115, -0.226), (0.55, 0.12, 0.02), GLASS),
            geometry.box((-0.58, 0.115, 0.0), (0.02, 0.12, 0.34), GLASS),  # rear window
            geometry.box((-0.18, 0.207, 0.0), (0.80, 0.006, 0.12), WHITE),  # roof stripe
            geometry.box((0.35, 0.026, 0.0), (0.46, 0.006, 0.12), WHITE),  # bonnet stripe
            geometry.box((-0.02, -0.065, 0.2535), (1.4, 0.04, 0.007), WHITE),  # side stripes
            geometry.box((-0.02, -0.065, -0.2535), (1.4, 0.04, 0.007), WHITE),
            geometry.box((0.995, -0.05, 0.15), (0.012, 0.045, 0.12), LIGHT),  # headlights
            geometry.box((0.995, -0.05, -0.15), (0.012, 0.045, 0.12), LIGHT),
            geometry.box((-0.973, -0.095, 0.2), (0.012, 0.04, 0.14), TAIL),  # tail lights
            geometry.box((-0.973, -0.095, -0.2), (0.012, 0.04, 0.14), TAIL),
            geometry.box((-0.86, 0.12, 0.22), (0.035, 0.30, 0.035), BLACK),  # wing posts
            geometry.box((-0.86, 0.12, -0.22), (0.035, 0.30, 0.035), BLACK),
            geometry.box((-0.9, 0.275, 0.0), (0.26, 0.02, 0.84), PAINT_DARK),  # wing
            geometry.box((-0.9, 0.3, 0.42), (0.26, 0.09, 0.015), PAINT_DARK),
            geometry.box((-0.9, 0.3, -0.42), (0.26, 0.09, 0.015), PAINT_DARK),
            geometry.cylinder((-0.7, 0.35, -0.17), 0.004, 0.5, BLACK, axis=1, segments=6, caps=False),  # RC aerial
            geometry.cylinder((-0.7, 0.61, -0.17), 0.012, 0.03, MARK, axis=1, segments=8),
        ]
        wheel = [
            geometry.cylinder((0, 0, 0), r, w, TYRE, axis=2, segments=48),
            geometry.cylinder((0, 0, 0), r * 0.64, w + 0.004, RIM, axis=2, segments=36),
            geometry.cylinder((0, 0, 0), r * 0.18, w + 0.012, SILVER, axis=2, segments=12),
        ]
        for spoke in range(5):
            angle = spoke * 2.0 * np.pi / 5.0
            wheel.append(geometry.box((np.cos(angle) * r * 0.42, np.sin(angle) * r * 0.42, 0.0), (0.03, 0.03, w + 0.01), MARK if spoke == 0 else BLACK))
        self.body = renderer.make_part(body)
        self.wheel = renderer.make_part(wheel)
        self.arm = renderer.make_part([geometry.box((0, 0, 0), (1.0, 0.016, 0.026), ARM)])
        self.damper = renderer.make_part([geometry.cylinder((0, 0, 0), 0.011, 1.0, SILVER, axis=0, segments=10)])
        self.spring = renderer.make_part([geometry.cylinder((0, 0, 0), 0.026, 1.0, SPRING, axis=0, segments=14)])

    def draw_list(self, sim: CarSim) -> list[tuple]:
        p = self.p
        R = sim.R
        base = np.eye(4)
        base[:3, :3], base[:3, 3] = R, sim.pos
        parts = [(self.body, base)]
        hub_local, _ = sim.hub_kinematics(sim.s)
        wheels = sim.wheel_matrices()
        for i in range(4):
            parts.append((self.wheel, wheels[i]))
            side = SIDE[i]
            hub = hub_local[i]
            pivot_z = side * (p.track / 2.0 - p.arm_length)  # where the arms are hinged on the body
            for height, spread in ((UPPER_ARM, ARM_SPREAD), (LOWER_ARM, ARM_SPREAD * 1.4)):
                for fore_aft in (-1.0, 1.0):
                    inner = np.array([hub[0] + fore_aft * spread, hub[1] + height, pivot_z])
                    outer = np.array([hub[0], hub[1] + height, hub[2]])
                    parts.append((self.arm, base @ bar_matrix(inner, outer, np.array([0.0, 1.0, 0.0]))))
            # coil-over: from the body to the lower arm, 70 % spring
            top = np.array([hub[0], p.hub_y + 0.16, side * (p.track / 2.0 - p.arm_length + 0.06)])
            bottom = np.array([hub[0], hub[1] + LOWER_ARM, hub[2] - side * p.arm_length * 0.35])  # a third of the way along the lower arm
            parts.append((self.damper, base @ bar_matrix(top, bottom, np.array([1.0, 0.0, 0.0]))))
            spring_end = bottom + (top - bottom) * 0.7
            parts.append((self.spring, base @ bar_matrix(bottom, spring_end, np.array([1.0, 0.0, 0.0]))))
        return parts
