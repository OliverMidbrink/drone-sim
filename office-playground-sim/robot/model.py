"""The robot drawn from plain geometry: a box with two wheels.

Model frame: x forward, y up, z to the robot's right; the origin is the middle of the axle. The
body is drawn leaning with the pitch about the axle, each wheel spins by its own angle, so the picture
shows exactly what the simulation computes.
"""

from __future__ import annotations

import numpy as np

from office import geometry
from office.geometry import srgb
from office.renderer import DynamicModel, OfficeRenderer

from .params import RobotParams

BODY = srgb(235, 235, 240)
TRIM = srgb(40, 110, 220)
DARK = srgb(30, 32, 36)
TYRE = srgb(22, 22, 24)
HUB = srgb(190, 190, 195)
MARK = srgb(255, 190, 40)
SENSOR = srgb(220, 60, 50)


class RobotModel:
    """Three dynamic models: the body and the two wheels. `matrices` turns a plant state into their world matrices."""

    def __init__(self, renderer: OfficeRenderer, params: RobotParams) -> None:
        self.params = params
        r, w = params.wheel_radius, 0.045
        depth, height = params.body_size[0], params.body_size[2]
        width = 0.7 * params.track  # the box between the wheels
        base = params.com_height - height / 2.0  # the underside of the body above the axle; the box is centred on the centre of mass
        top = base + height
        body = [
            geometry.box((0, base + height * 0.5, 0), (depth, height, width), BODY),  # the chassis box
            geometry.box((0, top + 0.006, 0), (depth + 0.02, 0.012, width + 0.02), TRIM),  # top plate
            geometry.box((depth / 2 + 0.003, base + height * 0.72, 0), (0.006, height * 0.16, width * 0.7), DARK),  # front panel
            geometry.box((depth / 2 + 0.007, base + height * 0.72, -width * 0.2), (0.006, 0.035, 0.035), SENSOR),  # the two "eyes"
            geometry.box((depth / 2 + 0.007, base + height * 0.72, width * 0.2), (0.006, 0.035, 0.035), SENSOR),
            geometry.box((0, base + height * 0.30, 0), (depth + 0.008, 0.03, width + 0.008), TRIM),  # a stripe round the lower part
            geometry.box((0, base + height * 0.52, 0), (depth + 0.008, 0.03, width + 0.008), TRIM),  # and one higher up
            geometry.cylinder((0, top + 0.1, 0), 0.008, 0.2, DARK),  # an antenna
            geometry.cylinder((0, top + 0.205, 0), 0.02, 0.025, MARK),
        ]
        motor_length = 0.025
        for side in (-1.0, 1.0):
            z = side * (width / 2 + motor_length / 2)
            body.append(geometry.cylinder((0, 0, z), 0.04, motor_length, srgb(150, 150, 155), axis=2))  # the gearmotors
            body.append(geometry.box((0, 0.5 * base, side * width * 0.5 * 0.9), (depth * 0.7, base, 0.03), DARK))  # their brackets
        wheel = [geometry.cylinder((0, 0, 0), r, w, TYRE, axis=2, segments=48)]
        wheel.append(geometry.cylinder((0, 0, 0), r * 0.62, w + 0.003, HUB, axis=2, segments=36))
        for spoke in range(3):  # three marks so the rolling can be seen
            angle = spoke * 2.0 * np.pi / 3.0
            wheel.append(geometry.box((np.cos(angle) * r * 0.38, np.sin(angle) * r * 0.38, 0), (0.025, 0.025, w + 0.008), MARK if spoke == 0 else DARK))
        self.body: DynamicModel = renderer.make_model(body)
        self.wheel: DynamicModel = renderer.make_model(wheel)

    def draw_list(self, plant) -> list[tuple[DynamicModel, np.ndarray]]:
        """(model, world matrix) pairs for the current plant state."""
        p = self.params
        psi, theta = plant.heading, plant.pitch
        c, s = np.cos(psi), np.sin(psi)
        heading = np.array([[c, 0, -s, 0], [0, 1, 0, 0], [s, 0, c, 0], [0, 0, 0, 1]], float)  # model x -> (cos psi, sin psi) in (x, z)

        def lean(angle: float) -> np.ndarray:
            ca, sa = np.cos(angle), np.sin(angle)
            return np.array([[ca, sa, 0, 0], [-sa, ca, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], float)  # y towards +x for a positive angle

        def move(x: float, y: float, z: float) -> np.ndarray:
            m = np.eye(4)
            m[:3, 3] = (x, y, z)
            return m

        axle = move(plant.x, p.wheel_radius, plant.z) @ heading
        result = [(self.body, axle @ lean(theta))]
        for index, side in enumerate((-1.0, 1.0)):
            result.append((self.wheel, axle @ move(0.0, 0.0, side * p.track / 2.0) @ lean(float(plant.q[3 + index]))))
        return result
