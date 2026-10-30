"""Walker tests (no GPU needed): python -m pytest office-playground-sim/tests/test_walker.py"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths  # noqa: E402,F401

from office.layout import Box, OfficeLayout  # noqa: E402
from office.walker import Walker  # noqa: E402

FORWARD_X = np.array([1.0, 0.0])


def run(walker: Walker, seconds: float, move=np.zeros(2), sprint=False, jump_first=False) -> None:
    for i in range(int(seconds * 60)):
        walker.update(1 / 60, move, sprint, jump_first and i == 0)


def test_gravity_and_jump_height():
    walker = Walker([Box(5, 5, 6, 6, 3.0)])
    walker.feet = 2.0
    run(walker, 1.0)
    assert walker.feet == 0.0 and walker.grounded
    peak = 0.0
    for i in range(90):
        walker.update(1 / 60, np.zeros(2), False, i == 0)
        peak = max(peak, walker.feet)
    assert 0.8 < peak < 1.1 and walker.grounded


def test_walls_stop_you_and_you_slide_along_them():
    walker = Walker([Box(2.0, -5.0, 2.2, 5.0, 3.0)])
    run(walker, 3.0, FORWARD_X)
    assert walker.position[0] <= 2.0 - Walker.RADIUS + 1e-6
    walker.velocity[:] = 0.0
    run(walker, 1.0, np.array([0.7, 0.7]))  # diagonally into the wall: slides in z
    assert walker.position[1] > 1.0 and walker.position[0] <= 2.0 - Walker.RADIUS + 1e-6


def test_jumping_onto_a_desk_and_walking_off_it():
    desk = Box(2.0, -1.0, 3.0, 1.0, 0.75)
    walker = Walker([desk], 1.0, 0.0)
    run(walker, 1.0, FORWARD_X, jump_first=False)  # too tall to just walk into
    assert walker.feet == 0.0 and walker.position[0] < 2.0
    walker.velocity[:] = 0.0
    for i in range(45):  # the jump, and a little walk across the top
        walker.update(1 / 60, FORWARD_X, False, i == 0)
    assert abs(walker.feet - 0.75) < 1e-6 and walker.position[0] > 2.0
    run(walker, 2.0, FORWARD_X)  # off the far side and down
    assert walker.feet == 0.0


def test_low_obstacles_are_stepped_over():
    walker = Walker([Box(2.0, -1.0, 2.2, 1.0, 0.1)])
    run(walker, 2.0, FORWARD_X)
    assert walker.position[0] > 2.3


def test_cannot_leave_the_office_or_pass_through_furniture():
    layout = OfficeLayout(0)
    walker = Walker(layout.boxes, 1.2, 10.5)
    rng = np.random.default_rng(0)
    for _ in range(40):
        direction = rng.normal(size=2)
        run(walker, 1.0, direction / np.linalg.norm(direction), sprint=True, jump_first=bool(rng.integers(2)))
        x, z = walker.position
        assert -0.2 < x < 18.2 and -0.2 < z < 12.2
        for b in layout.boxes:
            if b.height > walker.feet + Walker.STEP + 0.01:
                inside = b.x_min + 0.02 < x < b.x_max - 0.02 and b.z_min + 0.02 < z < b.z_max - 0.02
                assert not inside


def test_mesh_boxes_follow_the_shape_of_a_model():
    from office import geometry
    from office.collision import ShapeCache, mesh_boxes

    white = (1.0, 1.0, 1.0)
    table = [geometry.box((0, 0.725, 0), (1.0, 0.05, 0.6), white)] + [geometry.box((sx * 0.45, 0.35, sz * 0.25), (0.05, 0.7, 0.05), white) for sx in (-1, 1) for sz in (-1, 1)]
    boxes = mesh_boxes(ShapeCache().get("table", table), np.eye(4))
    top = max(b[5] for b in boxes)
    assert 0.7 < top <= 0.8 + 1e-9  # within a cell of the real 0.75
    slab = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    assert slab[4] > 0.6  # the biggest piece is the top, with air under it
    assert sum(1 for b in boxes if b[4] == 0.0) == 4  # four legs on the floor
    under = [b for b in boxes if b[4] == 0.0 and abs((b[0] + b[2]) / 2) < 0.3]
    assert not under  # nothing in the middle below the top
    rotated = np.eye(4)
    rotated[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]  # a quarter turn
    turned = mesh_boxes(ShapeCache().get("table", table), rotated)
    assert max(b[3] for b in turned) - min(b[1] for b in turned) > 0.9 and max(b[2] for b in turned) - min(b[0] for b in turned) < 0.75


def test_walking_through_the_legs_of_a_chair_but_not_through_its_back():
    layout = OfficeLayout(0)
    chairs = [b for b in layout.boxes if 0.3 < b.y_min < 0.6]  # seats: air below them
    assert chairs
    seat = chairs[0]
    walker = Walker(layout.boxes, (seat.x_min + seat.x_max) / 2, (seat.z_min + seat.z_max) / 2)
    run(walker, 0.5)
    assert walker.position is not None
    # the box of a seat blocks a person at body height
    blocker = Walker([seat], seat.x_min - 0.5, (seat.z_min + seat.z_max) / 2)
    run(blocker, 1.0, FORWARD_X)
    assert blocker.position[0] <= seat.x_min - Walker.RADIUS + 1e-6


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_"):
            function()
            print("ok", name)
