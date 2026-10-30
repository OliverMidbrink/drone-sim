"""The office building: rooms, walls, doors, windows, lights, furniture and the collision boxes that go with them.

Plan view (x to the right, z down, metres; every wall panel of the asset pack is 3 m wide, so the plan is a 3 m grid):

     0        3        6        9       12       15       18
   0 +--------+--------+--------+--------+--------+--------+
     |                                   | manager|  conference room
     |                                   +--  ----+                   |
   3 |            open  office           |        door               |
     |   (four-desk pods, partitions,    |  ==    |                   |
   6 |    printers, cabinets, whiteboard)|--------+--------+----------+
     |                                   |              server room  |
   9 +----  ------------  ----  ---------+----  ----------------------+
     |        corridor                            lounge             |
  12 +-----------------------------------------------------------------+

Doorways are 1.12 m wide (the pack's door frame); the corridor is 3 m wide and the open office aisles are
about 0.7 m, the smallest gap a robot has to find its way through. `OfficeLayout(seed)` moves chairs,
bins and desk clutter a little, so every seed is a slightly different office.

Asset conventions (found with tools/office_catalog.py): the main pack's models face +x and are centred on
their footprint; the low poly pack's face +z. `Placer.put` takes the direction a model faces in plan, as the
angle of (cos yaw, sin yaw) in (x, z), the same convention as the robot's heading.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import geometry
from .assets import Asset, AssetLibrary
from .renderer import Light, MeshBuilder

CELL = 3.0
WALL_HEIGHT = 3.0
DOOR_OPENING = 1.12
WALL_THICKNESS = 0.1
SPAWN = (3.0, 10.5, 0.0)  # x, z, heading: in the corridor, facing east


@dataclass
class Box:
    """Axis-aligned obstacle in plan; mirrors robot.sim.Box so the layout does not import the robot."""

    x_min: float
    z_min: float
    x_max: float
    z_max: float
    height: float = 2.5


@dataclass
class Room:
    name: str
    x_min: float
    z_min: float
    x_max: float
    z_max: float
    tint: tuple[float, float, float] = (1.0, 1.0, 1.0)  # floor tint

    def contains(self, x: float, z: float) -> bool:
        return self.x_min <= x <= self.x_max and self.z_min <= z <= self.z_max


ROOMS = (
    Room("open office", 0, 0, 12, 9, (0.80, 0.85, 0.95)),
    Room("corridor", 0, 9, 12, 12, (0.95, 0.92, 0.88)),
    Room("conference room", 12, 0, 18, 6, (0.75, 0.62, 0.55)),
    Room("server room", 12, 6, 18, 9, (0.55, 0.57, 0.60)),
    Room("lounge", 12, 9, 18, 12, (0.85, 0.78, 0.70)),
)

# wall runs: (x0, z0, x1, z1, kinds per 3 m panel). kinds: W wall, D doorway, N window, M wide window
WALLS = (
    (0, 0, 18, 0, "NNNNNN"),  # north, outside
    (0, 12, 18, 12, "MMNNNN"),  # south, outside
    (0, 0, 0, 12, "NWWN"),  # west, outside
    (18, 0, 18, 12, "WWWW"),  # east, outside
    (0, 9, 12, 9, "WDWD"),  # corridor <-> open office
    (12, 9, 18, 9, "DW"),  # lounge <-> server room: the server room door opens onto the lounge
    (12, 0, 12, 9, "WDWW"),  # open office <-> conference room (door), server room
    (12, 6, 18, 6, "WW"),  # conference <-> server room
    (9, 3, 12, 3, "D"),  # the manager's corner office (z 0..3, x 9..12)
    (9, 0, 9, 3, "W"),
)


class Placer:
    """Puts library models into a MeshBuilder at a pose and keeps the footprints of the solid ones as collision boxes."""

    def __init__(self, library: AssetLibrary, builder: MeshBuilder) -> None:
        self.library = library
        self.builder = builder
        self.boxes: list[Box] = []
        self.count = 0

    def put(
        self,
        name: str,
        x: float,
        z: float,
        yaw: float = 0.0,
        y: float | None = None,
        solid: bool = False,
        tint: tuple[float, float, float] = (1.0, 1.0, 1.0),
        scale: float = 1.0,
        run: bool = False,
    ) -> Asset:
        """Place `name` with its front towards `yaw` (plan angle). With `run`, the model's local z axis (a wall's run) points along yaw instead."""
        asset = self.library.get(name)
        if run:
            angle = -yaw + np.pi / 2.0
        else:
            angle = (np.pi / 2.0 - yaw) if name.startswith("lp/") else -yaw
        lift = (-asset.lo[1] * scale) if y is None else y
        cos, sin = np.cos(angle), np.sin(angle)
        matrix = np.eye(4)
        matrix[:3, :3] = scale * np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])
        matrix[:3, 3] = (x, lift, z)
        for primitive in asset.primitives:
            self.builder.add(primitive, matrix, tint)
        self.count += 1
        if solid:
            self.boxes.append(self.footprint(asset, matrix))
        return asset

    @staticmethod
    def footprint(asset: Asset, matrix: np.ndarray) -> Box:
        lo, hi = asset.lo, asset.hi
        corners = np.array([[lx, 0.0, lz] for lx in (lo[0], hi[0]) for lz in (lo[2], hi[2])])
        world = corners @ matrix[:3, :3].T + matrix[:3, 3]
        return Box(float(world[:, 0].min()), float(world[:, 2].min()), float(world[:, 0].max()), float(world[:, 2].max()), float(matrix[1, 3] + (hi[1] - lo[1])))


@dataclass
class OfficeLayout:
    seed: int = 0
    library: AssetLibrary | None = None
    building: MeshBuilder = field(default_factory=MeshBuilder)
    ceiling: MeshBuilder = field(default_factory=MeshBuilder)
    lights: list[Light] = field(default_factory=list)
    boxes: list[Box] = field(default_factory=list)
    rooms: tuple[Room, ...] = ROOMS
    items: int = 0

    def __post_init__(self) -> None:
        self.library = self.library or AssetLibrary()
        self.rng = np.random.default_rng(self.seed)
        self.placer = Placer(self.library, self.building)
        self._floor()
        self._walls()
        self._ceiling()
        self._lights()
        self._open_office()
        self._manager()
        self._conference()
        self._server_room()
        self._lounge()
        self._corridor()
        self.boxes += self.placer.boxes
        self.items = self.placer.count

    # --- shell ---------------------------------------------------------------------------------

    def _floor(self) -> None:
        put = self.placer.put
        for room in self.rooms:
            for i in range(int(round((room.x_max - room.x_min) / CELL))):
                for j in range(int(round((room.z_max - room.z_min) / CELL))):
                    put("Floor", room.x_min + CELL * (i + 0.5), room.z_min + CELL * (j + 0.5), tint=room.tint, y=0.0)
        outside = geometry.srgb(120, 150, 95)  # grass seen through the windows
        self.building.add(geometry.quad([(-60, -0.03, -60), (-60, -0.03, 70), (80, -0.03, 70), (80, -0.03, -60)], outside), np.eye(4), tint=(4.0, 4.0, 4.0))  # lit well above the room lights: it is daylight

    def _ceiling(self) -> None:
        for room in self.rooms:
            for i in range(int(round((room.x_max - room.x_min) / CELL))):
                for j in range(int(round((room.z_max - room.z_min) / CELL))):
                    asset = self.library.get("Ceiling")
                    matrix = np.eye(4)
                    matrix[:3, 3] = (room.x_min + CELL * (i + 0.5), WALL_HEIGHT, room.z_min + CELL * (j + 0.5))
                    for primitive in asset.primitives:
                        self.ceiling.add(primitive, matrix)

    def _walls(self) -> None:
        put = self.placer.put
        for x0, z0, x1, z1, kinds in WALLS:
            length = np.hypot(x1 - x0, z1 - z0)
            direction = np.array([(x1 - x0) / length, (z1 - z0) / length])
            yaw = float(np.arctan2(direction[1], direction[0]))  # the run direction
            for index, kind in enumerate(kinds):
                centre = np.array([x0, z0]) + direction * CELL * (index + 0.5)
                cx, cz = float(centre[0]), float(centre[1])
                exterior = (x0 == x1 and x0 in (0, 18)) or (z0 == z1 and z0 in (0, 12))
                if kind == "W":
                    put("Concrete_Wall" if exterior else "Matte_Wall", cx, cz, yaw, run=True)
                    self._wall_box(cx, cz, direction, 0.0)
                elif kind == "D":
                    put("Matte_Doorway_Drywall_Partition", cx, cz, yaw, run=True)
                    self._wall_box(cx, cz, direction, DOOR_OPENING)
                elif kind == "N":
                    put("Concrete_Wall_Window", cx, cz, yaw, run=True)
                    put("Window_Opening_Type_2", cx, cz, yaw, y=0.97, run=True)
                    self._wall_box(cx, cz, direction, 0.0)
                elif kind == "M":
                    put("Concrete_Wall_Window_Type_1", cx, cz, yaw, run=True)
                    put("Window_Opening_Type_1", cx, cz, yaw, y=0.97, run=True)
                    self._wall_box(cx, cz, direction, 0.0)

    def _wall_box(self, cx: float, cz: float, direction: np.ndarray, opening: float) -> None:
        """Collision boxes of one 3 m panel: the whole panel, or the two stubs either side of a doorway."""
        half = CELL / 2.0
        spans = [(-half, half)] if opening == 0.0 else [(-half, -opening / 2.0), (opening / 2.0, half)]
        t = WALL_THICKNESS / 2.0 + 0.01
        for a, b in spans:
            p0, p1 = np.array([cx, cz]) + direction * a, np.array([cx, cz]) + direction * b
            lo = np.minimum(p0, p1) - (np.abs(direction[::-1]) * t)
            hi = np.maximum(p0, p1) + (np.abs(direction[::-1]) * t)
            self.boxes.append(Box(float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]), WALL_HEIGHT))

    def _lights(self) -> None:
        def add(x: float, z: float, room: Room, strength: float = 1.0, color=(1.0, 0.95, 0.88), reach: float = 4.2) -> None:
            self.lights.append(Light((x, WALL_HEIGHT - 0.12, z), tuple(c * 1.7 * strength for c in color), reach, (room.x_min - 0.1, room.z_min - 0.1, room.x_max + 0.1, room.z_max + 0.1)))

        office, corridor, conference, server, lounge = self.rooms
        for x in (1.5, 4.5, 7.5, 10.5):
            for z in (1.5, 4.5, 7.5):
                if x > 9.0 and z < 3.0:
                    continue
                add(x, z, office)
        add(10.5, 1.5, office, 0.0)  # (the manager's corner is lit by its own lamp below)
        add(10.5, 1.5, Room("manager", 9, 0, 12, 3), 0.9, (1.0, 0.9, 0.75), 3.5)
        for x in (1.5, 4.5, 7.5, 10.5):
            add(x, 10.5, corridor, 0.9)
        for x, z in ((14.0, 2.0), (16.0, 4.0), (14.0, 4.0), (16.0, 2.0)):
            add(x, z, conference, 0.75)
        add(15.0, 7.5, server, 0.9, (0.85, 0.92, 1.0), 4.0)
        for x, z in ((13.5, 10.5), (16.5, 10.5)):
            add(x, z, lounge, 0.7, (1.0, 0.85, 0.7))

    # --- furniture -----------------------------------------------------------------------------

    def _jitter(self, amount: float) -> float:
        return float(self.rng.uniform(-amount, amount))

    def _workstation(self, x: float, z: float, user: float, chair_drift: float = 0.25) -> None:
        """A desk at (x, z) whose user sits on the side `user` (plan angle from the desk to the chair); the desk's front faces the user."""
        put, rng = self.placer.put, self.rng
        along = np.array([np.cos(user + np.pi / 2.0), np.sin(user + np.pi / 2.0)])  # the desk's long side
        away = np.array([np.cos(user), np.sin(user)])  # from the desk towards the user
        centre = np.array([x, z])
        put("lp/office_desk", x, z, user, solid=True)
        desk_top = 0.75
        put("lp/monitor", *(centre - away * 0.2), user, y=desk_top)
        put("lp/keyboard", *(centre + away * 0.1 + along * self._jitter(0.03)), user, y=desk_top)
        if rng.random() < 0.5:
            put("lp/coffee_mug", *(centre + along * (0.55 + self._jitter(0.1)) + away * 0.1), float(rng.uniform(0, 2 * np.pi)), y=desk_top)
        if rng.random() < 0.35:
            put("lp/desk_lamp", *(centre - along * 0.65 - away * 0.2), user, y=desk_top)
        if rng.random() < 0.4:
            put("lp/paper_stack", *(centre - along * 0.5 + away * 0.15), float(rng.uniform(0, 2 * np.pi)), y=desk_top)
        seat = centre + away * (0.75 + chair_drift * rng.random())
        put("lp/office_chair", float(seat[0]), float(seat[1]), user + np.pi + self._jitter(0.6), solid=True)

    def _open_office(self) -> None:
        put = self.placer.put
        for pod_x in (2.1, 6.0):
            for pod_z in (2.2, 6.4):
                for offset in (-0.8, 0.8):
                    self._workstation(pod_x + offset, pod_z - 0.45, -np.pi / 2.0)  # the user sits north of this desk
                    self._workstation(pod_x + offset, pod_z + 0.45, np.pi / 2.0)  # and south of this one
                put("Office_Partition", pod_x, pod_z, np.pi / 2.0, run=True, solid=True)  # a low screen between the two rows
        # the east end: copy room, cabinets
        put("lp/photocopier", 8.2, 1.0, np.pi / 2.0, solid=True)
        put("Standing_Printer", 8.4, 3.0, np.pi, solid=True)
        put("lp/water_cooler", 8.6, 4.4, np.pi, solid=True)
        for k in range(3):
            put("lp/filing_cabinet", 8.7, 5.6 + 0.5 * k, np.pi, solid=True)
        put("lp/bookshelf", 11.7, 7.4, np.pi, solid=True)
        put("lp/bookshelf", 11.7, 8.4, np.pi, solid=True)
        put("Whiteboard", 11.9, 5.0 + 1.0, np.pi, y=0.55)
        put("lp/office_plant", 0.5, 8.5, 0.0, solid=True)
        put("lp/office_plant", 8.6, 8.5, 0.0, solid=True)
        put("lp/trash_bin", 7.4 + self._jitter(0.1), 3.4, 0.0, solid=True)
        put("Locker", 0.4, 4.4, 0.0, solid=True)
        put("Locker", 0.4, 5.0, 0.0, solid=True)
        put("Closet_Type_1", 0.3, 6.6, 0.0, solid=True)

    def _manager(self) -> None:
        put = self.placer.put
        user = np.pi / 2.0 * 3.0  # the manager at the north wall, facing south
        put("Table_Type_1", 10.5, 1.2, 0.0, solid=True)
        put("Office_Chair", 10.5, 0.45, np.pi / 2.0, solid=True)
        put("Visitor_Chair", 10.5, 2.2, -np.pi / 2.0 + self._jitter(0.2), solid=True)
        put("lp/monitor", 10.5, 1.0, np.pi / 2.0, y=0.75)
        put("lp/laptop", 10.8, 1.4, np.pi / 2.0 * 3.0, y=0.75)
        put("lp/bookshelf", 9.35, 1.5, 0.0, solid=True)
        put("lp/office_plant", 11.6, 2.6, 0.0, solid=True)
        _ = user

    def _conference(self) -> None:
        put = self.placer.put
        put("Conference", 15.0, 3.0, 0.0, solid=True)
        for k in range(4):
            z = 1.75 + 0.5 + k * 0.85 - 0.6 + 0.0
            put("Visitor_Chair", 14.0, z, 0.0 + self._jitter(0.12), solid=True)
            put("Visitor_Chair", 16.0, z, np.pi + self._jitter(0.12), solid=True)
        put("Office_Chair", 15.0, 1.1, np.pi / 2.0, solid=True)
        put("Projection_Screen", 17.9, 3.0, np.pi, y=1.0)
        put("Projector", 15.0, 3.0, 0.0, y=2.45 - self.library.get("Projector").lo[1])
        put("Whiteboard", 12.3, 1.5, 0.0, y=0.6)
        put("Closet_Type_2", 17.7, 5.5, np.pi, solid=True)
        put("lp/office_plant", 12.5, 5.5, 0.0, solid=True)
        put("lp/trash_bin", 17.5, 0.5, 0.0, solid=True)

    def _server_room(self) -> None:
        put = self.placer.put
        names = ("Server_Cabinet", "lp/server_rack", "Rack_Type_2", "Server_Cabinet", "lp/server_rack")
        for k, name in enumerate(names):
            put(name, 17.5 if not name.startswith("lp/") else 17.45, 6.4 + 0.55 * k, np.pi, solid=True)
        put("Rack_Type_1", 13.0, 8.6, 0.0, solid=True)
        put("Air_Con", 15.0, 6.2, 0.0, y=2.4)
        put("Safe", 14.2, 6.45, np.pi / 2.0, solid=True)
        put("Computer_Case", 12.5, 6.4, 0.0 + self._jitter(0.5), solid=True)

    def _lounge(self) -> None:
        put = self.placer.put
        put("Sofa", 17.0, 10.5, np.pi, solid=True)
        put("Coffee_Table", 15.2, 10.5, 0.0, solid=True)
        put("lp/office_plant", 12.5, 11.5, 0.0, solid=True)
        put("lp/office_plant", 17.5, 9.4, 0.0, solid=True)
        put("Water_Cooler", 13.0, 9.4, 0.0, solid=True)
        put("Plasma", 12.1, 11.2, 0.0, y=1.0)
        put("lp/trash_bin", 12.6, 9.4, 0.0, solid=True)
        put("Visitor_Chair", 13.7, 10.9, 0.3 + self._jitter(0.2), solid=True)

    def _corridor(self) -> None:
        put = self.placer.put
        for k, x in enumerate((0.6, 1.2, 1.8)):
            put("Locker", x, 11.7, -np.pi / 2.0, solid=True)
        put("lp/office_plant", 7.2, 11.6, 0.0, solid=True)
        put("lp/trash_bin", 8.1 + self._jitter(0.2), 11.7, 0.0, solid=True)
        put("lp/filing_cabinet", 10.8, 11.7, -np.pi / 2.0, solid=True)
        put("lp/whiteboard", 4.5, 11.65, -np.pi / 2.0, solid=True)
        put("lp/wall_clock", 6.0, 8.95, np.pi / 2.0 * 3.0, y=2.3)
        for i, x in enumerate((2.4, 5.7)):
            put("Air_Con", x, 10.5, 0.0, y=2.7)

    # --- queries -------------------------------------------------------------------------------

    def room_at(self, x: float, z: float) -> str:
        for room in self.rooms:
            if room.contains(x, z):
                return room.name
        return "outside"

    def is_free(self, x: float, z: float, radius: float = 0.2) -> bool:
        if not (radius < x < 18.0 - radius and radius < z < 12.0 - radius):
            return False
        return not any(b.x_min - radius < x < b.x_max + radius and b.z_min - radius < z < b.z_max + radius for b in self.boxes)
