"""The on-screen display: speed, the keys that tune the car with their current values, a top view of the tyres and suspension charts."""

from __future__ import annotations

import cv2
import numpy as np

from car.params import DRIVETRAINS, Tuning
from hud import FONT, History, draw_graph

WHEEL_COLORS = ((90, 90, 255), (80, 230, 80), (255, 160, 60), (60, 220, 255))  # BGR: FL red, FR green, RL blue, RR yellow
TEXT = (235, 235, 235)
DIM = (150, 150, 150)
ACCENT = (60, 200, 255)

PANEL = (440, 400)
CHARTS = (380, 214)


def _text(canvas, text, org, scale=0.46, color=TEXT, thickness=1):
    cv2.putText(canvas, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def _bar(canvas, x, y, width, fraction, color) -> None:
    cv2.rectangle(canvas, (x, y), (x + width, y + 9), (70, 70, 70), 1)
    cv2.rectangle(canvas, (x + 1, y + 1), (x + 1 + int((width - 2) * float(np.clip(fraction, 0.0, 1.0))), y + 8), color, -1)


def _fraction(value: float, low: float, high: float) -> float:
    return float(np.log(value / low) / np.log(high / low))


def draw_panel(sim, tuning: Tuning, section: str, fps: float, assist: bool) -> np.ndarray:
    """RGBA image of the tuning panel (top left)."""
    width, height = PANEL
    canvas = np.zeros((height, width, 3), np.uint8)
    speed = sim.speed
    _text(canvas, f"{abs(speed) * 3.6:4.0f}", (10, 46), 1.5, TEXT, 2)
    _text(canvas, "km/h" if speed >= -0.3 else "km/h  REVERSE", (118, 46), 0.5, DIM)
    _, pitch, roll = sim.euler()
    _text(canvas, f"slip angle {sim.slip_angle():+5.0f} deg", (230, 22), 0.46)
    _text(canvas, f"lateral {sim.telemetry.body_acceleration[2] / 9.81:+4.1f} g   long {sim.telemetry.body_acceleration[0] / 9.81:+4.1f} g", (230, 42), 0.4, DIM)
    _text(canvas, f"pitch {pitch:+4.1f}  roll {roll:+4.1f} deg    {fps:3.0f} fps", (230, 60), 0.4, DIM)
    _text(canvas, section, (10, 76), 0.42, ACCENT)

    y = 100
    _text(canvas, "[T]", (10, y), 0.46, ACCENT)
    _text(canvas, "Drive", (50, y))
    x = 140
    for name in DRIVETRAINS:
        label = name.upper()
        on = tuning.drivetrain == name
        cv2.rectangle(canvas, (x, y - 15), (x + 62, y + 5), ACCENT if on else (80, 80, 80), -1 if on else 1)
        _text(canvas, label, (x + 8, y), 0.46, (20, 20, 20) if on else DIM)
        x += 72
    rows = (
        ("[1][2]", "Motor power", tuning.power, "power"),
        ("[3][4]", "Tyre grip", tuning.grip, "grip"),
        ("[5][6]", "Damping", tuning.damping, "damping"),
        ("[7][8]", "Spring stiffness", tuning.stiffness, "stiffness"),
    )
    for key, label, value, name in rows:
        y += 30
        low, high = Tuning.LIMITS[name]
        _text(canvas, key, (10, y), 0.46, ACCENT)
        _text(canvas, label, (74, y))
        _bar(canvas, 228, y - 10, 130, _fraction(value, low, high), ACCENT)
        cv2.line(canvas, (228 + int(130 * _fraction(1.0, low, high)), y - 13), (228 + int(130 * _fraction(1.0, low, high)), y + 3), (200, 200, 200), 1)
        _text(canvas, f"{value * 100:4.0f}%", (368, y), 0.46)
    y += 30
    _text(canvas, "[G]", (10, y), 0.46, ACCENT)
    _text(canvas, "Counter-steer assist", (74, y))
    _text(canvas, "ON" if assist else "off", (300, y), 0.46, (110, 230, 110) if assist else DIM)
    y += 24
    _text(canvas, "[0] reset tuning   [R] skid pad   [F] race track   [N][B] test sections", (10, y), 0.38, DIM)
    y += 20
    _text(canvas, "Arrows drive   Space handbrake   C distance   V view   H hide this", (10, y), 0.38, DIM)

    # top view of the car: tyres coloured by how hard they slide, with their load as the length
    cx, cy = 56, height - 52
    cv2.rectangle(canvas, (cx - 14, cy - 40), (cx + 14, cy + 40), (110, 110, 110), 1)
    for i, (dx, dy) in enumerate(((-24, -30), (24, -30), (-24, 30), (24, 30))):
        slip = float(sim.telemetry.slip[i])
        color = (70, 220, 70) if slip < 0.2 else (60, 200, 255) if slip < 0.4 else (60, 70, 255)
        load = float(sim.telemetry.load[i]) / 600.0
        half = int(np.clip(6 + 10 * load, 4, 20))
        cv2.rectangle(canvas, (cx + dx - 6, cy + dy - half), (cx + dx + 6, cy + dy + half), color, -1 if sim.telemetry.touching[i] else 1)
    _text(canvas, "tyres: green grips, red slides", (96, height - 58), 0.38, DIM)
    _text(canvas, "bar length = load", (96, height - 40), 0.38, DIM)
    rgba = np.zeros((height, width, 4), np.uint8)
    rgba[..., :3] = canvas[..., ::-1]
    rgba[..., 3] = np.where(canvas.any(axis=2), 255, 165)
    return rgba


class Charts:
    """Strip charts of what the suspension does."""

    def __init__(self) -> None:
        self.travel = History(4, 240)
        self.accel = History(1, 240)

    def record(self, sim) -> None:
        self.travel.push((sim.telemetry.travel - sim.mid) * 1000.0)
        self.accel.push([sim.telemetry.body_acceleration[1] - 9.81])

    def clear(self) -> None:
        self.travel.clear()
        self.accel.clear()

    def draw(self) -> np.ndarray:
        width, height = CHARTS
        canvas = np.zeros((height, width, 3), np.uint8)
        draw_graph(canvas, (2, 2, width - 2, 118), "suspension travel (+ = compressed)", self.travel, 70.0, "mm", 1.0, WHEEL_COLORS, ["FL ", "FR ", "RL ", "RR "])
        draw_graph(canvas, (2, 122, width - 2, height - 2), "body vertical acceleration", self.accel, 20.0, "m/s2", 1.0, ((255, 255, 255),), "")
        rgba = np.zeros((height, width, 4), np.uint8)
        rgba[..., :3] = canvas[..., ::-1]
        rgba[..., 3] = np.where(canvas.any(axis=2), 255, 165)
        return rgba
