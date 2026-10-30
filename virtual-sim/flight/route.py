"""Waypoint route planner: smooth, time-efficient legs between a few points, with the acceleration to fly them.

Each leg is a straight line flown rest to rest on a minimum-jerk profile

    s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5,   tau = t / T

whose peak speed is 1.875 D / T and peak acceleration 5.77 D / T^2 for a leg of length D. The duration T is the
shortest that keeps both below the limits, i.e. as fast as the speed and acceleration (= tilt) budget allows,
and the jerk stays finite, so the attitude controller is never asked for an instant tilt change. The planner
only knows positions in the estimator's world and the drone's limits; the flight controller turns the
(position, velocity, acceleration) it returns into thrust (see FlightController, "differential flatness").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PEAK_SPEED = 1.875  # peak of the minimum-jerk velocity profile, in D / T
PEAK_ACCEL = 5.7735  # ... and of its acceleration, in D / T^2


@dataclass
class Leg:
    start: np.ndarray
    end: np.ndarray
    duration: float
    dwell: float  # seconds spent stopped at the end
    target: int  # index of the waypoint this leg flies to

    @property
    def total(self) -> float:
        return self.duration + self.dwell


def leg_duration(distance: float, speed: float, accel: float) -> float:
    return max(PEAK_SPEED * distance / speed, float(np.sqrt(PEAK_ACCEL * distance / accel)), 0.2)


class Route:
    """Fly from `start` to waypoint 0, then round all the waypoints again and again (or once, with loop=False)."""

    def __init__(self, waypoints: np.ndarray, start: np.ndarray, t0: float, speed: float, accel: float, dwell: float = 1.0, loop: bool = True) -> None:
        self.waypoints = np.asarray(waypoints, dtype=float)
        self.t0, self.loop = t0, loop
        n = len(self.waypoints)
        self.lead = [self._leg(np.asarray(start, float), self.waypoints[0], 0, speed, accel, dwell)]
        self.cycle = [self._leg(self.waypoints[i], self.waypoints[(i + 1) % n], (i + 1) % n, speed, accel, dwell) for i in range(n)]
        if not loop:
            self.cycle = self.cycle[:-1]
        self.lead_time = sum(leg.total for leg in self.lead)
        self.cycle_time = sum(leg.total for leg in self.cycle)
        self.laps = 0
        self.leg: Leg = self.lead[0]
        self.moving = True

    @staticmethod
    def _leg(a: np.ndarray, b: np.ndarray, target: int, speed: float, accel: float, dwell: float) -> Leg:
        return Leg(a, b, leg_duration(float(np.linalg.norm(b - a)), speed, accel), dwell, target)

    def lap_time(self) -> float:
        return self.cycle_time

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Where the drone should be at time t, how fast it should move and accelerate (world frame)."""
        s = max(t - self.t0, 0.0)
        legs = self.lead
        if s >= self.lead_time:
            s -= self.lead_time
            if self.cycle_time <= 0.0:
                legs, s = [], 0.0
            elif self.loop:
                self.laps = int(s // self.cycle_time)
                s = s % self.cycle_time
                legs = self.cycle
            elif s >= self.cycle_time:
                legs, s = [], 0.0
            else:
                legs = self.cycle
        zero = np.zeros(3)
        if not legs:
            self.moving = False
            return (self.cycle[-1].end if self.cycle else self.lead[-1].end).copy(), zero, zero
        for leg in legs:
            if s < leg.total:
                break
            s -= leg.total
        self.leg = leg
        if s >= leg.duration:
            self.moving = False
            return leg.end.copy(), zero, zero
        self.moving = True
        tau = s / leg.duration
        d = leg.end - leg.start
        position = leg.start + d * (10 * tau**3 - 15 * tau**4 + 6 * tau**5)
        velocity = d * (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / leg.duration
        acceleration = d * (60 * tau - 180 * tau**2 + 120 * tau**3) / leg.duration**2
        return position, velocity, acceleration


def square(center: np.ndarray, side: float, height_step: float = 0.0) -> np.ndarray:
    """Four corners of a square (x, z plane) around `center`; opposite corners differ by `height_step` in height so the route is 3D."""
    h = side / 2.0
    c = np.asarray(center, float)
    return np.array(
        [
            c + [-h, 0.0, h],
            c + [h, 0.0, h],
            c + [h, height_step, -h],
            c + [-h, height_step, -h],
        ]
    )
