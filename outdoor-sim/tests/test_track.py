import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402,F401

from car.physics import CarSim  # noqa: E402
from ground import ASPHALT, Ground  # noqa: E402
from track import HALF_WIDTH, KERB, KERB_HEIGHT, LapTimer  # noqa: E402


@pytest.fixture(scope="module")
def ground():
    return Ground(0)


def at(track, i, offset):
    p = track.points[i] + track.right[i] * offset
    return np.array([p[0]]), np.array([p[1]])


def test_track_is_a_clear_closed_loop(ground):
    track = ground.track
    assert 300.0 < track.length < 370.0
    assert np.linalg.norm(track.points[-1] - track.points[0]) < 0.5
    n = len(track.points)
    gap = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    apart = np.minimum(gap, n - gap) * 0.25 > 30.0  # pairs of points far apart along the track
    distance = np.linalg.norm(track.points[::4, None] - track.points[None, ::4], axis=2)
    assert distance[apart[::4, ::4]].min() > 2.0 * (HALF_WIDTH + KERB) + 1.0  # no two parts of the track touch


def test_surface_asphalt_kerbs_and_grass(ground):
    track = ground.track
    assert ground.sample(*at(track, 0, 0.0))[0] == pytest.approx(ASPHALT, abs=1e-4)
    assert ground.paved(*at(track, 0, 1.0))[0] == 1.0
    assert ground.sample(*at(track, 0, 10.0))[0] == pytest.approx(0.0, abs=1e-6)
    assert ground.paved(*at(track, 0, 10.0))[0] == 0.0
    corner = int(np.flatnonzero(track.kerb)[30])
    for side in (-1.0, 1.0):
        top = ground.sample(*at(track, corner, side * (HALF_WIDTH + KERB / 2.0)))[0]
        assert top == pytest.approx(ASPHALT + KERB_HEIGHT, abs=2e-3)


def test_lap_timer_needs_the_whole_lap(ground):
    track = ground.track
    timer = LapTimer(track)
    speed = 10.0
    for t in np.arange(0.0, 2.2 * track.length / speed, 0.05):  # start mid-lap, then two crossings of the line
        x, z, _ = track.pose(track.length / 2.0 + speed * t)
        timer.update(float(t), x, z)
    assert timer.last == pytest.approx(track.length / speed, abs=0.2)
    cheat = LapTimer(track)
    for t, s in enumerate(np.concatenate([np.linspace(track.length - 10.0, track.length - 0.5, 20), np.linspace(0.5, 10.0, 20), np.linspace(track.length - 10.0, track.length - 0.5, 20), np.linspace(0.5, 10.0, 20)])):
        x, z, _ = track.pose(s)
        cheat.update(float(t), x, z)
    assert cheat.last is None  # crossing the line twice without going round is not a lap


def test_the_car_can_lap_the_track(ground):
    """A simple driver (aim 7 m ahead along the centre line, hold 9 m/s) gets round, kerbs and all."""
    track = ground.track
    sim = CarSim(ground)
    sim.place(*track.pose(track.length - 6.0))
    timer = LapTimer(track)
    worst_offset = 0.0
    for _ in range(int(60 * 1.4 * track.length / 9.0)):
        x, z = float(sim.pos[0]), float(sim.pos[2])
        s, offset = track.locate(x, z)
        worst_offset = max(worst_offset, abs(offset))
        tx, tz, _ = track.pose(s + 7.0)
        heading = sim.heading
        error = (np.arctan2(tz - z, tx - x) - heading + np.pi) % (2.0 * np.pi) - np.pi
        sim.controls.steer = float(np.clip(2.5 * error, -1.0, 1.0))
        sim.controls.throttle = float(np.clip(0.6 * (9.0 - sim.speed), -0.0, 1.0))
        sim.controls.brake = float(np.clip(0.5 * (sim.speed - 10.0), 0.0, 1.0))
        sim.advance(1.0 / 60.0)
        timer.update(sim.time, float(sim.pos[0]), float(sim.pos[2]))
        if timer.last is not None:
            break
    assert timer.last is not None, f"no lap (got to s = {timer.s})"
    assert worst_offset < HALF_WIDTH + KERB  # never left the track
    assert not sim.upside_down
