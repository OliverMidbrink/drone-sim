import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402,F401

from ground import ASPHALT, HUMP_HEIGHT, KICKER, POTHOLE_DEPTH, POTHOLES, SECTIONS, SINE_AMPLITUDE, Ground  # noqa: E402


@pytest.fixture(scope="module")
def ground():
    return Ground(0)


def heights(ground, x, z=0.0):
    x = np.atleast_1d(np.asarray(x, float))
    return ground.sample(x, np.full_like(x, z))


def test_pad_and_plain_road_are_flat(ground):
    xs, zs = np.meshgrid(np.linspace(-38, 38, 40), np.linspace(-28, 28, 30))
    assert np.allclose(ground.sample(xs, zs), ASPHALT, atol=1e-6)
    assert np.allclose(heights(ground, np.linspace(42, 55, 50)), ASPHALT, atol=1e-6)
    normal = ground.normal(np.array([0.0]), np.array([0.0]))[0]
    assert normal[1] > 0.9999


def test_sine_waves_have_the_stated_amplitude(ground):
    xs = np.linspace(70, 110, 2000)
    h = heights(ground, xs) - ASPHALT
    assert h.max() == pytest.approx(SINE_AMPLITUDE, abs=2e-3)
    assert h.min() == pytest.approx(-SINE_AMPLITUDE, abs=2e-3)
    assert np.allclose(heights(ground, xs, 2.5), heights(ground, xs, -2.5))  # straight across the road


def test_speed_humps_kicker_and_potholes(ground):
    section = next(s for s in SECTIONS if s.name == "Speed humps")
    humps = heights(ground, np.linspace(section.x0, section.x1, 3000)) - ASPHALT
    assert humps.max() == pytest.approx(HUMP_HEIGHT, abs=3e-3)
    start, length, rise, _ = KICKER
    assert heights(ground, start + length - 0.01)[0] - ASPHALT == pytest.approx(rise, abs=0.02)
    x, z, _ = POTHOLES[0]
    assert ASPHALT - ground.sample(np.array([x]), np.array([z]))[0] == pytest.approx(POTHOLE_DEPTH, abs=3e-3)


def test_paving_and_grass(ground):
    assert ground.paved(np.array([0.0, 200.0]), np.array([0.0, 0.0])).all()
    assert not ground.paved(np.array([0.0, 200.0]), np.array([45.0, 20.0])).any()
