import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402,F401

from car.params import G, Tuning  # noqa: E402
from car.physics import CarSim  # noqa: E402
from ground import SECTIONS, Ground  # noqa: E402


@pytest.fixture(scope="module")
def ground():
    return Ground(0)


def drive(sim: CarSim, seconds: float, throttle: float = 0.0, steer: float = 0.0, every=None):
    sim.controls.throttle, sim.controls.steer = throttle, steer
    out = []
    for _ in range(int(round(seconds * 60))):
        sim.advance(1.0 / 60.0)
        if every is not None:
            out.append(every(sim))
    return out


def rolling(sim: CarSim, x: float, speed: float) -> CarSim:
    sim.place(x, 0.0, 0.0)
    sim.vel = sim.forward * speed
    sim.spin[:] = speed / sim.params.wheel_radius
    return sim


def test_car_settles_at_mid_travel_carrying_its_weight(ground):
    sim = CarSim(ground, x=0.0)
    drive(sim, 2.0)
    assert np.allclose(sim.s, sim.mid, atol=3e-3)
    assert sim.telemetry.load.sum() == pytest.approx(sim.params.mass * G, rel=0.01)
    assert np.linalg.norm(sim.vel) < 0.01


@pytest.mark.parametrize("drivetrain", ["awd", "fwd", "rwd"])
def test_each_drivetrain_drives_only_its_wheels(ground, drivetrain):
    sim = CarSim(ground, x=-35.0)
    sim.tuning.drivetrain = drivetrain
    drive(sim, 2.0, throttle=1.0)
    assert sim.speed > 5.0
    tread = sim.spin * sim.params.wheel_radius / sim.speed
    driven = {"awd": [0, 1, 2, 3], "fwd": [0, 1], "rwd": [2, 3]}[drivetrain]
    free = [i for i in range(4) if i not in driven]
    assert (tread[driven] > 1.005).all()  # a driven tyre turns a little faster than the ground goes by
    assert np.allclose(tread[free], 1.0, atol=0.01)


def test_more_power_accelerates_harder(ground):
    speeds = []
    for power in (0.5, 1.0):
        sim = CarSim(ground, x=-35.0)
        sim.tuning.power = power
        drive(sim, 1.5, throttle=1.0)
        speeds.append(sim.speed)
    assert speeds[1] > speeds[0] * 1.5


def test_suspension_works_over_the_sine_waves_without_bottoming(ground):
    section = SECTIONS[0]
    sim = rolling(CarSim(ground), section.x0 - 5.0, 9.0)
    travel = np.array(drive(sim, 4.0, throttle=0.2, every=lambda s: s.s.copy()))
    swing = travel.max(axis=0) - travel.min(axis=0)
    assert (swing > 0.04).all()  # every wheel moves up and down by centimetres
    assert travel.max() < sim.params.travel and travel.min() > 0.0


def test_stiffer_springs_ride_harder_and_less_damping_lets_the_wheels_swing(ground):
    hump = next(s for s in SECTIONS if s.name == "Speed humps")

    def run(stiffness, damping):
        sim = rolling(CarSim(ground), hump.x0 - 3.0, 8.0)
        sim.tuning.stiffness, sim.tuning.damping = stiffness, damping
        samples = np.array(drive(sim, 2.0, throttle=0.25, every=lambda s: (np.abs(s.s - s.mid).max(), s.telemetry.body_acceleration[1] - G)))
        return samples[:, 0].max(), np.abs(samples[:, 1]).max()

    _, soft = run(0.5, 1.0)
    _, stiff = run(2.0, 1.0)
    assert stiff > soft * 1.5  # m/s^2 of the body's worst jolt
    loose, _ = run(1.0, 0.2)
    tight, _ = run(1.0, 2.5)
    assert loose > tight * 1.8  # m of the largest suspension movement


def test_less_grip_means_less_cornering(ground):
    lateral = []
    for grip in (0.5, 1.0):
        sim = rolling(CarSim(ground), -35.0, 9.0)
        sim.tuning.grip = grip
        sim.controls.assist = False
        g = drive(sim, 3.0, throttle=0.3, steer=1.0, every=lambda s: abs(s.telemetry.body_acceleration[2]))
        lateral.append(np.mean(g[-60:]))
    assert lateral[0] < lateral[1] * 0.75


def test_assist_holds_a_power_slide_without_spinning(ground):
    sim = CarSim(ground, x=-35.0)
    sim.tuning.drivetrain, sim.tuning.power = "rwd", 3.0
    drive(sim, 3.5, throttle=1.0)
    angles = drive(sim, 5.0, throttle=1.0, steer=1.0, every=lambda s: s.slip_angle())
    assert max(abs(a) for a in angles) > 20.0  # it does slide
    assert max(abs(a) for a in angles) < 80.0  # but never swaps ends
    assert sim.speed > 3.0


def test_kicker_launches_the_car_and_it_lands_upright(ground):
    kicker = SECTIONS[-1]
    sim = rolling(CarSim(ground), kicker.x0 - 10.0, 13.0)
    airborne = drive(sim, 4.0, throttle=0.3, every=lambda s: not s.telemetry.touching.any())
    assert sum(airborne) > 10  # several frames with all four wheels off the ground
    assert not sim.upside_down
    assert sim.R[1, 1] > 0.95


def test_tuning_limits():
    t = Tuning()
    for _ in range(50):
        t.nudge("power", 1)
    assert t.power == Tuning.LIMITS["power"][1]
    t.cycle_drivetrain()
    assert t.drivetrain == "fwd"
    t.reset()
    assert (t.drivetrain, t.power) == ("awd", 1.0)
