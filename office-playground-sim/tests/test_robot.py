"""Robot tests: python -m pytest office-playground-sim/tests  (or run this file directly)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths  # noqa: E402,F401

from robot.dynamics import TwipPlant  # noqa: E402
from robot.firmware import Command, design_lqr, linear_pitch_model  # noqa: E402
from robot.params import RobotParams  # noqa: E402
from robot.sensors import ImuSpec  # noqa: E402
from robot.sim import Box, RobotSim  # noqa: E402


def stand(seed: int = 0, seconds: float = 6.0) -> RobotSim:
    sim = RobotSim(seed=seed)
    sim.advance(seconds)
    return sim


def test_open_loop_robot_is_unstable():
    eigenvalues = np.linalg.eigvals(linear_pitch_model(RobotParams())[0])
    assert eigenvalues.real.max() > 3.0  # an inverted pendulum falls with a time constant of about 0.15 s


def test_lqr_stabilises_the_linear_model():
    from robot.firmware import discretize

    p = RobotParams()
    a, b = linear_pitch_model(p)
    gain = design_lqr(p)
    a5 = np.zeros((5, 5))
    a5[:4, :4], a5[4, 0] = a, 1.0
    ad, bd = discretize(a5, np.vstack([b, [[0.0]]]), 1.0 / 200.0)
    assert np.abs(np.linalg.eigvals(ad - bd @ gain.reshape(1, 5))).max() < 1.0


def test_unpowered_robot_falls_over():
    plant = TwipPlant(RobotParams(), pitch=np.radians(2.0))
    for _ in range(int(2.0 / 0.0005)):
        plant.step(0.0005, np.zeros(2))
    assert plant.fallen


def test_energy_is_conserved_without_losses():
    p = RobotParams(rolling_resistance=0.0, scrub_torque=0.0)
    p.motor.coulomb_friction = p.motor.gearbox_viscous = p.motor.rotor_friction = 0.0
    plant = TwipPlant(p, pitch=np.radians(5.0))
    plant.step(0.0005, np.zeros(2))

    def energy() -> float:
        m = plant.mass_matrix(plant.pitch)
        com = p.body_mass * 9.81 * plant.com_distance * np.cos(plant.pitch + plant.com_angle)
        return 0.5 * float(plant.qd @ m @ plant.qd) + com

    e0 = energy()
    for _ in range(int(0.25 / 0.0005)):
        plant.step(0.0005, np.zeros(2))
    assert abs(energy() - e0) < 2e-3  # joules


def test_balances_standing_still_for_several_builds():
    for seed in range(4):
        sim = stand(seed, 8.0)
        sim.history.clear()
        sim.advance(4.0)
        pitch = np.degrees([s.pitch for s in sim.history])
        assert not sim.plant.fallen
        assert np.abs(pitch).max() < 6.0
        assert abs(sim.plant.speed) < 0.1


def test_balances_with_ideal_sensors_and_nominal_parameters():
    sim = RobotSim(spread=0.0, imu_spec=ImuSpec().scaled(0.0))
    sim.advance(6.0)
    assert abs(np.degrees(sim.plant.pitch)) < 0.5 and abs(sim.plant.x) < 0.3


def test_drives_and_turns():
    sim = stand(1, 4.0)
    sim.command = Command(speed=0.4)
    sim.advance(4.0)
    assert not sim.plant.fallen
    assert 0.25 < sim.plant.speed < 0.6
    sim.command = Command(speed=0.0, yaw_rate=1.0)
    sim.advance(2.0)
    assert not sim.plant.fallen and 0.7 < sim.plant.yaw_rate < 1.3


def test_recovers_from_pushes():
    sim = stand(2, 5.0)
    for force in (2.0, -3.0, 5.0):
        sim.push(force, 0.15)
        sim.advance(4.0)
        assert not sim.plant.fallen and abs(np.degrees(sim.plant.pitch)) < 6.0


def test_falls_on_a_huge_shove_and_gets_up_again():
    sim = stand(0, 4.0)
    sim.push(30.0, 0.2)
    sim.advance(3.0)
    assert sim.plant.fallen and sim.firmware.state == "cutoff"  # the motors are cut when it is down
    sim.stand_up()
    sim.advance(3.0)
    assert not sim.plant.fallen and sim.firmware.state == "balancing"


def test_motor_current_is_limited_and_battery_sags():
    sim = stand(0, 4.0)
    sim.push(8.0, 0.2)
    peak = 0.0
    for _ in range(4000):
        sim.advance(0.0005)
        peak = max(peak, float(np.abs(sim.plant.currents).max()))
    assert peak <= sim.params.motor.stall_current_limit + 1e-6
    assert sim.plant.telemetry.battery_voltage < sim.params.battery.open_circuit(sim.plant.charge) + 1e-6


def test_wall_stops_the_robot():
    sim = RobotSim(seed=1, obstacles=[Box(1.5, -2.0, 1.7, 2.0)])
    sim.advance(3.0)
    sim.command = Command(speed=0.3)
    sim.advance(8.0)
    assert sim.plant.x < 1.5 and sim.collisions > 0 and not sim.plant.fallen


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_"):
            function()
            print("ok", name)
