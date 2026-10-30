import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import time
import numpy as np
from terrain import Terrain
from trees import load_models, place_trees
from collision import CollisionWorld
from drone import Quadrotor, QuadParams, attitude_angles
from sensors import SensorSuite
from autopilot import FlightController, PilotInput

terrain = Terrain.generate(3, 2000.0, 401, low=True)
models = load_models()
trees = place_trees(terrain, models, 3, 6000)
t0 = time.perf_counter()
world = CollisionWorld(terrain, trees, models)
print("collision build", round(time.perf_counter() - t0, 3), "s; trunks", len(world.trunks))
quad = Quadrotor()
quad.reset(np.array([0, terrain.height_at(0, 0) + 0.14, 0.0]), 0.0)
sensors = SensorSuite(seed=1)
sensors.reset(quad.velocity)
fc = FlightController(quad.params, quad.mixer_inverse)

script = [  # (seconds, pilot)
    (2.0, PilotInput()),
    (3.0, PilotInput(climb=1.0)),
    (4.0, PilotInput()),
    (3.0, PilotInput(forward=1.0)),
    (4.0, PilotInput()),
    (2.0, PilotInput(right=1.0)),
    (3.0, PilotInput(yaw=1.0)),
    (3.0, PilotInput()),
    (3.0, PilotInput(forward=1.0, sport=True)),
    (3.0, PilotInput()),
    (3.0, PilotInput(climb=-1.0)),
    (4.0, PilotInput()),
]
assist = len(sys.argv) > 1 and sys.argv[1] == "assist"
fc.velocity_assist = assist
dt = 1 / 500
tick = 0
t_cpu = time.perf_counter()
steps = 0
for seconds, pilot in script:
    n = int(seconds / dt)
    worst_att_err = 0.0
    for i in range(n):
        if tick % 2 == 0:
            s = sensors.sample(quad)
            cmd = fc.update(s, pilot, quad.on_ground)
        quad.step(dt, cmd, world, 990.0)
        tick += 1
        steps += 1
        if fc.estimator.initialised:
            true_r, true_p, true_y = attitude_angles(quad.rotation)
            er, ep, ey = attitude_angles(fc.estimator.rotation)
            worst_att_err = max(worst_att_err, abs(np.degrees(er - true_r)), abs(np.degrees(ep - true_p)))
    r, p, y = np.degrees(attitude_angles(quad.rotation))
    agl = quad.position[1] - terrain.height_at(quad.position[0], quad.position[2])
    print(f"t={quad.time:5.1f} pilot={pilot} agl={agl:6.2f} v={np.round(quad.velocity,2)} rpy=({r:6.1f},{p:6.1f},{y:6.1f}) est_err_max={worst_att_err:4.1f}deg crashed={quad.crashed} impacts={len(quad.impacts)} vz_est={fc.estimator.climb_rate:5.2f}")
print("cpu per sim second:", round((time.perf_counter() - t_cpu) / (steps * dt), 3))
