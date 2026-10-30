"""How well does the raw SLAM do on the drone's rectified fisheye pictures? (experiment, not part of the sim)

Flies a scripted wiggle with the stock attitude controller, runs one RawSlam per side camera on a
pinhole picture cut out of the fisheye one, and compares its poses with the truth.

    python virtual-sim/tools/slam_probe.py --seconds 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import paths  # noqa: E402,F401

from lens import GL_TO_CV, pinhole_matrix, rectify_maps  # noqa: E402
from imu_slam import ImuAidedSlam, rotation_vector_to_matrix  # noqa: E402
from raw_slam import RawSlam  # noqa: E402


def umeyama(source: np.ndarray, target: np.ndarray):
    """Scale, rotation, translation that best map source points onto target points."""
    mu_s, mu_t = source.mean(0), target.mean(0)
    s, t = source - mu_s, target - mu_t
    u, d, vt = np.linalg.svd(t.T @ s / len(s))
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2, 2] = -1
    rotation = u @ sign @ vt
    scale = np.trace(np.diag(d) @ sign) / s.var(0).sum()
    return scale, rotation, mu_t - scale * rotation @ mu_s


def rotation_log(r: np.ndarray) -> np.ndarray:
    angle = np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))
    if angle < 1e-9:
        return np.zeros(3)
    return angle / (2 * np.sin(angle)) * np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--cam-size", type=int, default=768)
    parser.add_argument("--cam-face", type=int, default=640)
    parser.add_argument("--slam-size", type=int, default=512)
    parser.add_argument("--slam-fov", type=float, default=110.0)
    parser.add_argument("--amp", type=float, default=0.5, help="stick amplitude of the wiggle")
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--kinematic", type=float, default=None, help="skip the physics: move on a prescribed wobble of this radius (m)")
    parser.add_argument("--no-fixes", action="store_true")
    parser.add_argument("--plain", action="store_true", help="the plain RawSlam instead of the gyro-aided one")
    parser.add_argument("--parallax", type=float, default=None, help="RawSlam keyframe parallax threshold in px")
    options = parser.parse_args()

    import fpv
    from autopilot import PilotInput
    from spawn import find_spawn

    sys.argv = ["fpv", "--seed", str(options.seed), "--size", "1000", "--resolution", "401", "--cam-size", f"{options.cam_size}x{options.cam_size}", "--cam-face", str(options.cam_face)]
    args = fpv.parse_args()
    window, ctx = fpv.create_hidden_context()
    app = fpv.FpvApp(args, options.seed, window, ctx, headless=True)
    spawn = find_spawn(app.world.terrain, app.collisions)
    print(f"spawn {np.round(spawn.position, 1)} nearest tree {spawn.nearest_tree:.1f} m, trees in range {spawn.trees_in_view_range}")
    app.quad.reset(spawn.position, 0.0)
    app.sensors.reset(app.quad.velocity)
    app.controller.reset(airborne=True)

    if options.parallax is not None:
        import raw_slam
        import imu_slam
        raw_slam.KEYFRAME_MIN_PARALLAX_PX = options.parallax
        imu_slam.MIN_DEROTATED_PARALLAX_PX = options.parallax
    if options.no_fixes:
        import raw_slam as _rs
        _rs.LOCALIZE_EVERY = 10**9
    from drone import quat_from_yaw, quat_multiply
    base = spawn.position.copy()
    maps = rectify_maps(options.cam_size, options.slam_size, options.slam_fov)
    intrinsic = pinhole_matrix(options.slam_size, options.slam_fov)
    slams: dict[str, RawSlam] = {}
    log = {name: [] for name in ("left", "right")}
    spent = {name: 0.0 for name in ("left", "right")}
    frames = 0
    dt = 1.0 / 500.0
    next_frame = 0.0
    body_cumulative = np.eye(3)  # body now -> body at the start, from the gyro alone
    rng = np.random.default_rng(5)
    fake_bias = np.array([0.010, -0.008, 0.012])
    last_rotation = app.quad.rotation.copy()

    def on_imu(sample):
        nonlocal body_cumulative
        if sample.dt > 0.0:
            rates = sample.gyro - app.controller.estimator.gyro_bias
            body_cumulative = body_cumulative @ rotation_vector_to_matrix(rates * sample.dt)

    if options.kinematic is None:
        app.imu_callbacks.append(on_imu)
    for i in range(int(options.seconds / dt)):
        t = i * dt
        if t > 1.0:
            w = 2 * np.pi * 0.35
            app.pilot = PilotInput(forward=options.amp * np.sin(w * (t - 1.0)), right=options.amp * np.sin(w * (t - 1.0) + 1.2), climb=0.3 * np.sin(0.7 * w * (t - 1.0)))
        if options.kinematic is None:
            app._step()
        else:
            app.quad.time = t
            r = options.kinematic
            w = 2 * np.pi * 0.3
            app.quad.position = base + r * np.array([np.sin(w * t), 0.4 * np.sin(1.3 * w * t), np.cos(0.8 * w * t) - 1.0])
            tilt = np.radians(4.0) * np.array([np.sin(1.1 * w * t), 0.0, np.sin(0.9 * w * t + 1)])
            half = tilt / 2
            app.quad.q = quat_multiply(quat_from_yaw(0.15 * np.sin(0.5 * w * t)), np.array([1.0, *half]) / np.linalg.norm([1.0, *half]))
            if i % 2 == 0:  # a gyro at 250 Hz: true rate, a bias it was calibrated for to within 0.002 rad/s, noise
                now = app.quad.rotation
                rate = rotation_log(last_rotation.T @ now) / (2 * dt)
                last_rotation = now.copy()
                measured = rate + fake_bias + rng.normal(0, 0.004, 3)
                body_cumulative = body_cumulative @ rotation_vector_to_matrix((measured - fake_bias - rng.normal(0, 0.002, 3) * 0 - np.array([0.002, -0.002, 0.001])) * 2 * dt)
        if app.quad.time >= next_frame:
            next_frame += 1.0 / 30.0
            frames += 1
            for name, camera in app.cameras.items():
                camera.render(app.quad.position, app.quad.rotation)
                picture = cv2.remap(camera.read_bgr(), *maps, cv2.INTER_LINEAR)
                began = time.perf_counter()
                body_to_camera = camera.rotation @ GL_TO_CV  # CV camera -> body
                camera_cumulative = body_to_camera.T @ body_cumulative @ body_to_camera
                if name not in slams:
                    slams[name] = (RawSlam if options.plain else ImuAidedSlam)(picture, intrinsic)
                    if options.save:
                        cv2.imwrite(str(options.save / f"{name}_rectified.png"), picture)
                        cv2.imwrite(str(options.save / f"{name}_fisheye.png"), camera.read_bgr())
                else:
                    slam = slams[name]
                    before = slam.commits, slam.fixes
                    if options.plain:
                        slam.process(picture)
                    else:
                        slam.process(picture, camera_cumulative)
                    spent[name] += time.perf_counter() - began
                    cam_rot = app.quad.rotation @ camera.rotation @ GL_TO_CV
                    cam_pos = app.quad.position + app.quad.rotation @ camera.mount.offset
                    log[name].append((app.quad.time, cam_pos, cam_rot, slam.world_position.copy(), slam.world_rotation.copy(), (slam.commits, slam.fixes) != before))
    print(f"simulated {options.seconds:.0f} s, {frames} frames")
    for name, entries in log.items():
        slam = slams[name]
        print(f"[{name}] {1000 * spent[name] / max(len(entries), 1):.1f} ms/frame  {slam.describe()}")
        t = np.array([e[0] for e in entries])
        truth = np.array([e[1] for e in entries])
        estimate = np.array([e[3] for e in entries])
        changed = np.array([e[5] for e in entries])
        moved = np.flatnonzero(np.any(estimate != estimate[0], axis=1))
        if len(moved) < 10:
            print("  SLAM never moved")
            continue
        start = moved[0]
        scale, rotation, shift = umeyama(estimate[start:], truth[start:])
        aligned = scale * estimate[start:] @ rotation.T + shift
        error = np.linalg.norm(aligned - truth[start:], axis=1)
        # rotation: truth camera rotation vs. slam rotation, relative to the first sample
        r0_true, r0_slam = entries[start][2], entries[start][4]
        angle = []
        for e in entries[start:]:
            relative_true = r0_true.T @ e[2]
            relative_slam = r0_slam.T @ e[4]
            angle.append(np.degrees(np.arccos(np.clip((np.trace(relative_true.T @ relative_slam) - 1) / 2, -1, 1))))
        print(f"  first motion at t={t[start]:.1f}s; scale (m per slam unit) {scale:.3f}; position error after best similarity fit: rms {np.sqrt((error**2).mean()):.3f} m, max {error.max():.3f} m")
        print(f"  rotation error vs truth: mean {np.mean(angle):.2f} deg, max {np.max(angle):.2f} deg; pose updates {changed.sum()} of {len(entries)} frames; truth path span {np.ptp(truth, axis=0).round(2)} m")


if __name__ == "__main__":
    main()
