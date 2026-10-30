"""Flight logger for finding out *why* the estimate wanders (--diag PREFIX writes PREFIX_state.csv and PREFIX_slam.csv).

It is an observer: it reads the simulator's ground truth next to what the estimator believed and writes both to
disk. Nothing it computes is ever fed back to the drone.

PREFIX_state.csv   every 0.5 s: truth vs estimate (position, velocity, heading, tilt), the planned point, the gyro and
                   accelerometer bias the filter believes vs what the sensor really has, scales, uncertainty
PREFIX_slam.csv    one row per SLAM pose that reached the filter: was it believed, how far did it disagree, how wrong is
                   the SLAM itself (position at the filter's scale, attitude), did an anchor fix just happen and how far
                   did it move the pose, how much is the SLAM's unit really worth in metres vs what the filter thinks

Summarise with  python virtual-sim/tools/diag_report.py PREFIX
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

STATE_EVERY = 250  # physics ticks (500 Hz): 0.5 s

STATE_COLUMNS = [
    "t", "mission",
    "truth_x", "truth_y", "truth_z", "est_x", "est_y", "est_z", "ref_x", "ref_y", "ref_z",
    "truth_vx", "truth_vy", "truth_vz", "est_vx", "est_vy", "est_vz",
    "yaw_err_deg", "tilt_err_deg", "att_err_deg",
    "gyro_bias_err_x", "gyro_bias_err_y", "gyro_bias_err_z", "accel_bias_err_x", "accel_bias_err_y", "accel_bias_err_z",
    "temperature", "scale_left", "scale_right", "sigma_left", "sigma_right", "pos_sigma", "turn_rate", "throttle",
]
TELEMETRY_COLUMNS = [  # numbers from inside the SLAM (latest step / fix)
    "rekeys_no_step", "hint_expected", "hint_raw", "hint_held", "step_scale", "step_landmarks", "depth_med", "step_spread", "tri_valid", "tri_angle_med", "tri_top_share",
    "fix_matches", "fix_inliers", "fix_candidates", "fix_age_med", "fix_age_min", "fix_age_max", "fix_live", "fix_rot_deg", "fix_move", "fix_scale", "fix_pairs", "fix_z", "fix_budget", "fix_refused", "fix_flagged", "cam_disagree_deg", "fix_cam_share", "step_cams", "anchors_fused", "guided_matches", "guided_inview", "anchors_banned",
]
SLAM_COLUMNS = [
    "t", "cam", "believed", "innovation", "distrust", "att_accepted",
    "slam_att_err_deg", "filter_att_err_deg", "slam_vs_filter_deg", "slam_pos_err_m", "true_scale", "filter_scale", "scale_ratio",
    "step_err_m", "fixes_total", "fix_now", "restarts", "steps", "landmarks", "tracks", "turn_rate",
    *TELEMETRY_COLUMNS,
    "slam_x", "slam_y", "slam_z", "cam_true_x", "cam_true_y", "cam_true_z",  # the pose (SLAM units, rotated into W by the filter's start attitude) and the camera's true displacement at the picture's time (m)
]


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1.0) / 2.0, -1.0, 1.0))))


class FlightDiagnostics:
    def __init__(self, prefix: Path) -> None:
        prefix.parent.mkdir(parents=True, exist_ok=True)
        self.state_file = open(f"{prefix}_state.csv", "w", newline="")
        self.slam_file = open(f"{prefix}_slam.csv", "w", newline="")
        self.state = csv.writer(self.state_file)
        self.slam = csv.writer(self.slam_file)
        self.state.writerow(STATE_COLUMNS)
        self.slam.writerow(SLAM_COLUMNS)
        self.previous: dict[str, tuple] = {}
        print(f"Diagnostics: {prefix}_state.csv, {prefix}_slam.csv")

    def reset(self) -> None:
        self.previous.clear()

    # ------------------------------------------------------------------ the slow state
    def state_row(self, app) -> None:
        est, quad = app.estimator, app.quad
        if getattr(est, "position", None) is None or not est.initialised:
            return
        truth_rotation = app.w_axes.T @ quad.rotation
        truth_position, truth_velocity = app.truth_in_w(), app.w_axes.T @ quad.velocity
        target = app.target_point()
        _, _, yaw_truth = _attitude(truth_rotation)
        _, _, yaw_est = _attitude(est.rotation)
        yaw_error = float(np.degrees((yaw_est - yaw_truth + np.pi) % (2.0 * np.pi) - np.pi))
        tilt_error = float(np.degrees(np.arccos(np.clip(est.rotation[1] @ truth_rotation[1], -1.0, 1.0))))
        gyro_error = est.gyro_bias - app.sensors.gyro_bias
        accel_error = est.x[6:9] - app.sensors.accel_bias
        names = list(est.cameras)
        pick = {"left": names[0], "right": names[-1]}  # the rig has a single scale: both columns show it
        scale = {n: est.scale(pick[n]) for n in ("left", "right")}
        sigma = {n: 100.0 * est.scale_uncertainty(pick[n]) for n in ("left", "right")}
        self.state.writerow([
            f"{quad.time:.2f}", app.mission, *_f(truth_position), *_f(est.position), *_f(target), *_f(truth_velocity), *_f(est.velocity),
            f"{yaw_error:.3f}", f"{tilt_error:.3f}", f"{_angle(est.rotation, truth_rotation):.3f}", *_f(gyro_error, 5), *_f(accel_error, 4),
            f"{app.sensors.temperature:.2f}", f"{scale['left']:.4f}", f"{scale['right']:.4f}", f"{sigma['left']:.2f}", f"{sigma['right']:.2f}",
            f"{float(np.sqrt(np.trace(est.P[0:3, 0:3]) / 3.0)):.4f}", f"{est.turn_rate:.3f}", f"{float(np.mean(quad.motors)):.3f}",
        ])

    # ------------------------------------------------------------------ one row per SLAM pose
    def before(self, est, name: str) -> tuple:
        return (est.updates[name], est.attitude_corrections)

    def pose(self, app, result, before: tuple) -> None:
        est, name = app.estimator, result.name
        if name not in est.start_rotation:
            return
        cam = est.cameras[name]
        r_ws = est.start_rotation[name] @ cam.body_from_cv
        slam_body = r_ws @ result.rotation @ cam.body_from_cv.T
        truth_rotation = app.w_axes.T @ app.quad.rotation
        lam = max(est.scale(name), 0.02)
        camera_truth = app.truth_in_w() - est.start_position[name] + (truth_rotation - est.start_rotation[name]) @ cam.lever_arm
        slam_position = r_ws @ result.position
        p_then, r_then = app.truth_at(result.t)
        picture_truth = p_then - est.start_position[name] + (r_then - est.start_rotation[name]) @ cam.lever_arm
        slam_pos_err = float(np.linalg.norm(slam_position / lam - camera_truth))
        distance = float(np.linalg.norm(camera_truth))
        true_scale = float(np.linalg.norm(slam_position) / distance) if distance > 0.5 else float("nan")  # SLAM units per real metre, from the truth
        fixes = int(result.stats.get("fixes", 0))
        last = self.previous.get(name)
        step_err = float("nan")
        fix_now = 0
        if last is not None:
            slam_step = (slam_position - last[0]) / lam  # in metres at the filter's scale
            true_step = camera_truth - last[1]
            step_err = float(np.linalg.norm(slam_step - true_step))
            fix_now = int(fixes > last[2])
        self.previous[name] = (slam_position.copy(), camera_truth.copy(), fixes)
        self.slam.writerow([
            f"{result.t:.3f}", name, int(est.updates[name] > before[0]), f"{est.last_innovation[name]:.2f}", f"{est.last_distrust[name]:.2f}", int(est.attitude_corrections > before[1]),
            f"{_angle(slam_body, truth_rotation):.3f}", f"{_angle(est.rotation, truth_rotation):.3f}", f"{_angle(slam_body, est.rotation):.3f}", f"{slam_pos_err:.3f}",
            f"{true_scale:.3f}", f"{lam:.3f}", f"{lam / true_scale if true_scale == true_scale and true_scale > 0 else float('nan'):.3f}",
            f"{step_err:.3f}", fixes, fix_now, result.stats.get("restarts", 0), result.stats.get("steps", 0), result.stats.get("landmarks", 0), result.stats.get("tracks", 0),
            f"{est.turn_rate:.3f}",
            *[f"{result.telemetry.get(k, float('nan')):.4g}" for k in TELEMETRY_COLUMNS],
            *map("{:.4f}".format, slam_position), *map("{:.4f}".format, picture_truth),
        ])

    def close(self) -> None:
        self.state_file.close()
        self.slam_file.close()


def _f(v: np.ndarray, digits: int = 3) -> list[str]:
    return [f"{x:.{digits}f}" for x in v]


def _attitude(rotation: np.ndarray):
    from drone import attitude_angles

    return attitude_angles(rotation)
