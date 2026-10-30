"""Summarise a --diag flight log in time windows.

    python virtual-sim/tools/diag_report.py PREFIX [--window 10]

Prints, per window: how far the drone really was from where it was meant to be, how wrong the estimate was
(position, heading, tilt, gyro and accelerometer bias), then what the SLAM poses did (how many were believed,
how wrong the SLAM itself was, how much its unit of length moved compared with what the filter thought, how big
its steps were and what the anchor fixes did).
"""

from __future__ import annotations

import argparse

import numpy as np


def load(path: str) -> dict[str, np.ndarray]:
    with open(path) as f:
        header = f.readline().strip().split(",")
    data = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=str)
    if data.ndim == 1:
        data = data[None, :]
    out = {}
    for i, name in enumerate(header):
        column = data[:, i]
        try:
            out[name] = column.astype(float)
        except ValueError:
            out[name] = column
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix")
    parser.add_argument("--window", type=float, default=10.0)
    args = parser.parse_args()
    state, slam = load(f"{args.prefix}_state.csv"), load(f"{args.prefix}_slam.csv")
    t = state["t"]
    truth = np.stack([state["truth_x"], state["truth_y"], state["truth_z"]], 1)
    est = np.stack([state["est_x"], state["est_y"], state["est_z"]], 1)
    ref = np.stack([state["ref_x"], state["ref_y"], state["ref_z"]], 1)
    to_ref = np.linalg.norm(truth - ref, axis=1)
    est_err = np.linalg.norm(est - truth, axis=1)
    gyro_err = np.degrees(np.linalg.norm(np.stack([state["gyro_bias_err_x"], state["gyro_bias_err_y"], state["gyro_bias_err_z"]], 1), axis=1))
    yaw_gyro_err = np.degrees(np.abs(state["gyro_bias_err_y"]))
    accel_err = np.linalg.norm(np.stack([state["accel_bias_err_x"], state["accel_bias_err_y"], state["accel_bias_err_z"]], 1), axis=1)
    print("STATE (mean per window)")
    print(f"{'t':>9} {'mission':>8} {'|truth-ref|':>11} {'|est err|':>9} {'yaw err':>8} {'tilt err':>8} {'gyro bias err':>13} {'(yaw axis)':>10} {'accel bias err':>14} {'temp':>5} {'scale L/R':>11} {'pos sigma':>9} {'turn':>5}")
    for lo in np.arange(0.0, t.max() + 1e-6, args.window):
        m = (t >= lo) & (t < lo + args.window)
        if not m.any():
            continue
        mission = max(set(state["mission"][m]), key=list(state["mission"][m]).count)
        print(
            f"{lo:4.0f}-{lo + args.window:<4.0f} {mission:>8} {to_ref[m].mean():11.2f} {est_err[m].mean():9.2f} {state['yaw_err_deg'][m].mean():+8.2f} {state['tilt_err_deg'][m].mean():8.2f} "
            f"{gyro_err[m].mean():10.2f}°/s {yaw_gyro_err[m].mean():8.2f}°/s {accel_err[m].mean():12.3f}   {state['temperature'][m].mean():5.1f} "
            f"{state['scale_left'][m].mean():5.2f}/{state['scale_right'][m].mean():<5.2f} {state['pos_sigma'][m].mean():9.3f} {state['turn_rate'][m].mean():5.2f}"
        )
    print("\nSLAM POSES (per window, both cameras)")
    ts = slam["t"]
    print(f"{'t':>9} {'poses':>5} {'believed':>8} {'innov':>6} {'SLAM pos err':>12} {'SLAM att':>8} {'filter att':>10} {'true scale L/R (SLAM units per m)':>34} {'filter scale L/R':>17} {'step err':>8} {'fixes':>5} {'restarts':>8}")
    for lo in np.arange(0.0, ts.max() + 1e-6, args.window):
        m = (ts >= lo) & (ts < lo + args.window)
        if not m.any():
            continue
        row = []
        for cam in ("left", "right"):
            c = m & (slam["cam"] == cam)
            row.append((np.nanmedian(slam["true_scale"][c]) if c.any() else np.nan, np.nanmedian(slam["filter_scale"][c]) if c.any() else np.nan))
        fixes = max([slam["fixes_total"][m & (slam["cam"] == cam)].max() if (m & (slam["cam"] == cam)).any() else 0 for cam in ("left", "right")])
        print(
            f"{lo:4.0f}-{lo + args.window:<4.0f} {m.sum():5d} {100 * slam['believed'][m].mean():7.0f}% {np.nanmean(slam['innovation'][m]):6.2f} {np.nanmean(slam['slam_pos_err_m'][m]):11.2f}m "
            f"{np.nanmean(slam['slam_att_err_deg'][m]):7.2f}° {np.nanmean(slam['filter_att_err_deg'][m]):9.2f}° "
            f"{row[0][0]:16.2f}/{row[1][0]:<16.2f} {row[0][1]:8.2f}/{row[1][1]:<8.2f} {np.nanmean(slam['step_err_m'][m]):8.3f} {int(fixes):5d} {int(slam['restarts'][m].max()):8d}"
        )
    fix = slam["fix_now"] == 1
    if fix.any():
        normal = ~fix
        print(
            f"\nANCHOR FIXES: {int(fix.sum())} of {len(ts)} poses. Step error (m) on a fix: mean {np.nanmean(slam['step_err_m'][fix]):.2f}, max {np.nanmax(slam['step_err_m'][fix]):.2f}; "
            f"on an ordinary step: mean {np.nanmean(slam['step_err_m'][normal]):.3f}, max {np.nanmax(slam['step_err_m'][normal]):.2f}; believed by the filter: {100 * slam['believed'][fix].mean():.0f}% of fixes vs {100 * slam['believed'][normal].mean():.0f}% of steps"
        )
        for k in np.flatnonzero(fix)[:12]:
            print(f"   t={ts[k]:6.1f} {slam['cam'][k]:>5}: step err {slam['step_err_m'][k]:.2f} m, SLAM att err after {slam['slam_att_err_deg'][k]:.1f}°, SLAM pos err after {slam['slam_pos_err_m'][k]:.2f} m, believed {int(slam['believed'][k])}")
    print(f"\nsteps with step error > 0.3 m that were not fixes: {int(((slam['step_err_m'] > 0.3) & ~fix).sum())}")
    fast = slam["turn_rate"] > 0.6
    if fast.any() and (~fast).any():
        print(f"SLAM position error while turning fast (>0.6 rad/s): {np.nanmean(slam['slam_pos_err_m'][fast]):.2f} m vs {np.nanmean(slam['slam_pos_err_m'][~fast]):.2f} m otherwise (n={int(fast.sum())})")


if __name__ == "__main__":
    main()
