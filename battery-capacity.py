#!/usr/bin/env python3
"""Estimate the battery's usable capacity from its energy counter, with a 95% confidence interval.

Keeps the last 30 minutes at 30-second spacing in recent-30min.csv. Whenever the integer charge percentage
steps, it records (energy now, percentage) as a point. A straight line through those points, energy against
percentage, has a slope equal to the capacity per percentage point. The confidence interval comes from the
scatter of the points around that line. Only the energy counter and the percentage are used.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("battery_flux", HERE / "battery-flux.py")
bf = importlib.util.module_from_spec(_spec)
sys.modules["battery_flux"] = bf  # dataclasses look the module up by name
_spec.loader.exec_module(bf)

DATA = Path.home() / ".local" / "share" / "battery-flux"
RECENT = DATA / "recent-30min.csv"
ESTIMATES = DATA / "capacity-estimates.csv"
MIN_POINTS = 3
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


@dataclass(frozen=True)
class Sample:
    t: float
    level: int
    energy_wh: float
    status: str
    design_wh: float | None
    power_w: float | None


def fit(points: list[tuple[float, int]]) -> dict | None:
    """Least-squares line energy = a + b * level. Returns capacity (b * 100) and its 95% interval, or None."""
    n = len(points)
    if n < MIN_POINTS:
        return None
    xs = [level for _, level in points]
    ys = [energy for energy, _ in points]
    if len(set(xs)) < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    rss = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    df = n - 2
    se_slope = math.sqrt(rss / df / sxx)
    half = T95.get(df, 1.96) * se_slope * 100.0
    capacity = slope * 100.0
    return {
        "points": n,
        "span_pct": max(xs) - min(xs),
        "capacity_wh": capacity,
        "ci_low_wh": capacity - half,
        "ci_high_wh": capacity + half,
        "ci_half_wh": half,
    }


def ticks(samples: list[Sample]) -> list[tuple[float, int]]:
    """(energy, level) at each step of the integer percentage, while charging or discharging the same way."""
    points = []
    for a, b in zip(samples, samples[1:]):
        if b.level == a.level or a.status != b.status:
            continue
        if "discharg" not in b.status.lower() and "charg" not in b.status.lower():
            continue
        points.append(((a.energy_wh + b.energy_wh) / 2.0, b.level))
    return points


def write_recent(samples: list[Sample]) -> None:
    tmp = RECENT.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "local_time", "level_pct", "energy_now_wh", "status", "power_now_w"])
        for s in samples:
            writer.writerow([round(s.t, 1), datetime.fromtimestamp(s.t).isoformat(timespec="seconds"),
                             s.level, round(s.energy_wh, 4), s.status, "" if s.power_w is None else round(s.power_w, 3)])
    os.replace(tmp, RECENT)


def append_estimate(now: float, est: dict, last: Sample, duration_min: float) -> None:
    new_file = not ESTIMATES.exists()
    with ESTIMATES.open("a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["time", "local_time", "points", "span_pct", "duration_min", "capacity_wh",
                             "ci_low_wh", "ci_high_wh", "ci_half_wh", "capacity_pct_of_design", "status", "level_pct", "energy_now_wh"])
        design = last.design_wh
        pct = "" if not design else round(100.0 * est["capacity_wh"] / design, 2)
        writer.writerow([round(now, 1), datetime.fromtimestamp(now).isoformat(timespec="seconds"), est["points"],
                         est["span_pct"], round(duration_min, 1), round(est["capacity_wh"], 3), round(est["ci_low_wh"], 3),
                         round(est["ci_high_wh"], 3), round(est["ci_half_wh"], 3), pct, last.status, last.level, round(last.energy_wh, 4)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate battery capacity from its energy counter with a 95% interval.")
    parser.add_argument("--sample-sec", type=float, default=30.0)
    parser.add_argument("--window-min", type=float, default=30.0)
    parser.add_argument("--data-dir", type=Path, default=DATA, help="Where the CSV files go (default: %(default)s)")
    args = parser.parse_args()

    global RECENT, ESTIMATES
    args.data_dir.mkdir(parents=True, exist_ok=True)
    RECENT = args.data_dir / "recent-30min.csv"
    ESTIMATES = args.data_dir / "capacity-estimates.csv"

    bat = bf.find_battery(None)
    keep = int(args.window_min * 60 / args.sample_sec) + 1
    buffer: deque[Sample] = deque(maxlen=keep)
    try:
        while True:
            try:
                reading = bf.read_battery(bat)
            except OSError:
                time.sleep(args.sample_sec)
                continue
            if reading.energy_now_uwh is None:
                time.sleep(args.sample_sec)
                continue
            now = time.time()
            buffer.append(Sample(
                t=now,
                level=int(reading.capacity_pct),
                energy_wh=reading.energy_now_uwh / 1_000_000.0,
                status=reading.status,
                design_wh=None if reading.energy_full_design_uwh is None else reading.energy_full_design_uwh / 1_000_000.0,
                power_w=reading.power_now_w,
            ))
            samples = list(buffer)
            write_recent(samples)

            step_now = len(samples) >= 2 and samples[-2].level != samples[-1].level
            if step_now:
                points = ticks(samples)
                est = fit(points)
                if est:
                    append_estimate(now, est, samples[-1], (samples[-1].t - samples[0].t) / 60.0)
            time.sleep(args.sample_sec)
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
