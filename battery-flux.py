#!/usr/bin/env python3
"""Track battery level over a rolling window and project charge/discharge ETA."""

from __future__ import annotations

import argparse
import curses
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

POWER_SUPPLY = Path("/sys/class/power_supply")
WINDOW_SEC = 30.0
SAMPLE_INTERVAL = 1.0
FLAT_SLOPE_PCT_PER_MIN = 0.05


@dataclass(frozen=True)
class BatteryReading:
    capacity_pct: float
    fine_pct: float
    status: str
    energy_full_uwh: float | None


def find_battery(name: str | None) -> Path:
    if name:
        bat = POWER_SUPPLY / name
        if not bat.is_dir():
            raise SystemExit(f"No power supply named {name!r} under {POWER_SUPPLY}")
        return bat

    for entry in sorted(POWER_SUPPLY.iterdir()):
        type_path = entry / "type"
        if type_path.is_file() and type_path.read_text().strip() == "Battery":
            return entry
    raise SystemExit(f"No battery found under {POWER_SUPPLY}")


def read_float(path: Path) -> float | None:
    try:
        return float(path.read_text().strip())
    except (OSError, ValueError):
        return None


def read_battery(bat: Path) -> BatteryReading:
    capacity = read_float(bat / "capacity")
    if capacity is None:
        raise OSError(f"Could not read {bat / 'capacity'}")

    status_path = bat / "status"
    status = status_path.read_text().strip() if status_path.is_file() else "Unknown"

    energy_now = read_float(bat / "energy_now")
    energy_full = read_float(bat / "energy_full")
    if (
        energy_now is not None
        and energy_full is not None
        and energy_full > 0
    ):
        fine_pct = 100.0 * energy_now / energy_full
    else:
        fine_pct = capacity

    return BatteryReading(
        capacity_pct=capacity,
        fine_pct=fine_pct,
        status=status,
        energy_full_uwh=energy_full,
    )


def linear_slope_pct_per_sec(samples: deque[tuple[float, float]]) -> float | None:
    if len(samples) < 2:
        return None
    t0 = samples[0][0]
    xs = [t - t0 for t, _ in samples]
    ys = [y for _, y in samples]
    n = len(xs)
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    den = sum((x - x_mean) ** 2 for x in xs)
    if den <= 0:
        return None
    return num / den


def trend_power_w(
    slope_pct_per_sec: float | None, energy_full_uwh: float | None
) -> float | None:
    """Average power implied by fine-% slope over the rolling window (watts)."""
    if slope_pct_per_sec is None or energy_full_uwh is None or energy_full_uwh <= 0:
        return None
    energy_uwh_per_sec = (slope_pct_per_sec / 100.0) * energy_full_uwh
    return abs(energy_uwh_per_sec) * 3600.0 / 1_000_000.0


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds != seconds:  # NaN
        return "—"
    if seconds <= 0:
        return "now"
    seconds = int(round(seconds))
    if seconds >= 86400:
        days, rem = divmod(seconds, 86400)
        hours, rem = divmod(rem, 3600)
        return f"{days}d {hours}h"
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def sparkline(values: list[float], width: int = 40) -> str:
    if not values:
        return " " * width
    lo, hi = min(values), max(values)
    if hi - lo < 1e-6:
        mid = "▄"
        return mid * min(len(values), width)
    chars = "▁▂▃▄▅▆▇█"
    if len(values) > width:
        step = len(values) / width
        picked = [values[int(i * step)] for i in range(width)]
    else:
        picked = values
    out = []
    for v in picked:
        idx = int((v - lo) / (hi - lo) * (len(chars) - 1))
        out.append(chars[idx])
    return "".join(out).ljust(width)


def project_eta(
    capacity: float, status: str, slope_pct_per_sec: float | None
) -> tuple[str, float | None]:
    if slope_pct_per_sec is None:
        return "Collecting samples…", None

    slope_pct_per_min = slope_pct_per_sec * 60.0
    if abs(slope_pct_per_min) < FLAT_SLOPE_PCT_PER_MIN:
        return "Level stable (no meaningful trend in window)", None

    charging = "charg" in status.lower()
    discharging = "discharg" in status.lower()

    if slope_pct_per_sec > 0:
        eta_sec = (100.0 - capacity) / slope_pct_per_sec
        if charging or not discharging:
            return f"Projected full (~100%) in {format_duration(eta_sec)}", eta_sec
        return (
            f"Trend rising (+{slope_pct_per_min:.2f}%/min) but status is {status!r}",
            eta_sec,
        )

    eta_sec = capacity / (-slope_pct_per_sec)
    if discharging or not charging:
        return f"Projected empty (~0%) in {format_duration(eta_sec)}", eta_sec
    return (
        f"Trend falling ({slope_pct_per_min:.2f}%/min) but status is {status!r}",
        eta_sec,
    )


def summarize_tray(
    reading: BatteryReading,
    samples: deque[tuple[float, float]],
    window_sec: float,
) -> tuple[str, str]:
    """Short top-bar label and longer tooltip/menu body."""
    slope = linear_slope_pct_per_sec(samples)
    slope_pct_per_min = slope * 60.0 if slope is not None else None
    trend_w = trend_power_w(slope, reading.energy_full_uwh)
    eta_line, _ = project_eta(reading.fine_pct, reading.status, slope)

    label = f"{reading.capacity_pct:.0f}%"
    if trend_w is not None and slope_pct_per_min is not None:
        if abs(slope_pct_per_min) >= FLAT_SLOPE_PCT_PER_MIN:
            arrow = "↑" if slope and slope > 0 else "↓"
            label = f"{label} {arrow}{trend_w:.0f}W"

    detail_lines = [
        f"{reading.capacity_pct:.0f}% ({reading.fine_pct:.2f}% fine) — {reading.status}",
    ]
    if slope_pct_per_min is None:
        detail_lines.append(f"Flux: warming up ({window_sec:.0f}s window)")
    else:
        sign = "+" if slope_pct_per_min >= 0 else ""
        detail_lines.append(f"Flux: {sign}{slope_pct_per_min:.2f}%/min")
    if trend_w is not None and slope_pct_per_min is not None:
        if abs(slope_pct_per_min) >= FLAT_SLOPE_PCT_PER_MIN:
            sign = "+" if slope and slope > 0 else "-"
            detail_lines.append(f"Trend power: {sign}{trend_w:.1f} W")
    detail_lines.append(f"ETA: {eta_line}")
    tooltip = "\n".join(detail_lines)
    return label, tooltip


def render_plain(
    bat_name: str,
    reading: BatteryReading,
    samples: deque[tuple[float, float]],
    window_sec: float,
) -> str:
    slope = linear_slope_pct_per_sec(samples)
    slope_pct_per_min = slope * 60.0 if slope is not None else None
    trend_w = trend_power_w(slope, reading.energy_full_uwh)
    eta_line, _ = project_eta(reading.fine_pct, reading.status, slope)
    values = [y for _, y in samples]
    age = time.time() - samples[0][0] if samples else 0.0

    lines = [
        f"Battery: {bat_name}  |  {reading.capacity_pct:.0f}% "
        f"({reading.fine_pct:.2f}% fine)  |  {reading.status}",
    ]
    if trend_w is not None and (
        slope_pct_per_min is None
        or abs(slope_pct_per_min) >= FLAT_SLOPE_PCT_PER_MIN
    ):
        sign = "+" if slope and slope > 0 else "-"
        lines[0] += f"  |  {sign}{trend_w:.1f} W (30s trend)"

    if slope_pct_per_min is None:
        lines.append(f"Flux (last {window_sec:.0f}s): collecting…")
    else:
        sign = "+" if slope_pct_per_min >= 0 else ""
        lines.append(
            f"Flux (last {window_sec:.0f}s, {age:.0f}s of data): "
            f"{sign}{slope_pct_per_min:.2f}%/min"
        )

    lines.append(f"ETA: {eta_line}")
    lines.append(f"Trend: [{sparkline(values)}]")
    lines.append("Ctrl+C to quit")
    return "\n".join(lines)


def run_curses(stdscr, bat: Path, window_sec: float, interval: float) -> None:
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(int(interval * 1000))

    samples: deque[tuple[float, float]] = deque()
    bat_name = bat.name
    last_draw = 0.0

    while True:
        now = time.time()
        cutoff = now - window_sec
        while samples and samples[0][0] < cutoff:
            samples.popleft()

        try:
            reading = read_battery(bat)
        except OSError as exc:
            stdscr.erase()
            stdscr.addstr(0, 0, f"Error reading battery: {exc}")
            stdscr.refresh()
            if stdscr.getch() in (3, 26, ord("q")):
                break
            continue

        samples.append((now, reading.fine_pct))

        if now - last_draw >= interval * 0.9:
            text = render_plain(bat_name, reading, samples, window_sec)
            stdscr.erase()
            for i, line in enumerate(text.splitlines()):
                try:
                    stdscr.addstr(i, 0, line[: curses.COLS - 1])
                except curses.error:
                    pass
            stdscr.refresh()
            last_draw = now

        ch = stdscr.getch()
        if ch in (3, ord("q")):
            break


def run_tray(bat: Path, window_sec: float, interval: float) -> None:
    try:
        gi = __import__("gi")
        gi.require_version("Gtk", "3.0")
        gi.require_version("AyatanaAppIndicator3", "0.1")
        from gi.repository import GLib, Gtk
        from gi.repository import AyatanaAppIndicator3 as AppIndicator
    except (ImportError, ValueError) as exc:
        raise SystemExit(
            "Tray mode needs GTK AppIndicator support on Fedora:\n"
            "  sudo dnf install libayatana-appindicator-gtk3 python3-gobject\n"
            "Also enable a GNOME tray extension, e.g. “AppIndicator and "
            "KStatusNotifierItem Support” from extensions.gnome.org.\n"
            f"({exc})"
        ) from exc

    samples: deque[tuple[float, float]] = deque()
    bat_name = bat.name

    indicator = AppIndicator.Indicator.new(
        "battery-flux",
        "battery-good-symbolic",
        AppIndicator.IndicatorCategory.SYSTEM_SERVICES,
    )
    indicator.set_status(AppIndicator.IndicatorStatus.ACTIVE)
    indicator.set_title("Battery flux")

    menu = Gtk.Menu()
    info_item = Gtk.MenuItem(label="Starting…")
    info_item.set_sensitive(False)
    menu.append(info_item)
    menu.append(Gtk.SeparatorMenuItem())

    quit_item = Gtk.MenuItem(label="Quit")
    quit_item.connect("activate", Gtk.main_quit)
    menu.append(quit_item)
    menu.show_all()
    indicator.set_menu(menu)

    def tick() -> bool:
        now = time.time()
        cutoff = now - window_sec
        while samples and samples[0][0] < cutoff:
            samples.popleft()
        try:
            reading = read_battery(bat)
        except OSError as exc:
            indicator.set_label("ERR", "")
            info_item.set_label(str(exc))
            return True

        samples.append((now, reading.fine_pct))
        label, tooltip = summarize_tray(reading, samples, window_sec)
        indicator.set_label(label, bat_name)
        indicator.set_title(f"Battery flux — {bat_name}\n{tooltip}")
        info_item.set_label(tooltip.replace("\n", " · "))
        return True

    GLib.timeout_add(int(interval * 1000), tick)
    tick()
    Gtk.main()


def run_plain(bat: Path, window_sec: float, interval: float) -> None:
    samples: deque[tuple[float, float]] = deque()
    bat_name = bat.name

    try:
        while True:
            now = time.time()
            cutoff = now - window_sec
            while samples and samples[0][0] < cutoff:
                samples.popleft()

            reading = read_battery(bat)
            samples.append((now, reading.fine_pct))

            text = render_plain(bat_name, reading, samples, window_sec)
            sys.stdout.write("\033[H\033[J" + text + "\n")
            sys.stdout.flush()
            time.sleep(interval)
    except KeyboardInterrupt:
        sys.stdout.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track battery level over a rolling window and project ETA from recent flux."
    )
    parser.add_argument(
        "--battery",
        help="Power supply name (default: first Battery under /sys/class/power_supply)",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=WINDOW_SEC,
        help=f"Rolling history length in seconds (default: {WINDOW_SEC:g})",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=SAMPLE_INTERVAL,
        help=f"Sample period in seconds (default: {SAMPLE_INTERVAL:g})",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Use ANSI terminal refresh instead of curses",
    )
    parser.add_argument(
        "--tray",
        action="store_true",
        help="Show a top-bar tray icon (GNOME needs AppIndicator extension)",
    )
    args = parser.parse_args()

    bat = find_battery(args.battery)
    if args.tray:
        if args.interval == SAMPLE_INTERVAL:
            args.interval = 5.0
        run_tray(bat, args.window, args.interval)
    elif args.plain or not sys.stdout.isatty():
        run_plain(bat, args.window, args.interval)
    else:
        curses.wrapper(run_curses, bat, args.window, args.interval)


if __name__ == "__main__":
    main()
