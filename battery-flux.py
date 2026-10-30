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
    energy_full_design_uwh: float | None = None
    power_now_w: float | None = None
    cycle_count: int | None = None
    energy_now_uwh: float | None = None
    voltage_uv: float | None = None
    charge_limit_pct: float | None = None


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

    power_now_uw = read_float(bat / "power_now")
    cycles = read_float(bat / "cycle_count")
    return BatteryReading(
        capacity_pct=capacity,
        fine_pct=fine_pct,
        status=status,
        energy_full_uwh=energy_full,
        energy_full_design_uwh=read_float(bat / "energy_full_design"),
        power_now_w=None if power_now_uw is None else power_now_uw / 1_000_000.0,
        cycle_count=None if cycles is None else int(cycles),
        energy_now_uwh=energy_now,
        voltage_uv=read_float(bat / "voltage_now"),
        charge_limit_pct=read_float(bat / "charge_control_end_threshold"),
    )


HWMON = Path("/sys/class/hwmon")


@dataclass(frozen=True)
class Sensor:
    id: str
    group: str
    label: str
    unit: str
    path: Path
    scale: float = 1.0


def read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def discover_sensors() -> list[Sensor]:
    """Every temperature and fan that the kernel's hardware monitors expose. Found once at start."""
    found: list[Sensor] = []
    for hw in sorted(HWMON.glob("hwmon*")):
        name = read_text(hw / "name") or hw.name
        for temp in sorted(hw.glob("temp*_input")):
            n = temp.name.split("_")[0]
            sub = read_text(hw / f"{n}_label") or n
            label = f"{name} {sub}" if sub == n else f"{name} {sub}"
            found.append(Sensor(f"temp.{hw.name}.{n}", "temperature", label, "°C", temp, 0.001))
        for fan in sorted(hw.glob("fan*_input")):
            n = fan.name.split("_")[0]
            found.append(Sensor(f"fan.{hw.name}.{n}", "fans", f"{name} {n}", "RPM", fan))
    return found


class CpuLoad:
    """Whole-machine CPU busy percentage from the change in /proc/stat between two samples."""

    def __init__(self) -> None:
        self.previous: tuple[int, int] | None = None

    def sample(self) -> float | None:
        try:
            fields = Path("/proc/stat").read_text().splitlines()[0].split()[1:9]
        except (OSError, IndexError):
            return None
        values = [int(x) for x in fields]
        idle = values[3] + values[4]  # idle + iowait
        total = sum(values)
        previous, self.previous = self.previous, (idle, total)
        if previous is None:
            return None
        d_total, d_idle = total - previous[1], idle - previous[0]
        if d_total <= 0:
            return None
        return 100.0 * (1.0 - d_idle / d_total)


def cpu_mhz() -> float | None:
    freqs = []
    for path in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_cur_freq"):
        value = read_float(path)
        if value is not None:
            freqs.append(value / 1000.0)
    return sum(freqs) / len(freqs) if freqs else None


def memory_used_pct() -> float | None:
    try:
        fields = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            fields[key] = float(rest.split()[0])
        return 100.0 * (1.0 - fields["MemAvailable"] / fields["MemTotal"])
    except (OSError, KeyError, ValueError, ZeroDivisionError):
        return None


def metric(id_: str, group: str, label: str, unit: str, value: float | None) -> dict:
    return {"id": id_, "group": group, "label": label, "unit": unit, "value": value}


def battery_metrics(reading: BatteryReading, slope: float | None, trend_w: float | None) -> list[dict]:
    voltage = None if reading.voltage_uv is None else reading.voltage_uv / 1_000_000.0
    power = reading.power_now_w
    current = power / voltage if power is not None and voltage else None
    energy_now = None if reading.energy_now_uwh is None else reading.energy_now_uwh / 1_000_000.0
    energy_full = None if reading.energy_full_uwh is None else reading.energy_full_uwh / 1_000_000.0
    health = None
    if reading.energy_full_uwh and reading.energy_full_design_uwh:
        health = 100.0 * reading.energy_full_uwh / reading.energy_full_design_uwh
    return [
        metric("battery.level", "battery", "Level", "%", reading.capacity_pct),
        metric("battery.fine", "battery", "Level (fine)", "%", reading.fine_pct),
        metric("battery.flux", "battery", "Flux", "%/min", None if slope is None else slope * 60.0),
        metric("battery.trend_power", "battery", "Trend power (from flux)", "W", trend_w),
        metric("battery.power", "battery", "Power now (sysfs)", "W", power),
        metric("battery.voltage", "battery", "Voltage", "V", voltage),
        metric("battery.current", "battery", "Current (power / voltage)", "A", current),
        metric("battery.energy_now", "battery", "Energy now", "Wh", energy_now),
        metric("battery.energy_full", "battery", "Energy full", "Wh", energy_full),
        metric("battery.health", "battery", "Health", "%", health),
        metric("battery.cycles", "battery", "Cycle count", "", None if reading.cycle_count is None else float(reading.cycle_count)),
        metric("battery.charge_limit", "battery", "Charge limit", "%", reading.charge_limit_pct),
    ]


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


def collect_payload(bat: Path, window_sec: float, samples: deque, sensors: list[Sensor], cpu: CpuLoad) -> dict:
    """One full sample: the battery fields, every metric, and the summary the top bar uses."""
    now = time.time()
    cutoff = now - window_sec
    while samples and samples[0][0] < cutoff:
        samples.popleft()
    reading = read_battery(bat)
    samples.append((now, reading.fine_pct))

    slope = linear_slope_pct_per_sec(samples)
    trend_w = trend_power_w(slope, reading.energy_full_uwh)
    eta_line, eta_sec = project_eta(reading.fine_pct, reading.status, slope)

    metrics = battery_metrics(reading, slope, trend_w)
    for sensor in sensors:
        raw = read_float(sensor.path)
        metrics.append(metric(sensor.id, sensor.group, sensor.label, sensor.unit, None if raw is None else raw * sensor.scale))
    metrics.append(metric("cpu.load", "cpu", "CPU load", "%", cpu.sample()))
    metrics.append(metric("cpu.frequency", "cpu", "CPU frequency (mean)", "MHz", cpu_mhz()))
    metrics.append(metric("memory.used", "memory", "Memory used", "%", memory_used_pct()))

    return {
        "time": now,
        "battery": bat.name,
        "level_pct": reading.capacity_pct,
        "fine_pct": reading.fine_pct,
        "status": reading.status,
        "flux_pct_per_min": None if slope is None else slope * 60.0,
        "trend_power_w": trend_w,
        "power_now_w": reading.power_now_w,
        "eta_sec": eta_sec,
        "eta_text": eta_line,
        "window_sec": window_sec,
        "metrics": metrics,
    }


def run_json(bat: Path, window_sec: float, interval: float) -> None:
    """One JSON object per line, for other programs (the GNOME Shell extension reads this)."""
    import json

    samples: deque[tuple[float, float]] = deque()
    sensors = discover_sensors()
    cpu = CpuLoad()
    try:
        while True:
            sys.stdout.write(json.dumps(collect_payload(bat, window_sec, samples, sensors, cpu)) + "\n")
            sys.stdout.flush()
            time.sleep(interval)
    except (KeyboardInterrupt, BrokenPipeError):
        return


DEFAULT_LOG = Path.home() / ".local" / "share" / "battery-flux" / "history.csv"


def run_log(bat: Path, window_sec: float, interval: float, path: Path, every_sec: float) -> None:
    """Headless: sample every interval, append one CSV row every every_sec seconds, for long-term history."""
    import csv
    from datetime import datetime

    samples: deque[tuple[float, float]] = deque()
    sensors = discover_sensors()
    cpu = CpuLoad()
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] | None = None
    last_written = 0.0

    try:
        while True:
            payload = collect_payload(bat, window_sec, samples, sensors, cpu)
            if payload["time"] - last_written >= every_sec:
                wanted = ["time", "local_time", "battery"] + [m["id"] for m in payload["metrics"]]
                if columns is None:
                    columns = wanted
                    if path.exists():
                        with path.open(newline="") as f:
                            existing = next(csv.reader(f), [])
                        if existing != columns:
                            # Metric list changed (new sensor, new kernel): keep the old file, start a fresh one.
                            path.rename(path.with_name(f"{path.stem}-{int(payload['time'])}{path.suffix}"))
                    new_file = not path.exists()
                    handle = path.open("a", newline="")
                    writer = csv.writer(handle)
                    if new_file:
                        writer.writerow(columns)
                values = {m["id"]: m["value"] for m in payload["metrics"]}
                values.update(time=round(payload["time"], 1),
                              local_time=datetime.fromtimestamp(payload["time"]).isoformat(timespec="seconds"),
                              battery=payload["battery"])
                writer.writerow(["" if values.get(c) is None else values[c] for c in columns])
                handle.flush()
                last_written = payload["time"]
            time.sleep(interval)
    except KeyboardInterrupt:
        return
    finally:
        if columns is not None:
            handle.close()


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
        "--json",
        action="store_true",
        help="Print one JSON object per sample (for other programs)",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Headless: append metrics to a CSV file for long-term history",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG,
        help=f"CSV file for --log (default: {DEFAULT_LOG})",
    )
    parser.add_argument(
        "--log-every",
        type=float,
        default=10.0,
        help="Minutes between CSV rows with --log (default: 10)",
    )
    parser.add_argument(
        "--tray",
        action="store_true",
        help="Show a top-bar tray icon (GNOME needs AppIndicator extension)",
    )
    args = parser.parse_args()

    bat = find_battery(args.battery)
    if args.log:
        run_log(bat, args.window, args.interval, args.log_file, args.log_every * 60.0)
    elif args.json:
        run_json(bat, args.window, args.interval)
    elif args.tray:
        if args.interval == SAMPLE_INTERVAL:
            args.interval = 5.0
        run_tray(bat, args.window, args.interval)
    elif args.plain or not sys.stdout.isatty():
        run_plain(bat, args.window, args.interval)
    else:
        curses.wrapper(run_curses, bat, args.window, args.interval)


if __name__ == "__main__":
    main()
