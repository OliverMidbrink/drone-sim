#!/usr/bin/env python3
"""Headless SLAM benchmark: FPS, RAM, and timestamp-aligned forward/back trajectory match.

Plays the first ``--seconds`` of a video forwards, then the same frames in reverse,
and compares the estimated camera position at each video timestamp.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import resource
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def _load_slam():
    path = Path(__file__).resolve().parent / "3d-slam.py"
    spec = importlib.util.spec_from_file_location("slam_app", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["slam_app"] = module
    spec.loader.exec_module(module)
    return module


slam = _load_slam()


def rss_mb() -> float:
    """Current resident set size in MiB (Linux)."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return peak_rss_mb()


def peak_rss_mb() -> float:
    """High-water RSS in MiB since process start."""
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def load_clip(path: Path, seconds: float, scale: float) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    n = max(2, int(round(seconds * fps)))
    frames: list[np.ndarray] = []
    while len(frames) < n:
        ok, raw = capture.read()
        if not ok:
            break
        frames.append(slam.resize_frame(raw, scale))
    capture.release()
    if len(frames) < 2:
        raise SystemExit("Not enough frames to benchmark")
    return frames, fps


def record_pass(
    session: object,
    frames: list[np.ndarray],
    times: list[float],
) -> np.ndarray:
    """Process frames; return (N, 4) [t_sec, x, y, z] after each one."""
    rows = []
    for frame, t in zip(frames, times, strict=True):
        session.process(frame)
        rows.append((t, *session.world_position.tolist()))
    return np.asarray(rows, dtype=np.float64)


def path_length(points: np.ndarray) -> float:
    """Total distance travelled along an (N, 3) polyline."""
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def match_by_timestamp(forward: np.ndarray, reverse: np.ndarray) -> dict[str, float]:
    """Compare poses that share the same video timestamp (nearest neighbour, ± half-frame)."""
    empty = {
        "pairs": 0,
        "rmse": float("nan"),
        "median_m": float("nan"),
        "mean_m": float("nan"),
        "p95_m": float("nan"),
        "max_m": float("nan"),
        "start_return_m": float("nan"),
        "turn_return_m": float("nan"),
        "forward_path": float("nan"),
        "reverse_path": float("nan"),
        "rmse_pct": float("nan"),
        "mean_pct": float("nan"),
        "p95_pct": float("nan"),
        "max_pct": float("nan"),
        "start_return_pct": float("nan"),
    }
    if len(forward) == 0 or len(reverse) == 0:
        return empty

    t_fwd, p_fwd = forward[:, 0], forward[:, 1:4]
    t_rev, p_rev = reverse[:, 0], reverse[:, 1:4]
    order = np.argsort(t_fwd)
    t_fwd, p_fwd = t_fwd[order], p_fwd[order]
    j = np.clip(np.searchsorted(t_fwd, t_rev), 1, len(t_fwd) - 1)
    left, right = j - 1, j
    pick_right = np.abs(t_fwd[right] - t_rev) < np.abs(t_fwd[left] - t_rev)
    idx = np.where(pick_right, right, left)
    dt = np.abs(t_fwd[idx] - t_rev)
    half = float(np.median(np.diff(t_fwd))) / 2.0 if len(t_fwd) > 1 else 0.05
    keep = dt <= max(half, 1e-3)
    if not np.any(keep):
        return empty

    err = np.linalg.norm(p_rev[keep] - p_fwd[idx[keep]], axis=1)
    fwd_len = path_length(p_fwd)
    rev_len = path_length(p_rev[np.argsort(t_rev)])
    scale = fwd_len if fwd_len > 1e-9 else float("nan")
    return {
        "forward_path": fwd_len,
        "reverse_path": rev_len,
        "rmse_pct": 100.0 * float(np.sqrt(np.mean(err**2))) / scale,
        "mean_pct": 100.0 * float(np.mean(err)) / scale,
        "p95_pct": 100.0 * float(np.percentile(err, 95)) / scale,
        "max_pct": 100.0 * float(np.max(err)) / scale,
        "start_return_pct": 100.0
        * float(np.linalg.norm(p_rev[np.argmin(t_rev)] - p_fwd[np.argmin(t_fwd)]))
        / scale,
        "pairs": int(keep.sum()),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "median_m": float(np.median(err)),
        "mean_m": float(np.mean(err)),
        "p95_m": float(np.percentile(err, 95)),
        "max_m": float(np.max(err)),
        "start_return_m": float(np.linalg.norm(p_rev[np.argmin(t_rev)] - p_fwd[np.argmin(t_fwd)])),
        "turn_return_m": float(np.linalg.norm(p_rev[np.argmax(t_rev)] - p_fwd[np.argmax(t_fwd)])),
    }


def stability(session: "slam.SlamSession") -> dict:
    """Pose-jitter counters from the session's commit log (lower is steadier)."""
    rows = [r for r in session.commit_log if r[0] == "step"]
    if not rows:
        return {"commits": 0, "reversals": 0, "turns_over_45": 0, "spins_over_5deg": 0, "repaired": 0, "sign_flips": 0}
    turn = np.array([r[5] for r in rows])
    spin = np.array([r[7] for r in rows])
    return {
        "commits": len(rows),
        "reversals": int((turn > 90).sum()),  # heading flips by more than 90 deg between commits
        "turns_over_45": int((turn > 45).sum()),
        "spins_over_5deg": int((spin > 5).sum()),  # implausible rotation for ~10-40 px of flow
        "repaired": int(session.stats.get("pose_repaired", 0)),
        "sign_flips": int(session.stats.get("sign_flips", 0)),
    }


def print_report(report: dict) -> None:
    print()
    print(f"profile        {report.get('profile', '-')}")
    print(f"video          {report['video']}")
    print(f"clip           {report['seconds']:.1f}s  ({report['frames']} frames, {report['fps_file']:.2f} file fps)")
    print(f"forward fps    {report['forward_fps']:.2f}   ({report['forward_s']:.2f}s wall)")
    print(f"reverse fps    {report['reverse_fps']:.2f}   ({report['reverse_s']:.2f}s wall)")
    print(f"overall fps    {report['overall_fps']:.2f}")
    print(f"RAM start      {report['ram_start_mb']:.1f} MiB")
    print(f"RAM after      {report['ram_end_mb']:.1f} MiB   (delta {report['ram_delta_mb']:+.1f})")
    print(f"RAM peak       {report['ram_peak_mb']:.1f} MiB")
    print(f"landmarks      {report['landmarks']}   anchors {report['anchors']}   loop fixes {report['fixes']} "
          f"(guided {report['fixes_guided']}, rejected {report['fixes_rejected']})")
    st = report.get("stability")
    if st:
        print(f"stability      {st['commits']} pose updates: {st['reversals']} heading reversals (>90°), "
              f"{st['turns_over_45']} turns >45°, {st['spins_over_5deg']} spins >5°; "
              f"repaired {st['repaired']}, sign tie-breaks {st['sign_flips']}")
    loc = report["relocalisation"]
    print()
    print("forward / reverse pose at the same video timestamp")
    print(f"  matched pairs     {loc['pairs']}")
    print(f"  RMSE              {loc['rmse']:.4f}")
    print(f"  median / mean     {loc['median_m']:.4f} / {loc['mean_m']:.4f}")
    print(f"  p95 / max         {loc['p95_m']:.4f} / {loc['max_m']:.4f}")
    print(f"  error at t=0      {loc['start_return_m']:.4f}   (should return home)")
    print(f"  error at turn     {loc['turn_return_m']:.4f}   (end of the forward clip)")
    print()
    print("error relative to distance travelled (scale-free)")
    print(f"  forward path      {loc['forward_path']:.4f}   reverse path {loc['reverse_path']:.4f}")
    print(f"  RMSE / path       {loc['rmse_pct']:.2f} %")
    print(f"  mean / path       {loc['mean_pct']:.2f} %")
    print(f"  p95 / path        {loc['p95_pct']:.2f} %")
    print(f"  max / path        {loc['max_pct']:.2f} %")
    print(f"  return-home / path {loc['start_return_pct']:.2f} %   (drift back at t=0)")


def run(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", nargs="?", type=Path, default=slam.DEFAULT_VIDEO)
    parser.add_argument("--seconds", type=float, default=40.0, help="clip length (default: 40)")
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--fov-deg", type=float, default=50.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-features", type=int, default=4000)
    parser.add_argument("--no-anchors", action="store_true")
    parser.add_argument("--no-guided", action="store_true", help="disable pose-guided anchor matching")
    parser.add_argument("--no-map-scale", action="store_true", help="experimental: per-step scene-depth scale")
    parser.add_argument("--profile", choices=sorted(slam.PROFILES), default=None, help="tracking profile (default: per video)")
    parser.add_argument("--json", type=Path, default=None, help="write the report as JSON")
    bench = parser.parse_args(argv)

    video = bench.video.expanduser().resolve()
    if not video.is_file():
        raise SystemExit(f"Video not found: {video}")

    extra = ["--no-anchors"] if bench.no_anchors else []
    if bench.no_guided:
        extra.append("--no-guided")
    if bench.no_map_scale:
        extra.append("--no-map-scale")
    if bench.profile:
        extra += ["--profile", bench.profile]
    slam_args = slam.parse_args(
        [
            str(video),
            "--scale", str(bench.scale),
            "--fov-deg", str(bench.fov_deg),
            "--stride", str(bench.stride),
            "--max-features", str(bench.max_features),
            "--no-disk-cache",
            "--no-mesh",
            *extra,
        ]
    )

    frames, fps = load_clip(video, bench.seconds, bench.scale)
    stride = max(bench.stride, 1)
    ram0 = rss_mb()
    session = slam.SlamSession(frames[0], slam_args, fps=fps)

    fwd_idx = [i for i in range(1, len(frames)) if i % stride == 0]
    rev_idx = [i for i in range(len(frames) - 2, -1, -1) if i % stride == 0]

    start = np.array([[0.0, *session.world_position.tolist()]], dtype=np.float64)
    t0 = time.perf_counter()
    forward = record_pass(session, [frames[i] for i in fwd_idx], [i / fps for i in fwd_idx])
    forward = np.vstack([start, forward])
    t1 = time.perf_counter()
    reverse = record_pass(session, [frames[i] for i in rev_idx], [i / fps for i in rev_idx])
    t2 = time.perf_counter()

    loc = match_by_timestamp(forward, reverse)
    n_fwd = max(len(fwd_idx), 1)
    n_rev = max(len(rev_idx), 1)
    ram1 = rss_mb()
    report = {
        "video": str(video),
        "seconds": float(len(frames) / fps),
        "frames": len(frames),
        "fps_file": fps,
        "forward_s": t1 - t0,
        "reverse_s": t2 - t1,
        "forward_fps": n_fwd / max(t1 - t0, 1e-9),
        "reverse_fps": n_rev / max(t2 - t1, 1e-9),
        "overall_fps": (n_fwd + n_rev) / max(t2 - t0, 1e-9),
        "ram_start_mb": ram0,
        "ram_end_mb": ram1,
        "ram_delta_mb": ram1 - ram0,
        "ram_peak_mb": peak_rss_mb(),
        "landmarks": len(session.map_store),
        "anchors": 0 if session.anchors is None else len(session.anchors),
        "fixes": int(session.stats["fixes"]),
        "fixes_guided": int(session.stats["guided_fixes"]),
        "fixes_rejected": int(session.stats["fix_rejected"]),
        "relocalisation": loc,
        "stability": stability(session),
        "profile": slam_args.profile_name,
    }
    print_report(report)
    if bench.json is not None:
        bench.json = bench.json.expanduser()
        bench.json.parent.mkdir(parents=True, exist_ok=True)
        bench.json.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {bench.json}")
    session.finalize()
    return report


if __name__ == "__main__":
    run()
