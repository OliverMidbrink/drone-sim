"""Is the SLAM map itself consistent, or is the filter reading it wrongly?

    python virtual-sim/tools/map_consistency.py out/jump/map_s7 [more prefixes]

Reads a `--diag` run's slam.csv (the raw SLAM position and the camera's true displacement at each picture) and, per camera:
    * fits ONE similarity (rotation, translation, scale) from the SLAM path to the true path over the first lap and shows how far
      the later laps fall from it (a consistent map keeps them close: the old anchors bind the pose to the first lap's frame);
    * fits a similarity per 20 s window and shows its scale (SLAM units per metre) next to the filter's scale: how the map's
      unit wanders and whether the filter follows it.
"""

from __future__ import annotations

import sys

import numpy as np

from diag_report import load


def similarity(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Umeyama: scale s, rotation R, translation t with target ~ s R source + t."""
    mu_s, mu_t = source.mean(0), target.mean(0)
    a, b = source - mu_s, target - mu_t
    u, d, vt = np.linalg.svd(b.T @ a / len(a))
    sign = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        sign[2, 2] = -1
    rotation = u @ sign @ vt
    scale = float(np.trace(np.diag(d) @ sign) / (a * a).sum(1).mean())
    return scale, rotation, mu_t - scale * rotation @ mu_s


def report(prefix: str) -> None:
    slam = load(f"{prefix}_slam.csv")
    for cam in sorted(set(slam["cam"])):
        rows = slam["cam"] == cam
        t = slam["t"][rows]
        p = np.stack([slam["slam_x"], slam["slam_y"], slam["slam_z"]], 1)[rows]
        q = np.stack([slam["cam_true_x"], slam["cam_true_y"], slam["cam_true_z"]], 1)[rows]
        lam = slam["filter_scale"][rows]
        ok = np.isfinite(p).all(1) & np.isfinite(q).all(1)
        t, p, q, lam = t[ok], p[ok], q[ok], lam[ok]
        start = t[0]
        first = (t > start + 5) & (t < start + 45)
        s, r, c = similarity(p[first], q[first])
        residual = np.linalg.norm((s * (r @ p.T)).T + c - q, axis=1)
        print(f"{prefix} [{cam}] first-lap fit: {1 / s:.2f} units/m; distance of the path from that fit per 20 s:")
        windows = np.arange(start, t[-1], 20.0)
        print("    t    from-lap1  window-fit-err  map units/m  filter units/m")
        for lo in windows:
            m = (t >= lo) & (t < lo + 20)
            if m.sum() < 10:
                continue
            ws, wr, wc = similarity(p[m], q[m])
            wres = np.linalg.norm((ws * (wr @ p[m].T)).T + wc - q[m], axis=1)
            print(f"  {lo - start:5.0f}   {np.median(residual[m]):7.2f}     {np.median(wres):7.2f}        {1 / ws:7.2f}      {np.median(lam[m]):7.2f}")


if __name__ == "__main__":
    for prefix in sys.argv[1:]:
        report(prefix)
