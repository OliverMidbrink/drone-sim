"""Run flight configurations over several seeds in parallel and tabulate route error and estimator jumps.

    python virtual-sim/tools/jump_experiment.py --seeds 7 21 33 --seconds 90 NAME="--flag value ..." NAME2="..." ...

Each flight is `fpv.py --slam --hold-test SECONDS --seed S --diag out/jump/NAME_sS`. Reported per configuration (mean over seeds, then the worst seed):
route error (true distance from the planned point), estimator error, and jumps: how often the estimate moved 0.3 m more than the drone really did
within half a second (from the 0.5 s state log), plus the largest such jump.
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from diag_report import load

HERE = Path(__file__).resolve().parent
FPV = HERE.parent / "fpv.py"
OUT = HERE.parent / "out" / "jump"


def run(name: str, flags: str, seed: int, seconds: float, extra: list[str]) -> dict:
    prefix = OUT / f"{name}_s{seed}"
    cmd = [sys.executable, str(FPV), "--slam", "--hold-test", str(seconds), "--seed", str(seed), "--diag", str(prefix), *extra, *shlex.split(flags)]
    text = subprocess.run(cmd, capture_output=True, text=True).stdout
    route = re.search(r"route flown for ([\d.]+) s .*?mean ([\d.]+) m\s+rms ([\d.]+) m\s+max ([\d.]+) m;\s+estimator error mean ([\d.]+) m\s+max ([\d.]+)", text)
    out = dict(name=name, seed=seed, crashed="CRASHED" in text)
    if route:
        out.update(flown=float(route[1]), mean=float(route[2]), rms=float(route[3]), max=float(route[4]), est=float(route[5]))
    else:
        out.update(flown=0.0, mean=np.nan, rms=np.nan, max=np.nan, est=np.nan)
    try:
        state = load(f"{prefix}_state.csv")
        slam = load(f"{prefix}_slam.csv")
        truth = np.stack([state["truth_x"], state["truth_y"], state["truth_z"]], 1)
        est = np.stack([state["est_x"], state["est_y"], state["est_z"]], 1)
        route_rows = state["mission"] == "route"
        jump = np.linalg.norm(np.diff(est, axis=0) - np.diff(truth, axis=0), axis=1)[route_rows[1:]]
        out.update(jumps=int((jump > 0.3).sum()), jump_max=float(jump.max()) if len(jump) else np.nan,
                   fixes=int(slam["fix_now"].sum()), refused=int((slam["fix_refused"] == 1).sum()), banned=int(max(0, np.nanmax(slam["fix_flagged"]) if len(slam["fix_flagged"]) else 0)))
    except Exception as error:  # noqa: BLE001
        out.update(jumps=-1, jump_max=np.nan, fixes=-1, refused=-1, banned=-1, error=str(error))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 21, 33])
    parser.add_argument("--seconds", type=float, default=90.0)
    parser.add_argument("--parallel", type=int, default=5)
    parser.add_argument("--extra", default="", help="flags added to every flight")
    parser.add_argument("configs", nargs="+", help='NAME="flags"')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    configs = [c.split("=", 1) for c in args.configs]
    jobs = [(name, flags, seed) for name, flags in configs for seed in args.seeds]
    with ThreadPoolExecutor(args.parallel) as pool:
        results = list(pool.map(lambda j: run(j[0], j[1], j[2], args.seconds, shlex.split(args.extra)), jobs))
    print(f"\n{'config':<18}{'route err mean (per seed)':<34}{'mean':>6}{'worst':>7}{'est err':>8}{'jumps>0.3m':>12}{'max jump':>9}{'fixes':>7}{'refused':>8}{'crash':>6}")
    for name, _ in configs:
        rows = [r for r in results if r["name"] == name]
        per_seed = " ".join(f"{r['mean']:5.2f}" for r in rows)
        print(f"{name:<18}{per_seed:<34}{np.nanmean([r['mean'] for r in rows]):6.2f}{np.nanmax([r['mean'] for r in rows]):7.2f}{np.nanmean([r['est'] for r in rows]):8.2f}"
              f"{np.mean([r['jumps'] for r in rows]):12.1f}{np.nanmax([r['jump_max'] for r in rows]):9.2f}{np.mean([r['fixes'] for r in rows]):7.0f}{np.mean([r['refused'] for r in rows]):8.1f}{sum(r['crashed'] for r in rows):6d}")


if __name__ == "__main__":
    main()
