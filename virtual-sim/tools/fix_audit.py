"""Audit the anchor fixes of --diag flight logs: which ones helped, which ones threw the pose away, and what tells them apart.

    python virtual-sim/tools/fix_audit.py PREFIX [PREFIX ...]

A fix is "bad" when the SLAM's own position error (against the truth, at the filter's scale) grows by more than --bad metres across it,
"good" when it shrinks by more than that. For both, the numbers the SLAM and the filter had at that moment are printed so a rule that
separates them can be read off (ground truth is used only here, to label; a rule may only use the columns that are inputs).
"""

from __future__ import annotations

import argparse

import numpy as np

from diag_report import load


def audit(prefix: str, bad: float):
    slam = load(f"{prefix}_slam.csv")
    rows = []
    for cam in ("left", "right"):
        idx = np.flatnonzero(slam["cam"] == cam)
        for a, b in zip(idx[:-1], idx[1:]):
            if slam["fix_now"][b] != 1:
                continue
            before, after = slam["slam_pos_err_m"][a], slam["slam_pos_err_m"][b]
            scale = max(slam["filter_scale"][b], 0.02)
            rows.append(dict(
                prefix=prefix, t=slam["t"][b], cam=cam, before=before, after=after, delta=after - before,
                move_m=slam["fix_move"][b] / scale, inliers=slam["fix_inliers"][b], matches=slam["fix_matches"][b], age_med=slam["fix_age_med"][b],
                age_min=slam["fix_age_min"][b], fix_scale=slam["fix_scale"][b], rot=slam["fix_rot_deg"][b], live=slam["fix_live"][b],
                pairs=slam["fix_pairs"][b], innovation=slam["innovation"][b], believed=slam["believed"][b], turn=slam["turn_rate"][b],
            ))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", nargs="+")
    parser.add_argument("--bad", type=float, default=0.3)
    args = parser.parse_args()
    rows = [r for p in args.prefix for r in audit(p, args.bad)]
    if not rows:
        print("no fixes")
        return
    delta = np.array([r["delta"] for r in rows])
    good, worse = delta < -args.bad, delta > args.bad
    print(f"{len(rows)} fixes: {int(good.sum())} good (error shrinks > {args.bad} m), {int(worse.sum())} bad (grows > {args.bad} m), {int((~good & ~worse).sum())} neutral")
    keys = ["move_m", "inliers", "matches", "age_med", "age_min", "fix_scale", "rot", "live", "pairs", "innovation", "believed", "turn", "before"]
    print(f"{'':>10}" + "".join(f"{k:>11}" for k in keys))
    for label, mask in (("good", good), ("neutral", ~good & ~worse), ("bad", worse)):
        if mask.any():
            print(f"{label:>10}" + "".join(f"{np.nanmedian([r[k] for r, m in zip(rows, mask) if m]):11.2f}" for k in keys) + f"   (n={int(mask.sum())}, median)")
    print("\nbad fixes:")
    for r, m in zip(rows, worse):
        if m:
            print(f"  {r['prefix'].split('/')[-1]:>10} t={r['t']:6.1f} {r['cam']:>5}: err {r['before']:.2f} -> {r['after']:.2f} m, move {r['move_m']:.2f} m, inliers {r['inliers']:.0f}, age med {r['age_med']:.0f}, "
                  f"scale x{r['fix_scale']:.2f}, rot {r['rot']:.1f} deg, believed {int(r['believed'])}")


if __name__ == "__main__":
    main()
