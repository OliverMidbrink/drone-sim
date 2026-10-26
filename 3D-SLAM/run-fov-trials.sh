#!/usr/bin/env bash
# Run three SLAM sessions with different horizontal FOV guesses; you pick the best map.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
SLAM="${ROOT}/3D-SLAM/3d-slam.py"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing venv: $PYTHON" >&2
  exit 1
fi

# Narrow / near-default / wide — bracket typical phone vs action-cam footage.
FOVS=(50 60 90)
LABELS=("narrow (50°)" "medium (60°)" "wide (90°)")

echo "FOV comparison: ${#FOVS[@]} trials, same video and SLAM settings."
echo "Watch the 3D map; press q in the track window when done with each run."
echo "Optional args are passed through, e.g.: $0 --scale 1.0"
echo

for i in "${!FOVS[@]}"; do
  fov="${FOVS[$i]}"
  label="${LABELS[$i]}"
  n=$((i + 1))
  echo "============================================================"
  echo " Trial ${n}/${#FOVS[@]}: horizontal FOV ${fov}° — ${label}"
  echo "============================================================"
  "$PYTHON" "$SLAM" --fov-deg "$fov" "$@"
  if [[ "$n" -lt "${#FOVS[@]}" ]]; then
    read -r -p "Enter for next trial (Ctrl+C to stop)… " _
  fi
done

echo
echo "Done. Note which trial (1=${FOVS[0]}°, 2=${FOVS[1]}°, 3=${FOVS[2]}°) looked best."
