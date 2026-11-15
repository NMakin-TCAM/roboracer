#!/bin/bash
# In the container, planning overlay and control sourced, a private ROS_DOMAIN_ID: the closed-loop clearance sweep on porto, one run per evasion_dist.
: "${TRACKS_DIR:?set TRACKS_DIR to a copy of the tracks (never the validated originals)}"
mkdir -p "${OUT:-/tmp/clearance_sweep}"
for ev in 0.35 0.30 0.25 0.20 0.15 0.10 0.05 0.00; do
  timeout 260 python3 "$(dirname "$0")"/closed_loop.py \
    porto_odom_trial $ev 150 "${OUT:-/tmp/clearance_sweep}"/ev_$ev > "${OUT:-/tmp/clearance_sweep}"/ev_$ev.out 2>&1
  echo "ev $ev: $(tail -2 "${OUT:-/tmp/clearance_sweep}"/ev_$ev.out | tr '\n' ' ')"
done
