#!/usr/bin/env bash
# Race a mapped track: Cartographer localization + pure pursuit on its minimum-time raceline.
#
#   ws/scripts/race.sh <name> [speed_scale:=0.5] [max_speed:=0.0] [pose_topic:=/tracked_pose]
#
# Needs the bridge running and tracks/<name>/ from ws/scripts/map_track.sh <name>.
# Resets the sim car first: localization starts from the pose mapping started from. Ctrl-C stops.
set -eo pipefail

name=${1:?usage: race.sh <track name> [launch args]}
shift
source "$(dirname "$0")/common.sh"

for f in map.pbstream raceline.csv; do
    [ -e "$tracks/$name/$f" ] || { echo "$tracks/$name/$f missing; run map-track $name  (scripts/map_track.sh)" >&2; exit 1; }
done

reset_sim_car

exec ros2 launch control race.launch.py track:="$name" "$@"
