#!/usr/bin/env bash
# Map an unseen track and plan its minimum-time raceline (docs/pipeline.md):
#   Cartographer mapping -> save_map -> centerline -> TUM mintime
#
#   ws/scripts/map_track.sh <name>
#
# Needs the bridge running (ros2 launch autodrive_bridge autodrive_launch.py) and the sim connected.
# Drive the lap(s) from another terminal with: ros2 run autodrive_bridge teleop_keyboard
# Writes tracks/<name>/: map.pbstream, map.pgm, map.yaml, trajectory.csv, centerline.csv,
# raceline.csv, preview.png. Re-plan an existing map with: ros2 run planning build_raceline <name>
set -eo pipefail

name=${1:?usage: map_track.sh <track name>}
tracks=${TRACKS_DIR:-$HOME/ws/tracks}
source /opt/ros/humble/setup.bash
source "$HOME/ws/install/setup.bash"

if [ -e "$tracks/$name/map.pbstream" ]; then
    echo "$tracks/$name already has a map; pick another name or delete it" >&2
    exit 1
fi

# localization assumes the race starts where mapping started: the sim reset pose
reset_sim_car() {
    if ros2 topic info /autodrive/reset_command 2>/dev/null | grep -q 'Subscription count: [1-9]'; then
        ros2 topic pub --times 3 -r 10 /autodrive/reset_command std_msgs/msg/Bool '{data: true}' >/dev/null
        sleep 1
    fi
}

log="$HOME/ws/log/map_track_$name.log"
mkdir -p "$(dirname "$log")"
reset_sim_car
# job control: the launch gets its own process group and keeps SIGINT (a script's background
# jobs ignore it otherwise), so stopping it stops every node it started
set -m
ros2 launch perception mapping.launch.py track:="$name" >"$log" 2>&1 &
mapping=$!
set +m
stop_mapping() {
    kill -INT -- "-$mapping" 2>/dev/null || return 0
    for _ in $(seq 20); do kill -0 -- "-$mapping" 2>/dev/null || return 0; sleep 0.5; done
    kill -TERM -- "-$mapping" 2>/dev/null || true
}
trap stop_mapping EXIT

sleep 3
kill -0 "$mapping" 2>/dev/null || { echo "mapping failed to start, see $log" >&2; exit 1; }
echo "Mapping '$name' (log: $log)."
echo "Drive 1-2 clean laps in another terminal: ros2 run autodrive_bridge teleop_keyboard"
read -rp "Stop the car, then press Enter to save the map... "

ros2 run perception save_map "$name"
stop_mapping
trap - EXIT

ros2 run planning build_raceline "$name"
echo "Done: $tracks/$name (check preview.png). Race with: race $name  (scripts/race.sh)"
