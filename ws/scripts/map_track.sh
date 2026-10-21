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
source "$(dirname "$0")/common.sh"

if [ -e "$tracks/$name/map.pbstream" ]; then
    echo "$tracks/$name already has a map; pick another name or delete it" >&2
    exit 1
fi
# the Enter prompts read the terminal: without one, `read` fails and the mapping run is lost
if ! { : </dev/tty; } 2>/dev/null; then
    echo "map_track.sh needs a terminal (run it in an interactive shell, e.g. docker compose exec ros bash)" >&2
    exit 1
fi

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

# ros2 launch stays alive when cartographer_node dies, so check the node's process (not
# `ros2 node list`, which keeps a dead node listed until its DDS lease runs out)
cartographer_up() { pgrep -g "$mapping" -f '^[^ ]*cartographer_ros/cartographer_node( |$)' >/dev/null; }
sleep 3   # a node that crashes on its config dies within a second or two
cartographer_up || { echo "Cartographer failed to start, see $log" >&2; exit 1; }
echo "Mapping '$name' (log: $log)."
echo "Drive 1-2 clean laps in another terminal: ros2 run autodrive_bridge teleop_keyboard"
read -rp "Stop the car, then press Enter to save the map... " </dev/tty

# keep mapping running on a failed save (the session is not on disk yet); save_map can be rerun
until ros2 run perception save_map "$name"; do
    cartographer_up || echo "cartographer_node is not running (see $log): the map cannot be saved" >&2
    read -rp "save_map failed. Enter to retry, q + Enter to quit without a map... " ans </dev/tty
    # a late failure may have written map.pbstream already, which blocks this name next time
    [ "$ans" != q ] || { echo "no map saved; delete $tracks/$name before reusing the name" >&2; exit 1; }
done
stop_mapping
trap - EXIT

ros2 run planning build_raceline "$name"
echo "Done: $tracks/$name (check preview.png). Race with: race $name  (scripts/race.sh)"
