#!/bin/bash
# One predictive_race.launch.py run in AutoDRIVE, from standing, recorded passively.
#
#   sim_run.sh OUT DURATION [launch args...]
#
# In the container, on the bridge's ROS domain (the default), with the bridge running
# (ros2 launch autodrive_bridge autodrive_launch.py) and AutoDRIVE connected in Autonomous
# mode. Set OVERLAY to source a planning overlay after ~/ws/install. Reads
# $TRACKS_DIR/porto_odom_trial (map.pbstream for Cartographer) and writes nothing there.
set -u
OUT=${1:?usage: sim_run.sh OUT DURATION [launch args]}
DURATION=${2:?}
shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
source /opt/ros/humble/setup.bash
source /root/ws/install/setup.bash
[ -n "${OVERLAY:-}" ] && source "$OVERLAY"
source /root/ws/scripts/common.sh
mkdir -p "$OUT"

# Preflight: the bridge is up and the simulator is streaming. Fail rather than guess.
if ! timeout 10 ros2 topic echo --once /autodrive/roboracer_1/lidar --field header >/dev/null 2>&1; then
    echo "no /autodrive/roboracer_1/lidar within 10 s: bridge not running or simulator not connected" >&2
    exit 2
fi
if ros2 node list 2>/dev/null | grep -qE 'spliner_node|pure_pursuit_node|local_path_node|cartographer'; then
    echo "a planner/controller/localizer is already running; refusing to start a second" >&2
    ros2 node list >&2
    exit 3
fi

reset_sim_car
date +%s.%N > "$OUT/start_wall"
python3 "$HERE/sim_record.py" "$DURATION" "$OUT" > "$OUT/recorder.log" 2>&1 &
REC=$!
setsid ros2 launch planning predictive_race.launch.py track:=porto_odom_trial \
    opponent_truth_topic:=/fake_opponent/truth "$@" > "$OUT/launch.log" 2>&1 &
LAUNCH=$!
wait $REC
kill -INT -- -$LAUNCH 2>/dev/null
for i in $(seq 1 30); do kill -0 -- -$LAUNCH 2>/dev/null || break; sleep 0.5; done
kill -KILL -- -$LAUNCH 2>/dev/null
sleep 1
echo "leftover nodes:"; ros2 node list 2>/dev/null | grep -v -E 'fast_bridge|adapter|wheel_odom|autodrive' || true
tail -1 "$OUT/recorder.log"
