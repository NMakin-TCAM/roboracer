#!/bin/bash
# One predictive_race.launch.py run in AutoDRIVE, from standing, recorded passively.
#
#   sim_run.sh OUT DURATION [launch args...]
#
# In the container, on the bridge's ROS domain (the default), with the bridge running
# (ros2 launch autodrive_bridge autodrive_launch.py) and AutoDRIVE connected in Autonomous
# mode. Set OVERLAY to source a planning overlay after ~/ws/install. Reads
# $TRACKS_DIR/porto_odom_trial (map.pbstream for Cartographer) and writes nothing there.
# (no set -u: ROS setup.bash reads unset variables)
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
# The recorder outlives the launch by up to 30 s, so the shutdown (the controller's stop
# burst, and whether every node exits) is recorded too.
python3 "$HERE/sim_record.py" "$((DURATION + 32))" "$OUT" > "$OUT/recorder.log" 2>&1 &
REC=$!
setsid ros2 launch planning predictive_race.launch.py track:=porto_odom_trial \
    opponent_truth_topic:=/fake_opponent/truth "$@" > "$OUT/launch.log" 2>&1 &
LAUNCH=$!
sleep "$DURATION"
for pid in $(pgrep -f "lib/control/pure_pursuit_node"); do
    echo "pure_pursuit_node $pid $(grep SigIgn /proc/$pid/status) (bit 1 = SIGINT ignored)" | tee -a "$OUT/shutdown.log"
done
date +%s.%N > "$OUT/sigint_wall"
kill -INT -- -$LAUNCH 2>/dev/null
for i in $(seq 1 60); do kill -0 -- -$LAUNCH 2>/dev/null || break; sleep 0.5; done
if kill -0 -- -$LAUNCH 2>/dev/null; then
    echo "launch still alive 30 s after SIGINT:" | tee -a "$OUT/shutdown.log"
    ps -o pid,args -g $LAUNCH | tee -a "$OUT/shutdown.log"
    kill -KILL -- -$LAUNCH 2>/dev/null
else
    echo "launch exited $(python3 -c "import time;print(round(time.time()-float(open('$OUT/sigint_wall').read()),1))") s after SIGINT" | tee -a "$OUT/shutdown.log"
fi
wait $REC
echo "leftover nodes:"; ros2 node list 2>/dev/null | grep -v -E 'fast_bridge|adapter|wheel_odom|autodrive' || true
tail -1 "$OUT/recorder.log"
