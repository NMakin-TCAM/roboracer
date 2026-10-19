# Sourced by map_track.sh and race.sh: ROS environment, tracks dir, sim reset.
tracks=${TRACKS_DIR:-$HOME/ws/tracks}
source /opt/ros/humble/setup.bash
source "$HOME/ws/install/setup.bash"

# Teleport the sim car to its start pose, if the bridge is running (no-op on the car).
# Localization assumes the race starts where mapping started: this reset pose.
reset_sim_car() {
    if ros2 topic info /autodrive/reset_command 2>/dev/null | grep -q 'Subscription count: [1-9]'; then
        ros2 topic pub --times 3 -r 10 /autodrive/reset_command std_msgs/msg/Bool '{data: true}' >/dev/null
        sleep 1
    fi
}
