# ==============================================================================
# ROS 2 Core & Workspace Management
# ==============================================================================
# Quickly source ROS 2 Humble base and your local workspace
alias sb="source /opt/ros/humble/setup.bash && source ~/ws/install/setup.bash"

# Build options using colcon
alias cb="cd ~/ws && colcon build --symlink-install"
alias cbp="cd ~/ws && colcon build --symlink-install --packages-select"
alias cbclean="cd ~/ws && rm -rf build/ install/ log/ && cb"

# ==============================================================================
# F1TENTH Gym & Sim Launchers
# ==============================================================================
# Run the F1TENTH Gym ROS 2 node/bringup directly[cite: 1]
alias f1sim="ros2 launch f1tenth_gym_ros gym_bridge_launch.py"
alias kb="ros2 run teleop_twist_keyboard teleop_twist_keyboard"

# Open RViz2 configured with standard ROS 2 topic visualization
alias rviz="rviz2"

# Reset car to global origin (0, 0, 0)
alias origin-car="ros2 topic pub -1 /initialpose geometry_msgs/msg/PoseWithCovarianceStamped '{header: {frame_id: \"map\"}, pose: {pose: {position: {x: 0.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}}'"

# Stop car instantly (publish 0 m/s speed and 0 rad steer)
alias stop-car="ros2 topic pub -1 /drive ackermann_msgs/msg/AckermannDriveStamped '{header: {frame_id: \"base_link\"}, drive: {speed: 0.0, steering_angle: 0.0}}'"

# Full reset: stops the vehicle AND teleports it back to start
alias reset-car="stop-car && origin-car"

# ==============================================================================
# Topic Inspection & Command Shortcuts
# ==============================================================================
# List all active topics
alias rtl="ros2 topic list"

# Echo LiDAR scan stream[cite: 1]
alias echo-scan="ros2 topic echo /scan"

# Echo Odometry stream (sim vs real shortcuts)[cite: 1]
alias echo-odom="ros2 topic echo /ego_racecar/odom"
alias echo-real-odom="ros2 topic echo /odom"

# Echo drive command output[cite: 1]
alias echo-drive="ros2 topic echo /drive"

# ==============================================================================
# ROS 2 Node & Parameter Utilities
# ==============================================================================
alias rnl="ros2 node list"
alias rpl="ros2 param list"
