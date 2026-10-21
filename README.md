# SIGRobotics F1Tenth Stack

ROS 2 Humble in Docker, connected to the AutoDRIVE RoboRacer simulator.

Built to race an unseen track: map it, plan a minimum-time raceline, race it.

```
Dockerfile, docker-compose.yml   the ROS 2 environment (any OS, amd64 or arm64)
ws/src/autodrive_bridge/         AutoDRIVE <-> ROS 2 bridge, keyboard teleop (sim only)
ws/src/perception/               Cartographer mapping + localization, save_map; experimental IMU EKF velocity
ws/src/planning/                 centerline extraction + TUM minimum-time raceline
ws/src/control/                  pure pursuit, follow the gap
ws/scripts/                      map_track.sh, race.sh (+ common.sh, sourced by both)
ws/tracks/<name>/                generated per track (git-ignored)
docs/pipeline.md                 interfaces, frames and track file formats
```

## Setup (once)

Install [Docker](https://docs.docker.com/get-docker/) and download the AutoDRIVE RoboRacer
simulator for your OS (release `2026-iros`).

```bash
git clone https://github.com/SIGRobotics-UIUC/roboracer.git && cd roboracer
docker compose build
```

## Run

```bash
docker compose up -d && docker compose exec ros bash
colcon build --symlink-install && source install/setup.bash     # inside the container
ros2 launch autodrive_bridge autodrive_launch.py
```

Start the simulator, **Connect** to `127.0.0.1:4567`, and set **Driving Mode** to **Autonomous**.
Restarting the bridge while the simulator is connected makes it reconnect in a loop: **Disconnect** first.

## Race an unseen track

In new container shells (`docker compose exec ros bash`), with the bridge running:

```bash
scripts/map_track.sh <name>                  # Cartographer mapping; drive, press Enter -> map, centerline, mintime raceline
ros2 run autodrive_bridge teleop_keyboard    # meanwhile, in another shell: drive 1-2 clean laps (W/S speed, A/D steer, X stop)
scripts/race.sh <name> speed_scale:=0.5      # reset the car, localize on the saved map, pure pursuit on the raceline
```

Check `tracks/<name>/preview.png` before racing. To re-plan an existing map (e.g. new car parameters in
`ws/src/planning/config/racecar.ini`): `ros2 run planning build_raceline <name>`.
No map (mapping failed)? `ros2 launch control follow_the_gap.launch.py [speed:=1.0]` drives on the LiDAR alone.
Stop launches with Ctrl-C, not `kill`/`timeout` (SIGTERM to `ros2 launch` leaves its nodes running, see `docs/pipeline.md`).
The aliases in `.bash_aliases` (`map-track`, `race`, `ftg`, `kb`, `reset-car`, ...) wrap these.

| Topic | Direction | |
|---|---|---|
| `/scan` | out | LiDAR (`sensor_msgs/LaserScan`) |
| `/odom` | out | wheel-encoder odometry, `odom -> base_link` (`nav_msgs/Odometry`); `vesc_to_odom` on the car |
| `/sensors/imu/raw` | out | IMU (`sensor_msgs/Imu`) |
| TF `base_link -> laser, imu, front_camera` | out | static sensor mounts |
| `/drive` | in | `ackermann_msgs/AckermannDriveStamped` (speed m/s, steering rad) |
| TF `map -> odom`, `/tracked_pose`, `/localized_pose` | out | Cartographer pose in the `map` frame; `/localized_pose` is the same pose, only once localized (localization_gate; what race.launch drives on) |
| `/ground_truth/odom`, `/ground_truth/pose`, TF `world -> ground_truth` | out | sim ground truth in the sim `world` frame, debugging only; off with `publish_restricted_topics:=false` |
| `/autodrive/roboracer_1/sim_reset` | out | fast_bridge: one `std_msgs/String` event per sim reset (respawns only with restricted topics); localization_gate relocalizes on it |

Topic names follow the official [f1tenth_system](https://github.com/f1tenth/f1tenth_system), so the same
stack runs on the car. `fast_bridge`, `adapter`, `wheel_odom` and `teleop_keyboard` are sim-only: on the car, `vesc_to_odom` publishes `/odom`
and the VESC does the speed control.

Raw sim topics are under `/autodrive/roboracer_1/*`. Foxglove: `ws://localhost:8765`.
