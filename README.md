# SIGRobotics F1Tenth Stack

ROS 2 Humble in Docker, connected to the AutoDRIVE RoboRacer simulator.

```
Dockerfile, docker-compose.yml   the ROS 2 environment (any OS, amd64 or arm64)
ws/src/autodrive_bridge/         AutoDRIVE <-> ROS 2 bridge
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

| Topic | Direction | |
|---|---|---|
| `/scan` | out | LiDAR (`sensor_msgs/LaserScan`) |
| `/odom` | out | wheel-encoder odometry, `odom -> base_link` (`nav_msgs/Odometry`); `vesc_to_odom` on the car |
| `/sensors/imu/raw` | out | IMU (`sensor_msgs/Imu`) |
| TF `base_link -> laser, imu, front_camera` | out | static sensor mounts |
| `/drive` | in | `ackermann_msgs/AckermannDriveStamped` (speed m/s, steering rad) |
| `/ground_truth/odom`, TF `map -> ground_truth` | out | sim ground truth, debugging only; off with `publish_restricted_topics:=false` |

Topic names follow the official [f1tenth_system](https://github.com/f1tenth/f1tenth_system), so the same
stack runs on the car. `wheel_odom` and `adapter` are sim-only: on the car, `vesc_to_odom` publishes `/odom`
and the VESC does the speed control.

Raw sim topics are under `/autodrive/roboracer_1/*`. Foxglove: `ws://localhost:8765`.
