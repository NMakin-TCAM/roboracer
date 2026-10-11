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
# AutoDRIVE Devkit

<p align="justify">
This directory hosts ROS 2 API (a meta-package), which supports modular algorithm development targetted towards autonomous driving. It uses client libraries for Python and C++, and can be therefore exploited by the users to develop their algorithms flexibly.
</p>

## SETUP

1. Clone the `AutoDRIVE-RoboRacer-Sim-Racing` repository.
    ```bash
    $ git clone https://github.com/AutoDRIVE-Ecosystem/AutoDRIVE-RoboRacer-Sim-Racing.git
    ```
2. Give executable permissions to the Python scripts.
   ```bash
   $ cd autodrive_devkit
   $ sudo chmod +x *.py
   ```
4. Install the necessary dependencies as mentioned below.
    [AutoDRIVE Devkit's ROS 2 API](https://github.com/AutoDRIVE-Ecosystem/AutoDRIVE-RoboRacer-Sim-Racing/tree/main/autodrive_devkit) has the following dependencies (tested with Python 3.8, 3.9 and 3.10):
    
    - Websocket-related dependencies for communication bridge between [AutoDRIVE Simulator](https://github.com/AutoDRIVE-Ecosystem/AutoDRIVE-RoboRacer-Sim-Racing/tree/main/autodrive_simulator) and [AutoDRIVE Devkit](https://github.com/AutoDRIVE-Ecosystem/AutoDRIVE-RoboRacer-Sim-Racing/tree/main/autodrive_devkit) (version sensitive):
    
      | Package            | Python 3.8 | Python 3.9 | Python 3.10 |
      |--------------------|------------|------------|-------------|
      | eventlet           | 0.33.3     | 0.33.3     | 0.33.3      |
      | Flask              | 1.1.1      | 1.1.1      | 1.1.1       |
      | Flask-SocketIO     | 4.1.0      | 4.1.0      | 4.1.0       |
      | python-socketio    | 4.2.0      | 4.2.0      | 4.2.0       |
      | python-engineio    | 3.13.0     | 3.13.0     | 3.13.0      |
      | greenlet           | 1.0.0      | 1.0.0      | 1.1.0       |
      | gevent             | 21.1.2     | 21.1.2     | 21.12.0     |
      | gevent-websocket   | 0.10.1     | 0.10.1     | 0.10.1      |
      | Jinja2             | 3.0.3      | 3.0.3      | 3.0.3       |
      | itsdangerous       | 2.0.1      | 2.0.1      | 2.0.1       |
      | werkzeug           | 2.0.3      | 2.0.3      | 2.0.3       |
    
    - Generic dependencies for data processing and visualization (usually any version will do the job):
    
      | Package               | Tested Version |
      |-----------------------|----------------|
      | attrdict              | 2.0.1          |
      | numpy                 | 1.22.2         |
      | pillow                | 9.0.1          |
      | opencv-contrib-python | 4.10.0.84      |
      | transforms3d          | 0.4.2          |
  
    - Install dependencies using `requirements.txt` file (use the file specific to your Python version &#8594; check using `python3 --version`):

      ```bash
      $ pip3 install -r requirements_python_3.8.txt # Python 3.8
      $ pip3 install -r requirements_python_3.9.txt # Python 3.9
      $ pip3 install -r requirements_python_3.10.txt # Python 3.10
      ```

    - ROS 2 dependencies for data processing and visualization (usually any version will do the job):

      ```bash
      $ sudo apt install ros-$ROS_DISTRO-tf-transformations
      $ sudo apt install ros-$ROS_DISTRO-imu-tools
      ```

## USAGE

- **Bringup:**
    - **Headless Mode Bringup:**
      ```bash
      $ ros2 launch autodrive_roboracer bringup_headless.launch.py
      ```
      **[OR]**
    - **Graphics Mode Bringup:**
      ```bash
      $ ros2 launch autodrive_roboracer bringup_graphics.launch.py
      ```

- **Teleoperation:**
  ```bash
  $ ros2 run autodrive_roboracer teleop_keyboard
  ```
