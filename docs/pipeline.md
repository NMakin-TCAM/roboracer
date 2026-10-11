# Unseen-track pipeline

The stack is built to race a track it has never seen. Nothing track-specific is committed:
every map and raceline is generated on the day, into `ws/tracks/<name>/` (git-ignored).

```
new track ─► Cartographer mapping ─► centerline ─► TUM mintime raceline ─► localization + pure pursuit
             (keyboard teleop lap)
```

## Commands

Inside the container (`docker compose exec ros bash`), with the simulator connected:

```bash
ros2 launch autodrive_bridge autodrive_launch.py     # terminal 1: bridge (always running)
ws/scripts/map_track.sh <name>                      # terminal 2: map -> save -> centerline -> mintime
ros2 run autodrive_bridge teleop_keyboard           # terminal 3: drive the mapping lap(s)
ws/scripts/race.sh <name>                           # localization + pure pursuit on tracks/<name>/raceline.csv
```

`map_track.sh` starts Cartographer mapping, waits for you to finish driving and press Enter,
saves the map, then runs `ros2 run planning build_raceline <name>`.

## Packages (`ws/src/`)

| Package | Contents |
|---|---|
| `autodrive_bridge` | `fast_bridge`, `adapter`, `wheel_odom`, `teleop_keyboard`; `autodrive_launch.py`. Sim only: replaced by f1tenth_system on the car. |
| `perception` | Cartographer configs, `mapping.launch.py`, `localization.launch.py`, `save_map` |
| `planning` | `build_raceline` = `centerline` extraction + TUM `mintime` optimization |
| `control` | `pure_pursuit_node`, `pure_pursuit.launch.py` |

## Interfaces

| Topic / frame | Type | Producer |
|---|---|---|
| `/scan` (frame `laser`) | `sensor_msgs/LaserScan` | adapter (sim) / urg_node (car) |
| `/odom`, TF `odom -> base_link` | `nav_msgs/Odometry` | wheel_odom (sim) / vesc_to_odom (car) |
| `/sensors/imu/raw` | `sensor_msgs/Imu` | adapter (sim) / vesc (car) |
| `/drive` | `ackermann_msgs/AckermannDriveStamped` | teleop_keyboard, pure_pursuit |
| TF `map -> odom` | | Cartographer (mapping and localization) |
| `/tracked_pose` | `geometry_msgs/PoseStamped`, frame `map` | Cartographer (`publish_tracked_pose`) |
| `/ground_truth/pose` | `geometry_msgs/PoseStamped`, sim world frame | adapter, debugging only |

Static TF `base_link -> laser, imu, front_camera` comes from the adapter. `base_link` is the rear axle.

## Track directory `ws/tracks/<name>/`

| File | Written by | Format |
|---|---|---|
| `map.pbstream` | `save_map` | Cartographer state, loaded by `localization.launch.py` |
| `map.pgm`, `map.yaml` | `save_map` | ROS map_server occupancy grid, `map` frame |
| `trajectory.csv` | `save_map` | `# t,x_m,y_m,yaw_rad`: the mapping lap's `/tracked_pose`, gives driving direction |
| `map_to_world.yaml` | `save_map` | ground-truth pose of the `map` origin in the sim world frame (debugging) |
| `centerline.csv` | `build_raceline` | TUM input: `# x_m,y_m,w_tr_right_m,w_tr_left_m`, closed loop, driving direction |
| `raceline.csv` | `build_raceline` | TUM output: `# s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2`, `map` frame |
| `preview.png` | `build_raceline` | map, centerline, track bounds, raceline coloured by speed |

## Car parameters (AutoDRIVE RoboRacer)

| | |
|---|---|
| wheelbase | 0.33 m |
| max steering | 0.524 rad |
| wheel radius | 0.0585 m (calibrated against ground truth) |
| top speed | ~25 m/s per unit throttle (sim), throttle loop in `adapter.yaml` |
