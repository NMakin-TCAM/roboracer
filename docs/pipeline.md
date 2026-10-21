# Unseen-track pipeline

The stack is built to race a track it has never seen. Nothing track-specific is committed:
every map and raceline is generated on the day, into `ws/tracks/<name>/` (git-ignored).

```
new track ─► Cartographer mapping ─► centerline ─► TUM mintime raceline ─► localization + pure pursuit
             (keyboard teleop lap)
```

## Commands

Inside the container (`docker compose exec ros bash`, working directory `~/ws`), with the simulator connected:

```bash
ros2 launch autodrive_bridge autodrive_launch.py     # shell 1: bridge (always running)          alias: sim
scripts/map_track.sh <name>                         # shell 2: map -> save -> centerline -> mintime   map-track
ros2 run autodrive_bridge teleop_keyboard           # shell 3: drive the mapping lap(s)            kb
scripts/race.sh <name> [speed_scale:=0.5]           # localization + pure pursuit                  race
ros2 launch control follow_the_gap.launch.py [speed:=1.0]   # no-map fallback, instead of race.sh   ftg
```

`map_track.sh` resets the sim car, starts Cartographer mapping, waits for you to finish driving and
press Enter, runs `save_map`, stops mapping, then runs `ros2 run planning build_raceline <name>`.
`race.sh` resets the sim car (localization starts at the map origin = where mapping started) and
launches `control race.launch.py` (perception `localization.launch.py` + control `pure_pursuit.launch.py`).
If mapping fails on race day, `follow_the_gap.launch.py` drives on `/scan` alone (no map); run it instead of
`race.sh`, never alongside it, since both publish `/drive`.

## Packages (`ws/src/`)

| Package | Contents |
|---|---|
| `autodrive_bridge` | `fast_bridge`, `adapter`, `wheel_odom`, `teleop_keyboard`; `autodrive_launch.py`. Sim only: replaced by f1tenth_system on the car. |
| `perception` | Cartographer configs, `mapping.launch.py`, `localization.launch.py`, `save_map`, `map_origin`. Experimental, not in the pipeline: `ekf.launch.py` (robot_localization IMU EKF + `corrector`, a drift-corrected forward speed on `/odometry/corrected`, sim only) |
| `planning` | `build_raceline` = `centerline` extraction + TUM `mintime` optimization |
| `control` | `pure_pursuit_node`, `pure_pursuit.launch.py`, `race.launch.py`. `follow_the_gap_node` (`follow_the_gap.launch.py`): reactive `/scan` -> `/drive`, needs no map, the fallback |

## Interfaces

| Topic / frame | Type | Producer |
|---|---|---|
| `/scan` (frame `laser`) | `sensor_msgs/LaserScan` | adapter (sim) / urg_node (car) |
| `/odom`, TF `odom -> base_link` | `nav_msgs/Odometry` | wheel_odom (sim) / vesc_to_odom (car) |
| `/sensors/imu/raw` (frame `imu`) | `sensor_msgs/Imu`, accel includes gravity; Cartographer uses it | adapter (sim) / vesc (car) |
| `/drive` | `ackermann_msgs/AckermannDriveStamped` | teleop_keyboard, pure_pursuit, follow_the_gap |
| TF `map -> odom` | | Cartographer (mapping and localization) |
| `/tracked_pose` | `geometry_msgs/PoseStamped`, frame `map` | Cartographer (`publish_tracked_pose`) |
| `/localized_pose` | `geometry_msgs/PoseStamped`, frame `map` | localization_gate: `/tracked_pose` relayed once localized; withheld after a sim reset (`sim_reset`) until relocalized (race.launch.py's `pose_topic`) |
| `/autodrive/roboracer_1/sim_reset` | `std_msgs/String`, the reason | fast_bridge (sim only): one event per sim reset. Reset commands always; respawns and other teleports (collision count change, position jump) only with restricted topics. fast_bridge also replies zeros for `reset_frames` frames |
| `/ground_truth/pose`, TF `world -> ground_truth` | `geometry_msgs/PoseStamped`, frame `world` (sim) | adapter, debugging only |

Static TF `base_link -> laser, imu, front_camera` comes from the adapter. `base_link` is the rear axle.
Cartographer runs with `use_imu_data = true`, which requires `imu` at the origin of `base_link`
(rotation allowed). On the car, publish `base_link -> imu` with zero translation and the VESC's
mounting rotation, or Cartographer stops waiting for an IMU it cannot transform.

## Track directory `ws/tracks/<name>/`

| File | Written by | Format |
|---|---|---|
| `map.pbstream` | `save_map` | Cartographer state, loaded by `localization.launch.py` |
| `map.pgm`, `map.yaml` | `save_map` | ROS map_server occupancy grid, `map` frame |
| `trajectory.csv` | `save_map` | `# t,x_m,y_m,yaw_rad`: the mapping lap's optimized poses (`/trajectory_query`), gives driving direction |
| `map_to_world.yaml` | `save_map` | ground-truth pose of the `map` origin in the sim world frame (debugging) |
| `centerline.csv` | `build_raceline` | TUM input: `# x_m,y_m,w_tr_right_m,w_tr_left_m`, closed loop, driving direction |
| `raceline.csv` | `build_raceline` | TUM output: `# s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2`, `map` frame |
| `preview.png` | `build_raceline` | map, centerline, track bounds, raceline coloured by speed |

## Car parameters (AutoDRIVE RoboRacer)

| | |
|---|---|
| wheelbase, max steering | `ws/src/control/config/vehicle.yaml` (0.33 m, 0.524 rad), loaded by every node that uses them |
| wheel radius | 0.0585 m (calibrated against ground truth) |
| throttle | motor torque (sim VehicleController; ~25 m/s per unit throttle is only its steady state), speed loop in `adapter.yaml` |
| acceleration limit | 2.5 m/s^2 up, 6.9 m/s^2 down (`adapter.yaml` `max_acceleration`, `max_deceleration`): the adapter ramps the /drive speed as the car's speed controller would, so a launch from a standstill does not spin the wheels; after a /drive gap the ramp restarts from the measured speed |
