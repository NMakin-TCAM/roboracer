# control

Pure pursuit on a generated raceline, and a wall follower for the mapping lap(s) (see below). `pure_pursuit_node` takes the car's pose in the `map` frame,
finds the closest raceline waypoint, picks the waypoint one lookahead distance ahead (shorter in
curves: `min(3, 1/(7 kappa))`), steers on the arc through it and drives at the raceline speed scaled
by `speed_scale`. See `docs/pipeline.md` for where it sits in the stack.

## Run

```bash
ros2 launch control pure_pursuit.launch.py track:=<name>
ros2 launch control pure_pursuit.launch.py track:=<name> pose_topic:=/ground_truth/pose speed_scale:=0.3
```

Localization (Cartographer, `/tracked_pose`) must be running for the default pose topic.

## Topics

| Topic | Type | |
|---|---|---|
| `pose_topic` (`/tracked_pose`) | `geometry_msgs/PoseStamped` | subscribed, `map` frame |
| `/drive` | `ackermann_msgs/AckermannDriveStamped` | published, one command per pose |

## Parameters

| Parameter | Default | |
|---|---|---|
| `track` | `''` | loads `$TRACKS_DIR/<track>/raceline.csv`, `TRACKS_DIR` defaults to `~/ws/tracks` |
| `raceline_path` | `''` | explicit raceline file, overrides `track` |
| `pose_topic` | `/tracked_pose` | `/ground_truth/pose` for debugging in the sim |
| `speed_scale` | `0.5` | speed = raceline `vx_mps` * `speed_scale` |
| `lookahead_distance` | `3.0` | initial lookahead (m), then updated from curvature |
| `wheelbase` | `0.33` | m, AutoDRIVE RoboRacer |
| `max_steering_angle` | `0.524` | rad, steering is clipped to this |

The raceline is the TUM output format: `;` separated, `#` header
`s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2`.

## Wall follower

`wall_follow_node` drives the mapping lap(s) of an unseen track: it needs only `/scan`, no map or
localization. It measures the left and right wall distances (two beams per side, projected
`lookahead` m ahead), steers with a PD controller to stay centred, and slows down in corners. With
`laps > 0` it counts returns to its start position on `pose_topic`, then stops the car and exits.

```bash
scripts/map_track.sh <name> --wall-follow [--laps 2] [speed:=1.0]   # mapping with the wall follower
ros2 launch control wall_follow.launch.py speed:=1.0 debug:=true     # on its own, until Ctrl+C
```

| Topic | Type | |
|---|---|---|
| `/scan` | `sensor_msgs/LaserScan` | subscribed |
| `pose_topic` (`/tracked_pose`) | `geometry_msgs/PoseStamped` | subscribed when `laps > 0` (Cartographer, also while mapping) |
| `/drive` | `ackermann_msgs/AckermannDriveStamped` | published, one command per scan; zero on exit |

| Parameter | Default | |
|---|---|---|
| `speed` | `1.0` | m/s on straights; keep it low for clean scan matching |
| `kp`, `kd` | `1.05`, `0.10` | rad per m off centre, rad per m/s of drift |
| `lookahead` | `0.8` | m ahead at which the wall distances are projected |
| `beam_angle_deg` | `45` | angle between the side beam and the forward beam |
| `max_range` | `5.0` | m, a side with no return within this has no wall |
| `track_width` | `1.2` | m, holds half of it from the wall when only one wall is visible |
| `corner_slowdown` | `0.5` | fraction of `speed` shed at full steering |
| `max_steering_angle` | `0.524` | rad, steering is clipped to this |
| `laps` | `0` | stop after this many laps, 0 = never |
| `lap_leave_radius`, `lap_return_radius`, `lap_min_distance` | `3.0`, `1.5`, `10.0` | m: a lap is leaving the start by the first, coming back within the second, after driving at least the third |
| `debug` | `false` | log wall distances, steering and speed |

## Test

```bash
python3 -m pytest ws/src/control/test   # needs a sourced ROS 2 environment and the built package
```

The tests publish `/scan`, `/tracked_pose` and read `/drive`: with the bridge running, give them their
own `ROS_DOMAIN_ID` (e.g. `ROS_DOMAIN_ID=42 python3 -m pytest ...`) so they don't drive the sim car.

## Credit

Adapted from the F1TENTH / UPenn ESE 6150 pure-pursuit lab template (MIT, see `LICENSE`).
