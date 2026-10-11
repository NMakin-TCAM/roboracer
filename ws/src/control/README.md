# control

Pure pursuit on a generated raceline. `pure_pursuit_node` takes the car's pose in the `map` frame,
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

## Test

```bash
python3 -m pytest ws/src/control/test   # needs a sourced ROS 2 environment and the built package
```

## Credit

Adapted from the F1TENTH / UPenn ESE 6150 pure-pursuit lab template (MIT, see `LICENSE`).
