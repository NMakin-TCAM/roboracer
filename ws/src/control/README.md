# control

Pure pursuit on a generated raceline. `pure_pursuit_node` takes the car's pose in the `map` frame,
finds the closest raceline waypoint, picks the first waypoint one lookahead ahead (lookahead =
`lookahead_time` * drive speed, clipped to [`lookahead_min`, `lookahead_distance`]), steers on the arc
through it and drives at the raceline speed scaled by `speed_scale`. See `docs/pipeline.md` for where
it sits in the stack.

`follow_the_gap_node` is a reactive alternative that needs no map: it steers toward the farthest point
of the widest free gap in `/scan` at a constant `speed` (`ros2 run control follow_the_gap_node`).
`tools/lidar_plot.py` plots its scan processing live (WebAgg, not installed).

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
| `max_speed` | `0.0` | m/s cap on the drive speed, 0 = no cap |
| `lookahead_time` | `0.4` | s; lookahead = `lookahead_time` * drive speed, clipped to [`lookahead_min`, `lookahead_distance`] |
| `lookahead_min` | `0.6` | m, about twice the wheelbase |
| `lookahead_distance` | `3.0` | m, maximum lookahead |
| `search_window` | `2.0` | m; the closest point is searched this far along the line either side of the previous one |
| `reacquire_distance` | `1.0` | m; farther than this from that point, the whole line is searched (points heading within 90 deg of the car) |
| `wheelbase` | none | m, from `config/vehicle.yaml` (required) |
| `max_steering_angle` | none | rad, steering is clipped to this; from `config/vehicle.yaml` (required) |

The raceline is the TUM output format: `;` separated, `#` header
`s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2`.

## Test

```bash
python3 -m pytest ws/src/control/test   # needs a sourced ROS 2 environment and the built package
```

## Credit

Adapted from the F1TENTH / UPenn ESE 6150 pure-pursuit lab template (MIT, see `LICENSE`).
