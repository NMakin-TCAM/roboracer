# control

Pure pursuit on a generated raceline. `pure_pursuit_node` takes the car's pose in the `map` frame,
finds the closest raceline waypoint, picks the first waypoint one lookahead ahead (lookahead =
`lookahead_time` * drive speed, clipped to [`lookahead_min`, `lookahead_distance`]), steers on the arc
through it and drives at the raceline speed scaled by `speed_scale`. The speed is taken at the target
waypoint, not the closest one: the car brakes early into corners, but on corner exits it can exceed the
profile at its own position near `speed_scale` 1.0. See `docs/pipeline.md` for where
it sits in the stack.

`follow_the_gap_node` is a reactive alternative that needs no map: it steers toward the farthest point
(the middle of the run of farthest beams, since ranges are capped at `max_range`) of the widest free
gap within `fov` (default ±90°) of straight ahead in `/scan`, at a constant `speed` (`ros2 run control follow_the_gap_node`, or
`ros2 launch control follow_the_gap.launch.py`, which loads `config/vehicle.yaml`). Its steering is
clipped to `max_steering_angle` (default 0.524 rad, the value in `config/vehicle.yaml`).
`tools/lidar_plot.py` plots its scan processing live (WebAgg, not installed).

## Run

```bash
ros2 launch control pure_pursuit.launch.py track:=<name>
ros2 launch control pure_pursuit.launch.py track:=<name> speed_scale:=0.3
```

Localization (Cartographer, `/tracked_pose`) must be running for the default pose topic.
`race.launch.py` instead uses `/localized_pose` (localization_gate), so the car waits until
Cartographer has localized (~2-3 s at the start and after each sim reset). The sim's `/ground_truth/pose` is in the sim `world` frame, not `map`
(the offset is in `tracks/<name>/map_to_world.yaml`), so pure pursuit refuses it.

## Safety

Each pose is checked before it is used; a failed check publishes a stop (speed 0, steering 0) and a
warning (at most once a second), and driving resumes on the next good pose:

- `header.frame_id` must be `map`;
- the pose must be at most `max_pose_age` s old (an unstamped pose counts as stale);
- the car must be within `max_offset` m of the raceline.

These are coarse. Cartographer extrapolates `/tracked_pose` (`use_pose_extrapolator` defaults to true;
`pose_publish_period_sec` 1e-2), so its stamps stay fresh even when SLAM stalls: the age check only
catches DDS backlog or a non-Cartographer publisher. `max_offset` only catches gross errors: a wrong
pose on a nearby leg (iros2026's legs are 1.07 m apart) passes it. There is no `/scan`-based stop; if
poses stop arriving, `fast_bridge` stops the car after 0.5 s without `/drive`.

## Topics

| Topic | Type | |
|---|---|---|
| `pose_topic` (`/tracked_pose`; race.launch.py: `/localized_pose`) | `geometry_msgs/PoseStamped` | subscribed, must be in the `map` frame |
| `/drive` | `ackermann_msgs/AckermannDriveStamped` | published, one command per pose |

## Parameters

| Parameter | Default | |
|---|---|---|
| `track` | `''` | loads `$TRACKS_DIR/<track>/raceline.csv`, `TRACKS_DIR` defaults to `~/ws/tracks` |
| `raceline_path` | `''` | explicit raceline file, overrides `track` |
| `pose_topic` | `/tracked_pose` (race.launch.py: `/localized_pose`) | any `PoseStamped` of `base_link` in the `map` frame |
| `speed_scale` | `0.5` | speed = raceline `vx_mps` * `speed_scale` |
| `max_speed` | `0.0` | m/s cap on the drive speed, 0 = no cap |
| `lookahead_time` | `0.4` | s; lookahead = `lookahead_time` * drive speed, clipped to [`lookahead_min`, `lookahead_distance`] |
| `lookahead_min` | `0.6` | m, about twice the wheelbase |
| `lookahead_distance` | `3.0` | m, maximum lookahead |
| `search_window` | `2.0` | m; the closest point is searched this far along the line either side of the previous one |
| `reacquire_distance` | `1.0` | m; farther than this from that point, the whole line is searched (points heading within 90 deg of the car) |
| `max_pose_age` | `0.5` | s; an older pose gets a stop (same as `fast_bridge`'s `/drive` watchdog) |
| `max_offset` | `3.0` | m; farther than this from the raceline, the pose is taken as bad and the car stops |
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
