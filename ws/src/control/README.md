# control

Pure pursuit on a generated raceline, and a wall follower for the mapping lap(s) (see below). `pure_pursuit_node` takes the car's pose in the `map` frame,
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

`disparity_gap_node` is a second follow the gap, with disparity extension (see below).

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
| `max_steering_angle` | `0.524` | rad, steering is clipped to this; `wall_follow.launch.py` loads it from `config/vehicle.yaml` |
| `laps` | `0` | stop after this many laps, 0 = never |
| `lap_leave_radius`, `lap_return_radius`, `lap_min_distance` | `3.0`, `1.5`, `10.0` | m: a lap is leaving the start by the first, coming back within the second, after driving at least the third |
| `debug` | `false` | log wall distances, steering and speed |

## Follow the gap with disparity extension

`disparity_gap_node` needs only `/scan`, like `follow_the_gap_node`. At every jump of more than
`disparity_jump` between neighbouring beams it extends the nearer range sideways by half the car's
width plus `safety_margin`, so every beam that is still long is a direction the whole car fits through.
It then takes the runs of beams longer than the first of `free_thresholds` (relaxing it if there are
none) within `fov`, picks the deepest run (the widest among near-equal depths), aims at its deepest
point, shifted to the middle of the narrowest spot along the way, and never turns toward something
beside the car's body. Speed scales with the clear distance straight ahead, between `min_speed` and
`max_speed`.

```bash
ros2 launch control disparity_gap.launch.py [speed:=1.0]   # speed sets max_speed
```

Run it instead of `follow_the_gap.launch.py` or `race.sh`, never alongside them: they all publish `/drive`.
It was tuned in the F1TENTH gym (Levine), not yet in AutoDRIVE.

| Topic | Type | |
|---|---|---|
| `/scan` | `sensor_msgs/LaserScan` | subscribed |
| `/drive` | `ackermann_msgs/AckermannDriveStamped` | published, one command per scan; a stop when no beam ahead has a return |

| Parameter | Default | |
|---|---|---|
| `max_speed` | `1.0` | m/s cap; the launch file's `speed` |
| `min_speed` | `0.5` | m/s, unless `max_speed` is lower |
| `speed_per_metre` | `0.6` | m/s per metre of clear road within ±8° ahead, halved at full steering |
| `max_range` | `10.0` | m, longer returns are clipped to this |
| `fov` | `π/2` | rad, only beams within ± this can be aimed at |
| `car_width`, `safety_margin` | `0.31`, `0.05` | m, the extension radius is half the width plus the margin |
| `disparity_jump` | `0.35` | m between neighbouring beams that counts as an obstacle edge |
| `free_thresholds` | `[3.0, 2.0, 1.2, 0.6]` | m, tried in order until a gap exists |
| `path_lookahead`, `path_cap` | `6.0`, `1.5` | m: how far along the path to look for its narrowest spot; room beyond the cap on a side counts as open |
| `side_clearance`, `push_gain` | `0.08`, `2.0` | m kept beside the body; rad per metre something is inside it |
| `body_behind_lidar`, `body_ahead_of_lidar` | `0.45`, `0.15` | m, the car's body along x relative to the LiDAR |
| `max_steering_angle` | `0.524` | rad, steering is clipped to this; the launch file loads it from `config/vehicle.yaml` |

## Test

```bash
python3 -m pytest ws/src/control/test   # needs a sourced ROS 2 environment and the built package
```

The tests publish `/scan`, `/tracked_pose` and read `/drive`: with the bridge running, give them their
own `ROS_DOMAIN_ID` (e.g. `ROS_DOMAIN_ID=42 python3 -m pytest ...`) so they don't drive the sim car.

## Credit

Adapted from the F1TENTH / UPenn ESE 6150 pure-pursuit lab template (MIT, see `LICENSE`).
