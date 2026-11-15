# Spliner clearance sweep on porto_odom_trial, 2026-11-15

The question: on porto, with Predictive Spliner's dense GP feeding
collision_prediction, and our spliner planning around its regions
(`obs_topic:=/collision_prediction/obstacles`), how does the spliner's required
clearance (`evasion_dist`) trade passable regions against how close the car
actually gets to the opponent? This is the curve, not a recommendation.

Code: the `predictive-spliner-shared` commit adding this file. That includes
the spliner's near-edge ranging and collision_prediction's liveness
(`docs/forzaeth-quirks.md` §1, §8).

## Setup

- **Graph** (`closed_loop.py`):
  - `shared_graph.launch.py gp:=dense state:=TRAILING localizer:=external`,
    with obstacle_tracker and a weaving `fake_opponent_node` (2.0 m/s,
    ±0.4 m, 13.45 m wavelength, starting 6 m ahead, noise-free, 10 m forward
    range).
  - `spliner_node` on `/collision_prediction/obstacles` with `evasion_dist`
    swept. All its other parameters are defaults: car_width 0.30 m,
    bound_margin 0.15 m, lookahead 10 m.
  - `local_path_node`, and control's `pure_pursuit.launch.py avoid:=true`
    (speed_scale 1.0, max_speed 4.0 m/s).
- **The car is a kinematic bicycle in the harness, not AutoDRIVE.** It has
  wheelbase 0.33 m, steering lock 0.524 rad, steering rate 4 rad/s, and
  +4/−8 m/s² acceleration. It integrates `/drive` at 100 Hz, publishes
  `/localized_pose` and `/odom` at 50 Hz, and stops if `/drive` is older than
  0.5 s. There is no slip, no sensor noise and no localization error.
- **Warm-up.** Nothing lets the car move before collision_prediction
  publishes, and that needs the GP, which needs the car following the
  opponent for half a lap. So the harness places the car 3 m behind the
  tracked opponent until the GP has published lapcount ≥ 1 and an obstacle
  array has arrived (about 15 s). It then hands over at 2 m/s. All numbers
  are from after the handover: about 135 s, roughly 5 opponent laps.
- **state:=TRAILING, deliberately.** In GB_TRACK, update_waypoints overwrites
  the ego's waypoint speeds with its measured speed: 2 m/s after the warm-up.
  collision_prediction then rolls the ego forward at the opponent's own speed,
  predicts no catch and publishes empty arrays, while pure pursuit drives up
  to 4 m/s. The one GB_TRACK smoke run touched the opponent 1.4 s after
  handover, with no region published. In TRAILING, update_waypoints writes
  nothing, and collision_prediction rolls the ego at raceline speed (up to
  6.3 m/s, faster than the 4 m/s cap, so conservative).
- **Gaps** are signed distances between footprints (negative is
  penetration):
  - ego: the spliner's box, car_front 0.45 m and car_rear 0.10 m from the
    rear axle, 0.30 m wide;
  - opponent: the fake's 0.50 × 0.30 m box at its true pose.

  The true pose comes from the fake's own motion model, phase-fitted to its
  marker. The marker itself disappears once the opponent is not ahead of the
  car, which is exactly when they are alongside. The fit is within 3–4 cm of
  every marker sample, so treat gaps as ±4 cm.
- **One run per value.** The minimum gap is an extreme of a single run and is
  noisy. The region counts differ between runs because the regions depend on
  how the car drove.

## Results

| evasion_dist (m) | regions | walls | passable % (all / ended) | evading % of blocked cycles | evasion starts | passes | min gap (m) | time in contact (s) | 1st pct gap (m) | distance (m) | mean v (m/s) | stopped (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.35 (default) | 1352 | 149 | 14.9 / 16.7 | 3.1 | 7 | 0 | 2.139 | 0.00 | 2.251 | 271 | 2.00 | 7.8 |
| 0.30 | 1089 | 83 | 23.9 / 25.8 | 27.3 | 14 | 2 | 0.151 | 0.00 | 0.372 | 324 | 2.40 | 4.6 |
| 0.25 | 843 | 70 | 34.6 / 37.8 | 51.3 | 14 | 4 | 0.069 | 0.00 | 0.222 | 372 | 2.76 | 4.3 |
| 0.20 | 697 | 30 | 53.5 / 55.9 | 63.9 | 13 | 5 | 0.021 | 0.00 | 0.178 | 406 | 3.00 | 1.0 |
| 0.15 | 686 | 32 | 57.4 / 60.2 | 64.8 | 13 | 5 | −0.242 | 0.68 | 0.101 | 402 | 3.00 | 1.6 |
| 0.10 | 830 | 22 | 63.6 / 65.3 | 65.8 | 14 | 4 | −0.074 | 0.20 | 0.157 | 386 | 2.85 | 5.0 |
| 0.05 | 1087 | 10 | 68.0 / 68.6 | 64.1 | 36 | 2 | −0.300 | 1.64 | −0.043 | 340 | 2.52 | 11.1 |
| 0.00 | 579 | 18 | 75.1 / 77.5 | 77.3 | 11 | 6 | −0.073 | 0.84 | 0.031 | 435 | 3.19 | 2.7 |

### What the columns mean

- **regions:** the non-empty `/collision_prediction/obstacles` messages
  (10 Hz) after handover. **walls:** those with no end, spanning d from −10
  to 10.
- **passable %:** the share of those regions for which the spliner's own
  `choose_side`, at that `evasion_dist`, finds a side. That needs room of
  at least car_width + evasion_dist between the region and the track edge
  less bound_margin. Walls are never passable.
- **evading %:** of the spliner cycles whose status was not "clear" (something
  in the way), the share that were evading. The rest were "no room" or "too
  close".
- **passes:** the car's rear axle getting from less than 3 m behind the
  opponent to more than 1 m ahead of it.

At 0.35 m the car never got past the opponent. It trailed at the
opponent's speed, stopping and going (2.00 m/s mean), and never got closer
than 2.1 m.

## Reproduce

Run this in the container, on a private ROS_DOMAIN_ID, with the planning
overlay and control sourced, and with a copy of the track (never the
validated directory):

    TRACKS_DIR=/tmp/tracks OUT=/tmp/clearance_sweep bash sweep.sh
    python3 sweep_analyze.py /tmp/tracks/porto_odom_trial /tmp/clearance_sweep/ev_*

It takes about 22 minutes. The scripts publish `/drive`, `/localized_pose` and
`/odom`, so never point them at a live stack's domain.
