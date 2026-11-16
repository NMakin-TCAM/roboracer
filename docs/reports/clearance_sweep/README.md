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

## Results (with the clearance violation present)

These numbers were measured before the planner fix (see "After the fix"). The
passable shares, and especially the minimum gaps, at 0.25, 0.20 and 0.15 are
superseded by the re-sweep there.

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

## Why there is contact from 0.15 m down

The hypothesis was pure pursuit tracking error. It is not that.

Three new instrumented runs (0.15, 0.05, 0.25) record each planner output
(`/planner/avoidance/otwpnts`, paired with its status by stamp). They also
record the opponent's true pose from its marker stamps, fitted to the motion
model (median residual 0.1 cm, worst 1.6 cm). `diag_analyze.py` compares, for
every 20 ms sample:
- **achieved:** the car as driven, against the true opponent;
- **plan:** the car put exactly on the plan at its own s (the evasion's d
  while evading, the raceline otherwise), against the true opponent;
- **region:** the car on the plan, against the region it was planned around.

**Tracking error is small.** While alongside and evading, |car d − plan d|:

| clearance | median | 95th pct | max | worst plan − achieved |
|---|---|---|---|---|
| 0.15 | 2.4 cm | 8.5 cm | 9.3 cm | 6.8 cm |
| 0.05 | 2.7 cm | 9.2 cm | 10.0 cm | 9.8 cm |
| 0.25 | 3.5 cm | 9.8 cm | 11.1 cm | 12.3 cm |

That is real margin to budget for, around 10 cm, but no contact came from it.
(A first version of this analysis, and the report that quoted it, gave worse
maxima, and an overlap of an *evading* plan with the opponent at 0.25. That
was a measurement artefact. The published remaining path starts at the first
sample at or ahead of the car, up to 0.1 m in front of it. The analysis
treated the car in that sliver as off the path, on the raceline. Fixed by
holding the path's end values; no evading plan overlapped the opponent in any
of the three runs: minimum planned gap while evading +0.17, +0.20, +0.23 m.)

**Every contact happens while the plan itself overlaps the opponent:**

| clearance | contact | achieved (m) | plan (m) | region (m) | status at worst | the second before |
|---|---|---|---|---|---|---|
| 0.15 | 43.6 s, 0.06 s long | −0.019 | −0.022 | −0.160 | clear (raceline) | evading → too close → clear |
| 0.15 | 124.6 s, 0.22 s long | −0.104 | −0.091 | −0.294 | clear (raceline) | evading → too close → clear |
| 0.05 | 16.6 s, 0.68 s long | −0.259 | −0.179 | −0.443 | too close / clear / no room, alternating at 0.8 m/s | evading → too close ⇄ clear |

The mechanism, all three times:
1. collision_prediction recomputes the region every cycle, and it moves.
2. The committed evasion is released mid-pass. At 0.15: 17 "gone" (moved
   more than match_tolerance) and 14 "blocked" (moved onto the path).
3. The stateless replan then looks for obstacles whose near edge is ahead of
   the rear axle. The region the car is already inside has its `s_start`
   behind it, so it is not seen, and the answer is "clear".
4. The car steers back onto the raceline beside the opponent. Or, at 0.05, it
   flaps between "too close" and "clear" while half-alongside.

The planner violates its own constraint, and only through the replan after
a release. At 0.25 that run had no contact (minimum +0.072 m).

## Cold start, 0.25 m clearance, before the fix

The car stands at the raceline's start with the opponent 6 m ahead; there is
no warm-up and no hand placement (`CL_COLD=1`). spliner_node runs with
`trail_obs_topic:=/perception/obstacles` and `evasion_dist:=0.25`; the stub
state machine is in TRAILING. Times are seconds from launch.

- **0.4 s:** it plans on the tracked obstacles (trailing, "no collision
  prediction yet"). It says clear and the car moves at 0.5 s.
- **Lap 1 (12.1 s, 2.23 m/s mean):** stop-and-go behind the opponent. "No
  room" whenever the opponent's near edge comes within 2 m of the nose
  (closest 1.5 m), then clear again: 3 stops, 1.2 s stopped. The gap behind
  was 2.4 / 3.3 / 6.4 m (p10 / p50 / p90).
- **7.3 s:** the GP's first half lap. collision_prediction speaks with
  force_trailing True, so the planner keeps trailing ("forces trailing").
- **14.3 s:** the GP's whole lap. force_trailing goes False, the planner
  switches to the regions, and it starts an evasion at once.
- **Lap 2 (8.8 s, 3.05 m/s mean):** the evasion is released once ("blocked",
  0.3 s of no room) and re-planned. The car passes the opponent at 17.3 s,
  minimum gap +0.156 m, and is clear after.
- **Laps 3–12:** a regular cycle. The car pulls away (100% clear), catches up
  (trailing on "no room", 3–4 short stops), then passes, once every three
  laps (laps 5, 8, 11). Minimum gap per passing lap: +0.137, +0.060,
  +0.134 m. No contact in 12 laps. force_trailing flickers True for about a
  second at the start line on every lap (`docs/forzaeth-quirks.md` §4), switching the planner to
  trailing briefly.

Run `cold_analyze.py` on the record for the full timeline.

## After the fix

**What changed** (`planning/spliner.py`, COMMITMENT and `obstacles_ahead`):

1. **`obstacles_ahead` ranges on the far edge.** An obstacle counts until the
   car has fully passed it (`s_end` still ahead of the rear axle). The range
   cap stays on the near edge (`s_start` within the lookahead). A replan
   inside a region now refuses ("too close", the car stops) instead of saying
   "clear".
2. **"blocked" is held alongside, for the committed obstacle only.** Once the
   car's nose reaches the committed obstacle, that obstacle can no longer
   release the commitment as "blocked"; only "passed" ends it, as for "gone".
   "The committed obstacle" means the same id. collision_prediction's region
   is always id 0, and the tracker's ids are per object. On porto the
   recomputed region slid 1.3 m forward in 0.7 s, clear of its old span, so
   position alone misidentified it as new. Obstacles without an id fall back
   to matching within match_tolerance or overlapping the old span. Any other
   obstacle on the rest of the path still releases, alongside or not.

**The permanent test.** `planning/test/test_closed_loop_clearance.py` is the
full chain, from cold, on porto's geometry (`test/fixtures/porto_odom_trial`),
130 s at each of 0.25 and 0.15. Every 20 ms it compares the planned gap and
the achieved gap against the opponent's true pose, taken from
fake_opponent_node's evaluation-only truth topic. It fails if the plan ever
overlaps the opponent (while CLEAR or EVADING), on any contact, or if it never
passes the opponent at all.

| planner | 0.25 | 0.15 |
|---|---|---|
| before the fix | **failed**: 34 cycles planned through, worst −0.184 m, while "clear" | passed (+0.163 planned, +0.086 achieved) |
| after the fix | passed: closest planned +0.115 m, achieved +0.052 m, 4 passes | passed: +0.165 / +0.085 m, 4 passes |

So the violation is intermittent: in that pair of runs it showed at 0.25,
where the diagnosis had found it at 0.15. That is why the test runs both.

**Re-sweep, fixed planner.** Same protocol as the first sweep (warm-up,
TRAILING, 150 s). Gaps are from the stamped truth.

| evasion_dist (m) | regions | walls | passable % (all / ended) | evading % of blocked cycles | evasion starts | passes | closest planned (m) | planned-through cycles | closest achieved (m) | contact (s) | 1st pct gap (m) | mean v (m/s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 733 | 35 | 40.4 / 42.4 | 59.2 | 8 | 5 | +0.043 | 0 | +0.037 | 0 | 0.108 | 2.96 |
| 0.20 | 708 | 53 | 45.6 / 49.3 | 65.1 | 5 | 5 | +0.046 | 0 | +0.006 | 0 | 0.138 | 3.00 |
| 0.15 | 696 | 33 | 57.9 / 60.8 | 66.2 | 9 | 5 | +0.082 | 0 | +0.007 | 0 | 0.145 | 3.00 |

**Where the closest approaches are:**

| clearance | status | planned (m) | tracking (m) | achieved (m) | where |
|---|---|---|---|---|---|
| 0.25 | clear | +0.043 | −0.007 | +0.037 | car 0.34 m ahead of the opponent: the cut-back after the pass |
| 0.20 | evading | +0.083 | −0.076 | +0.006 | car 0.35 m ahead: the rejoin |
| 0.15 | evading | +0.096 | −0.086 | +0.007 | car 0.35 m ahead: the rejoin |

With the planner fixed, the closest approaches are what the tracking-error
hypothesis predicted. At 0.20 and 0.15 the plan kept 8–10 cm, and pure
pursuit's 8–9 cm error during the rejoin took nearly all of it. One run per
value; treat the minima as single samples.

**The cut-back is not covered by the clearance.** At 0.25 the closest approach
is the cut-back to the raceline once the rear axle has passed the region's
far edge. collision_prediction ends the region where the rear axle is 0.4 m
(upstream's `save_distance_back`) past the opponent's centre. With
car_rear 0.10 m and the opponent's half-length 0.25 m, that leaves about
5 cm along the track. Clearance is only enforced laterally while the region
lasts, so `evasion_dist` does not govern this approach.

## Cold start after the fix, 0.25 m clearance

Same setup as before (`CL_COLD=1`); 120 s, 12 car laps.

- **First two laps.** The car moves at 1.6 s, trailing on the tracked
  obstacles.
  - The GP's half lap comes at 11.5 s and its whole lap at 18.4 s. This is
    slower than before (7.3 and 14.3 s): the trailing gap often opened beyond
    opponent_trajectory's 6 m (p90 8.1 m on lap 1), so it got fewer points.
  - The car plans on the regions from 18.5 s. Its one evasion on lap 2
    (22.9 s) was released as "gone" after 0.1 s, before the nose reached
    the region.
  - Lap 1: 12.3 s, 2.19 m/s, 3 stops. Lap 2: 13.8 s, 1.95 m/s, 6 stops, no
    pass.
- **Laps 3–12.** Passes on laps 3, 6 and 10, and the same pull away, catch
  up, trail, pass cycle as before.
- **No contact in 12 laps.**
  - Closest planned gap: +0.084 m (evading, 98.3 s), with 0 planned-through
    cycles.
  - Closest achieved gap: +0.024 m at 98.2 s, evading on the rejoin, with
    the car 0.38 m ahead of the opponent and 9.4 cm off its plan (tracking).
  - Closest per passing lap: +0.060, +0.170, +0.024 m.

## Reproduce

Run this in the container, on a private ROS_DOMAIN_ID, with the planning
overlay and control sourced, and with a copy of the track (never the
validated directory):

    TRACKS_DIR=/tmp/tracks OUT=/tmp/clearance_sweep bash sweep.sh
    python3 sweep_analyze.py /tmp/tracks/porto_odom_trial /tmp/clearance_sweep/ev_*

It takes about 22 minutes. The scripts publish `/drive`, `/localized_pose` and
`/odom`, so never point them at a live stack's domain.
