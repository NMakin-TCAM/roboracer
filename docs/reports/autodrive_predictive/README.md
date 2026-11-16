# The predictive chain in AutoDRIVE, porto, 2026-11-16

The first runs of the moving-opponent chain against the simulator, not the
kinematic stand-in: `planning predictive_race.launch.py` with AutoDRIVE's car
through the bridge, Cartographer localization, and everything downstream.

## Setup

- **Simulator:** SIGRobotics' AutoDRIVE fork (Udog-ILLINOIS/AutoDRIVE,
  `f1tenth_multitrack`) on macOS, track Porto, Autonomous mode, connected to
  the bridge on 127.0.0.1:4567. It ran at 70–113 frames/s through the bridge.
  The fork's control port (127.0.0.1:4580: `status`, `screenshot`, `click`,
  `obstacle`, ...) and its `-track <name> -autoconnect` flags mean a future run
  needs no GUI action.
- **Graph** (`predictive_race.launch.py track:=porto_odom_trial`):
  - the bridge, Cartographer and localization_gate;
  - global waypoints, frenet_state_node, fake_opponent_node (weaving, ±0.4 m,
    noise-free, 10 m forward range) and obstacle_tracker_node;
  - the dense GP, the predictor and collision_prediction;
  - spliner_node at evasion_dist 0.25 with trailing (trail_gap 2.0 m);
  - local_path_node and pure pursuit at speed_scale 0.5, as avoid_graph was
    validated.

  The state stand-in is TRAILING and `use_sim_time` is true (the bridge
  publishes `/clock`; it ran at exactly 1.000 × wall time).
- **The opponent exists only in software.** The car drives round a car the
  simulator does not have. The simulator's collisions below are with its own
  walls.
- **Measurement** (`sim_record.py`, passive: it subscribes and publishes
  nothing; `sim_analyze.py`, `crash_detail.py`, `wall_clearance.py`):
  - The car's truth is `/autodrive/roboracer_1/odom`, put in the map frame by
    porto_odom_trial's map→world transform: rotation −92.693°, translation
    (0.5065, −4.3836) m, verified by SIGRobotics to 4.3 cm at 7.4 m. This
    differs slightly from `map_to_world.yaml` (−92.33°, (0.5037, −4.3818)),
    which this work does not use.
  - The opponent's truth is fake_opponent_node's evaluation-only truth topic.
  - Truth is used only to measure. The sim odometry's reference point on the
    car is undocumented; the localized pose sits a steady 2.7–4.3 cm behind
    it along the car, taken as that offset.

## Runs

| run | duration | settings | outcome |
|---|---|---|---|
| 1 | 90 s | opponent 1.2 m/s, bound_margin 0.15 (spliner default) | 3.8 laps; 4 simulator collisions, one per lap from the first evasion |
| 2 | 240 s | bound_margin 0.30 | 10.9 laps, no collisions, no passes |
| 3 | 300 s | opponent 0.8 m/s | 8.5 laps, no collisions; the opponent drove through the standing car at start-up (setup bug, below) |
| 4 | 600 s | opponent_s0 24.3 (5 m ahead of the car's real start), pure pursuit fix | 17.65 laps, no collisions, no passes, clean shutdown |

## What came up, what blocked, what changed, in order

1. **The simulator needed a GUI connect.** The headless `-autonomous` start in
   the repo is for the patched Windows build only. You connected it; the fork's
   control port and `-autoconnect` flag remove that step next time.
2. **My run script: `set -u` breaks sourcing ROS's `setup.bash`** (it reads
   unset variables). Removed.
3. **Every evasion along the inner wall hit it** (run 1: collisions at 54.3,
   65.6, 76.9, 88.2 s, 2 s after the first evasion and then once per lap at
   the same two places, s ≈ 8 and 21 m). The simulator respawned the car each
   time (truth jumps to a standstill at s 21.44 or 8.68, d −0.3) and
   localization was lost for about 2.5 s. In the 2 s before each, the car's
   footprint was −5.4, +0.4, −5.0 and +0.2 cm from the mapped track edge.

   **Cause:** the spliner keeps the planned line, the car's *centre*,
   `bound_margin` (0.15 m) inside the edge, and 0.15 m is the car's half
   width. So an evasion against the wall puts the car's side on it; the
   raceline itself already runs 0.21 m from the walls at the apexes (its
   provenance header).

   **Changed:** `predictive_race.launch.py` passes `bound_margin` 0.30 (the
   half width plus the intended 0.15). avoid_graph keeps 0.15 and is
   unchanged. What the parameter *means* is the open decision: as written it
   lets avoid_graph's car touch walls too. Runs 2–4 had no collisions, and the
   footprint never came closer than +5.6 cm to the edge (1st percentile).
4. **Pure pursuit could not be stopped** (runs 1–3: still alive 30 s after
   SIGINT, the launch waiting on it, no stop burst).

   **Cause:** pure_pursuit_node opts out of rclpy's signal handler to send its
   stop burst, and relies on Python's KeyboardInterrupt. A background job of a
   non-interactive shell inherits SIGINT as *ignored*, and Python then
   installs no handler. Its `/proc` mask showed it: `SigIgn 0000000081001006`,
   bit 1. Every other node uses rclpy's handler and exited.

   **Changed:** `main()` restores Python's SIGINT handler explicitly.
   `test_shutdown_stops_when_started_with_sigint_ignored` failed on the old
   node and passes now. Run 4: mask `...1004`, and the launch exited 1.0 s
   after SIGINT.
5. **The fake opponent's start is in raceline s, but the car does not start at
   s = 0.** After the reset the car stands at raceline s = 19.3 m (where
   mapping started), so `opponent_s0:=5.0` put the opponent 14 m behind it.
   In run 3 (0.8 m/s) it caught up while the car waited 12 s for localization
   and drove through it. The fake sensor only looks forward, so the planner
   said "clear" for 14 cycles with the footprints overlapping by 0.245 m.

   **Changed:** nothing in code; run 4 passes `opponent_s0:=24.3`. This is a
   setup problem of the software opponent, not a planner failure, but any run
   whose opponent starts behind the car will show it. Planner and opponent
   should agree on where "ahead" starts.
6. **The recorder heard no `/clock`** (QoS). It now subscribes best effort.

## Measurements

**Time to first plan, Cartographer against a perfect pose.** About 12 s from
launch, all of it localization:

| | run 1 | run 2 | run 3 | run 4 |
|---|---|---|---|---|
| first `/tracked_pose` | 1.12 | 0.47 | 0.64 | 0.36 |
| localization_gate localized | 12.56 | 11.65 | 12.12 | 12.08 |
| first plan (collision-free) | 12.63 | 11.69 | 12.19 | 12.14 |
| car moves | 12.73 | 11.81 | 12.28 | 12.23 |
| GP whole lap (lapcount 1) | 44.7 | 58.7 | 46.5 | 46.1 |
| first evasion | 52.4 | 65.2 | 55.0 | 52.9 |

With the stand-in, the first plan came 0.4 s after launch and the GP's whole
lap came at 14–18 s. The planner answers within 70 ms of the gate localizing.
The GP is slower because of the 12 s wait (the opponent is far ahead by then)
and the slower car.

**Rates with the simulator in the loop** (run 4, 632 s; arrival at the
recorder):

| topic | Hz | max gap |
|---|---|---|
| `/scan` | 40.0 | 356 ms |
| `/odom`, IMU, sim odometry | 86.4 | 356 ms |
| `/tracked_pose` | 99.7 | 362 ms |
| `/localized_pose`, `/car_state/odom_frenet` | 99.7 | 70 ms |
| `/perception/detections`, `/perception/obstacles` | 40.0 | 63 ms |
| `/collision_prediction/obstacles` | 10.0 | 114 ms |
| spliner status and otwpnts | 20.0 | 63 ms |
| `/local_path`, `/local_path/valid` | 20.0 | 356 ms |
| `/drive` | 99.7 | 76 ms |

Every node held its nominal rate for 10 minutes, with no header-stamp
reversals. The chain itself dropped nothing.

The simulator did not keep its rate. Once in run 4, every bridge topic (scan,
odometry, IMU, `/clock`) stopped for 355 ms at the same time: a simulator
frame stall. Downstream, `/localized_pose` never paused for more than 70 ms
through it, because Cartographer extrapolates fresh-stamped poses during a
stall (AGENTS.md lists this). That is 0.36 s of driving on a pose nothing
measured. run 1's 2.5 s pose gaps were the respawns.

`/drive` had 66 duplicate stamps in run 4: pure pursuit's repeated stops,
which share a stamp.

**Localization against truth** (runs 2–4, consistent):
- across the car: p5..p95 −6.6..+3.3 cm, worst 7.8 cm;
- yaw: worst 3.2°;
- along the car: a steady −2.7 to −4.3 cm, the truth reference offset.

**Pure pursuit tracking on the simulator's dynamics** (truth, rear axle,
minus the plan's d at the car's own s). While evading, at speed_scale 0.5
(1.0–3.1 m/s):

| run | samples | p50 | p95 | max |
|---|---|---|---|---|
| 2 | 683 | 2.7 cm | 4.3 cm | 4.6 cm |
| 3 | 3,636 | 1.9 cm | 5.3 cm | 9.0 cm |
| 4 | 7,519 | 1.8 cm | 5.5 cm | 9.8 cm |

The stand-in's 8–9 cm was on the rejoin at 3–4 m/s, so these are not like
for like: half the speed, and no rejoin past the opponent (no passes, below).
Over all driving, "clear" included, run 4 gives p95 31.9 cm and max 48.4 cm.
That is not tracking. Split by how long the plan had said "clear":

| run 4, plan "clear" for | samples | p50 | p95 | max |
|---|---|---|---|---|
| ≥ 1 s (steady raceline following) | 351 | 1.0 cm | 5.6 cm | 8.9 cm |
| < 1 s (just after a switch) | 13,336 | 7.3 cm | 36.2 cm | 48.4 cm |

When an evasion is released, the reference jumps back to the raceline while
the car is still 30–48 cm off it. The statuses switch so often in this
stop-and-go trailing that most driving falls within a second of a switch.
Steady tracking on the simulator's dynamics is about 5.5 cm at p95 and under
10 cm at worst. Those lateral jumps of the planned line are a planner finding
in their own right.

**Closest gaps to the opponent** (run 4, 17.65 laps):
- closest planned: +1.032 m, with 0 planned-through cycles;
- closest achieved: +1.059 m;
- no contact.

**No passes in 37 laps across runs 2–4.** The car caught the opponent and
trailed it, but never got by. Run 4 spent 1,834 cycles evading, and every
evasion was released before the car's nose reached the region, where the
alongside hold does not apply:

| release | count |
|---|---|
| "gone" (the recomputed region moved more than 0.5 m), re-planned as a new evasion | 72 |
| "gone", then no room or clear | 5 |
| "blocked" (the region moved onto the path), then no room | 32 |
| "blocked", re-planned as a new evasion | 10 |

The replans then mostly refused: "no feasible side" 231 times, and "the way
left is blocked by another obstacle" 70 times. That second message names the
region itself: the 0.30 wall margin clips the apex back towards the region.

So the gaps above are trailing distances, and nothing here tests clearance in
a pass on the real dynamics. Getting passes on porto needs room the chain
does not find at evasion_dist 0.25 with a 0.30 wall margin: either a wider
track, a smaller margin, or evasions that survive the region moving.

## Follow-up on the stand-in: holding the commitment, and the jump

**The question.** Run 4 dropped every evasion before the car reached the
opponent. Would holding the commitment by obstacle id from the moment it starts
(not only once alongside) produce passes at the 0.30 wall margin?

**Setup.** The stand-in (`clearance_sweep/closed_loop.py`, porto geometry,
from cold), no simulator:
- **run-4-like:** speed_scale 0.5, opponent 0.8 m/s, 600 s. Its baseline
  reproduces run 4's drops: 120 against 119.
- **test-like:** speed_scale 1.0, opponent 2.0 m/s, 300 s.

Both at evasion_dist 0.25 and bound_margin 0.30.

| variant | drops (run-4-like) | laps | passes | closest planned | closest achieved | contact |
|---|---|---|---|---|---|---|
| baseline | 120 (81 gone, 39 blocked) | 19 | 0 | +1.661 m | +1.644 m | 0 |
| hold from commit (gone and blocked) | 25 (8 gone, 17 off line) | 29 | 7 | −0.011 m, 21 cycles through | −0.043 m | 2.1 s |
| hold gone only | 78 (all blocked) | 19 | 0 | +1.674 m | +1.653 m | 0 |
| no hold + REJOIN (committed) | 116 (80 gone, 36 blocked) | 19 | 0 | +1.649 m | +1.618 m | 0 |

Test-like at 0.30: 23 laps, 0 passes and 0 drops for every variant (no evasion
ever starts).

The closed-loop clearance test runs at bound_margin 0.15, so these outcomes are
independent of the 0.30 question:

| variant | evasion_dist 0.25 | evasion_dist 0.15 |
|---|---|---|
| hold from commit | fails: 3 cycles through, −0.014 m | fails: 70 cycles, −0.170 m, contact −0.199 m |
| hold gone only | fails: contact −0.014 m (planned +0.023) | passes |
| no hold + REJOIN | passes: planned +0.172, achieved +0.101 m | passes: +0.136 / +0.075 m |

**Why no hold.**
- **Holding "blocked" is what produced passes at 0.30, and it is unsafe.** A
  region moving onto the committed path is the prediction saying the opponent
  will be on it; holding drove through the opponent.
- **Holding only "gone" bought nothing.** The same evasions were dropped as
  "blocked" a moment later. A held path is re-checked at evasion_dist / 2, so
  its margin to the moving region eroded, and it touched the opponent at 0.25.

The finding: on porto at a 0.30 wall margin, a planner that keeps its
clearance never passed, in 42 laps. That is the track being the binding
constraint for this planner, not dropped plans. Untested: re-planning the
committed evasion against the moved region on the same side, instead of
holding or dropping it.

**The jump (REJOIN, committed).** When an evasion ends with the car more than
rejoin_threshold (0.10 m) off the raceline, the new plan starts at the car's d
and returns within the lateral-slope limit, re-derived each cycle. Gap between
the car and the plan within 1 s of a status switch (`step_analyze.py`):

| run-4-like | p50 | p95 | max |
|---|---|---|---|
| before | 2.2 cm | 38.6 cm | 47.2 cm |
| after | 0.9 cm | 7.2 cm | 16.4 cm |

AutoDRIVE's run 4 showed 36.2 cm at p95 and 48.4 cm at worst. Steady tracking is
unchanged (p95 about 8–9 cm).

## Follow-up: re-planning on the committed side, and why nothing starts at speed

**The change** (`spliner.py`, RE-PLAN ON THE COMMITTED SIDE). This applies
before the car reaches the committed obstacle, and only to obstacles with an
id. When that obstacle has moved beyond match_tolerance, or the rest of the
committed path no longer clears it:
- the evasion is re-planned from the car's current s and d;
- round the obstacle as it is now;
- on the committed side only;
- at full evasion_dist;
- and checked against every obstacle.

The side carries over; the geometry does not. These still release:
- gone (the id is no longer published);
- another obstacle on the path;
- off line;
- passed;
- and now "no room on the committed side", "too close" and "blocked", when the
  re-plan fails.

**Gate:** the closed-loop clearance test passes at 0.25 (planned +0.207,
achieved +0.123 m) and at 0.15 (+0.178 / +0.108 m).

**Measurements** (stand-in, bound_margin 0.30, evasion_dist 0.25):

| | drops | re-plans | laps | passes | closest planned | closest achieved | contact |
|---|---|---|---|---|---|---|---|
| half speed (run-4-like), before | 116 (80 gone, 36 blocked) | – | 19 | 0 | +1.649 m | +1.618 m | 0 |
| half speed, side re-plan | 34 (blocked, then no room) | 111 | 19 | 0 | +1.660 m | +1.646 m | 0 |
| full speed (opponent 2 m/s), before | 0 | – | 23 | 0 | +1.846 m | +1.563 m | 0 |
| full speed, side re-plan | 0 | 0 | 23 | 0 | +1.843 m | +1.580 m | 0 |

At half speed the re-plan does what it should. "Gone" disappears (80 to 0),
and evasions follow the moving region (111 re-plans) without ever planning
through it (0 planned-through cycles). But each one still ends before the car
gets alongside: the re-plan on the committed side finds the region has moved
into a part of the track where the path would hit it ("blocked", 34). No passes.

At full speed no evasion ever starts, before or after. The re-plan cannot
change that, since there is never a commitment to re-plan.

**Why nothing starts at speed: the track.** `speed_diag.py` and
`where_regions.py`, on the before/after records:

- **The refusals are geometric.** 1,505 cycles of "no feasible side" and 262
  of "the way left is blocked by another obstacle". Only 12.8% of predicted
  regions leave car_width + evasion_dist (0.55 m) on either side.
- **Most of porto is too narrow.** With 0.30 off each wall, the usable width
  is 0.55–0.92 m on about 70% of the lap: s 4–14 m and 18–26 m. A region's d
  band (0.5–0.8 m) plus 0.55 m does not fit there at any speed. It fits only
  around s 0–4 and 14–18 m, about 8 m of the 26.9 m lap.
- **Speed makes it worse in two ways.**
  - At 2 m/s the weaving opponent's d changes about 2.5× faster than at
    0.8 m/s, so over a region it moves further, and the region's band is
    wider: median 0.64 m against 0.53 m.
  - Fewer catches happen in the wide stretches (s 16–18 m: 3.5% of regions
    against 9.3%). So the share of regions that fit at all falls from 32% to
    16%.
- **The "blocked by another obstacle" refusals name the region itself.**
  choose_side measures room only at the region's centre; the path, which holds
  its offset along the whole region and its lead-in, is clamped by the wall
  margin where the track narrows and then fails against the same region. Those
  refusals are right. Their message is wrong (it is not another obstacle), and
  the room check is optimistic.

So nothing in the planner stops an evasion starting at speed; porto's width
at a 0.30 margin does. With the side re-plan, porto still never passes safely,
at either speed, in 42 laps. The track is the remaining blocker.

## Follow-up: a non-weaving opponent

**The question.** Does porto pass at the 0.30 wall margin if the opponent does
not weave (fake_opponent_node `constant_speed`, d = 0)? This used the
stand-in, from cold, with evasion_dist 0.25, at two speeds:
- half speed: speed_scale 0.5, opponent 0.8 m/s, 600 s;
- full speed: speed_scale 1.0 (max_speed 4.0), opponent 2.0 m/s, 300 s.

**Region band and room** (`where_regions.py`, `speed_diag.py`):

| | band p50 (p5..p95) | band + 0.55 fits the usable width | a side has 0.55 m |
|---|---|---|---|
| weaving, half speed | 0.53 m | 32% | 27% |
| weaving, full speed | 0.64 m | 16% | 13–14% |
| constant, half speed | 0.50 m (0.50..0.51) | 36% | 37% |
| constant, full speed | 0.50 m (0.50..0.50) | 44% | 46% |

0.50 m is the floor: the region's ±0.25 m margin round a single d.

**A safety bug the weaving runs never reached.** The first constant-speed run
at full speed made 4 passes in 28 laps, but with 79 cycles planned through the
opponent (closest −0.300 m) and 2.1 s of contact.

- **What happened.** An evasion was committed on a region at s 1.7–5.7 m; the
  real catch came at s 8.1–9.2 m. Because the car's nose had passed the
  *committed* region's start, it counted as "alongside". The hold kept the
  old path, whose rejoin then steered back to the raceline into the opponent.
- **Fixed** (`spliner.py` release_reason). "Alongside" is now judged against
  the current region of the same id. A region that slid ahead of the car is
  re-planned on the committed side, or released if that side has no room.
  `test_a_region_that_slid_past_the_committed_one_is_re_planned_not_held`
  pins it.
- **Ruled out:** a re-plan every cycle (there were none in the 2 s before any
  contact), and the speed model (with no speed cap it got worse: 181 cycles,
  6.5 s of contact).
- **The gate missed it.** The closed-loop test uses a weaving opponent at
  bound_margin 0.15; it passes before and after (now: planned +0.232 /
  +0.157 m, achieved +0.150 / +0.080 m).

**With the fix:**

| constant-speed opponent | laps | passes | closest planned | closest achieved | contact | drops |
|---|---|---|---|---|---|---|
| half speed | 20 | 0 | +0.908 m | +0.900 m | 0 | 33 (no room on the committed side) |
| full speed | 25 | 0 | +0.370 m | +0.366 m | 0 | 31 blocked, 22 no room on the committed side |

**Still no pass. The limiting numbers** (`room_needed.py`):
- **Width:** an evasion needs band 0.50 + car 0.30 + clearance 0.25 = 1.05 m
  usable through the region, i.e. 1.65 m wall to wall at the 0.30 margin.
- **Length:** about 7.7 m in all. That is a 3.0 m lead-in, the region itself
  (median 2.0 m, p90 3.6 m) at full width, a 0.25 m hold and a 2.4 m rejoin.
- **What porto has:** usable width of at least 1.05 m on 27% of the lap, in
  two stretches, 4.4 m (across the start line, from s 26.1) and 2.9 m (from
  s 15.0). Its median usable width is 0.78 m, and its minimum 0.47 m.
- **Why the evasions end.** They start, but the region slides as the car
  closes, out of those stretches and into the narrow ones. There the re-plan
  on the committed side finds no room, or the path hits the region.

A track for this chain needs at least 1.65 m wall to wall over stretches of
7–8 m where the car will catch the opponent. That is a lower bound; a weaving
opponent needs more width.

No simulator run: nothing passed on the stand-in.

## Not done

- No physical obstacle. The fork's `obstacle <x> <y>` puts one in the world
  frame, with map→world as above; it was not used.
- GB_TRACK is untested here (documented, not fixed: forzaeth-quirks §8).
- One configuration of speed and opponent per run.
