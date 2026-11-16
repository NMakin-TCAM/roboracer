# Preserved upstream quirks in the shared Predictive Spliner / FSDP code

Four files are byte-identical between ForzaETH's Predictive Spliner and ZJU's
FSDP, and are ported once into `ws/src/planning/planning/` for both comparison
branches to import. They are the whole of the shared code: every other
same-named file in the two repositories differs (`gaussian_process_opp_traj.py`,
`dynamic_collision_server.py` and `dynamic_sqp_server.py` were checked).

| Upstream file | Port | Tests |
|---|---|---|
| `update_waypoints.py` | `update_waypoints.py`, `update_waypoints_node.py` | `test_update_waypoints.py` |
| `collision_prediction.py` | `collision_prediction.py`, `collision_prediction_node.py` | `test_collision_prediction.py` |
| `opponent_trajectory.py` | `opponent_trajectory.py`, `opponent_trajectory_node.py` | `test_opponent_trajectory.py` |
| `predictor_opponent_trajectory.py` | `predictor_opponent_trajectory.py`, `predictor_opponent_trajectory_node.py` | `test_predictor_opponent_trajectory.py` |

Predictive Spliner's own main GP, `gaussian_process_opp_traj.py` (the dense
sklearn GP), is ported too, as `dense_gp_opp_traj.py` and
`dense_gp_opp_traj_node.py`, tested by `test_dense_gp_opp_traj.py`. It is not
shared code: FSDP's file of the same name is a different, sparse GP. It is
started only by `shared_graph.launch.py gp:=dense` (§6), and §8 covers it.

All five ports keep upstream's defects, and for the same main reason: if
the shared code is "fixed", both branches stop running the code they claim
to port, and our numbers can no longer be compared with ForzaETH's or ZJU's
published results. Wherever a quirk had another reason as well, its entry
says so. The module docstrings hold the full detail; this page collects it
in one place.

**Pinned** names the unit test that fails if the behaviour changes. **Not
pinned** means there is no such test.

**Differential** marks behaviour covered by
`ws/src/planning/test/test_upstream_differential.py`. That test runs
upstream's unmodified file under stub ROS against the port on random
scenarios, requires identical output, and asserts that the run reached each
listed branch. It is opt-in: set `FORZAETH_SPLINER_SRC` to a directory
holding upstream's files (Predictive Spliner's `src/` or FSDP's `src/gp/`).
Without that it skips, so CI and fresh clones are unaffected. A file with a
different md5 from the one ported fails.

    FORZAETH_SPLINER_SRC=~/predictive-spliner/src \
        python3 -m pytest src/planning/test/test_upstream_differential.py -q

It covers `opponent_trajectory`, `predictor_opponent_trajectory` and the dense
GP:

- **opponent_trajectory:** 300 scenarios by default (`DIFFERENTIAL_SCENARIOS`),
  about 13 s.
- **predictor:** 100 by default (`DIFFERENTIAL_PREDICTOR_SCENARIOS`), about
  22 s, because every prediction is a GP fit run twice. It also needs
  scikit-learn 1.1 or later and skips otherwise (see §1).
- **dense GP:** 40 by default (`DIFFERENTIAL_GP_SCENARIOS`), about 25 s. Same
  sklearn requirement. Upstream is given our `FrenetConverter` behind
  ForzaETH's `get_cartesian`/`get_frenet` interface, and the vendored ccma as
  `ccma`, so both sides do the same CCMA round trip through x, y. Scenarios are
  closed tracks at 0.1 to 0.3 m spacing with half laps shaped like
  opponent_trajectory's, plus off-trajectory, empty, NaN and waypoint-change
  messages, and two planted boundary inputs (a first detection exactly on a
  global s, one not wrapped into the lap). The port's crash guards must fire
  on exactly the frame upstream dies on, and output up to and including that
  frame must match.

Those three ports were accepted at 3,000 scenarios on `DIFFERENTIAL_SEED` 1
and 2. For the predictor and the dense GP that run was sharded with
`DIFFERENTIAL_START`. Use
`OMP_NUM_THREADS=1` when running shards in parallel. collision_prediction's
3,000-scenario check was run once, during its port, and is not in the
repository.

Quirks that no unit test pins, and that the differential now covers:

- `is_visible` never checked. About 5,600 published detections with
  `is_visible` False per 300 scenarios, identical to upstream's.
- The AttributeError deviation (§1). The run must reach upstream's crash,
  and the port's output before it must match.
- The predictor feeding on its own output (§3), across sequences of
  predictions. Counted as `built_on_own_prediction`, about 5,300 per 3,000
  scenarios.
- The predictor's two crash-avoidance deviations (§1). Upstream's
  `ValueError` and `ZeroDivisionError` crashes are counted, though not
  asserted: on seed 1, 33 and 37 in 3,000 scenarios. The 33 `ValueError`
  crashes equal the port's 33 skipped empty spans.
- Every dense-GP quirk in §8 except the zero top speed, by branch counters
  asserted on the default run: first, later and whole-lap fits, the double
  publish at lapcount 1, start-line crossings in half-lap mode, thinning,
  start-line copies. The non-finite guard (§1) is asserted frame for frame:
  224 of upstream's `ValueError` crashes on seed 1, 206 on seed 2.
- The dense GP's mutation check: 16 single-line mutants of the port (one per
  quirk: window ends, thinning, the double count, sigma alignment, CCMA mode,
  the republish, normalize_y, ...) all fail the default 40-scenario run on seed
  7. Two survived the first version of the scenarios, and the two planted
  boundary inputs exist because of them.

Still covered by nothing:

- The exactly-half-lap backwards step. Random floats never land on it.
- The dense GP's empty half-lap window guard (§1), unreachable with lapcount
  0.5 then 1.0, and its zero-top-speed markers (pinned by a unit test).
- The clock constraint.
- The loop-timing deviations. The differential compares per cycle, so it
  says nothing about cycle times.

Section 1 covers deliberate deviations from upstream, which anyone comparing
our results with published ones needs first. Sections 2 to 5 are the
preserved quirks, ordered by how much they can change what the comparison
is able to claim. Section 6 covers the stand-ins this repository puts where
upstream's own publishers would be, so that the four ports can run at all.
Section 7 covers our obstacle tracker, which decides the order in which the
ports see obstacles, and so which car each of them follows.

## 1. Deviations from upstream (behaviour differs)

**obstacle_tracker: coasted tracks are not published.** Upstream's perception
keeps publishing a lost obstacle at its predicted position, with `is_visible`
False, for a while after it was last seen. opponent_trajectory never reads
`is_visible` (§2), so every such prediction enters the GP's training data as a
sighting. The GP exists to infer from sightings, so our tracker, by default,
publishes only obstacles detected this cycle: `publish_coasted` (parameter of
`obstacle_tracker_node`) is False. Nothing downstream needs the coasted ones:
collision_prediction holds a lost opponent through a gap with its own
`expire_counter`, and the spliner commits to an evasion and holds its "gone"
release until "passed". The track still coasts internally, so an obstacle seen
again within `max_misses` cycles returns under the same id. Set
`publish_coasted:=true` to restore upstream's behaviour. Pinned:
`test_coasted_tracks_are_not_published_by_default`,
`test_every_published_obstacle_is_visible_by_default`.

**predictor_opponent_trajectory: depends on the sklearn version.** Before
scikit-learn 1.1, `GaussianProcessRegressor.predict` returns the mean as
(n, 1). This applies to 1.0.2 and to Ubuntu 22.04's apt `python3-sklearn`
0.23.2. Upstream passes that mean to ForzaETH's `get_cartesian`, whose
in-place `x += d * cos(...)` then raises ValueError on **every** prediction,
so upstream never publishes. This was checked against ForzaETH's ROS 1
converter with sklearn 0.23.2, 1.0.2, 1.1.3, 1.2.2 and 1.3.2. ForzaETH's
requirements leave scikit-learn unpinned, so their runs had a working
version.

The port flattens the mean. With sklearn 1.1 or later it matches upstream
exactly; below that it publishes where upstream would not. The image
therefore needs pip `scikit-learn` 1.1 or later (1.3.2, matching the TUM
venv), not the apt package.

Pinned: not by a unit test, since that would need two sklearn versions.
Differential: skips below 1.1.

**predictor_opponent_trajectory: no crash on an empty span.** If no
`/global_waypoints` s falls inside the fitted span, upstream's `predict`
raises ValueError on an empty array and the node dies. The port skips that
half lap. Pinned: `test_span_with_no_global_s_is_skipped`. Differential:
`upstream_crash_ValueError` against `skipped_empty_span`.

**predictor_opponent_trajectory: no crash on a zero top speed.** If every
global waypoint speed is 0, upstream's marker heights divide by zero. The
port draws them with height 0. Pinned:
`test_marker_height_is_zero_without_a_top_speed`.

**predictor_opponent_trajectory: no messages dropped while busy.** Upstream
calls `wait_for_message` on `/proj_opponent_trajectory` once per pass, so a
message arriving during a GP fit, or between passes, is lost. The port
handles every message in its callback, and messages arriving mid-fit queue
(depth 10). Off-trajectory messages are rare, because opponent_trajectory
waits up to 3 s after each, so the same messages are normally processed.
Not pinned (node only).

**predictor_opponent_trajectory: node name and header.** Upstream named the
node `gaussian_process_opp_traj`, the same name as the main GP node, which
only worked because both were anonymous. The port is
`predictor_opponent_trajectory`. Upstream's `header.seq = lap_count` is
dropped: ROS 2 headers have no seq, and rospy overwrote seq on publish
anyway. Not pinned.

**dense GP: depends on the sklearn version,** as the predictor does. Before
1.1 the half-lap d mean reaches ForzaETH's `get_cartesian` as (n, 1) and
upstream raises on every half lap. The port flattens both means. Pinned: not by
a unit test. Differential: skips below 1.1.

**dense GP: no crash on non-finite input.** A non-finite s, d or vs among the
points about to be fitted makes sklearn's `fit` (or, in whole-lap mode,
ForzaETH's `get_frenet`) raise, and the node dies. opponent_trajectory's NaN
poisoning (§2) produces exactly that, from its stale flush. In whole-lap mode,
thinning may drop the bad point before the fit, and upstream then carries on;
so does the port, because the guard sits at the fit, not on arrival. Where
upstream would raise, the port publishes nothing for that message, removes
the non-finite detections from the accumulated list, and wraps a half lap's s
back into the lap as upstream would have after publishing. A NaN vd alone
crashes neither. Pinned:
`test_half_lap_with_a_nan_speed_publishes_nothing_and_drops_the_point`,
`test_nan_at_lap_one_publishes_nothing`, `test_a_nan_vd_does_not_stop_anything`.
Differential: `port_survived_nonfinite` on the same frame as every
`upstream_crash_ValueError`.

**dense GP: no crash on an empty half-lap window,** an IndexError upstream.
The port publishes nothing for that half lap. It needs a third half lap with
lapcount at or below 1, which opponent_trajectory does not produce. Not
pinned.

**dense GP: no crash on a zero top speed.** Upstream divides marker heights by
the top /global_waypoints speed: ZeroDivisionError while any waypoint still
has its initial integer 100, inf once all are numpy floats. The port draws
height 0. Pinned: `test_marker_height_is_zero_without_a_top_speed`. Not in the
differential.

**dense GP: no messages dropped while busy, node name, header.** As for the
predictor: upstream's per-pass `wait_for_message` became a depth-10
subscription, the node is `dense_gp_opp_traj` (upstream's anonymous
`gaussian_process_opp_traj` is also FSDP's name), and `header.seq = lap_count`
is dropped. Both publishes at lapcount 1 carry the first one's stamp, as
upstream's republished message object did. Not pinned (node only).

**dense GP: the Frenet converter sets published d.** As in the other ports the
converter is `FrenetConverter.from_track`, not ForzaETH's, built from
/global_waypoints. Here that is more than markers: whole-lap d is CCMA-smoothed
in x, y and converted back, so the converter sets every published `d_m`.
ForzaETH's `get_frenet` guesses s as "index of the nearest waypoint × 0.1 m"
and corrects it by at most 5 cm, so it is only right at 0.1 m waypoint
spacing; ours projects onto the polyline. Our tracks are at 0.1 m
(porto_odom_trial: 270 waypoints, 0.1 m apart), where the two
should agree to millimetres, but that was not measured. Not pinned.

**dense GP: ccma is vendored,** byte for byte, from ForzaETH's fork at the
commit race_stack pins (`planning/vendor/ccma/README.md` gives the reasons and
the provenance). Pinned: `test_vendored_ccma_is_the_pinned_upstream_file`
(sha256).

**opponent_trajectory: no crash on predictions received only during OVERTAKE.**
Upstream raises AttributeError and the node dies once lapcount ≥ 1 and
`/opponent_trajectory` has been received more than once, if every one of
those messages arrived while the state machine said OVERTAKE (so none was
stored). The port skips the on-trajectory check until a trajectory is
stored, and tracking continues. This hit 69 of 3,000 random differential
scenarios. A run that upstream would have ended keeps feeding the GP here.
Pinned: `test_overtake_predictions_are_counted_but_not_stored`.
Differential: `upstream_crash_AttributeError`.

**opponent_trajectory: loop timing.** Upstream's main loop never sleeps when no
opponent is in range, and its off-trajectory wait blocks in `time.sleep(0.02)`
for up to 3 s. Both are now one 25 Hz timer, and the wait is tracker state.
The wait polls at 25 Hz instead of 50 Hz, and a stale flush can come up to
40 ms later. The differential test shows the logic is identical per cycle,
but the cycle times differ from upstream. Not pinned (node only).

**collision_prediction: loop rate.** Upstream's `rate.sleep()` sits outside its
`while` loop, so the loop never sleeps. The port runs at the 10 Hz upstream
set. The expire counter counts cycles, so a region now lingers
`max_expire_counter / rate` seconds (1 s at the defaults). Upstream's
lingered for however long 10 unthrottled cycles took. Not pinned (node
only).

**collision_prediction: one obstacle array every cycle (liveness).** Upstream
publishes `/collision_prediction/obstacles` only when it finds a region and,
once it has expired, an empty array every cycle. In between, for up to
`max_expire_counter / rate` (1 s) after the last region found, and before the
first expiry, it is silent. A ROS 1 consumer keeps the last message, so to it
the region just persists until the empty array arrives. Our spliner_node treats
an obstacle array older than 0.5 s as a dead input and stops planning, and
local_path_node then invalidates the path, so that silence stopped the car for
up to 0.45 s at a time (8.9 s of 112 s on porto). The port now publishes in
every cycle what a ROS 1 consumer would have held: the region when found, the
last region (freshly stamped, no new markers) while it lingers, otherwise the
empty array. The alternative, a staleness budget over 1 s, would also hide a
dead collision_prediction for over a second. An empty array in the linger
cycles would cancel upstream's expire hysteresis, so the region would vanish a
second early, and a committed evasion would be released as "gone". Region
contents and expiry timing are unchanged. Pinned:
`test_region_lingers_until_it_expires`,
`test_a_new_region_replaces_the_lingering_one`,
`test_nothing_lingers_before_the_first_region`.

**collision_prediction: in-process Frenet conversion.**
`convert_frenet2globarr_service` became `FrenetConverter.to_cartesian`. The
service was only used for marker positions. Pinned:
`test_markers_are_placed_by_to_cartesian`.

**update_waypoints: clock.** `time.time()` became the node clock, so the
GB_TRACK hysteresis follows sim time when `use_sim_time` is set. Not pinned.

**opponent_trajectory, collision_prediction and predictor: Frenet converter.**
Upstream built its converter from `/global_waypoints` x and y. The ports
use `FrenetConverter.from_track`, so the nodes need `track_name`. For
opponent_trajectory and collision_prediction this affects markers only. For
the predictor it also sets the published `x_m`/`y_m` of predicted
waypoints. Their s and d are unaffected. Not pinned.

## 2. What the GP is given: opponent_trajectory

These decide the `ProjOppTraj` that Predictive Spliner's dense GP and FSDP's
sparse GP learn from. That is the difference the comparison turns on, so
treat these as part of the experiment, not as noise.

Predictive Spliner's `gaussian_process_opp_traj.py` switches on lapcount: it
treats `lapcount <= 1` as half-lap mode, and runs its whole-lap initialisation
only when `lapcount == 1` exactly. FSDP's GP was not checked for the same
tests.

**Density is set by the waypoints, not the sensor.** A detection is recorded
only when the opponent moves into a new `/global_waypoints` interval, so a
half lap holds at most one point per waypoint. Changing waypoint spacing
changes how dense the GP input is, whichever GP is downstream. Preserved
because this is the input both GPs were designed and published against.
Pinned: `test_one_detection_per_waypoint_interval`.

**lapcount is opponent travel, not start-line crossings.** It grows by 0.5 each
time more than half a lap of detection-to-detection distance has built up
since the first detection. Lap boundaries therefore fall wherever the
opponent was first seen, and overshoot past the half lap is discarded.
Pinned: `test_lapcount_counts_opponent_travel_from_its_first_detection`,
`test_half_lap_publishes_and_shares_its_last_point`.

**Nothing is stale before lapcount 1.** If the opponent is lost and found
again further on, the gap becomes one point whose vs is the average speed
across it, and the gap counts towards the half lap. Pinned:
`test_nothing_is_stale_before_lap_one`,
`test_gaps_before_lap_one_are_bridged_with_the_average_speed`.

**Only the first half lap is checked for gaps.** If two consecutive detections
are more than 2.5 m apart, everything before the gap is dropped, lapcount
goes back to 0, and that half lap is not published. Later half laps are
published with whatever gaps they have. Pinned:
`test_first_half_lap_with_a_gap_is_trimmed_and_not_published`.

**Stale flush from lapcount 1, measured from the last change of waypoint.**
When the node clock is more than 0.5 s past the header stamp of the last
accepted detection, the half lap is published (if it has more than 5
points) or dropped, and restarts. An opponent that is in view but stays
inside one waypoint interval for 0.5 s (under about 0.2 m/s at 0.1 m
spacing) also goes stale. Pinned:
`test_stale_half_lap_is_flushed_with_the_last_step_added_to_lapcount`,
`test_stale_half_lap_with_few_points_is_dropped`,
`test_opponent_in_view_but_in_one_interval_goes_stale`.

**The stale flush adds the wrong distance to lapcount.** It adds
`trav_dist / track_length`, where `trav_dist` is the last single step
(accepted or not), not the half lap's distance. After the first flush,
lapcount stops being a multiple of 0.5: the node smoke run published
2.0055. Since flushes only happen from lapcount 1, the GP's `== 1` test is
only reachable before any flush. Pinned:
`test_stale_half_lap_is_flushed_with_the_last_step_added_to_lapcount`.

**Coasted tracks count as detections.** The last `/perception/obstacles`
message is reused every cycle, and `is_visible` is copied into each point
but never checked. A tracker that keeps publishing a lost opponent's
predicted position feeds those predictions to the GP as if they were seen.
This matters as soon as real detection replaces `fake_opponent_node`. Not
pinned by a unit test. Differential: `invisible_detections_published`. Our
obstacle_tracker does not publish coasted tracks by default, so the quirk is
preserved but not exercised (§1).

**An opponent is only found if it is last in the obstacle array.** The pick is
the dynamic obstacle whose `s_start` is nearest ahead of the ego (modulo
the lap) and within 6 m, recorded at its `s_center`. But the loop resets its
pick to None on every obstacle that is not a nearer dynamic one, so with a
cone or a second car listed after the opponent, nothing is tracked. Nothing
is learned while the opponent is behind or more than 6 m ahead. See §7 for
the order obstacle_tracker publishes in because of this. Pinned:
`test_opponent_is_lost_unless_it_is_last_in_the_array`,
`test_opponent_is_the_nearest_dynamic_obstacle_ahead_within_6_m`,
`test_opponent_is_recorded_at_s_center_and_ranged_by_s_start`.

**Jumps over 1 m after lapcount 1.** The jump is dropped, and the next
detection becomes a "first point": vs = vd = 0, with no bounds, speed or
direction checks, and the oldest point of the half lap is popped to make
room. The gap does not count towards the half lap. Before lapcount 1 the
same jump is kept. Pinned:
`test_jump_over_1m_after_lap_one_restarts_with_an_unchecked_first_point`,
`test_jump_over_1m_before_lap_one_is_kept`.

**Off-trajectory points are sent twice.** From lapcount 1, once
`/opponent_trajectory` has been received twice, each detection is compared
with the first predicted waypoint further on in s (first in list order, not
nearest). More than 5 consecutive detections 0.3 m or more off in d, while
under half a lap has been covered, publish the half lap so far with
`opp_is_on_trajectory` False. The node then waits up to 3 s for a new
prediction. During the wait, detections are still added, but nothing is
checked for staleness, off-ness or a completed half lap. Afterwards the
half lap carries on, so the off points go out again in the next
on-trajectory message. Pinned:
`test_six_points_off_the_prediction_publish_and_wait`,
`test_points_are_still_added_while_waiting_and_kept_after`,
`test_wait_ends_after_3_s_without_a_new_prediction`,
`test_half_lap_is_not_checked_while_waiting`,
`test_on_trajectory_check_uses_the_first_waypoint_ahead_in_list_order`.

**Predictions received during OVERTAKE are counted but not stored.** They still
count towards the "received twice" condition. Pinned:
`test_overtake_predictions_are_counted_but_not_stored`.

**NaN poisoning.** If the off-trajectory wait ends on a cycle with no
opponent, and the half lap completes on that cycle, upstream writes `None`
into the raw row the next detection is differenced against. numpy stores
it as NaN. The next point's vs, vd and the half lap's distance become NaN,
so the half lap never completes. Only a stale flush, possible from lapcount
1, recovers it. Pinned:
`test_wait_ending_without_an_opponent_poisons_the_half_lap_with_nan`.

**Filters on each detection.** The following are dropped:
- points going backwards by less than half a lap (treated as duplicates)
- points off the track according to the waypoint's `d_left`/`d_right`
- points faster than 15 m/s
- points whose tracker speed changes faster than 20 m/s²
- points stamped under 10 ms after the previous one

A backwards step of more than half a lap is a wrap: vs is copied from the
previous kept point, or the point is dropped if the half lap has none. A
step of exactly half a lap backwards counts as forwards, with negative vs.
Pinned: `test_backwards_detection_is_dropped`,
`test_off_track_and_too_fast_detections_are_dropped`,
`test_detections_under_10_ms_apart_are_dropped`,
`test_wrap_copies_vs_from_the_previous_detection`. The exactly-half-lap case
is not pinned.

**nrofpoints is not len(detections).** The first detection of a half lap
that starts from nothing is counted but not kept, because its velocity is
unknown. A published half lap's last point is kept as the next half lap's
first, so consecutive messages share a point. Pinned:
`test_first_detection_is_counted_but_not_kept`,
`test_half_lap_publishes_and_shares_its_last_point`.

**Clocks.** Staleness and velocities compare the node clock with obstacle
header stamps. If `use_sim_time` is set on one side only, every detection
from lapcount 1 is stale, or none ever is. This is a deployment constraint
rather than a defect. Not pinned.

## 3. After the opponent leaves its line: predictor_opponent_trajectory

The predictor acts only on the half laps opponent_trajectory publishes with
`opp_is_on_trajectory` False. It fits a local sklearn GP to the latest
detections and to the stored trajectory 7 m beyond them, and writes the
result back into `/opponent_trajectory`, the message collision_prediction
reads for the opponent's d and speed.

The stage is identical on both branches, but its base is not: it edits
whichever trajectory the branch's main GP (dense or sparse) last published.
So the behaviour after an off-trajectory event is not a clean dense-versus-sparse
comparison. It is each GP's output, patched by this shared GP over the span
of the return.

**It feeds on its own output.** It publishes on `/opponent_trajectory`, the
topic it subscribes to and the main GP publishes on, and edits the stored
waypoints in place. A second off-trajectory prediction therefore builds on
the first, until the main GP publishes again. Pinned:
`test_stored_trajectory_is_edited_in_place`. Differential:
`built_on_own_prediction`.

**It is what ends opponent_trajectory's wait.** opponent_trajectory's 3 s
wait after an off-trajectory cut ends on any new `/opponent_trajectory`
message, and the predictor's answer is normally the first. A slow or
skipped prediction (see the empty span in §1) lets the wait run its full
3 s. Not pinned (two nodes).

**Only d is predicted.** `proj_vs_mps` keeps the stored trajectory's value
(upstream's vs resampling is commented out), so the predicted return keeps
the old speed profile, and collision_prediction's timing uses it. vd is
linearly interpolated over the merged points, which assumes their s is
increasing; `np.interp` is silently wrong otherwise. Pinned:
`test_waypoints_in_the_span_are_replaced_and_marked`,
`test_vd_is_interpolated_linearly_across_the_merged_points`.

**`d_var` is a standard deviation, and `vs_var` = 69 is a flag.** Predicted
waypoints get the GP's sigma in `d_var` and the sentinel 69 in `vs_var`,
which only the markers use (red). The main GP fills `vs_var` with its
sigma_vs, so anything reading `vs_var` as a variance on predicted points
gets 69 m²/s². Pinned: `test_waypoints_in_the_span_are_replaced_and_marked`,
`test_markers_follow_speed_and_variance`.

**The GP's training points.** It takes the last 15 detections, then the first
15 trajectory points, in list order and not by nearest s, that lie more than
7 m beyond the last detection. The search runs over the trajectory repeated
for a second lap, so a return past the start line lands at the start of the
lap. Near the end of the second lap fewer points, or none, are found, and the
GP is fitted to detections alone. Pinned:
`test_merge_takes_the_last_15_detections_and_15_points_7m_beyond`,
`test_merge_takes_trajectory_points_in_list_order`,
`test_merge_near_the_end_of_the_second_lap_finds_fewer_points`,
`test_a_return_past_the_start_line_lands_at_the_start`.

**Unwrapping.** From the first consecutive pair of detections more than half a
lap apart, every later detection gets a lap added, even one close to the
shifted s before it, and a forward jump of more than half a lap counts as a
wrap too. The shift edits the received message. Pinned:
`test_unwrap_adds_a_lap_from_the_first_big_step_onwards`,
`test_unwrap_keeps_shifting_once_it_has_started`,
`test_unwrap_also_shifts_a_forward_jump_of_more_than_half_a_lap`,
`test_detections_are_unwrapped_in_the_message`. Differential:
`unwrapped_detections`.

**Only waypoints on a global s are replaced.** A waypoint changes only if its s
is within 1e-8 of a `/global_waypoints` s in the fitted span (taken modulo
the lap). The main GP lays its waypoints on those s, so this holds for its
output. Pinned: `test_waypoint_not_on_a_global_s_is_left_alone`.

**Kernel.** C(0.5) · Matern(1.0, ν = 1.5) + C(0.2) · White(1.0), with one
L-BFGS-B run from those values and no restarts, so the fit is
deterministic. Only the two constants are bounded, to [1e-6, 1e3]. The
length scale and noise level keep sklearn's defaults, (1e-5, 1e5), and fits
do run into them (sklearn's ConvergenceWarning). Pinned:
`test_kernel_is_upstreams` (the kernel; the default bounds are not
pinned).

**Which half laps it acts on.** On-trajectory half laps, and half laps with
no detections, are ignored. Nothing is done until both `/global_waypoints`
and an `/opponent_trajectory` have arrived. Pinned:
`test_on_trajectory_and_empty_half_laps_are_ignored`,
`test_nothing_before_waypoints_and_a_trajectory`.

**Second-lap copies and two track lengths.** The second-lap copies of the
trajectory carry no `d_var` or `vs_var`. The double-lap s list and the top
speed are fixed from the first `/global_waypoints`, while lap arithmetic uses
the latest one. Pinned: `test_second_lap_copies_carry_no_variances`,
`test_waypoints_fixed_from_the_first_message_but_lap_length_follows`.

## 4. What the planners do with it: collision_prediction and update_waypoints

These shape the region the avoidance planner treats as blocked, and whether
it may overtake at all. Both comparison branches share them, but they
change absolute lap times and overtake counts.

**collision_prediction: force_trailing is tied to the GP's lap count.** Overtaking
is forbidden when the opponent is more than 0.25 m off its predicted line,
or fewer than one opponent lap (the `lap_count` on `/opponent_trajectory`)
has been observed. Combined with section 2, overtakes can only start once
the lapcount chain has reached 1. Pinned:
`test_trailing_forced_before_one_opponent_lap`,
`test_trailing_not_forced_on_the_predicted_line`,
`test_trailing_forced_off_the_predicted_line`.

**collision_prediction: the force-trailing lookup uses the unwrapped opponent
s.** After a lap is added, the nearest predicted waypoint is the last one
rather than the right one. Pinned:
`test_trailing_check_uses_the_unwrapped_opponent_s`.

**collision_prediction: only the first dynamic obstacle is the opponent**
(upstream's own TODO). Note that this is a different rule from
opponent_trajectory's "last in the array". With several obstacles, the two
nodes can be tracking different cars. Pinned:
`test_first_dynamic_obstacle_is_the_opponent`. obstacle_tracker orders its
output so that they never do (§7).

**collision_prediction: the ego lap is one waypoint short.** The last updated
waypoint is dropped as a duplicate of the first, and the lap length is read
from what is left. Every wrap uses that length. The opponent lap keeps its
last waypoint. Pinned: `test_ego_lap_is_one_waypoint_short`,
`test_opponent_lap_keeps_its_last_waypoint`,
`test_markers_wrap_by_the_short_ego_lap`.

**collision_prediction: start-of-lap wrap heuristic.** Ego past 2/3 of the lap
and opponent before 1/3 means the opponent is a lap ahead, and gets one lap
added. Exercised by `test_region_s_is_not_wrapped`; the thresholds
themselves are not pinned.

**collision_prediction: region s is not wrapped.** `s_start` and `s_center`
can be at or past the lap length, which breaks `Obstacle.msg`'s
`[0, lap_length)` convention. Pinned: `test_region_s_is_not_wrapped`.

**collision_prediction: a region with no end is a wall.** If the gap never
reopens within the horizon, the region ends at the opponent's final
predicted s and spans d from -10 to 10. Pinned:
`test_region_without_an_end_is_a_wall_ending_at_the_opponent`.

**collision_prediction: expiry counts cycles.** A region is deleted
`max_expire_counter` cycles after the last one was found. After that, the
empty array and marker deletes go out every cycle. (Until then the port
republishes the last region every cycle where upstream was silent: §1.) Pinned:
`test_region_expires_after_max_expire_counter_cycles`,
`test_region_published_and_deleted_in_one_cycle_when_counter_is_one`,
`test_no_opponent_publishes_nothing_but_counts_down`.

**collision_prediction: no wait for odometry.** The ego is at s = 0, v = 0 until
odometry arrives. Pinned: `test_ego_defaults_to_start_line_at_rest`.

**update_waypoints: nearest waypoint without a lap wrap.** It is found by
`|s_m - ego_s|`, so near the line either duplicated endpoint can be picked.
Pinned: `test_no_lap_wrap_when_picking_nearest`.

**update_waypoints: endpoints not kept in sync.** Upstream's two lines meant to
copy speeds between the duplicated first and last waypoint only read the
values. Pinned: `test_endpoints_are_not_kept_in_sync`.

**update_waypoints: only the first waypoint arrays are used.** A one-shot
publish before the node subscribes is missed, because the port uses
VOLATILE QoS. Pinned: `test_only_the_first_scaled_array_is_used`,
`test_only_the_first_global_array_is_used`.

**update_waypoints: waits for a real GB_TRACK message.** It starts in
GB_TRACK with no start time, so nothing is updated until the state machine
publishes. Pinned: `test_nothing_updated_before_any_state_message`.

**opponent_trajectory: two track lengths.** Bins, bounds, the half-lap test
and the wrap use the length fixed when the loop starts. The first-half-lap
gap test uses the latest `/global_waypoints`. The two only differ if the
waypoints change while running. Pinned:
`test_later_waypoints_change_only_the_gap_test_track_length`.

## 5. Cosmetic

- **predictor_opponent_trajectory:** cylinders are as tall as the waypoint's
  speed over the top speed, 5 × `d_var` wide (clamped to [0.07, 0.7]), and
  red where predicted, yellow elsewhere. Pinned:
  `test_markers_follow_speed_and_variance`.
- **opponent_trajectory:** after a few cycles the marker array alternates
  DELETEALL with single markers, so RViz shows only the newest. Pinned:
  `test_marker_array_settles_to_deleteall_between_single_markers`.
- **collision_prediction:** the published Obstacle has size and variances 0
  and `is_visible` False, although `Obstacle.msg` asks for positive
  variances. Not pinned (node only).
- **collision_prediction:** upstream read `speed_offset` from the
  `max_expire_counter` parameter and never used it. Dropped. Not pinned.
- **collision_prediction:** marker topics are `/collision_predict/*` while the
  rest is `/collision_prediction/*`, and `force_trailing` is a relative
  name. Kept as upstream. Not pinned.
- **update_waypoints:** upstream opened a `convert_frenet2globarr_service`
  proxy and never called it. Dropped. Not pinned.
- **All three nodes:** upstream marker stamps were zero or fixed at startup.
  The ports stamp markers at publish time, and opponent_trajectory's get an
  identity quaternion where upstream left it all zeros. rclpy needs ints
  converted to float and bool. Not pinned.

## 6. Stand-ins for upstream's publishers

The four ports wait on `/global_waypoints`, `/global_waypoints_scaled` and
`/state_machine`. Upstream, ForzaETH's global planner and state machine
publish them; nothing in this repository did, so none of the ports could
leave its wait loop. `planning shared_graph.launch.py` starts the ports
together with the stand-ins below. None of these is a port, and each one
simplifies in a way that the comparison has to account for.

**global_waypoints_publisher_node: the raceline as global waypoints.**
`tracks/<name>/raceline.csv` (build_raceline's output, the same file
pure_pursuit and FrenetConverter read) becomes one `WpntArray`, published once
on both topics. The conversion is in `global_waypoints_publisher.py`, which
imports no ROS. Its docstring gives the source of every `Wpnt` field. In
brief:

- `d_m` is 0, because the waypoints are the reference line that
  `frenet_state_node` measures d against. Only `reference=raceline` is
  accepted.
- `d_left` and `d_right` are positive distances from the raceline to the
  track edges, as upstream's are. They are computed from `centerline.csv`,
  because the raceline file has no widths.
- `psi_rad` is ROS yaw. The file's TUM heading (yaw - pi/2) is converted and
  checked against the path direction first.
- The closing row is kept, because `opponent_trajectory` and the predictor
  take `wpnts[-1].s_m` as the track length.
- A raceline without positive speeds is refused rather than zero-filled.
  Curvature and acceleration are recomputed only when the file has none, and
  the log says so when that happens.

Pinned: `test_global_waypoints_publisher.py`.

**/global_waypoints_scaled is unscaled.** Upstream it is the raceline with
the speeds multiplied by a velocity factor that the state machine owns. There
is no velocity scaler here, so the factor is 1 and both topics carry
identical messages. This is a deliberate simplification. `update_waypoints`
republishes the scaled array as `/global_waypoints_updated`, and
`collision_prediction` predicts the ego's motion from that, so every collision
region is computed at full raceline speed. Upstream would compute it at
whatever speed its scaler had set. The node logs a WARN about this at startup.
Not pinned.

**Latched waypoints (a deviation in all four node wrappers).** Both waypoint
topics are TRANSIENT_LOCAL, KEEP_LAST(1), RELIABLE, on the publisher and on
every port's subscription (`planning/waypoint_qos.py`). The ports use only the
first message, and a VOLATILE subscription never receives one published before
it started, so with the original VOLATILE subscriptions a port that started
after the publisher waited forever. The side effect is that a VOLATILE
publisher on these topics now reaches no port. To publish by hand, use
`ros2 topic pub --qos-durability transient_local --qos-depth 1`. Pinned by
the late-subscriber check in `test_shared_graph_launch.py`.

**state_machine_stub_node: a fixed state, not a state machine.** It publishes
the `state` parameter (`GB_TRACK`, `TRAILING` or `OVERTAKE`) on
`/state_machine` as `std_msgs/String` at 10 Hz. The state can be changed at
runtime with `ros2 param set /state_machine_stub state OVERTAKE`. The stub
never changes state by itself, and it ignores
`collision_prediction/force_trailing`, the output that upstream's state
machine uses to decide whether an overtake is allowed. By the ports' code:

- `update_waypoints` writes measured speeds only in GB_TRACK, after
  `hysteresis_time`.
- `opponent_trajectory` stops storing GP output in OVERTAKE.
- `collision_prediction` stores the state and never reads it.
- The predictor does not subscribe.

Unknown or lower-case state names are refused, since they would leave both
gated branches inert without any error. Pinned (validation only):
`test_known_states_pass`, `test_unknown_states_are_refused`.

**fake_opponent_node's marker moves to `/fake_opponent/marker` in the
launch.** `opponent_trajectory_node` publishes a `MarkerArray` on
`/opponent_marker` (upstream's name), and the fake opponent defaulted to a
`Marker` on the same topic. Two types on one topic break every subscriber.
Not pinned.

**The main GP is a launch argument.** `predictor_opponent_trajectory` and
`collision_prediction` both wait for an `/opponent_trajectory`. Upstream,
that comes first from `gaussian_process_opp_traj.py`, which differs between
Predictive Spliner and FSDP and so is not shared code. `shared_graph.launch.py`
therefore starts none by default (`gp:=none`), and neither node ever publishes.
`gp:=dense` starts Predictive Spliner's dense GP (§8). FSDP's sparse GP is not
ported. `test_shared_graph_launch.py` asserts all three cases: with no GP the two
nodes are silent; a synthetic `/opponent_trajectory` along the raceline
unblocks both; and with `gp:=dense` both publish with nothing synthetic.

## 7. Obstacle order: obstacle_tracker

`obstacle_tracker_node` (`obstacle_tracker.py`) sits between a detector and
the planners. It takes detections with a position and nothing else, and adds
ids, velocities, variances and `is_static`. It is our code, not a port, so it
is free to choose the order of `/perception/obstacles`, and that order
decides which car each port follows:

- collision_prediction takes the **first** obstacle that is not static (§4).
- opponent_trajectory takes the **last** obstacle, only if it is dynamic and
  no nearer-ahead dynamic obstacle comes before it. Distance is
  `(s_start - ego_s) % lap`. Any obstacle after it, static or not, resets its
  pick (§2).

With two or more dynamic obstacles **no order makes both take the same one**.
The first rule needs the target before every other dynamic obstacle; the
second needs it after them.

**The order chosen.** Static obstacles first, by s. Then dynamic obstacles by
`(s_start - ego_s) % lap`, ascending, so the nearest ahead comes first. A car
behind the ego sorts to the end.

- **No dynamic obstacle:** neither port has an opponent, as before.
- **One dynamic obstacle, any number of static ones:** both ports take it.
  Statics have to come first for this: one listed after the car would reset
  opponent_trajectory's pick.
- **Two or more dynamic obstacles:** collision_prediction takes the nearest
  one ahead. opponent_trajectory takes none, because the last obstacle is
  farther ahead than its first pick. The two ports never follow different
  cars, but while a second car is tracked the GP gets no new points. That
  includes a car behind the ego, if the detector sees astern (and a coasted
  track, if `publish_coasted` is set).

**Why this order, not the reverse.** Dynamic obstacles nearest-ahead last
would give opponent_trajectory the nearest car and collision_prediction the
farthest: the force-trailing and collision-region decisions would be made by
rolling one car forward along a GP trajectory learned from another. A GP that
pauses is a smaller error than a GP of the wrong car. The order that serves
collision_prediction also serves the obstacle that matters most for safety,
the nearest one ahead.

Without an ego position (no `/car_state/odom_frenet` yet) dynamic obstacles
are ordered by id. The spliner sorts obstacles itself and does not depend on
the order.

Pinned (against the ports' own selection functions, `first_dynamic` and
`find_nearest_dyn_obstacle`):
`test_statics_first_then_dynamic_nearest_ahead_first`,
`test_one_opponent_among_cones_is_taken_by_both_consumers`,
`test_two_opponents_never_split_the_consumers`,
`test_a_car_behind_does_not_split_the_consumers`,
`test_without_an_ego_position_dynamic_obstacles_are_ordered_by_id`.

**Coasted tracks are not published by default** (§1). A confirmed track
survives `max_misses` cycles (10, 0.25 s at 40 Hz) without a detection, at its
constant-velocity prediction with growing variances, and keeps its id if seen
again, but nothing is published for it meanwhile. With `publish_coasted:=true`
it is published with `is_visible` False, and since opponent_trajectory does not
check that flag (§2) the GP then receives up to `max_misses` predicted points
per occlusion. A coasted track never counts towards "two or more dynamic
obstacles" by default. Pinned:
`test_coasted_tracks_are_not_published_by_default`,
`test_a_track_survives_a_dropout_shorter_than_max_misses` (with
`publish_coasted`).

## 8. Predictive Spliner's dense GP: dense_gp_opp_traj

`dense_gp_opp_traj_node` turns opponent_trajectory's half laps
(`/proj_opponent_trajectory`, §2) into the `/opponent_trajectory` that
collision_prediction rolls the opponent forward on (§4) and the predictor
patches (§3). It is Predictive Spliner's alone, so everything here is part of
what the comparison with FSDP measures, not shared noise. Deviations are in §1;
these are preserved.

**Unobserved waypoints run at 100 m/s.** The trajectory starts as the raceline
(d = 0) with `proj_vs_mps` 100 on every waypoint, and every message carries the
whole lap, filled in or not. After the first half lap the unfilled half is the
one the opponent is about to enter, so collision_prediction rolls it away at
100 m/s and predicts no collision there until the whole-lap fit. Pinned:
`test_initial_trajectory_is_the_raceline_at_100_mps`,
`test_first_half_lap_fills_its_window_and_leaves_the_rest_at_100`.

**Half-lap mode (lapcount ≤ 1): two fixed windows.** The first half lap is fitted
on the global s strictly between its first detection and half a lap on. The
second is fitted on the complement, [end of the first, its start + lap], from
its own detections wherever they are; the GP extrapolates to cover the
difference. Global s past the lap are wrapped only if the half lap's detections
crossed the start line; otherwise those waypoints are not written. Pinned:
`test_first_window_is_strictly_inside_first_detection_to_half_a_lap_on`,
`test_second_half_lap_window_is_the_complement_wherever_its_detections_are`,
`test_half_lap_window_past_the_lap_is_dropped_without_a_start_line_crossing`.

**Means fall back to 0 away from the data.** `normalize_y` is False, so where
a half-lap window extends beyond its detections, predicted d and vs return to
0, and the opponent is predicted to stop there. Pinned:
`test_half_lap_means_fall_back_to_zero_away_from_the_detections`.

**Variances are standard deviations, and whole-lap d has none.** In half-lap
mode `d_var` and `vs_var` are the GPs' sigmas. In whole-lap mode d comes from
CCMA, so `d_var` is 0 everywhere, and the predictor's and markers' widths fall
to their minimum. Pinned: `test_half_lap_variances_are_standard_deviations`,
`test_lap_one_publishes_twice_and_the_second_keeps_lap_one_and_the_stamp`.

**Whole-lap `vs_var` is the sigma of the point about 3 m earlier.** The vs
prediction runs over the lap with the 3 m either side of the start line
copied across it. The mean is cut back to the lap, the sigma is not, so
waypoint j gets the sigma of the j-th point of the longer list. Pinned:
`test_whole_lap_vs_var_is_the_sigma_of_the_point_3_m_earlier`.

**lapcount == 1 exactly publishes twice.** First the second half lap's
trajectory, then the same message object again after the whole-lap fit edited
its waypoints in place, so the second keeps lap_count 1 and the first one's
stamp. Upstream compares with `== 1` and sets lapcount 1.1 internally for the
whole-lap fit. Pinned:
`test_lap_one_publishes_twice_and_the_second_keeps_lap_one_and_the_stamp`.

**The last half lap counts twice from lapcount 1.** Its detections are
appended once in half-lap mode and again for the whole-lap fit (the same
objects), so until thinning removes them they weigh double in the vs GP and
in CCMA. Pinned: `test_lap_one_counts_the_last_half_lap_twice`.

**Whole-lap d is CCMA, not a GP.** The accumulated detections, plus one copied
point at each end, go to x, y, through `CCMA(w_ma=5, w_cc=3)` in its default
padding mode (endpoints repeated, not cyclic, so the single copied point is
all the wrapping there is), back to Frenet, sorted by s, and interpolated onto
every global s. Pinned: `test_whole_lap_d_is_ccma_smoothed_and_follows_the_opponent`.
Differential mutant: `ccma_wrapping_mode`.

**The vs GP wraps by copying 3 m.** Detections and prediction s within 3 m of
either end of the lap are copied a lap the other way before fitting. Pinned:
`test_whole_lap_copies_the_3_m_either_side_of_the_start_line`.

**Detection s is edited in place.** In half-lap mode every detection from the
first decrease in s onwards gets a lap added, and all accumulated s are
wrapped back after publishing. In whole-lap mode every detection but the first
is taken modulo the lap. Pinned:
`test_half_lap_across_the_start_line_is_shifted_then_wrapped_back`,
`test_whole_lap_wraps_every_detection_but_the_first`.

**Thinning keeps the newest per bin and drops the top of the lap.** Once
more than 200 detections are accumulated (from lapcount 1), they are binned
every two waypoint spacings and only the newest in each bin is kept. The bins
stop one to two bins below the highest s, so the detections just before the
start line are dropped every time it runs. Pinned:
`test_thinning_keeps_the_newest_per_bin_and_drops_the_top_bins`.

**Kernels and fitting.** vs: C(0.5) · RBF(1.0) + C(0.2) · White(1.0). d (half
lap only): C(0.5) · Matern(1.0, ν = 1.5) + C(0.2) · White(1.0). Only the
constants are bounded, to [1e-6, 1e3]; one L-BFGS-B run, no restarts, so fits
are deterministic. Pinned: `test_kernels_are_upstreams`.

**Fixed from the first waypoints.** The global s list, the top speed and the
initial trajectory come from the first `/global_waypoints`; the lap length
follows the latest. Pinned:
`test_waypoints_fixed_from_the_first_message_but_lap_length_follows`.

**What it ignores.** Off-trajectory half laps (the predictor's job) and half
laps without detections. Nothing happens before `/global_waypoints`. Pinned:
`test_off_trajectory_and_empty_half_laps_are_ignored`,
`test_nothing_before_waypoints`.

**Cosmetic.** `OppWpnt.id` is never set (0). Markers are all yellow, as tall
as speed over top speed and 5 × `d_var` wide, clamped to [0.07, 0.7]. Pinned:
`test_initial_trajectory_is_the_raceline_at_100_mps`,
`test_markers_follow_speed_and_d_var`.

### In the graph: what collision_prediction and the spliner do with it

Run once, 2026-11-15, not in CI: `shared_graph.launch.py gp:=dense` with
`localizer:=external`, the tracker, and a weaving fake opponent (2 m/s, ±0.4 m,
noise-free). A scripted localizer put the ego 5.5 m behind the opponent and
closed at 1 m/s (odometry 3 m/s) to 0.5 m, then jumped back, for 120 s. The
ego never drives an evasion: nothing here is a lap, a collision count or an
overtake. Two tracks: porto_odom_trial (26.9 m lap, 1.07–2.49 m wide) and a
synthetic 3 m wide oval (35.2 m lap).

- **The GP.** First publish after the first half lap, with half the waypoints
  observed. From lapcount 1 all are observed: vs 2.00 everywhere, d within
  0.007 m rms of the true weave (0.044 m max).
- **collision_prediction publishes, with nothing faked.** force_trailing goes
  False at lapcount 1, as §4 says it must. It flicks True for about a second
  once per opponent lap, at the start line, which is consistent with the
  unwrapped-s lookup (§4).
- **Its regions are sane where they end.** About 75% end within the
  horizon. Their `s_start` ahead of the ego matches a constant-speed catch
  estimate to a few centimetres at the median. They are 2.7–3.0 m long and
  0.54–0.91 m wide, and the true opponent d at the catch lies inside
  [d_right, d_left] every time. d_center is 0.12–0.14 m rms from it, since the
  band spans the opponent's d at both ends.
- **The rest are walls** (d −10 to 10), mostly when the catch is near the end
  of the 4 s horizon. About 20–30% of region `s` values are at or past the
  lap length (§4).
- **The spliner on `/collision_prediction/obstacles` (porto):** no evasion in
  120 s. 1,149 cycles were "no feasible side": only 19% of ended regions leave
  the 0.65 m it needs (car_width + evasion_dist) on either side inside its
  0.15 m margin. Where they did, it was "too close" (157) or "blocked" (103).
- **The spliner on the wide oval:** it plans around ended regions. 973 cycles
  evading (left 484, right 489), 405 "too close" (regions starting within
  about 1.5 m), 63 releases (24 "gone", as the recomputed region moves more
  than match_tolerance; 39 "off line", an artefact of the ego not following).
- **On both tracks, walls read as "clear".** The spliner ranged obstacles by
  `s_center`, which for a wall lies beyond its 10 m lookahead even when
  `s_start` does not. Fixed since: it ranges on `s_start` (`spliner.py`
  `obstacles_ahead`, `test_a_wall_region_that_starts_close_is_not_clear`).
- **On both tracks, it stopped on stale inputs.** collision_prediction
  published nothing for up to 1 s after a region vanished, so the spliner
  stopped on its 0.5 s staleness check: 162 times on porto, 7.9 to 8.9 s of
  110 s silent. Fixed since by the liveness deviation (§1).
- **With both fixes, porto again (same run):** still no evasion. Now 0.0 s
  silent, and 161 cycles of walls starting within the lookahead are "no room",
  where they had been "clear". Only 19% of ended regions leave the room the
  spliner needs, so on porto the limit is geometric. The closed-loop sweep of
  the spliner's clearance on porto is in
  `docs/reports/clearance_sweep/README.md`.
- **The car could not start cold.** With the spliner on
  `/collision_prediction/obstacles`, local_path_node needs the spliner, the
  spliner needed an obstacle array, collision_prediction publishes none until
  the GP has published, and the GP needs the car following the opponent for
  half a lap. Fixed in spliner_node, not local_path_node: with
  `trail_obs_topic:=/perception/obstacles` it trails on the tracked obstacles
  until collision_prediction first publishes, and while its
  `force_trailing` is True (fewer than one opponent lap learned). It never
  starts an evasion round a moving obstacle there, and refuses (NO_ROOM) within
  `trail_gap` (2 m) of one (`spliner.py` TRAILING). Off by default, so
  avoid_graph is unchanged. From standing at 0.25 m clearance the car trailed
  for its first lap, switched to the regions at the GP's whole lap (14.3 s), and
  passed on its second; 12 laps, no contact
  (`docs/reports/clearance_sweep/README.md`).
- **Contacts at small clearances were planning, not tracking.** The planner
  released a committed evasion mid-pass ("gone" or "blocked": the
  recomputed region moves), and the stateless replan ignored the region the car
  was inside (its near edge behind the rear axle), said "clear" and steered
  back into the opponent. Fixed: `obstacles_ahead` ranges on the far edge, and
  "blocked" by the committed obstacle (same id) is held alongside until
  "passed" (`spliner.py` COMMITMENT). Pinned closed-loop by
  `test_closed_loop_clearance.py`: the planned gap against the opponent's true
  pose must never be negative. That test failed on the old planner (34 cycles,
  −0.184 m at 0.25). With the fix, the closest approaches are tracking error
  on the rejoin, and the cut-back after the region's far edge (upstream's
  `save_distance_back` 0.4 m, about 5 cm along the track for our footprints).
  `docs/reports/clearance_sweep/README.md` has the re-sweep.
- **GB_TRACK teaches collision_prediction the ego's past speed.** With the
  stand-in state machine fixed at GB_TRACK, update_waypoints (§4) overwrites
  `/global_waypoints_updated` speeds with the ego's measured speed. After a
  lap of trailing at the opponent's speed, collision_prediction rolls the ego
  forward at that same speed, predicts no catch and publishes empty arrays,
  while pure pursuit drives the raceline speed. In one closed-loop GB_TRACK
  run the car touched the opponent with no region published. Upstream's real
  state machine would leave GB_TRACK while trailing; the stub does not (§6).
- **In AutoDRIVE** (`predictive_race.launch.py`, porto, 2026-11-16;
  `docs/reports/autodrive_predictive/README.md`):
  - Evasions along the inner wall hit it until the spliner's `bound_margin`
    was raised to 0.30. It bounds the car's centre, and 0.15 is the car's half
    width; the launch passes 0.30, avoid_graph keeps 0.15.
  - pure_pursuit_node could not be stopped when started from a script's
    background job (SIGINT inherited as ignored). Fixed in its `main()`.
  - First plan takes about 12 s, all of it Cartographer localizing.
  - Every node held its rate.
  - Tracking while evading: p95 5.5 cm at speed_scale 0.5.
  - No passes in 37 laps on porto at evasion_dist 0.25 with that wall margin.

Not pinned (one-off run; the launch test checks only that the GP and
collision_prediction publish).
