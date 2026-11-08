# Preserved upstream quirks in the shared Predictive Spliner / FSDP code

Three files are byte-identical between ForzaETH's Predictive Spliner and ZJU's
FSDP, and are ported once into `ws/src/planning/planning/` for both comparison
branches to import:

| Upstream file | Port | Tests |
|---|---|---|
| `update_waypoints.py` | `update_waypoints.py`, `update_waypoints_node.py` | `test_update_waypoints.py` |
| `collision_prediction.py` | `collision_prediction.py`, `collision_prediction_node.py` | `test_collision_prediction.py` |
| `opponent_trajectory.py` | `opponent_trajectory.py`, `opponent_trajectory_node.py` | `test_opponent_trajectory.py` |

All three ports keep upstream's defects, and for the same main reason: if
the shared code is "fixed", both branches stop running the code they claim
to port, and our numbers can no longer be compared with ForzaETH's or ZJU's
published results. Wherever a quirk had another reason as well, its entry
says so. The module docstrings hold the full detail; this page collects it
in one place.

**Pinned** names the unit test that fails if the behaviour changes. **Not
pinned** means there is no such test. `opponent_trajectory` was also
checked against upstream's unmodified file under stub ROS: 3,000 random
scenarios on each of two seeds gave identical output. That harness is not
in the repository, so it is listed separately from the unit tests.

Section 1 covers deliberate deviations from upstream, which anyone comparing
our results with published ones needs first. Sections 2 to 4 are the
preserved quirks, ordered by how much they can change what the comparison
is able to claim.

## 1. Deviations from upstream (behaviour differs)

**opponent_trajectory: no crash on predictions received only during OVERTAKE.**
Upstream raises AttributeError and the node dies once lapcount ≥ 1 and
`/opponent_trajectory` has been received more than once, if every one of
those messages arrived while the state machine said OVERTAKE (so none was
stored). The port skips the on-trajectory check until a trajectory is
stored, and tracking continues. This hit 69 of 3,000 random differential
scenarios. A run that upstream would have ended keeps feeding the GP here.
Pinned: `test_overtake_predictions_are_counted_but_not_stored`.

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

**collision_prediction: in-process Frenet conversion.**
`convert_frenet2globarr_service` became `FrenetConverter.to_cartesian`. The
service was only used for marker positions. Pinned:
`test_markers_are_placed_by_to_cartesian`.

**update_waypoints: clock.** `time.time()` became the node clock, so the
GB_TRACK hysteresis follows sim time when `use_sim_time` is set. Not pinned.

**opponent_trajectory and collision_prediction: marker converter.** Upstream
built its converter from `/global_waypoints` x and y. The ports use
`FrenetConverter.from_track`, so the nodes need `track_name`. This affects
markers only. Not pinned.

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
pinned. The differential test covers it: random `is_visible`, with identical
output.

**An opponent is only found if it is last in the obstacle array.** The pick is
the dynamic obstacle whose `s_start` is nearest ahead of the ego (modulo
the lap) and within 6 m, recorded at its `s_center`. But the loop resets its
pick to None on every obstacle that is not a nearer dynamic one, so with a
cone or a second car listed after the opponent, nothing is tracked. Nothing
is learned while the opponent is behind or more than 6 m ahead. Pinned:
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

## 3. What the planners do with it: collision_prediction and update_waypoints

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
`test_first_dynamic_obstacle_is_the_opponent`.

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
empty array and marker deletes go out every cycle. Pinned:
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

## 4. Cosmetic

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
