"""Opponent trajectory accumulator, checked against upstream's arithmetic by hand.

The track is 20 m with waypoints every 0.1 m, so half a lap is 10 m. The
opponent moves 0.3 m every 0.04 s (7.5 m/s), placed mid-interval so the bin it
falls in is never ambiguous. Each accepted step adds 0.3 m, so a half lap
completes on the 34th step after the first detection (10.2 > 10; 33 gives 9.9).
"""
import math

import numpy as np
import pytest

from planning.opponent_trajectory import (OpponentTrajectoryTracker, check_on_trajectory,
                                          find_nearest_dyn_obstacle, get_index)

L = 20.0
N = 200            # waypoints at s = 0, 0.1, ... 20.0, the last a duplicate of the first
STEP = 0.3
DT = 0.04
V = STEP / DT      # 7.5 m/s
WIDTH = 1.0
HALF_LAP_STEPS = 34
OFF_D = 0.4         # 0.3 or more off the prediction; vd 10 m/s keeps it under 15 m/s


class FakeWpnt:
    """Stands in for f1t_msgs/Wpnt, which needs a built workspace to import."""

    def __init__(self, s_m):
        self.s_m = s_m
        self.d_left = WIDTH
        self.d_right = WIDTH


class FakeObs:
    """Stands in for f1t_msgs/Obstacle."""

    def __init__(self, s_center, d_center=0.0, vs=V, is_static=False, s_start=None):
        self.s_center = s_center
        self.s_start = s_center - 0.25 if s_start is None else s_start
        self.d_center = d_center
        self.vs = vs
        self.vd = 0.0
        self.is_static = is_static
        self.is_visible = True
        self.s_var = self.d_var = self.vs_var = self.vd_var = 0.0


class FakeOppWpnt:
    """Stands in for f1t_msgs/OppWpnt."""

    def __init__(self, s_m, d_m=0.0):
        self.s_m = s_m
        self.d_m = d_m


def prediction(d=0.0):
    return [FakeOppWpnt(i * 0.5, d) for i in range(41)]


class Drive:
    """Moves the opponent, keeps the ego 1 m behind it, and runs one cycle per call."""

    def __init__(self, s=0.05, t=100.0):
        self.tr = OpponentTrajectoryTracker()
        self.tr.set_global_waypoints([FakeWpnt(i * L / N) for i in range(N + 1)])
        self.tr.set_obstacles([], t)
        self.s, self.t = s - STEP, t
        self.published = []

    def see(self, ds=STEP, d=0.0, dt=DT, stamp_dt=None):
        self.s += ds
        self.t += dt
        s = self.s % L
        self.tr.set_obstacles([FakeObs(s, d)], self.t - dt + (dt if stamp_dt is None else stamp_dt))
        self.tr.set_ego((self.s - 1.0) % L, 0.0, V, 0.0)
        return self._step()

    def blind(self, dt=DT):
        """A cycle with an empty obstacle array."""
        self.t += dt
        self.tr.set_obstacles([], self.t)
        return self._step()

    def _step(self):
        out = self.tr.step(self.t)
        self.published += out.trajectories
        return out

    def run(self, n, **kw):
        return [self.see(**kw) for _ in range(n)]

    def to_lap_one(self):
        """Two clean half laps: lapcount 0.5 then 1.0."""
        self.run(1 + 2 * HALF_LAP_STEPS)
        assert [p.lapcount for p in self.published] == [0.5, 1.0]
        self.published.clear()
        return self


# ------------------------------------------------------------------ helpers

def test_index_is_the_first_waypoint_ahead_or_zero_past_the_end():
    s = [0.0, 1.0, 2.0]
    assert get_index(s, 0.5) == 1
    assert get_index(s, 1.0) == 2
    assert get_index(s, 2.5) == 0


def test_opponent_is_the_nearest_dynamic_obstacle_ahead_within_6_m():
    ego = np.array([10.0, 0.0, 0.0, 0.0])
    assert find_nearest_dyn_obstacle([FakeObs(13.0)], 1.0, ego, L)[0] == 13.0
    assert find_nearest_dyn_obstacle([FakeObs(17.0)], 1.0, ego, L) is None    # 6.75 ahead
    assert find_nearest_dyn_obstacle([FakeObs(9.0)], 1.0, ego, L) is None     # behind
    assert find_nearest_dyn_obstacle([FakeObs(13.0, is_static=True)], 1.0, ego, L) is None
    assert find_nearest_dyn_obstacle([FakeObs(13.0)], 1.0, np.array([]), L) is None


def test_opponent_is_lost_unless_it_is_last_in_the_array():
    """Upstream resets its pick on every obstacle that is not a nearer dynamic one."""
    ego = np.array([10.0, 0.0, 0.0, 0.0])
    opp, cone = FakeObs(13.0), FakeObs(12.0, is_static=True)
    assert find_nearest_dyn_obstacle([cone, opp], 1.0, ego, L)[0] == 13.0
    assert find_nearest_dyn_obstacle([opp, cone], 1.0, ego, L) is None
    assert find_nearest_dyn_obstacle([opp, FakeObs(15.0)], 1.0, ego, L) is None


def test_opponent_is_recorded_at_s_center_and_ranged_by_s_start():
    ego = np.array([10.0, 0.0, 0.0, 0.0])
    row = find_nearest_dyn_obstacle([FakeObs(17.0, s_start=15.5)], 42.0, ego, L)
    assert row[0] == 17.0 and row[6] == 42.0


def test_on_trajectory_check_uses_the_first_waypoint_ahead_in_list_order():
    """Not the nearest: the list is scanned in order for the first s ahead."""
    class P:
        s, d = 5.0, 0.0
    wps = [FakeOppWpnt(9.0, 1.0), FakeOppWpnt(5.5, 0.0)]
    assert check_on_trajectory(P, 0, wps) == 1
    assert check_on_trajectory(P, 3, [FakeOppWpnt(5.5, 0.29)]) == 0
    assert check_on_trajectory(P, 3, [FakeOppWpnt(5.5, 0.3)]) == 4        # 0.3 is off
    assert check_on_trajectory(P, 3, [FakeOppWpnt(4.0, 2.0)]) == 3        # none ahead


# ------------------------------------------------------------------ accumulation

def test_first_detection_is_counted_but_not_kept():
    dr = Drive()
    dr.run(2)
    assert len(dr.tr.detections) == 1
    assert dr.tr.points_in_lap == 2
    assert dr.tr.detections[0].vs == pytest.approx(V)
    assert dr.tr.detections[0].vd == pytest.approx(0.0)


def test_nothing_happens_before_odometry():
    dr = Drive()
    dr.tr.set_obstacles([FakeObs(5.0)], 100.0)
    assert dr.tr.step(100.0).trajectories == [] and dr.tr.index is None


def test_one_detection_per_waypoint_interval():
    """A second detection in the same 0.1 m interval is not recorded."""
    dr = Drive()
    dr.run(2)
    dr.see(ds=0.02)
    assert dr.tr.points_in_lap == 2


def test_backwards_detection_is_dropped():
    dr = Drive()
    dr.run(3)
    dr.see(ds=-0.5)
    assert dr.tr.points_in_lap == 3


def test_detections_under_10_ms_apart_are_dropped():
    dr = Drive()
    dr.run(2)
    dr.see(stamp_dt=0.005)
    assert dr.tr.points_in_lap == 2


def test_off_track_and_too_fast_detections_are_dropped():
    dr = Drive()
    dr.run(2)
    dr.see(d=WIDTH + 0.1)
    assert dr.tr.points_in_lap == 2
    dr = Drive()
    dr.run(2)
    dr.see(ds=0.7)              # 17.5 m/s
    assert dr.tr.points_in_lap == 2


def test_half_lap_publishes_and_shares_its_last_point():
    dr = Drive()
    dr.run(HALF_LAP_STEPS)
    assert dr.published == []
    dr.see()
    (msg,) = dr.published
    assert msg.lapcount == 0.5 and msg.opp_is_on_trajectory
    assert msg.nrofpoints == HALF_LAP_STEPS + 1
    assert len(msg.detections) == HALF_LAP_STEPS
    assert dr.tr.detections == [msg.detections[-1]]
    assert dr.tr.points_in_lap == 1


def test_wrap_copies_vs_from_the_previous_detection():
    """Crossing s = 20 is a 19.7 m step backwards, more than half a lap, so a wrap."""
    dr = Drive(s=19.25)
    dr.run(4)                   # 19.25, 19.55, 19.85, then 20.15 -> 0.15
    last = dr.tr.detections[-1]
    assert last.s == pytest.approx(0.15)
    assert last.vs == dr.tr.detections[-2].vs == pytest.approx(V)
    assert dr.tr.traversed == pytest.approx(3 * STEP)


def test_lapcount_counts_opponent_travel_from_its_first_detection():
    dr = Drive(s=7.05)
    dr.to_lap_one()
    assert dr.tr.lap_count == 1.0


# ------------------------------------------------------------------ first lap gaps

def test_gaps_before_lap_one_are_bridged_with_the_average_speed():
    dr = Drive()
    dr.run(3)
    dr.blind(dt=2.0)
    dr.see(ds=3.0, dt=DT)
    p = dr.tr.detections[-1]
    assert p.vs == pytest.approx(3.0 / (2.0 + DT))
    assert dr.tr.traversed == pytest.approx(2 * STEP + 3.0)


def test_first_half_lap_with_a_gap_is_trimmed_and_not_published():
    """The 3 m gap counts towards the half lap; at completion the points before it go."""
    dr = Drive()
    dr.run(4)                                   # 3 kept, s up to 0.95
    dr.blind(dt=1.0)
    dr.see(ds=3.0)
    steps = math.ceil((10.0 - 3 * STEP - 3.0) / STEP)
    dr.run(steps)
    assert dr.published == []
    assert dr.tr.lap_count == 0
    assert dr.tr.detections[0].s == pytest.approx(0.95 + 3.0)
    assert dr.tr.traversed == pytest.approx(steps * STEP)
    assert dr.tr.points_in_lap == 1 + 4 + steps - 3
    assert dr.tr.initial_lap


def test_later_waypoints_change_only_the_gap_test_track_length():
    """Bins, bounds and the half-lap test keep the first message's 20 m lap.

    The gap test uses the latest, here 2 m, so the 3 m gap is 3 % 2 = 1 m and
    the same drive as above publishes instead of trimming.
    """
    dr = Drive()
    dr.tr.set_global_waypoints([FakeWpnt(i * 2.0 / N) for i in range(N + 1)])
    assert dr.tr.track_length == L and dr.tr.live_track_length == 2.0
    dr.run(4)
    dr.blind(dt=1.0)
    dr.see(ds=3.0)
    dr.run(math.ceil((10.0 - 3 * STEP - 3.0) / STEP))
    (msg,) = dr.published
    assert msg.lapcount == 0.5 and len(msg.detections) == 3 + 1 + 21


# ------------------------------------------------------------------ staleness

def test_nothing_is_stale_before_lap_one():
    dr = Drive()
    dr.run(10)
    dr.blind(dt=5.0)
    assert dr.tr.points_in_lap == 10 and dr.published == []


def test_stale_half_lap_is_flushed_with_the_last_step_added_to_lapcount():
    """lapcount grows by the last 0.3 m step over 20 m, not the 1.8 m covered."""
    dr = Drive().to_lap_one()
    dr.run(6)
    dr.blind(dt=0.3)
    assert dr.published == []
    dr.blind(dt=0.3)
    (msg,) = dr.published
    assert msg.opp_is_on_trajectory and msg.nrofpoints == 7
    assert msg.lapcount == pytest.approx(1.0 + STEP / L)
    assert dr.tr.index is None and dr.tr.detections == [] and dr.tr.points_in_lap == 1


def test_stale_half_lap_with_few_points_is_dropped():
    dr = Drive().to_lap_one()
    dr.run(4)
    dr.blind(dt=0.6)
    assert dr.published == [] and dr.tr.detections == []
    assert dr.tr.lap_count == 1.0


def test_opponent_in_view_but_in_one_interval_goes_stale():
    """Fresh stamps do not help: the time is the last detection that changed waypoint."""
    dr = Drive().to_lap_one()
    dr.run(6)
    for _ in range(15):
        dr.see(ds=0.0)
    (msg,) = dr.published
    assert msg.nrofpoints == 7
    # Flushed on the 13th (0.52 s) and re-acquired at once as a new first point.
    assert dr.tr.detections == [] and dr.tr.points_in_lap == 2


# ------------------------------------------------------------------ jumps after lap one

def test_jump_over_1m_after_lap_one_restarts_with_an_unchecked_first_point():
    dr = Drive().to_lap_one()
    dr.run(5)
    n, oldest, travelled = len(dr.tr.detections), dr.tr.detections[0], dr.tr.traversed
    dr.see(ds=1.5)
    assert len(dr.tr.detections) == n           # dropped
    dr.see(d=WIDTH + 2.0)                       # off the track, still kept
    assert len(dr.tr.detections) == n           # appended, oldest popped
    assert dr.tr.detections[0] is not oldest
    assert dr.tr.detections[-1].vs == 0 and dr.tr.detections[-1].d == WIDTH + 2.0
    assert dr.tr.traversed == travelled


def test_jump_over_1m_before_lap_one_is_kept():
    dr = Drive()
    dr.run(3)
    dr.see(ds=1.5, dt=0.2)
    assert dr.tr.detections[-1].vs == pytest.approx(1.5 / 0.2)


# ------------------------------------------------------------------ off trajectory

def off_trajectory_drive():
    dr = Drive().to_lap_one()
    dr.tr.set_opponent_trajectory(prediction())
    dr.tr.set_opponent_trajectory(prediction())
    dr.run(5, d=OFF_D)
    assert dr.published == []
    dr.see(d=OFF_D)
    return dr


def test_six_points_off_the_prediction_publish_and_wait():
    dr = off_trajectory_drive()
    (msg,) = dr.published
    assert not msg.opp_is_on_trajectory
    assert msg.lapcount == 1.0 and msg.nrofpoints == 7
    assert dr.tr.waiting is not None


def test_points_are_still_added_while_waiting_and_kept_after():
    dr = off_trajectory_drive()
    sent = dr.published[0].detections
    dr.run(3, d=OFF_D)
    dr.tr.set_opponent_trajectory(prediction())
    dr.see(d=OFF_D)
    assert dr.tr.waiting is None and dr.tr.off_count == 0
    assert dr.tr.detections[:len(sent)] == sent
    assert dr.tr.points_in_lap == 7 + 4


def test_wait_ends_after_3_s_without_a_new_prediction():
    dr = off_trajectory_drive()
    dr.blind(dt=2.9)
    assert dr.tr.waiting is not None
    dr.blind(dt=0.2)
    assert dr.tr.waiting is None


def test_half_lap_is_not_checked_while_waiting():
    """Completed during the wait, it is only published when the wait ends."""
    dr = off_trajectory_drive()
    dr.run(HALF_LAP_STEPS, d=OFF_D)
    assert len(dr.published) == 1
    dr.tr.set_opponent_trajectory(prediction())
    dr.see(d=OFF_D)
    assert [m.lapcount for m in dr.published] == [1.0, 1.5]
    assert dr.published[1].opp_is_on_trajectory


def test_wait_ending_without_an_opponent_poisons_the_half_lap_with_nan():
    """The completed half lap is published, then the raw row the next step uses is NaN."""
    dr = off_trajectory_drive()
    dr.run(HALF_LAP_STEPS, d=OFF_D)
    dr.tr.set_opponent_trajectory(prediction())
    dr.blind()
    assert [m.lapcount for m in dr.published] == [1.0, 1.5]
    assert np.isnan(dr.tr.raw[dr.tr.index]).all()
    dr.see(d=OFF_D)
    assert math.isnan(dr.tr.detections[-1].vs) and math.isnan(dr.tr.traversed)
    dr.run(3 * HALF_LAP_STEPS)
    assert len(dr.published) == 2


def test_overtake_predictions_are_counted_but_not_stored():
    """Upstream crashed here with AttributeError; the check is skipped instead."""
    dr = Drive().to_lap_one()
    dr.tr.set_state('OVERTAKE')
    dr.tr.set_opponent_trajectory(prediction())
    dr.tr.set_opponent_trajectory(prediction())
    assert dr.tr.opp_traj_count == 2 and dr.tr.opponent_trajectory is None
    dr.run(8, d=OFF_D)
    assert dr.tr.off_count == 0 and dr.published == []


# ------------------------------------------------------------------ markers

def test_marker_array_settles_to_deleteall_between_single_markers():
    dr = Drive()
    out = dr.run(12)
    last = out[-1].markers[-1]
    assert [m.delete_all for m in last] == [True, False, True, False, True, False]
    assert last[-1].id == 11
    assert last[-1].s == pytest.approx(dr.tr.detections[-1].s)


def test_marker_snapshots_do_not_change_afterwards():
    dr = Drive()
    out = dr.run(8)
    snap = out[5].markers[-1]
    assert not any(m.delete_all for m in snap)
