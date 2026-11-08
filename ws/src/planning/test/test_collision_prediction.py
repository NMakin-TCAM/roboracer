"""Collision predictor, checked against upstream's arithmetic on straight-line cases.

The expected numbers are worked by hand from upstream's loop, not read back from
the port: with dt = 0.02 a closing speed c moves the gap by 0.02 * c per step, so
the step on which a threshold is crossed is a ceiling, and the ego s there follows.
"""
import numpy as np
import pytest

from planning.collision_prediction import (WALL_D, CollisionParams, CollisionPredictor,
                                           Region, acceleration, first_dynamic,
                                           predict_region)
from planning.frenet import FrenetConverter

LAP = 100      # waypoints at s = 0..100, the last a duplicate of the first
EGO_V = 6.0    # updated waypoint speed everywhere
OPP_V = 4.0    # opponent trajectory speed everywhere
OPP_D = 0.3


class FakeWpnt:
    """Stands in for f1t_msgs/Wpnt, which needs a built workspace to import."""

    def __init__(self, s_m, vx_mps=EGO_V):
        self.s_m = s_m
        self.vx_mps = vx_mps


class FakeOppWpnt:
    """Stands in for f1t_msgs/OppWpnt."""

    def __init__(self, s_m, proj_vs_mps=OPP_V, d_m=OPP_D):
        self.s_m = s_m
        self.proj_vs_mps = proj_vs_mps
        self.d_m = d_m


class FakeObs:
    """Stands in for f1t_msgs/Obstacle."""

    def __init__(self, s_center, d_center=OPP_D, is_static=False):
        self.s_center = s_center
        self.d_center = d_center
        self.is_static = is_static


def ego_line(v=EGO_V):
    return [FakeWpnt(float(i), v) for i in range(LAP + 1)]


def opp_line(v=OPP_V, d=OPP_D):
    return [FakeOppWpnt(float(i), v, d) for i in range(LAP + 1)]


def predictor(opp=None, ego=None, lap_count=1.0, **params):
    c = CollisionPredictor(CollisionParams(**params))
    c.set_opponent_trajectory(opp or opp_line(), lap_count)
    c.set_updated_waypoints(ego or ego_line())
    return c


def region_for(ego_s, ego_v, opp_s, opp=None, ego=None, **params):
    c = predictor(opp, ego, **params)
    return predict_region(ego_s, ego_v, opp_s, c.opp_wpnts, c.ego_wpnts,
                          c.max_s_opp, c.max_s_ego, c.params)


# ------------------------------------------------------------------ inputs

def test_first_dynamic_obstacle_is_the_opponent():
    a, b = FakeObs(5.0), FakeObs(9.0)
    assert first_dynamic([FakeObs(1.0, is_static=True), a, b]) is a
    assert first_dynamic([FakeObs(1.0, is_static=True)]) is None
    assert first_dynamic([]) is None


def test_ego_lap_is_one_waypoint_short():
    """Upstream drops the duplicate last waypoint, then reads the lap off what is left."""
    c = predictor()
    assert len(c.ego_wpnts) == LAP
    assert c.max_s_ego == LAP - 1


def test_opponent_lap_keeps_its_last_waypoint():
    c = predictor()
    assert len(c.opp_wpnts) == LAP + 1
    assert c.max_s_opp == LAP


def test_empty_arrays_are_ignored_rather_than_crashing():
    c = CollisionPredictor()
    assert not c.set_opponent_trajectory([], 1.0)
    assert not c.set_updated_waypoints([FakeWpnt(0.0)])
    assert c.max_s_opp is None and c.max_s_ego is None


def test_acceleration_interpolates_between_limits():
    p = CollisionParams()
    assert acceleration(p.max_v, p) == pytest.approx(p.min_a)
    assert acceleration(p.min_v, p) == pytest.approx(p.max_a)
    assert acceleration(5.0, p) == pytest.approx(5.25)


# ------------------------------------------------------------------ region

def test_region_starts_and_ends_where_the_gap_crosses_the_margins():
    """Gap 5.01 closing at 2 m/s, 0.04 m per step.

    Starts when gap < 0.6: step 111, ego s = 6 * 0.02 * 111 = 13.32.
    Ends when gap < -0.4: step 136, ego s = 16.32.
    """
    r = region_for(0.0, EGO_V, 5.01)
    assert r.ended
    assert r.s_start == pytest.approx(13.32)
    assert r.s_end == pytest.approx(16.32)
    assert r.s_center == pytest.approx((13.32 + 16.32) / 2)
    assert (r.d_right, r.d_left) == pytest.approx((OPP_D - 0.25, OPP_D + 0.25))
    assert r.d_center == pytest.approx(OPP_D)


def test_d_bounds_take_the_lower_and_upper_opponent_d():
    """Opponent d steps from 0.3 to -0.2 at s = 15, between its s at start (~13.9) and end (~16.7)."""
    opp = [FakeOppWpnt(float(i), OPP_V, OPP_D if i < 15 else -0.2) for i in range(LAP + 1)]
    r = region_for(0.0, EGO_V, 5.01, opp=opp)
    assert r.begin_d == pytest.approx(OPP_D) and r.end_d == pytest.approx(-0.2)
    assert (r.d_right, r.d_left) == pytest.approx((-0.45, 0.55))


def test_region_without_an_end_is_a_wall_ending_at_the_opponent():
    """Closing at 0.2 m/s from 1.01: starts at step 103, never gets 0.4 m past.

    The end is the opponent's s after all 200 steps, 1.01 + 5.8 * 4 = 24.21.
    """
    r = region_for(0.0, EGO_V, 1.01, opp=opp_line(v=5.8))
    assert not r.ended
    assert r.s_start == pytest.approx(6 * 0.02 * 103)
    assert r.s_end == pytest.approx(24.21)
    assert (r.d_right, r.d_left) == (-WALL_D, WALL_D)


def test_no_region_when_the_opponent_pulls_away():
    assert region_for(0.0, EGO_V, 5.0, opp=opp_line(v=8.0)) is None


def test_no_region_when_the_horizon_is_too_short():
    assert region_for(0.0, EGO_V, 5.01, time_steps=100) is None


def test_ego_below_waypoint_speed_accelerates():
    """At 2 m/s against 6 m/s waypoints the ego speeds up and catches the opponent.

    Without the acceleration branch it would move at the 2 m/s waypoint speed and
    never catch a 4 m/s opponent, which the second case shows.
    """
    assert region_for(0.0, 2.0, 5.01) is not None
    assert region_for(0.0, 2.0, 5.01, ego=ego_line(v=2.0)) is None


def test_ego_above_waypoint_speed_drops_to_it():
    """Upstream moves the ego at the waypoint speed, not its own, when it is faster."""
    assert region_for(0.0, 20.0, 5.01) == region_for(0.0, EGO_V, 5.01)


def test_region_s_is_not_wrapped():
    """Upstream leaves the rolled-forward s as is, so it can run past the lap.

    Ego at 95, opponent at 1.02 is treated as 100.02 (a lap ahead). The region
    then starts at about 95 + 13.32, past the ego lap length of 99.
    """
    c = predictor()
    c.set_ego(95.0, EGO_V)
    c.set_obstacles([FakeObs(1.02)])
    r = c.step().region
    assert r.s_start == pytest.approx(95.0 + 6 * 0.02 * 111)
    assert r.s_start > c.max_s_ego and r.s_center > c.max_s_ego


# ------------------------------------------------------------------ force trailing

def test_trailing_forced_before_one_opponent_lap():
    c = predictor(lap_count=0.0)
    assert c.should_force_trailing(10.0, OPP_D)


def test_trailing_not_forced_on_the_predicted_line():
    c = predictor()
    assert not c.should_force_trailing(10.0, OPP_D + 0.2)


def test_trailing_forced_off_the_predicted_line():
    c = predictor()
    assert c.should_force_trailing(10.0, OPP_D + 0.3)


def test_trailing_check_uses_the_unwrapped_opponent_s():
    """After the lap is added the nearest opponent waypoint is the last one.

    Only that last waypoint is off the line here, so forcing trailing proves the
    check looked there and not at s = 10 where the opponent actually is.
    """
    opp = opp_line()
    opp[-1] = FakeOppWpnt(float(LAP), OPP_V, OPP_D + 1.0)
    c = predictor(opp=opp)
    c.set_ego(80.0, EGO_V)
    c.set_obstacles([FakeObs(10.0)])
    assert c.step().force_trailing is True


# ------------------------------------------------------------------ cycles

def test_no_opponent_publishes_nothing_but_counts_down():
    c = predictor(max_expire_counter=2)
    first = c.step()
    assert first.force_trailing is None and first.region is None and not first.delete
    assert c.step().delete


def test_region_expires_after_max_expire_counter_cycles():
    """Found: counter 0 then 1. Two empty cycles take it to 3, and that deletes."""
    c = predictor(max_expire_counter=3)
    c.set_ego(0.0, EGO_V)
    c.set_obstacles([FakeObs(5.01)])
    hit = c.step()
    assert hit.region is not None and not hit.delete
    c.set_obstacles([])
    assert not c.step().delete
    assert c.step().delete
    assert c.step().delete       # and keeps deleting every cycle
    assert c.expire_counter == 3


def test_region_published_and_deleted_in_one_cycle_when_counter_is_one():
    c = predictor(max_expire_counter=1)
    c.set_ego(0.0, EGO_V)
    c.set_obstacles([FakeObs(5.01)])
    s = c.step()
    assert s.region is not None and s.delete


def test_ego_defaults_to_start_line_at_rest():
    """Upstream does not wait for odometry; until it arrives the ego is at s = 0, v = 0."""
    c = CollisionPredictor()
    assert (c.ego_s, c.ego_v) == (0.0, 0.0)


# ------------------------------------------------------------------ markers

R, N = 10.0, 600


@pytest.fixture(scope='module')
def circle(tmp_path_factory):
    d = tmp_path_factory.mktemp('circle')
    th = np.linspace(0, 2 * np.pi, N, endpoint=False)
    xy = np.column_stack((R * np.cos(th), R * np.sin(th)))
    np.savetxt(d / 'centerline.csv',
               np.column_stack((xy, np.full(N, 1.5), np.full(N, 1.5))),
               delimiter=',', header='x_m,y_m,w_tr_right_m,w_tr_left_m')
    np.savetxt(d / 'raceline.csv',
               np.column_stack((np.arange(N) * (2 * np.pi * R / N), xy[:, 0], xy[:, 1],
                                th + np.pi / 2, np.full(N, 1 / R), np.full(N, 5.0),
                                np.zeros(N))),
               delimiter=';', header='s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2')
    return FrenetConverter.from_track(str(d))


def circle_predictor(conv):
    step = conv.length / N
    c = CollisionPredictor()
    c.set_updated_waypoints([FakeWpnt(i * step) for i in range(N + 1)])
    return c


def region(s_start, s_end, begin_d, end_d):
    return Region(s_start=s_start, s_end=s_end, s_center=(s_start + s_end) / 2,
                  d_right=-1.0, d_left=1.0, d_center=0.0,
                  begin_d=begin_d, end_d=end_d, ended=True)


def test_markers_are_placed_by_to_cartesian(circle):
    """Counter-clockwise circle: d > 0 is towards the centre."""
    c = circle_predictor(circle)
    (bx, by), (ex, ey) = c.marker_positions(region(1.0, 3.0, 0.3, -0.2), circle)
    assert np.hypot(bx, by) == pytest.approx(R - 0.3, abs=1e-3)
    assert np.hypot(ex, ey) == pytest.approx(R + 0.2, abs=1e-3)
    assert (bx, by) == pytest.approx(circle.to_cartesian(1.0, 0.3))


def test_markers_wrap_by_the_short_ego_lap(circle):
    """s just past max_s_ego wraps to near 0, one waypoint spacing early."""
    c = circle_predictor(circle)
    s = c.max_s_ego + 0.01
    (bx, by), _ = c.marker_positions(region(s, s + 1.0, 0.0, 0.0), circle)
    assert (bx, by) == pytest.approx(circle.to_cartesian(0.01, 0.0))
    assert (bx, by) != pytest.approx(circle.to_cartesian(s, 0.0))
