"""Reactive overtaking planner, checked on a circular track with known geometry."""
import numpy as np
import pytest

from planning.frenet import FrenetConverter
from planning.spliner import (LEFT, RIGHT, SplinerParams, build_evasion, choose_side,
                              obstacles_ahead, plan)

R, W_LEFT, W_RIGHT = 10.0, 1.5, 1.5


class FakeObs:
    """Stands in for f1t_msgs/Obstacle, which needs a built workspace to import."""

    def __init__(self, s_center, d_center=0.0, length=0.5, width=0.3):
        self.s_center = s_center
        self.s_start = s_center - length / 2
        self.s_end = s_center + length / 2
        self.d_center = d_center
        self.d_left = d_center + width / 2
        self.d_right = d_center - width / 2
        self.is_static = True
        self.is_visible = True


@pytest.fixture(scope='module')
def track(tmp_path_factory):
    d = tmp_path_factory.mktemp('circle')
    n = 600
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xy = np.column_stack((R * np.cos(th), R * np.sin(th)))
    np.savetxt(d / 'centerline.csv',
               np.column_stack((xy, np.full(n, W_RIGHT), np.full(n, W_LEFT))),
               delimiter=',', header='x_m,y_m,w_tr_right_m,w_tr_left_m')
    np.savetxt(d / 'raceline.csv',
               np.column_stack((np.arange(n) * (2 * np.pi * R / n), xy[:, 0], xy[:, 1],
                                th + np.pi / 2, np.full(n, 1 / R), np.full(n, 5.0),
                                np.zeros(n))),
               delimiter=';', header='s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2')
    return FrenetConverter.from_track(str(d))


@pytest.fixture
def params():
    return SplinerParams()


# ------------------------------------------------------------------ filtering

def test_only_obstacles_ahead_and_close_are_considered(track, params):
    near = FakeObs(5.0)
    far = FakeObs(40.0)
    behind = FakeObs(track.length - 5.0)
    got = obstacles_ahead([far, behind, near], 0.0, track, params)
    assert [o.s_center for o in got] == [5.0]


def test_filtering_uses_lap_distance_not_subtraction(track, params):
    """An obstacle 3 m past the line is ahead of a car 2 m before it."""
    obs = FakeObs(3.0)
    assert obstacles_ahead([obs], track.length - 2.0, track, params) == [obs]


def test_obstacles_off_the_line_are_ignored(track, params):
    """Something parked against the wall is not in our way."""
    assert obstacles_ahead([FakeObs(5.0, d_center=2.0)], 0.0, track, params) == []


def test_nearest_obstacle_comes_first(track, params):
    a, b, c = FakeObs(8.0), FakeObs(2.0), FakeObs(5.0)
    assert [o.s_center for o in obstacles_ahead([a, b, c], 0.0, track, params)] == [2.0, 5.0, 8.0]


# ------------------------------------------------------------------ side choice

def test_obstacle_on_the_right_is_passed_on_the_left(track, params):
    """The apex clears the obstacle's left edge. It is not necessarily left of the
    reference line: an obstacle far enough right is passed without crossing over."""
    obs = FakeObs(5.0, d_center=-0.8)
    choice = choose_side(obs, track, params)
    assert choice.side == LEFT
    assert choice.d_apex - obs.d_left == pytest.approx(params.car_width / 2 + params.evasion_dist)


def test_obstacle_on_the_left_is_passed_on_the_right(track, params):
    obs = FakeObs(5.0, d_center=0.8)
    choice = choose_side(obs, track, params)
    assert choice.side == RIGHT
    assert obs.d_right - choice.d_apex == pytest.approx(params.car_width / 2 + params.evasion_dist)


def test_a_wall_to_wall_obstacle_has_no_feasible_side(track, params):
    """Refusing to plan is the correct answer, not squeezing through."""
    blocker = FakeObs(5.0, d_center=0.0, width=2.8)
    choice = choose_side(blocker, track, params)
    assert choice.side is None
    assert 'no feasible' in choice.reason


def test_apex_stays_inside_the_track(track, params):
    """An obstacle hard against one edge must not push the apex through the other."""
    choice = choose_side(FakeObs(5.0, d_center=-1.0), track, params)
    assert choice.d_apex <= W_LEFT - params.bound_margin + 1e-9


def test_hysteresis_keeps_the_committed_side(track, params):
    """Both sides clear and nearly equal: stay where we are rather than flip."""
    obs = FakeObs(5.0, d_center=0.02)       # a hair left, so right is marginally roomier
    assert choose_side(obs, track, params, current_side=LEFT).side == LEFT
    assert choose_side(obs, track, params, current_side=None).side == RIGHT


def test_hysteresis_still_yields_when_the_other_side_is_much_better(track, params):
    obs = FakeObs(5.0, d_center=1.0)        # firmly left, right is far roomier
    assert choose_side(obs, track, params, current_side=LEFT).side == RIGHT


# ------------------------------------------------------------------ the spline

def test_evasion_starts_at_the_car_and_ends_on_the_raceline(track, params):
    obs = FakeObs(6.0, d_center=-0.5)
    s, d, v = build_evasion(0.0, 0.0, obs, choose_side(obs, track, params), track, params)
    assert d[0] == pytest.approx(0.0, abs=1e-6)
    assert d[-1] == pytest.approx(0.0, abs=1e-6)


def test_evasion_clears_the_obstacle_at_its_closest_point(track, params):
    """The whole job: at the obstacle, be far enough off it to not hit it."""
    obs = FakeObs(6.0, d_center=-0.5)
    choice = choose_side(obs, track, params)
    s, d, v = build_evasion(0.0, 0.0, obs, choice, track, params)
    at_obs = [d[i] for i in range(len(s)) if obs.s_start <= s[i] <= obs.s_end]
    assert at_obs, 'the path never reaches the obstacle'
    clearance = min(at_obs) - obs.d_left
    assert clearance >= params.car_width / 2


def test_evasion_stays_within_the_track(track, params):
    obs = FakeObs(6.0, d_center=-1.0)
    s, d, v = build_evasion(0.0, 0.0, obs, choose_side(obs, track, params), track, params)
    assert np.all(d <= W_LEFT - params.bound_margin + 1e-9)
    assert np.all(d >= -W_RIGHT + params.bound_margin - 1e-9)


def test_evasion_is_smooth(track, params):
    """A kink is a path the controller cannot follow. Check curvature of d(u) is bounded."""
    obs = FakeObs(8.0, d_center=-0.5)
    s, d, v = build_evasion(0.0, 0.0, obs, choose_side(obs, track, params), track, params)
    second = np.diff(d, 2) / params.resolution ** 2
    assert np.max(np.abs(second)) < 5.0


def test_evasion_wraps_across_the_start_line(track, params):
    """Planning from just before s=0 around an obstacle just after it."""
    ego = track.length - 3.0
    obs = FakeObs(2.0, d_center=-0.5)
    s, d, v = build_evasion(ego, 0.0, obs, choose_side(obs, track, params), track, params)
    assert np.all(s >= 0) and np.all(s < track.length)
    assert np.all(np.abs(np.diff(d)) < 0.2), 'd jumped, so the wrap was mishandled'


def test_speed_is_scaled_down_through_the_manoeuvre(track, params):
    obs = FakeObs(6.0, d_center=-0.5)
    s, d, v = build_evasion(0.0, 0.0, obs, choose_side(obs, track, params), track, params)
    assert np.allclose(v, 5.0 * params.speed_scale)


def test_an_obstacle_on_top_of_us_is_not_planned_around(track, params):
    """No room for a lead-in means no spline; trailing is the honest answer."""
    assert build_evasion(0.0, 0.0, FakeObs(0.2), choose_side(FakeObs(0.2), track, params),
                         track, params) is None


# ------------------------------------------------------------------ end to end

def test_clear_track_produces_no_trajectory(track, params):
    s, d, v, side, note = plan(0.0, 0.0, [], track, params)
    assert len(s) == 0 and side is None and note == 'clear'


def test_an_obstacle_produces_a_trajectory(track, params):
    s, d, v, side, note = plan(0.0, 0.0, [FakeObs(6.0, d_center=-0.5)], track, params)
    assert len(s) > 10 and side == LEFT
    assert 'overtaking left' in note


def test_a_blocked_track_trails_instead_of_planning(track, params):
    s, d, v, side, note = plan(0.0, 0.0, [FakeObs(6.0, width=2.8)], track, params)
    assert len(s) == 0 and side is None
    assert note.startswith('trailing')


def test_the_nearest_obstacle_is_the_one_planned_around(track, params):
    """Two in the way: handle the one about to hit us, not the far one."""
    near, far = FakeObs(4.0, d_center=-0.6), FakeObs(9.0, d_center=0.6)
    s, d, v, side, note = plan(0.0, 0.0, [far, near], track, params)
    assert side == LEFT, 'planned around the far obstacle instead of the near one'
