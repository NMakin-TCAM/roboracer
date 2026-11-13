"""Reactive overtaking planner, checked on a circular track with known geometry."""
import numpy as np
import pytest

from planning.frenet import FrenetConverter
from planning.spliner import (CLEAR, EVADING, LEFT, NO_ROOM, RIGHT, STATUSES, TOO_CLOSE,
                              SplinerParams, build_evasion, choose_side, obstacles_ahead, plan)

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
    # psi_rad in TUM convention (yaw - pi/2), as build_raceline writes it; yaw here is th + pi/2
    np.savetxt(d / 'raceline.csv',
               np.column_stack((np.arange(n) * (2 * np.pi * R / n), xy[:, 0], xy[:, 1],
                                th, np.full(n, 1 / R), np.full(n, 5.0),
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

def test_clear_track_is_clear_with_no_trajectory(track, params):
    r = plan(0.0, 0.0, [], track, params)
    assert r.status == CLEAR and r.collision_free
    assert len(r.s) == 0 and r.side is None and r.obstacle is None and r.note == 'clear'


def test_an_obstacle_produces_an_evasion(track, params):
    obs = FakeObs(6.0, d_center=-0.5)
    r = plan(0.0, 0.0, [obs], track, params)
    assert r.status == EVADING and r.collision_free
    assert len(r.s) > 10 and len(r.s) == len(r.d) == len(r.v)
    assert r.side == LEFT and r.obstacle is obs
    assert 'overtaking left' in r.note


def test_a_blocked_track_is_no_room_and_not_collision_free(track, params):
    """Same empty trajectory as CLEAR, opposite meaning: this one must stop the car."""
    blocker = FakeObs(6.0, width=2.8)
    r = plan(0.0, 0.0, [blocker], track, params)
    assert r.status == NO_ROOM and not r.collision_free
    assert len(r.s) == 0 and r.side is None and r.obstacle is blocker
    assert r.note.startswith('trailing')


def test_an_obstacle_on_top_of_us_is_too_close_and_not_collision_free(track, params):
    """The lead-in collapses (see build_evasion), so there is nothing safe to publish."""
    obs = FakeObs(0.5, d_center=-0.5)
    r = plan(0.0, 0.0, [obs], track, params)
    assert r.status == TOO_CLOSE and not r.collision_free
    assert len(r.s) == 0 and r.side is None and r.obstacle is obs


def test_the_empty_cases_differ_only_in_status(track, params):
    """Before PlanResult these three were indistinguishable to a caller."""
    results = [plan(0.0, 0.0, obs, track, params)
               for obs in ([], [FakeObs(6.0, width=2.8)], [FakeObs(0.5, d_center=-0.5)])]
    assert all(len(r.s) == 0 for r in results)
    assert [r.status for r in results] == [CLEAR, NO_ROOM, TOO_CLOSE]
    assert [r.collision_free for r in results] == [True, False, False]


@pytest.mark.parametrize('obstacles', [[], [FakeObs(6.0, d_center=-0.5)], [FakeObs(6.0, width=2.8)],
                                       [FakeObs(0.5, d_center=-0.5)], [FakeObs(40.0)]])
def test_status_and_trajectory_always_agree(track, params, obstacles):
    r = plan(0.0, 0.0, obstacles, track, params)
    assert r.status in STATUSES
    assert (len(r.s) > 0) == (r.status == EVADING)
    assert (r.side is not None) == (r.status == EVADING)


def test_the_nearest_obstacle_is_the_one_planned_around(track, params):
    """Two in the way: handle the one about to hit us, not the far one."""
    near, far = FakeObs(4.0, d_center=-0.6), FakeObs(9.0, d_center=0.6)
    r = plan(0.0, 0.0, [far, near], track, params)
    assert r.status == EVADING and r.obstacle is near
    assert r.side == LEFT, 'planned around the far obstacle instead of the near one'


# ------------------------------------------------------------------ problem 1: the shape

from planning.spliner import Spliner, path_clear  # noqa: E402


def _apex_index(s, d, apex):
    return int(np.argmax(np.isclose(d, apex, atol=1e-9)))


@pytest.mark.parametrize('ego_d', [-0.3, 0.0, 0.3])
@pytest.mark.parametrize('obs_d', [-0.5, 0.5])
def test_the_path_only_moves_toward_the_apex(track, params, ego_d, obs_d):
    """The cubic bulged up to 0.3 m the wrong way first. From any start, d must head toward
    the chosen side at once and never back away before the apex."""
    obs = FakeObs(7.0, d_center=obs_d)
    choice = choose_side(obs, track, params)
    s, d, v = build_evasion(0.0, ego_d, obs, choice, track, params)
    i = _apex_index(s, d, choice.d_apex)
    assert i > 0, 'the apex is never reached'
    toward = np.sign(choice.d_apex - ego_d)
    steps = np.diff(d[:i + 1]) * toward
    assert np.all(steps >= -1e-12), f'moved away from the apex by {-steps.min():.4f} m'
    assert steps[1] > 0, 'the first step does not head toward the chosen side'


def test_the_rejoin_only_moves_back_toward_the_raceline(track, params):
    obs = FakeObs(7.0, d_center=-0.5)
    choice = choose_side(obs, track, params)
    s, d, v = build_evasion(0.0, 0.0, obs, choice, track, params)
    last_apex = len(d) - 1 - int(np.argmax(np.isclose(d[::-1], choice.d_apex, atol=1e-9)))
    back = np.diff(d[last_apex:]) * np.sign(-choice.d_apex)
    assert np.all(back >= -1e-12) and d[-1] == pytest.approx(0.0, abs=1e-9)


def test_transitions_respect_the_slope_limit(track, params):
    obs = FakeObs(7.0, d_center=-0.5)
    s, d, v = build_evasion(0.0, 0.0, obs, choose_side(obs, track, params), track, params)
    slope = np.abs(np.diff(d)) / params.resolution
    assert slope.max() <= params.max_lateral_slope * 1.02


def test_the_apex_is_held_across_the_whole_footprint(track, params):
    """Full offset before the nose meets the obstacle, until the tail has left it."""
    obs = FakeObs(7.0, d_center=-0.5)
    choice = choose_side(obs, track, params)
    s, d, v = build_evasion(0.0, 0.0, obs, choice, track, params)
    held = (s >= obs.s_start - params.car_front - params.apex_margin + 0.01) & \
           (s <= obs.s_end + params.car_rear + params.apex_margin - 0.01)
    assert held.any() and np.allclose(d[held], choice.d_apex)


def test_too_close_when_even_the_steepest_transition_does_not_fit(track, params):
    obs = FakeObs(1.2, d_center=-0.5)      # reach by 1.2 - 0.25 - 0.45 - 0.15 = 0.35 m
    choice = choose_side(obs, track, params)
    assert build_evasion(0.0, 0.0, obs, choice, track, params) is None
    assert plan(0.0, 0.0, [obs], track, params).status == TOO_CLOSE


def test_plan_refuses_a_way_round_that_another_obstacle_blocks(track, params):
    """Room at the first obstacle, but a second sits on the evasion line."""
    first = FakeObs(6.0, d_center=-0.5)
    s, d, v = build_evasion(0.0, 0.0, first, choose_side(first, track, params), track, params)
    second = FakeObs(6.8, d_center=float(d[np.argmin(np.abs(s - 6.8))]), width=0.3)
    assert not path_clear(s, d, [first, second], track, params)
    r = plan(0.0, 0.0, [first, second], track, params)
    assert r.status == NO_ROOM and not r.collision_free


def test_path_clear_accepts_the_evasion_against_its_own_obstacle(track, params):
    obs = FakeObs(6.0, d_center=-0.5)
    s, d, v = build_evasion(0.0, 0.0, obs, choose_side(obs, track, params), track, params)
    assert path_clear(s, d, [obs], track, params)
    # this one sits off the line: the raceline itself clears it (car edge -0.15, its edge -0.35)
    assert path_clear(s, np.zeros_like(d), [obs], track, params)
    on_line = FakeObs(6.0, d_center=0.0)
    assert not path_clear(s, np.zeros_like(d), [on_line], track, params)   # the raceline hits it


# ------------------------------------------------------------------ problem 2: commitment


@pytest.fixture
def committed(track, params):
    """A Spliner that has just committed to passing an obstacle 7 m ahead, on the line."""
    sp = Spliner(track, params)
    obs = FakeObs(7.0, d_center=0.0)
    r = sp.step(0.0, 0.0, [obs])
    assert r.status == EVADING and not r.committed and sp.committed is not None
    return sp, obs


def _d_on(sp, s):
    c = sp.committed
    return float(np.interp(s, c.s, c.d))


def test_the_commitment_holds_where_stateless_planning_gives_up(track, params, committed):
    """0.25 m before the obstacle the stateless plan() says too close (the bug); the
    committed evasion carries on, unchanged."""
    sp, obs = committed
    first = sp.committed.s.copy(), sp.committed.d.copy()
    ego = obs.s_start - 0.25
    assert plan(ego, _d_on(sp, ego), [obs], track, params).status == TOO_CLOSE
    r = sp.step(ego, _d_on(sp, ego), [obs])
    assert r.status == EVADING and r.committed and r.released is None
    assert np.array_equal(sp.committed.s, first[0]) and np.array_equal(sp.committed.d, first[1])
    assert r.s[0] >= ego - params.resolution and len(r.s) < len(first[0])   # the rest of it


def test_release_when_passed(track, params, committed):
    sp, obs = committed
    end = sp.committed.s[-1]
    r = sp.step(end + 0.2, 0.0, [obs])
    assert r.released == 'passed' and r.status == CLEAR and sp.committed is None


def test_release_when_the_obstacle_is_gone_while_still_ahead(track, params, committed):
    sp, obs = committed
    r = sp.step(2.0, _d_on(sp, 2.0), [])
    assert r.released == 'gone' and r.status == CLEAR and sp.committed is None


def test_an_obstacle_lost_from_view_alongside_is_held_until_passed(track, params, committed):
    """A forward sensor drops it once it is beside the car. Releasing then would steer the car
    back into it, so the commitment holds until 'passed'."""
    sp, obs = committed
    beside = obs.s_center - 0.2                  # nose well past the obstacle's start
    r = sp.step(beside, _d_on(sp, beside), [])
    assert r.status == EVADING and r.committed and r.released is None
    r = sp.step(sp.committed.s[-1] + 0.2, 0.0, [])
    assert r.released == 'passed'


def test_a_moved_obstacle_is_still_the_committed_one_within_tolerance(track, params, committed):
    sp, obs = committed
    nudged = FakeObs(obs.s_center + 0.1, d_center=obs.d_center + 0.03)
    r = sp.step(2.0, _d_on(sp, 2.0), [nudged])
    assert r.committed and r.released is None


def test_release_when_the_rest_is_no_longer_collision_free(track, params, committed):
    """Something new on the committed line: drop the evasion, and never return its path."""
    sp, obs = committed
    stale = sp.committed.s.copy(), sp.committed.d.copy()
    blocker = FakeObs(6.0, d_center=_d_on(sp, 6.0))
    r = sp.step(2.0, _d_on(sp, 2.0), [obs, blocker])
    assert r.released == 'blocked'
    assert not (r.status == EVADING and np.allclose(np.interp(6.0, r.s, r.d),
                                                    np.interp(6.0, *stale), atol=0.05))
    assert path_clear(r.s, r.d, [obs, blocker], track, params) or not r.collision_free


def test_release_when_the_car_is_too_far_off_the_committed_line(track, params, committed):
    sp, obs = committed
    on = _d_on(sp, 2.0)
    assert sp.step(2.0, on + params.max_commit_offset * 0.9, [obs]).committed
    actual = _d_on(sp, 2.1) + params.max_commit_offset + 0.05
    r = sp.step(2.1, actual, [obs])
    assert r.released == 'off line' and not r.committed
    assert r.status == EVADING                    # replanned, from where the car actually is
    assert r.s[0] == pytest.approx(2.1) and r.d[0] == pytest.approx(actual)


def test_a_commitment_is_not_kept_after_release(track, params, committed):
    sp, obs = committed
    sp.step(2.0, _d_on(sp, 2.0), [])              # gone
    assert sp.committed is None
    r = sp.step(2.05, 0.0, [])
    assert r.status == CLEAR and not r.committed and r.released is None
