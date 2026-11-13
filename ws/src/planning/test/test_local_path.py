"""The spliner -> /local_path adapter: splice, validity, pairing. No ROS: plain pytest."""
import numpy as np
import pytest

from planning.frenet import FrenetConverter
from planning.local_path import Pairer, Plan, decide, load_raceline_rows, splice
from planning.raceline import write_raceline
from planning.spliner import (CLEAR, EVADING, LEFT, NO_ROOM, TOO_CLOSE, Spliner,
                              SplinerParams, plan)

R, N, W = 10.0, 600, 1.5
MAX_PLAN, MAX_EGO = 0.25, 0.25


class FakeObs:
    def __init__(self, s_center, d_center=0.0, length=0.5, width=0.3):
        self.s_center, self.s_start, self.s_end = s_center, s_center - length / 2, s_center + length / 2
        self.d_center, self.d_left, self.d_right = d_center, d_center + width / 2, d_center - width / 2
        self.is_static, self.is_visible = True, True


@pytest.fixture(scope='module')
def track(tmp_path_factory):
    """A closed TUM-format circle raceline + centerline, as build_raceline writes them."""
    d = tmp_path_factory.mktemp('circle')
    th = np.linspace(0, 2 * np.pi, N, endpoint=False)
    xy = np.column_stack((R * np.cos(th), R * np.sin(th)))
    np.savetxt(d / 'centerline.csv', np.column_stack((xy, np.full(N, W), np.full(N, W))),
               delimiter=',', header='x_m,y_m,w_tr_right_m,w_tr_left_m')
    traj = np.column_stack((th * R, xy, th, np.full(N, 1 / R), np.full(N, 5.0), np.zeros(N)))
    traj = np.vstack((traj, traj[0]))
    traj[-1, 0] = 2 * np.pi * R
    write_raceline(str(d / 'raceline.csv'), traj, {'method': 'mintime', 'clean': True})
    s, rxy = load_raceline_rows(str(d / 'raceline.csv'))
    return FrenetConverter.from_track(str(d)), s, rxy


def ev(conv, ego_s, obstacles):
    return plan(ego_s, 0.0, obstacles, conv, SplinerParams())


def fresh(status, s=(), d=(), note='', received=10.0):
    return Plan(status, note, np.asarray(s, float), np.asarray(d, float), received)


# ------------------------------------------------------------------ rows and splice


def test_rows_are_the_ones_pure_pursuit_indexes(track):
    """One per raceline row with the TUM closing row dropped: pure pursuit rejects any other count."""
    conv, s, xy = track
    assert len(s) == len(xy) == N
    assert not np.allclose(xy[0], xy[-1])


def test_no_evasion_is_exactly_the_raceline(track):
    conv, s, xy = track
    out = splice(s, xy, conv.length, [], [], conv.to_cartesian)
    assert out.shape == (N, 2) and np.array_equal(out, xy)


def test_splice_moves_only_the_evasion_rows_and_by_d(track):
    conv, s, xy = track
    r = ev(conv, 0.0, [FakeObs(6.0, d_center=-0.5)])
    assert r.status == EVADING
    out = splice(s, xy, conv.length, r.s, r.d, conv.to_cartesian)
    assert out.shape == (N, 2)
    inside = s <= r.s[-1]
    assert np.array_equal(out[~inside], xy[~inside])                  # untouched exactly
    # moved rows sit d to the left: on this anticlockwise circle, radius R - d
    d_at = np.interp(s[inside], r.s, r.d)
    assert np.hypot(*out[inside].T) == pytest.approx(R - d_at, abs=0.01)
    at_obs = (s > 5.75) & (s < 6.25)
    assert np.all(np.interp(s[at_obs], r.s, r.d) > -0.5 + 0.15 + 0.15)   # clear of it, to its left


def test_splice_wraps_across_the_start_line(track):
    conv, s, xy = track
    ego = conv.length - 3.0
    r = ev(conv, ego, [FakeObs(3.0, d_center=-0.5)])
    assert r.status == EVADING and r.s[0] > r.s[-1]                    # the evasion wraps
    out = splice(s, xy, conv.length, r.s, r.d, conv.to_cartesian)
    moved = np.any(out != xy, axis=1)
    assert moved[s < 3.0].any() and moved[s > conv.length - 3.0].any()
    assert not moved[(s > 10.0) & (s < conv.length - 5.0)].any()
    step = np.hypot(*np.diff(np.vstack((out, out[:1])), axis=0).T)
    assert step.max() < 0.2, 'the spliced path jumps somewhere'


def test_splice_rejects_mismatched_arrays(track):
    conv, s, xy = track
    with pytest.raises(ValueError):
        splice(s, xy, conv.length, [1.0, 2.0], [0.1], conv.to_cartesian)


# ------------------------------------------------------------------ the decision


def test_clear_is_valid_raceline():
    dec = decide(10.0, fresh(CLEAR), 10.0, MAX_PLAN, MAX_EGO)
    assert dec.valid and not dec.spliced


def test_evading_is_valid_spliced():
    dec = decide(10.0, fresh(EVADING, [1.0, 1.1], [0.0, 0.1]), 10.0, MAX_PLAN, MAX_EGO)
    assert dec.valid and dec.spliced


# Every stop case. Each must leave the path invalid, which pure pursuit turns into a stop.

def test_stop_when_planner_says_no_room():
    """The case that would otherwise drive the car into something: an empty trajectory, as for
    CLEAR, but the planner has refused. Before PlanResult this was indistinguishable."""
    dec = decide(10.0, fresh(NO_ROOM, note='trailing: no feasible side'), 10.0, MAX_PLAN, MAX_EGO)
    assert not dec.valid and 'no room' in dec.reason


def test_stop_when_planner_says_too_close():
    dec = decide(10.0, fresh(TOO_CLOSE), 10.0, MAX_PLAN, MAX_EGO)
    assert not dec.valid and 'too close' in dec.reason


def test_stop_before_any_planner_output():
    assert not decide(10.0, None, 10.0, MAX_PLAN, MAX_EGO).valid


def test_stop_when_planner_output_is_stale():
    dec = decide(10.0 + MAX_PLAN + 0.01, fresh(CLEAR), 10.0 + MAX_PLAN, MAX_PLAN, MAX_EGO)
    assert not dec.valid and 'planner output' in dec.reason
    assert decide(10.0 + MAX_PLAN - 0.01, fresh(CLEAR), 10.0, MAX_PLAN, MAX_EGO).valid


def test_stop_before_any_ego_state():
    dec = decide(10.0, fresh(CLEAR), None, MAX_PLAN, MAX_EGO)
    assert not dec.valid and 'ego' in dec.reason


def test_stop_when_ego_state_is_stale():
    dec = decide(10.0, fresh(CLEAR), 10.0 - MAX_EGO - 0.01, MAX_PLAN, MAX_EGO)
    assert not dec.valid and 'ego state' in dec.reason


@pytest.mark.parametrize('status', ['', 'overtaking', 'Clear', 'trailing'])
def test_stop_on_an_unknown_status(status):
    assert not decide(10.0, fresh(status), 10.0, MAX_PLAN, MAX_EGO).valid


def test_stop_when_evading_has_no_trajectory():
    assert not decide(10.0, fresh(EVADING), 10.0, MAX_PLAN, MAX_EGO).valid


def test_stop_when_clear_comes_with_a_trajectory():
    assert not decide(10.0, fresh(CLEAR, [1.0, 1.1], [0.0, 0.1]), 10.0, MAX_PLAN, MAX_EGO).valid


def test_every_planner_outcome_maps_to_the_right_decision(track):
    """End to end through plan(): only CLEAR and EVADING let the car drive."""
    conv, s, xy = track
    cases = {CLEAR: [], EVADING: [FakeObs(6.0, d_center=-0.5)],
             NO_ROOM: [FakeObs(6.0, width=2.8)], TOO_CLOSE: [FakeObs(0.5, d_center=-0.5)]}
    for status, obstacles in cases.items():
        r = ev(conv, 0.0, obstacles)
        assert r.status == status
        dec = decide(10.0, fresh(r.status, r.s, r.d, r.note), 10.0, MAX_PLAN, MAX_EGO)
        assert dec.valid == (status in (CLEAR, EVADING)), status


# ------------------------------------------------------------------ pairing by stamp


def test_a_status_is_only_used_with_its_own_trajectory():
    p = Pairer()
    p.add_trajectory((1, 0), [1.0, 1.1], [0.0, 0.2], now=1.0)
    p.add_status((1, 0), EVADING, 'overtaking left', now=1.0)
    assert p.plan.status == EVADING
    # the next cycle's status arrives before its trajectory: keep the last complete pair
    p.add_status((1, 50), NO_ROOM, 'trailing', now=1.05)
    assert p.plan.status == EVADING and p.plan.received == 1.0
    p.add_trajectory((1, 50), [], [], now=1.06)
    assert p.plan.status == NO_ROOM and len(p.plan.s) == 0 and p.plan.received == 1.06


def test_an_unpaired_stream_never_becomes_a_plan():
    p = Pairer()
    for k in range(20):
        p.add_trajectory((1, k), [], [], now=float(k))
    assert p.plan is None
    assert len(p.trajectories) <= p.keep


def test_a_late_old_pair_does_not_replace_a_newer_one():
    p = Pairer()
    p.add_trajectory((2, 0), [], [], 2.0)
    p.add_status((2, 0), CLEAR, 'clear', 2.0)
    p.add_trajectory((1, 0), [], [], 2.1)
    p.add_status((1, 0), NO_ROOM, 'x', 2.1)
    assert p.plan.status == CLEAR


# ------------------------------------------------------------------ closed loop
#
# The only tests here that ask whether the planner WORKS, not whether it is self-consistent:
# a car steered as pure pursuit steers, replanning every cycle through Spliner (what
# spliner_node runs), must get past a stationary obstacle without touching it.


def closed_loop(conv, obstacle, speed, start_s=0.0, dt=0.05, distance=14.0, planner=None):
    """The car follows the plan as pure pursuit would: toward the point one lookahead ahead.

    Returns (statuses seen, distance travelled, smallest footprint gap, legacy hit flag).
    The footprint gap is the lateral gap between the car's rectangle (car_rear behind to
    car_front ahead of the rear axle, car_width wide) and the obstacle's, over every cycle
    their s spans overlap; negative means they overlapped. The legacy flag is the original
    criterion of test_a_stationary_obstacle_is_never_hit, kept as it was.
    """
    lookahead = min(max(0.4 * speed, 0.6), 3.0)        # control/config/vehicle.yaml's law
    params = SplinerParams()
    planner = planner or Spliner(conv, params)
    s, d, travelled = start_s, 0.0, 0.0
    seen, gap, hit = [], np.inf, False
    half = params.car_width / 2.0
    for _ in range(int(distance / speed / dt)):
        r = planner.step(s, d, [obstacle])
        if not seen or seen[-1] != r.status:
            seen.append(r.status)
        if not r.collision_free:
            break
        if r.status == EVADING:
            u = np.concatenate(([0.0], np.cumsum(np.diff(r.s) % conv.length)))
            u_car = conv.ds(r.s[0], s)
            d += (float(np.interp(u_car + lookahead, u, r.d)) - d) * speed * dt / lookahead
        else:
            d -= np.clip(d, -speed * dt / lookahead, speed * dt / lookahead)
        s = (s + speed * dt) % conv.length
        travelled += speed * dt
        rear = conv.ds(obstacle.s_end, s - params.car_rear)      # > 0: tail past its end
        front = conv.ds(obstacle.s_start, s + params.car_front)  # > 0: nose past its start
        if front > 0 and rear < 0:
            gap = min(gap, max(obstacle.d_right - (d + half), (d - half) - obstacle.d_left))
        u_obs = conv.ds(s, obstacle.s_center)
        if -obstacle_half(obstacle) - 0.25 <= -u_obs <= obstacle_half(obstacle) + 0.25 \
                and abs(d - obstacle.d_center) < 0.15 + half:
            hit = True
    return seen, travelled, gap, hit


def obstacle_half(o):
    return (o.s_end - o.s_start) / 2.0


CASES = {                                 # name: (obstacle, car start s)
    'on the line': (FakeObs(8.0), 0.0),
    'right of it': (FakeObs(8.0, d_center=-0.4), 0.0),
    'left of it': (FakeObs(8.0, d_center=0.4), 0.0),
    'past the start line': (FakeObs(2.0), 2 * np.pi * R - 6.0),
}


@pytest.mark.parametrize('speed', [1.5, 3.0])
def test_a_stationary_obstacle_is_never_hit(track, speed):
    """Whatever the planner does, it must not drive the car into the obstacle."""
    conv, _, _ = track
    seen, travelled, gap, hit = closed_loop(conv, FakeObs(8.0), speed)
    assert not hit


@pytest.mark.parametrize('case', CASES)
@pytest.mark.parametrize('speed', [1.5, 3.0])
def test_a_stationary_obstacle_is_passed_without_touching_it(track, speed, case):
    """Reaches the far side, never inside the obstacle's footprint, and never stops.

    Before the smoothstep path and the commitment this stopped the car 0.25 m short of the
    obstacle (too close), after first steering it up to 0.3 m the wrong way."""
    conv, _, _ = track
    obstacle, start = CASES[case]
    seen, travelled, gap, hit = closed_loop(conv, obstacle, speed, start_s=start)
    assert TOO_CLOSE not in seen and NO_ROOM not in seen, seen
    far_side = conv.ds(start, obstacle.s_end) + SplinerParams().car_rear + 1.0
    assert travelled > far_side, f'stopped after {travelled:.2f} m, statuses {seen}'
    assert np.isfinite(gap), 'the car never drew level with the obstacle'
    assert gap > 0.0, f'the car\'s footprint overlapped the obstacle by {-gap:.3f} m'


@pytest.mark.parametrize('speed', [1.5, 3.0])
def test_the_evasion_starts_toward_the_chosen_side_in_closed_loop(track, speed):
    """Problem 1 in the loop: the car's d never moves away from the side it is passing on
    before it reaches the obstacle (the cubic drove it up to 0.3 m the wrong way)."""
    conv, _, _ = track
    params = SplinerParams()
    planner = Spliner(conv, params)
    obstacle = FakeObs(8.0, d_center=-0.4)          # passed on the left: d must not go negative
    lookahead = min(max(0.4 * speed, 0.6), 3.0)
    s = d = 0.0
    worst = 0.0
    while s < obstacle.s_start:
        r = planner.step(s, d, [obstacle])
        assert r.status == EVADING and r.side == LEFT
        u = np.concatenate(([0.0], np.cumsum(np.diff(r.s) % conv.length)))
        d += (float(np.interp(conv.ds(r.s[0], s) + lookahead, u, r.d)) - d) * speed * 0.05 / lookahead
        worst = min(worst, d)
        s += speed * 0.05
    assert worst >= -1e-9, f'drifted {-worst:.3f} m the wrong way'


def test_stateless_planning_alone_still_stops_short(track):
    """Why spliner_node uses Spliner: plan() every cycle, with no commitment, gives up as the
    lead-in collapses. If this ever passes, the commitment may no longer be needed."""
    conv, _, _ = track

    class Stateless:
        def step(self, s, d, obstacles):
            return plan(s, d, obstacles, conv, SplinerParams())

    seen, travelled, gap, hit = closed_loop(conv, FakeObs(8.0), 1.5, planner=Stateless())
    assert TOO_CLOSE in seen and not hit
