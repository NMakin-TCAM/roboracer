"""The obstacle tracker against opponent_models: does it recover what detection cannot report?

Every observation is stripped before the tracker sees it: id, vs, vd and the four variances
become NaN or -1, and is_static becomes True, which is what a scan-based detector reports.
The tests then check the tracker puts back what was taken, measured against the motion
model's ground truth. No ROS.
"""
import dataclasses

import numpy as np
import pytest

from planning.collision_prediction import first_dynamic
from planning.frenet import FrenetConverter
from planning.obstacle_tracker import ObstacleTracker, TrackerParams
from planning.opponent_models import (VAR_FLOOR, ConstantSpeed, ObstacleObservation, Replay,
                                      SensorModel, StaticObstacle, Weaving, observe)
from planning.opponent_trajectory import find_nearest_dyn_obstacle

NAN = float('nan')
RATE = 40.0
DT = 1.0 / RATE


@pytest.fixture(scope='module')
def conv():
    th = np.linspace(0, 2 * np.pi, 721)[:-1]
    r = 100.0 / (2 * np.pi)
    return FrenetConverter(np.column_stack((r * np.cos(th), r * np.sin(th))))


@pytest.fixture
def lap(conv):
    return conv.length


def strip(o):
    """What a detector without a tracker reports: a position and a footprint, nothing else."""
    return dataclasses.replace(o, id=-1, vs=NAN, vd=NAN, s_var=NAN, d_var=NAN, vs_var=NAN,
                               vd_var=NAN, is_static=True)


def det(s, d=0.0, lap=None, half_l=0.3, half_w=0.2):
    """A stripped detection at (s, d)."""
    s_start, s_end = s - half_l, s + half_l
    if lap is not None and s_start < 0:
        s_start, s_end = s_start + lap, s_end + lap
    return ObstacleObservation(id=-1, s_start=s_start, s_end=s_end, d_right=d - half_w,
                               d_left=d + half_w, is_actually_a_gap=False, s_center=s,
                               d_center=d, size=0.7, vs=NAN, vd=NAN, s_var=NAN, d_var=NAN,
                               vs_var=NAN, vd_var=NAN, is_static=True, is_visible=True)


NOISY = SensorModel(max_range=10.0, detection_prob=0.9, sigma_s=0.05, sigma_d=0.05, latency=0.05)


def simulate(conv, motions, sensor=NOISY, duration=20.0, seed=0, ego_gap=3.0, extra=None):
    """Run the tracker over motion models observed through the sensor model, with the ego
    trailing the first motion by ego_gap m. Returns a list of (t, truths, published)."""
    lap = conv.length
    tracker = ObstacleTracker(conv, TrackerParams(sigma_s=0.05, sigma_d=0.05))
    rng = np.random.default_rng(seed)
    log = []
    for k in range(int(duration * RATE)):
        t = k * DT
        truths = [m.state_at(t - sensor.latency) for m in motions]
        ego = (truths[0].s - ego_gap) % lap
        dets = [o for m in motions if (o := observe(m, t, ego, lap, sensor, rng)) is not None]
        if extra:
            dets += extra(t, rng)
        out = tracker.step([strip(o) for o in dets], t, ego)
        log.append((t, truths, out))
    return log


def velocity_errors(log, settle=2.0):
    err = np.array([(o[0].vs - tr[0].vs, o[0].vd - tr[0].vd)
                    for t, tr, o in log if t >= settle and o])
    return np.sqrt(np.mean(err ** 2, axis=0))


def replay_line(lap):
    s = np.linspace(0, lap, 40, endpoint=False)
    return s, 0.3 * np.sin(6 * np.pi * s / lap) + 0.1 * np.cos(14 * np.pi * s / lap)


# ---------------------------------------------------------------- velocity

@pytest.mark.parametrize('name, tol_vs, tol_vd', [
    ('static', 0.12, 0.12),
    ('constant_speed', 0.12, 0.12),
    # constant velocity lags a sine: vd swings +-0.6 m/s at 1.5 rad/s here
    ('weaving', 0.12, 0.25),
    ('replay', 0.12, 0.15),
])
def test_velocity_is_recovered_for_each_motion_model(conv, name, tol_vs, tol_vd):
    lap = conv.length
    rs, rd = replay_line(lap)
    motion = {'static': StaticObstacle(20.0, 0.2),
              'constant_speed': ConstantSpeed(lap, 4.0, 10.0, 0.1),
              'weaving': Weaving(lap, 3.0, 0.4, 12.0, 10.0),
              'replay': Replay(lap, rs, rd, 3.0, 10.0)}[name]
    log = simulate(conv, [motion])
    rms_vs, rms_vd = velocity_errors(log)
    assert rms_vs < tol_vs and rms_vd < tol_vd, (rms_vs, rms_vd)


def test_constant_speed_is_unbiased(conv):
    log = simulate(conv, [ConstantSpeed(conv.length, 4.0, 10.0, 0.1)])
    vs = [o[0].vs for t, tr, o in log if t >= 2.0 and o]
    assert np.mean(vs) == pytest.approx(4.0, abs=0.03)


def test_noiseless_constant_speed_is_exact(conv):
    ideal = SensorModel(max_range=10.0)
    log = simulate(conv, [ConstantSpeed(conv.length, 5.0, 10.0, -0.2)], sensor=ideal, duration=5.0)
    t, tr, out = log[-1]
    assert out[0].vs == pytest.approx(5.0, abs=1e-3)
    assert out[0].vd == pytest.approx(0.0, abs=1e-3)
    assert out[0].s_center == pytest.approx(tr[0].s, abs=1e-3)


# ---------------------------------------------------------------- identity and lifecycle

def test_one_obstacle_keeps_one_id_through_a_noisy_run(conv):
    log = simulate(conv, [Weaving(conv.length, 3.0, 0.4, 12.0, 10.0)], duration=30.0)
    ids = {o.id for _, _, out in log for o in out}
    assert ids == {0}
    assert sum(1 for _, _, out in log if out) > 0.85 * len(log)   # 90% detected; coasting is silent


def test_two_obstacles_keep_their_ids_as_one_passes_the_other(conv):
    """A car at 3 m/s passes a cone 0.6 m to its side; neither identity swaps."""
    lap = conv.length
    car = ConstantSpeed(lap, 3.0, 10.0, -0.3)
    cone = StaticObstacle(18.0, 0.3)
    log = simulate(conv, [car, cone], duration=6.0, ego_gap=2.0)
    id_of = {}
    for t, tr, out in log:
        for o in out:
            truth = min(tr, key=lambda x: abs(conv.ds(x.s, o.s_center)) + abs(x.d - o.d_center))
            who = 'car' if truth is tr[0] else 'cone'
            assert id_of.setdefault(o.id, who) == who
    assert sorted(id_of.values()) == ['car', 'cone']


def test_coasted_tracks_are_not_published_by_default(conv):
    """opponent_trajectory never reads is_visible, so a published prediction would enter the
    GP as a sighting. Hidden while coasting; back under the same id when seen again."""
    p = TrackerParams()
    assert not p.publish_coasted
    tr = ObstacleTracker(conv, p)
    t = 0.0
    for k in range(20):
        out = tr.step([det(10.0 + 2.0 * t)], t)
        t += DT
    assert [o.id for o in out] == [0]
    for k in range(p.max_misses):
        assert tr.step([], t) == []
        t += DT
    out = tr.step([det(10.0 + 2.0 * t)], t)
    assert [o.id for o in out] == [0] and out[0].is_visible


def test_every_published_obstacle_is_visible_by_default(conv):
    sensor = dataclasses.replace(NOISY, detection_prob=0.6)
    log = simulate(conv, [ConstantSpeed(conv.length, 4.0, 10.0, 0.1)], sensor=sensor, duration=10.0)
    assert all(o.is_visible for _, _, out in log for o in out)


def test_a_track_survives_a_dropout_shorter_than_max_misses(conv, lap):
    p = TrackerParams(publish_coasted=True)
    tr = ObstacleTracker(conv, p)
    t = 0.0
    for k in range(20):
        out = tr.step([det(10.0 + 2.0 * t)], t)
        t += DT
    assert [o.id for o in out] == [0]
    for k in range(p.max_misses):          # occluded: nothing reported
        out = tr.step([], t)
        assert [o.id for o in out] == [0] and not out[0].is_visible
        assert out[0].s_center == pytest.approx(10.0 + 2.0 * t, abs=0.05)   # coasted, not frozen
        t += DT
    out = tr.step([det(10.0 + 2.0 * t)], t)
    assert [o.id for o in out] == [0] and out[0].is_visible


def test_a_track_dies_after_more_than_max_misses(conv):
    p = TrackerParams()
    tr = ObstacleTracker(conv, p)
    for k in range(10):
        tr.step([det(10.0)], k * DT)
    for k in range(10, 11 + p.max_misses):
        out = tr.step([], k * DT)
    assert out == []


def test_dropout_from_the_sensor_model_does_not_break_tracks(conv):
    sensor = dataclasses.replace(NOISY, detection_prob=0.6)
    log = simulate(conv, [ConstantSpeed(conv.length, 4.0, 10.0, 0.1)], sensor=sensor, duration=20.0)
    assert {o.id for _, _, out in log for o in out} == {0}


def test_no_track_from_a_single_spurious_detection(conv):
    tr = ObstacleTracker(conv)
    assert tr.step([det(40.0, 0.5)], 0.0) == []
    for k in range(1, 30):
        assert tr.step([], k * DT) == []


@pytest.mark.parametrize('frames', [1, 2])
def test_fewer_than_confirm_frames_never_publish(conv, frames):
    tr = ObstacleTracker(conv)
    for k in range(frames):
        assert tr.step([det(40.0)], k * DT) == []
    for k in range(frames, 40):
        assert tr.step([], k * DT) == []


def test_confirmed_on_the_third_consecutive_frame(conv):
    tr = ObstacleTracker(conv)
    assert tr.step([det(40.0)], 0.0) == []
    assert tr.step([det(40.0)], DT) == []
    assert len(tr.step([det(40.0)], 2 * DT)) == 1


def test_random_clutter_spawns_no_obstacle(conv):
    """A spurious return somewhere on the lap in 30% of cycles, never twice in one place."""
    lap = conv.length

    def clutter(t, rng):
        if rng.random() < 0.3:
            return [det(rng.uniform(0, lap), rng.uniform(-1, 1), lap)]
        return []

    log = simulate(conv, [ConstantSpeed(lap, 4.0, 10.0, 0.1)], duration=20.0, extra=clutter)
    assert {o.id for _, _, out in log for o in out} == {0}


def test_a_spurious_return_absorbed_during_a_dropout_does_not_spawn_a_duplicate(conv):
    """Replays clutter seed 10, which produced a duplicate before the matching cascade: the
    car's detection drops while a spurious return 0.4 m behind it falls in the gate and is
    absorbed, moving the gate's centre back. The car's next detection is then outside the gate
    and starts a tentative track. On the cycle after, both tracks gate the car; the tentative
    one, sitting on the last detection, is nearer. Confirmed tracks choose first, so the real
    track takes it back and the tentative one dies instead of being confirmed."""
    tr = ObstacleTracker(conv, TrackerParams(sigma_s=0.05, sigma_d=0.05))

    def car(k):
        return 20.0 + 4.0 * k * DT

    for k in range(40):
        tr.step([det(car(k), 0.15)], k * DT)
    k = 40
    tr.step([det(car(k) - 0.4, -0.4)], k * DT)           # dropout; a spurious return absorbed
    out = tr.step([det(car(k + 1), 0.06)], (k + 1) * DT)  # outside the moved gate: tentative
    assert tr.tracks[-1].id == -1
    for k in range(k + 2, k + 20):
        out = tr.step([det(car(k), 0.2)], k * DT)
        assert [o.id for o in out] == [0]


def test_input_ids_velocities_and_variances_are_not_read(conv):
    """NaN in every field the tracker must not read, and nothing NaN comes out."""
    log = simulate(conv, [ConstantSpeed(conv.length, 4.0, 10.0, 0.1)], duration=3.0)
    for _, _, out in log:
        for o in out:
            assert all(np.isfinite(getattr(o, f)) for f in
                       ('s_center', 'd_center', 'vs', 'vd', 's_var', 'd_var', 'vs_var', 'vd_var'))


def test_remembered_detections_are_not_measurements(conv):
    tr = ObstacleTracker(conv)
    ghost = dataclasses.replace(det(40.0), is_visible=False)
    for k in range(10):
        assert tr.step([ghost], k * DT) == []


# ---------------------------------------------------------------- is_static

def test_a_stationary_obstacle_becomes_static(conv):
    log = simulate(conv, [StaticObstacle(20.0, 0.2)], duration=5.0)
    flags = [o[0].is_static for t, _, o in log if t >= 1.0 and o]
    assert flags and all(flags)


def test_a_moving_obstacle_is_never_static(conv):
    for motion in (ConstantSpeed(conv.length, 2.0, 10.0, 0.1),
                   Weaving(conv.length, 3.0, 0.4, 12.0, 10.0)):
        log = simulate(conv, [motion], duration=10.0)
        assert not any(o.is_static for _, _, out in log for o in out)


def test_a_slow_walker_is_moving(conv):
    """1 m/s is a car crawling out of the pits, not a cone."""
    log = simulate(conv, [ConstantSpeed(conv.length, 1.0, 10.0, 0.1)], duration=10.0)
    assert not any(o.is_static for _, _, out in log for o in out)


def test_new_tracks_are_dynamic_until_the_window_has_passed(conv):
    tr = ObstacleTracker(conv)
    for k in range(int(TrackerParams().static_window * RATE) - 1):
        out = tr.step([det(40.0)], k * DT)
        assert all(not o.is_static for o in out)
    for k in range(k + 1, k + 10):
        out = tr.step([det(40.0)], k * DT)
    assert out[0].is_static


def test_a_car_that_stops_becomes_static_and_moving_again_when_it_leaves(conv):
    tr = ObstacleTracker(conv)
    s, flags = 10.0, []
    for k in range(int(12 * RATE)):
        t = k * DT
        v = 3.0 if t < 4.0 or t >= 8.0 else 0.0
        s += v * DT
        out = tr.step([det(s)], t)
        flags.append((t, out[0].is_static if out else None))
    assert not any(f for t, f in flags if 1.0 <= t < 4.0)
    assert all(f for t, f in flags if 5.5 <= t < 8.0)
    assert not any(f for t, f in flags if t >= 9.0)


# ---------------------------------------------------------------- variances

def test_variances_are_floored_and_grow_while_coasting(conv):
    tr = ObstacleTracker(conv, TrackerParams(publish_coasted=True))
    for k in range(80):
        out = tr.step([det(10.0 + 2.0 * k * DT)], k * DT)
    seen = out[0]
    assert all(getattr(seen, f) >= VAR_FLOOR for f in ('s_var', 'd_var', 'vs_var', 'vd_var'))
    assert seen.vs_var < 0.05     # converged: a few cm/s, not the 25 m^2/s^2 prior
    for k in range(80, 85):
        out = tr.step([], k * DT)
    assert out[0].s_var > seen.s_var and out[0].vs_var > seen.vs_var


def test_variance_floor_holds_for_a_noiseless_tracker(conv):
    tr = ObstacleTracker(conv, TrackerParams(sigma_s=1e-6, sigma_d=1e-6, accel_sigma=1e-6))
    for k in range(200):
        out = tr.step([det(10.0)], k * DT)
    assert all(getattr(out[0], f) >= VAR_FLOOR for f in ('s_var', 'd_var', 'vs_var', 'vd_var'))


# ---------------------------------------------------------------- the start line

def test_a_car_crossing_the_start_line_keeps_its_id_and_speed(conv):
    lap = conv.length
    car = ConstantSpeed(lap, 4.0, lap - 6.0, 0.1)
    log = simulate(conv, [car], duration=4.0)
    assert {o.id for _, _, out in log for o in out} == {0}
    crossing = [o[0] for t, tr, o in log if t >= 1.0 and o]
    assert any(o.s_center < 2.0 for o in crossing) and any(o.s_center > lap - 2.0 for o in crossing)
    assert all(abs(o.vs - 4.0) < 0.4 for o in crossing)     # no lap-length spike
    for o in crossing:
        assert o.s_start < o.s_end and 0.0 <= o.s_center < lap


def test_a_cone_on_the_start_line_is_one_obstacle(conv):
    """Noise puts it either side of s = 0 from one scan to the next."""
    log = simulate(conv, [StaticObstacle(0.0, 0.0)], duration=5.0, ego_gap=5.0)
    assert {o.id for _, _, out in log for o in out} == {0}
    last = log[-1][2][0]
    assert last.is_static
    assert abs(last.vs) < 0.3
    assert last.s_start < last.s_end


# ---------------------------------------------------------------- time

def test_time_going_backwards_drops_every_track(conv):
    tr = ObstacleTracker(conv)
    for k in range(10):
        tr.step([det(10.0)], 100.0 + k * DT)
    assert tr.step([], 50.0) == []


def test_a_long_silence_drops_every_track(conv):
    tr = ObstacleTracker(conv)
    for k in range(10):
        tr.step([det(10.0)], k * DT)
    assert tr.step([det(10.0)], 5.0) == []     # new tentative track; the old one is gone
    assert tr.tracks[0].id == -1


# ---------------------------------------------------------------- output order

def picks(out, ego_s, lap):
    """(collision_prediction's opponent, opponent_trajectory's opponent), by s_center."""
    a = first_dynamic(out)
    b = find_nearest_dyn_obstacle(out, 0.0, np.array([ego_s, 0.0, 0.0, 0.0]), lap)
    return (None if a is None else a.s_center), (None if b is None else b[0])


def converge(tr, positions, lap, ego_s, frames=60):
    """Feed fixed (s, d, speed) obstacles until classified; return the last output."""
    for k in range(frames):
        t = k * DT
        out = tr.step([det((s + v * t) % lap, d, lap) for s, d, v in positions], t, ego_s)
    return out


def test_statics_first_then_dynamic_nearest_ahead_first(conv, lap):
    tr = ObstacleTracker(conv)
    out = converge(tr, [(30.0, 0.0, 3.0), (15.0, 0.5, 0.0), (20.0, -0.5, 3.0), (5.0, 0.0, 0.0)],
                   lap, ego_s=12.0)
    assert [o.is_static for o in out] == [True, True, False, False]
    assert [o.s_center for o in out[:2]] == sorted(o.s_center for o in out[:2])
    assert out[2].s_center < out[3].s_center      # 20 + ... is nearer ahead of 12 than 30 + ...


def test_one_opponent_among_cones_is_taken_by_both_consumers(conv, lap):
    tr = ObstacleTracker(conv)
    out = converge(tr, [(8.0, 0.4, 0.0), (11.0, 0.0, 2.0), (16.0, -0.4, 0.0)], lap, ego_s=10.0)
    a, b = picks(out, 10.0, lap)
    assert a is not None and a == b


def test_two_opponents_never_split_the_consumers(conv, lap):
    """With two dynamic obstacles no order makes both consumers take the same one. This one
    makes collision_prediction take the nearest ahead and opponent_trajectory take none."""
    tr = ObstacleTracker(conv)
    out = converge(tr, [(11.0, 0.3, 2.0), (13.0, -0.3, 2.0)], lap, ego_s=10.0)   # 3.7, 5.7 m ahead
    a, b = picks(out, 10.0, lap)
    near = min((o for o in out if not o.is_static), key=lambda o: (o.s_start - 10.0) % lap)
    assert a == near.s_center
    assert b is None


def test_a_car_behind_does_not_split_the_consumers(conv, lap):
    tr = ObstacleTracker(conv)
    out = converge(tr, [(11.0, 0.3, 2.0), (4.0, -0.3, 2.0)], lap, ego_s=10.0)
    a, b = picks(out, 10.0, lap)
    assert b is None or a == b
    assert a > 10.0


def test_without_an_ego_position_dynamic_obstacles_are_ordered_by_id(conv, lap):
    tr = ObstacleTracker(conv)
    tr.step([det(30.0)], 0.0)                       # starts first, confirmed first: id 0
    for k in range(1, 60):
        out = tr.step([det(30.0 + 2 * k * DT), det(20.0 + 2 * k * DT)], k * DT)
    assert [o.id for o in out] == sorted(o.id for o in out)
