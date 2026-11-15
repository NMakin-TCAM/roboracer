"""Predictive Spliner's dense GP port: upstream's behaviour, pinned quirk by quirk.

Synthetic: a 40 m circle at 0.1 m waypoint spacing, and half laps built the way
opponent_trajectory publishes them. No ROS.
"""
import hashlib
import math
import os
import warnings
from types import SimpleNamespace as NS

import numpy as np
import pytest

pytest.importorskip('sklearn')
from planning import dense_gp_opp_traj as dg  # noqa: E402
from planning.frenet import FrenetConverter  # noqa: E402

L = 40.0
N = 400                                    # waypoints, 0.1 m apart
SPACING = L / N
CCMA_SHA256 = 'ea60ae29ae95b756adc0548f2d06e30e140accd69db26bb937e40ad3397e8b03'


@pytest.fixture(autouse=True)
def quiet():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')     # sklearn's ConvergenceWarning on upstream's bounds
        yield


def circle():
    r = SPACING / (2 * math.sin(math.pi / N))      # chord exactly SPACING, so the lap is L
    th = np.arange(N) * 2 * math.pi / N
    xy = np.column_stack((r * np.cos(th), r * np.sin(th)))
    conv = FrenetConverter(xy)
    s = list(conv.s) + [conv.length]
    wpnts = [NS(s_m=float(s[i]), x_m=float(xy[i % N, 0]), y_m=float(xy[i % N, 1]), vx_mps=5.0)
             for i in range(N + 1)]
    return conv, wpnts


def make_gp(wpnts=None):
    conv, w = circle()
    gp = dg.DenseOpponentGP(conv.to_cartesian, conv.to_frenet)
    gp.set_global_waypoints(w if wpnts is None else wpnts)
    return gp, conv


def det(s, d=0.2, vs=3.0, vd=0.0, t=0.0):
    return NS(s=s, d=d, vs=vs, vd=vd, time=t)


def half_lap(s0, length=L / 2, step=0.12, d=lambda s: 0.2, vs=lambda s: 3.0, t0=0.0):
    """Detections every `step` from s0 over `length` of travel, s wrapped into the lap."""
    out, s, t = [], s0, t0
    while s < s0 + length:
        out.append(det(s % L, d(s), vs(s), t=t))
        s += step
        t += step / 3.0
    return out


def by_s(wpnts):
    return {round(w.s_m, 6): w for w in wpnts}


def nearest(wpnts, s):
    return min(wpnts, key=lambda w: abs(w.s_m - s))


# ------------------------------------------------------------------ vendored ccma

def test_vendored_ccma_is_the_pinned_upstream_file():
    path = os.path.join(os.path.dirname(dg.__file__), 'vendor', 'ccma', 'ccma.py')
    with open(os.path.realpath(path), 'rb') as f:
        assert hashlib.sha256(f.read()).hexdigest() == CCMA_SHA256
    from planning.vendor.ccma import CCMA
    assert dg.CCMA is CCMA


# ------------------------------------------------------------------ setup

def test_kernels_are_upstreams():
    kvs, kd = dg.make_kernels()
    assert repr(kvs) == '0.707**2 * RBF(length_scale=1) + 0.447**2 * WhiteKernel(noise_level=1)'
    assert repr(kd) == ('0.707**2 * Matern(length_scale=1, nu=1.5) + '
                        '0.447**2 * WhiteKernel(noise_level=1)')
    for k in (kvs, kd):
        assert k.k1.k1.constant_value_bounds == (1e-6, 1e3)
        assert k.k2.k1.constant_value_bounds == (1e-6, 1e3)
        assert k.k1.k2.length_scale_bounds == (1e-5, 1e5)       # sklearn's default, unbounded
        assert k.k2.k2.noise_level_bounds == (1e-5, 1e5)


def test_initial_trajectory_is_the_raceline_at_100_mps():
    gp, conv = make_gp()
    assert len(gp.oppwpnts) == N                 # the duplicated last waypoint is dropped
    x, y = conv.to_cartesian(gp.oppwpnts[7].s_m, 0.0)
    for w in gp.oppwpnts:
        assert (w.d_m, w.proj_vs_mps, w.vd_mps, w.d_var, w.vs_var, w.id) == (0, 100, 0, 0, 0, 0)
    assert (gp.oppwpnts[7].x_m, gp.oppwpnts[7].y_m) == pytest.approx((x, y))


def test_nothing_before_waypoints():
    conv, _ = circle()
    gp = dg.DenseOpponentGP(conv.to_cartesian, conv.to_frenet)
    assert not gp.ready
    assert gp.on_proj_trajectory(0.5, True, half_lap(5.05)) == []


def test_off_trajectory_and_empty_half_laps_are_ignored():
    gp, _ = make_gp()
    assert gp.on_proj_trajectory(0.5, False, half_lap(5.05)) == []
    assert gp.on_proj_trajectory(0.5, True, []) == []
    assert gp.first_half_lap and gp.sorted_detections == []


def test_waypoints_fixed_from_the_first_message_but_lap_length_follows():
    gp, _ = make_gp()
    _, w = circle()
    longer = [NS(s_m=p.s_m * 1.01, x_m=p.x_m, y_m=p.y_m, vx_mps=9.0) for p in w]
    gp.set_global_waypoints(longer)
    assert gp.track_length == pytest.approx(L * 1.01)
    assert gp.max_velocity == 5.0
    assert gp.ego_s_original[-1] == pytest.approx(w[-2].s_m)
    assert len(gp.oppwpnts) == N


# ------------------------------------------------------------------ half-lap mode

def test_first_half_lap_fills_its_window_and_leaves_the_rest_at_100():
    gp, _ = make_gp()
    pubs = gp.on_proj_trajectory(0.5, True, half_lap(5.05, d=lambda s: 0.3))
    assert len(pubs) == 1 and pubs[0].lap_count == 0.5 and pubs[0].stamp_of is None
    w = by_s(pubs[0].oppwpnts)
    inside = [p for s, p in w.items() if 5.05 < s < 25.05]      # starts between waypoints
    outside = [p for s, p in w.items() if s <= 5.05 or s >= 25.05]
    assert inside and all(p.proj_vs_mps != 100 for p in inside)
    assert all(p.proj_vs_mps == 100 and p.d_m == 0 for p in outside)
    mid = nearest(pubs[0].oppwpnts, 15.0)
    assert mid.d_m == pytest.approx(0.3, abs=0.05)
    assert mid.proj_vs_mps == pytest.approx(3.0, abs=0.1)


def test_first_window_is_strictly_inside_first_detection_to_half_a_lap_on():
    gp, _ = make_gp()
    s0 = gp.ego_s_original[50]
    gp.on_proj_trajectory(0.5, True, half_lap(s0))
    ego = gp.ego_s_doublelap
    assert gp.first_ego_s == ego[51]                   # s0 itself is excluded
    assert gp.last_ego_s == max(s for s in ego if s < s0 + gp.track_length / 2)
    assert gp.last_ego_s < s0 + gp.track_length / 2


def test_second_half_lap_window_is_the_complement_wherever_its_detections_are():
    gp, _ = make_gp()
    gp.on_proj_trajectory(0.5, True, half_lap(5.05))
    # Detections only between 30 and 35 m, but the window is [25.0, 45.1] regardless.
    pubs = gp.on_proj_trajectory(1.0, True, half_lap(30.0, length=5.0))
    assert (gp.first_ego_s, gp.last_ego_s) == pytest.approx((25.0, 45.1), abs=1e-6)
    w = pubs[0].oppwpnts
    assert nearest(w, 26.0).proj_vs_mps != 100
    # The window's 40 .. 45.1 is never wrapped (these detections do not cross the line),
    # so the first 5 m of the lap keep 100 until the whole-lap fit.
    assert nearest(w, 4.0).proj_vs_mps == 100


def wavy(s):
    return 3.0 + 0.5 * math.sin(2 * math.pi * s / 2.0)


def sklearn_fit(kernel, dets, y, s_pred):
    """The same fit, done here directly, as the reference."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    gpr = GaussianProcessRegressor(kernel=kernel, optimizer=dg._optimizer)
    gpr.fit(np.array([d.s for d in dets]).reshape(-1, 1), np.array(y).reshape(-1, 1))
    return gpr.predict(np.array(s_pred).reshape(-1, 1), return_std=True)


def test_half_lap_means_fall_back_to_zero_away_from_the_detections():
    # normalize_y is False: away from the data the GP mean returns to 0. Here that is
    # inside the window, so an opponent seen over a short stretch is predicted to stop.
    gp, _ = make_gp()
    pubs = gp.on_proj_trajectory(0.5, True, half_lap(5.05, length=4.0, vs=wavy))
    far = nearest(pubs[0].oppwpnts, 22.0)
    assert abs(far.proj_vs_mps) < 0.5
    assert nearest(pubs[0].oppwpnts, 7.0).proj_vs_mps == pytest.approx(wavy(7.0), abs=0.3)


def test_half_lap_variances_are_standard_deviations():
    gp, _ = make_gp()
    dets = half_lap(5.05, length=4.0, vs=wavy)
    vs = [d.vs for d in dets]
    pubs = gp.on_proj_trajectory(0.5, True, dets)
    w = [nearest(pubs[0].oppwpnts, s) for s in (7.0, 15.0, 22.0)]
    _, std = sklearn_fit(dg.make_kernels()[0], dets, vs, [p.s_m for p in w])
    assert [p.vs_var for p in w] == pytest.approx(list(std), rel=1e-9)
    assert w[0].vs_var < w[2].vs_var and w[0].d_var < w[2].d_var


def test_half_lap_across_the_start_line_is_shifted_then_wrapped_back():
    gp, _ = make_gp()
    dets = half_lap(30.0)                           # 30 .. 50, i.e. past 40 back to 10
    gp.on_proj_trajectory(0.5, True, dets)
    assert max(d.s for d in dets) < L               # wrapped back after publishing
    assert gp.sorted_detections == sorted(gp.sorted_detections, key=lambda d: d.s)


def test_half_lap_window_past_the_lap_is_dropped_without_a_start_line_crossing():
    # First half lap from 30 m: the window runs to 50 m. The detections stop at 39 m,
    # so they never decrease and nothing wraps: global s 40..50 match no waypoint.
    gp, _ = make_gp()
    pubs = gp.on_proj_trajectory(0.5, True, half_lap(30.0, length=9.0))
    w = pubs[0].oppwpnts
    assert nearest(w, 35.0).proj_vs_mps != 100
    assert all(p.proj_vs_mps == 100 for p in w if p.s_m < 9.9)


# ------------------------------------------------------------------ lapcount == 1

def two_half_laps(gp, d=lambda s: 0.2, step=0.12):
    first = half_lap(5.05, d=d, step=step)
    second = half_lap(first[-1].s, d=d, step=step, t0=first[-1].time)
    second[0] = first[-1]                       # consecutive half laps share a point
    p1 = gp.on_proj_trajectory(0.5, True, first)
    p2 = gp.on_proj_trajectory(1.0, True, second)
    return first, second, p1, p2


def test_lap_one_publishes_twice_and_the_second_keeps_lap_one_and_the_stamp():
    gp, _ = make_gp()
    _, _, _, pubs = two_half_laps(gp)
    assert [p.lap_count for p in pubs] == [1.0, 1.0]
    assert pubs[1].stamp_of == 0
    assert any(p.d_var != 0 for p in pubs[0].oppwpnts)       # the half lap's GP d
    assert all(p.d_var == 0 for p in pubs[1].oppwpnts)       # CCMA d
    assert all(p.proj_vs_mps != 100 for p in pubs[1].oppwpnts)
    assert gp.lap_count == 1.1


def test_lap_one_counts_the_last_half_lap_twice():
    gp, _ = make_gp()
    first, second, _, _ = two_half_laps(gp, step=0.4)
    # first + second (the shared point is one object, listed in both) + second again
    assert len(gp.sorted_detections) == len(first) + 2 * len(second) <= dg.THIN_ABOVE
    assert sum(d is second[5] for d in gp.sorted_detections) == 2


def test_whole_lap_d_is_ccma_smoothed_and_follows_the_opponent():
    gp, _ = make_gp()
    d = lambda s: 0.3 * math.sin(2 * math.pi * s / L)
    _, _, _, pubs = two_half_laps(gp, d=d)
    for s in (5.0, 12.0, 30.0):
        assert nearest(pubs[1].oppwpnts, s).d_m == pytest.approx(d(s), abs=0.03)


# ------------------------------------------------------------------ whole-lap mode

def test_whole_lap_wraps_every_detection_but_the_first():
    dets = [det(41.0), det(42.0), det(43.0)]
    out, around = dg.sort_detections(dets, [], 1.5, L, N)
    assert [d.s for d in dets] == pytest.approx([41.0, 2.0, 3.0]) and not around
    assert [d.s for d in out] == pytest.approx([2.0, 3.0, 41.0])


def test_thinning_keeps_the_newest_per_bin_and_drops_the_top_bins():
    delta = 2 * L / N                                   # 0.2 m
    dets = []
    for k in range(int(L / 0.1)):                       # two per bin, the second newer
        dets.append(det(k * 0.1 + 0.01, t=float(k)))
    out, _ = dg.sort_detections(dets, [], 1.5, L, N)
    assert len(dets) > dg.THIN_ABOVE
    times = [d.time for d in out]
    assert all(t % 2 == 1 for t in times)               # the newer of each pair
    # Bins stop at the last x with (x + 1) * delta < max s: [x - 1, x) * delta.
    assert max(d.s for d in out) < L - 2 * delta + 1e-9
    assert max(d.s for d in dets) > L - delta


def test_whole_lap_vs_var_is_the_sigma_of_the_point_3_m_earlier():
    # Detections only between 10 and 20 m, so none is copied across the start line, but
    # the prediction s list gets the lap's last 3 m prepended (as negative s). The vs mean
    # is cut back to the lap; sigma is not, so waypoint j gets the sigma of the j-th
    # point of the longer list: the one about 3 m (29 or 30 waypoints) before it.
    gp, _ = make_gp()
    dets = half_lap(10.0, length=10.0, step=0.1, vs=wavy)
    vs = [d.vs for d in dets]
    pubs = gp.on_proj_trajectory(1.5, True, dets)
    ego = list(gp.ego_s_original)
    nr = dg.wrap_training([], [], [], [], ego, gp.track_length)
    assert nr in (29, 30)       # 37.0 m is exactly 3 m from the end, give or take a float
    mean, std = sklearn_fit(dg.make_kernels()[0], dets, vs, ego)
    w = pubs[0].oppwpnts
    assert [p.vs_var for p in w] == pytest.approx(list(std[:len(w)]), rel=1e-9)
    assert [p.proj_vs_mps for p in w] == pytest.approx(list(mean[nr:nr + len(w)]), rel=1e-9)
    assert w[180].vs_var < w[100].vs_var             # 18 m shows the sigma at 15 m, in the data


def test_whole_lap_copies_the_3_m_either_side_of_the_start_line():
    opp_s = [0.5, 1.0, 20.0, 38.5, 39.5]
    opp_d, opp_vs, opp_vd = [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]
    ego = [i * 0.5 for i in range(80)]
    nr = dg.wrap_training(opp_s, opp_d, opp_vs, opp_vd, ego, L)
    assert opp_s == pytest.approx([-1.5, -0.5, 0.5, 1.0, 20.0, 38.5, 39.5, 40.5, 41.0])
    assert opp_vs == [4, 5, 1, 2, 3, 4, 5, 1, 2]
    assert nr == 5 and ego[:nr] == pytest.approx([-2.5, -2.0, -1.5, -1.0, -0.5])
    assert ego[-6:] == pytest.approx([40.0, 40.5, 41.0, 41.5, 42.0, 42.5])


# ------------------------------------------------------------------ deviations

def test_half_lap_with_a_nan_speed_publishes_nothing_and_drops_the_point():
    gp, _ = make_gp()
    dets = half_lap(30.0)
    dets[3].vs = float('nan')
    assert gp.on_proj_trajectory(0.5, True, dets) == []      # upstream: ValueError, dead
    assert all(np.isfinite(d.vs) for d in gp.sorted_detections)
    assert len(gp.sorted_detections) == len(dets) - 1
    assert max(d.s for d in gp.sorted_detections) < L        # wrapped back, as upstream would
    pubs = gp.on_proj_trajectory(1.0, True, half_lap(10.0))
    assert len(pubs) == 2


def test_nan_at_lap_one_publishes_nothing():
    gp, _ = make_gp()
    first = half_lap(5.05)
    gp.on_proj_trajectory(0.5, True, first)
    second = half_lap(first[-1].s, t0=first[-1].time)
    second[1].vs = float('nan')
    pubs = gp.on_proj_trajectory(1.0, True, second)
    # The half lap's own fit raises first; nothing goes out.
    assert pubs == []


def test_a_nan_vd_does_not_stop_anything():
    gp, _ = make_gp()
    dets = half_lap(5.05)
    dets[4].vd = float('nan')
    pubs = gp.on_proj_trajectory(0.5, True, dets)
    assert len(pubs) == 1


def test_marker_height_is_zero_without_a_top_speed():
    _, w = circle()
    for p in w:
        p.vx_mps = 0.0
    gp, _ = make_gp(w)
    pubs = gp.on_proj_trajectory(0.5, True, half_lap(5.05))
    assert all(m.height == 0.0 for m in pubs[0].markers)


def test_markers_follow_speed_and_d_var():
    gp, _ = make_gp()
    pubs = gp.on_proj_trajectory(0.5, True, half_lap(5.05))
    for w, m in zip(pubs[0].oppwpnts, pubs[0].markers):
        assert m.height == pytest.approx(w.proj_vs_mps / 5.0)
        assert m.width == pytest.approx(min(max(5 * w.d_var, 0.07), 0.7))
        assert (m.x, m.y) == (w.x_m, w.y_m)


def test_publication_is_a_snapshot():
    gp, _ = make_gp()
    pubs = gp.on_proj_trajectory(0.5, True, half_lap(5.05))
    before = [p.proj_vs_mps for p in pubs[0].oppwpnts]
    gp.on_proj_trajectory(1.0, True, half_lap(25.0))
    assert [p.proj_vs_mps for p in pubs[0].oppwpnts] == before
