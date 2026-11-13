"""Return predictor, checked on upstream's selection rules and where it writes its result.

The GP's numbers are not worked by hand (upstream's own code is the reference for
those, in test_upstream_differential.py). These tests pin the deterministic parts:
which points the GP is fitted to, which waypoints it overwrites and with what, and
when nothing happens. The track is 20 m with waypoints every 0.5 m, and the stored
trajectory has one waypoint on each, as the main GP lays it out.
"""
import importlib.util

import numpy as np
import pytest

# Not pytest.importorskip at module level: under the image's pytest 6.2.5, a
# module-level skip during collection left the whole session with nothing
# collected. A per-test mark skips only these tests.
HAVE_SKLEARN = importlib.util.find_spec('sklearn') is not None
pytestmark = pytest.mark.skipif(not HAVE_SKLEARN, reason='scikit-learn is not installed')

if HAVE_SKLEARN:
    from planning.predictor_opponent_trajectory import (
        PREDICTED, ReturnPredictor, double_lap, double_trajectory, make_kernel,
        marker_records, merge_points, unwrap_detections)

L = 20.0
SPACING = 0.5
N = int(L / SPACING)      # 40 waypoints, plus the duplicate at s = 20
TOP = 8.0


class FakeWpnt:
    """Stands in for f1t_msgs/Wpnt."""

    def __init__(self, s_m, vx_mps=TOP):
        self.s_m = s_m
        self.vx_mps = vx_mps


class FakeOppWpnt:
    """Stands in for f1t_msgs/OppWpnt."""

    def __init__(self, s_m, d_m=0.0):
        self.s_m = s_m
        self.d_m = d_m
        self.x_m = self.y_m = 0.0
        self.proj_vs_mps = 5.0
        self.vd_mps = 0.0
        self.d_var = 0.05
        self.vs_var = 0.3


class FakePoint:
    """Stands in for f1t_msgs/ProjOppPoint."""

    def __init__(self, s, d=0.6, vd=0.0):
        self.s, self.d, self.vd = s, d, vd


def to_xy(s, d):
    return np.asarray(s, float), np.asarray(d, float)


def trajectory():
    return [FakeOppWpnt(i * SPACING) for i in range(N)]


def predictor(traj=None, top=TOP):
    pr = ReturnPredictor()
    pr.set_global_waypoints([FakeWpnt(i * SPACING, top) for i in range(N + 1)])
    pr.set_opponent_trajectory(trajectory() if traj is None else traj)
    return pr


def detections(s0, n=15, step=0.1, d=0.6, vd=0.0):
    return [FakePoint((s0 + i * step) % L, d, vd) for i in range(n)]


def by_s(ws):
    return {round(w.s_m, 6): w for w in ws}


# ------------------------------------------------------------------ selection

def test_double_lap_drops_the_duplicate_and_repeats_one_lap_on():
    assert double_lap([0.0, 1.0, 2.0, 3.0], 3.0) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_second_lap_copies_carry_no_variances():
    traj = trajectory()
    dl = double_trajectory(traj, L)
    assert dl[:N] == traj
    assert dl[N].s_m == L and dl[N].d_m == traj[0].d_m
    assert (dl[N].d_var, dl[N].vs_var) == (0.0, 0.0)


def test_unwrap_adds_a_lap_from_the_first_big_step_onwards():
    dets = [FakePoint(s) for s in (19.0, 19.5, 0.2, 0.6)]
    unwrap_detections(dets, L)
    assert [p.s for p in dets] == pytest.approx([19.0, 19.5, 20.2, 20.6])


def test_unwrap_keeps_shifting_once_it_has_started():
    """12 is only 9 m from the shifted 21, but everything after the wrap gets a lap."""
    dets = [FakePoint(s) for s in (15.0, 1.0, 12.0)]
    unwrap_detections(dets, L)
    assert [p.s for p in dets] == pytest.approx([15.0, 21.0, 32.0])


def test_unwrap_also_shifts_a_forward_jump_of_more_than_half_a_lap():
    dets = [FakePoint(s) for s in (5.0, 5.5, 16.0, 16.5)]
    unwrap_detections(dets, L)
    assert [p.s for p in dets] == pytest.approx([5.0, 5.5, 36.0, 36.5])


def test_merge_takes_the_last_15_detections_and_15_points_7m_beyond():
    dets = detections(1.0, n=20)                    # s = 1.0 .. 2.9
    s, d, vd = merge_points(dets, double_trajectory(trajectory(), L))
    assert s[:15] == pytest.approx([1.5 + 0.1 * i for i in range(15)])
    assert s[15:] == pytest.approx([10.0 + 0.5 * i for i in range(15)])    # > 2.9 + 7
    assert d[:15] == [0.6] * 15 and d[15:] == [0.0] * 15


def test_merge_takes_trajectory_points_in_list_order():
    traj = trajectory()[::-1]
    s, _, _ = merge_points(detections(1.0), double_trajectory(traj, L))
    assert s[15:] == pytest.approx([19.5 - 0.5 * i for i in range(15)])


def test_merge_near_the_end_of_the_second_lap_finds_fewer_points():
    dets = [FakePoint(30.0 + 0.1 * i) for i in range(10)]          # already unwrapped
    s, _, _ = merge_points(dets, double_trajectory(trajectory(), L))
    assert s[10:] == pytest.approx([38.0, 38.5, 39.0, 39.5])      # beyond 30.9 + GAP
    s, _, _ = merge_points([FakePoint(35.0)], double_trajectory(trajectory(), L))
    assert s == [35.0]


# ------------------------------------------------------------------ prediction

def test_waypoints_in_the_span_are_replaced_and_marked():
    """Detections 1.0 .. 2.4, trajectory points 9.5 .. 16.5: waypoints 1.0 .. 16.5 change."""
    pr = predictor()
    pred = pr.on_proj_trajectory(1.5, False, detections(1.0), to_xy)
    ws = by_s(pred.oppwpnts)
    changed = sorted(s for s, w in ws.items() if w.vs_var == PREDICTED)
    assert changed == pytest.approx([1.0 + 0.5 * i for i in range(32)])
    for s in (0.5, 17.0):
        assert ws[s].vs_var == 0.3 and ws[s].d_m == 0.0 and ws[s].d_var == 0.05
    w = ws[1.5]
    assert w.proj_vs_mps == 5.0                         # vs is not predicted
    assert isinstance(w.d_m, float) and w.d_var > 0     # d_var holds a standard deviation
    assert ws[1.5].d_m > ws[16.0].d_m                   # from the detections' 0.6 towards 0
    assert (w.x_m, w.y_m) == (1.5, w.d_m)               # placed by to_xy
    assert pred.lap_count == 1.5


def test_vd_is_interpolated_linearly_across_the_merged_points():
    """vd 1.0 on the detections (to 2.4 m), 0 on the trajectory (from 9.5 m)."""
    pr = predictor()
    ws = by_s(pr.on_proj_trajectory(1.0, False, detections(1.0, vd=1.0), to_xy).oppwpnts)
    assert ws[1.5].vd_mps == pytest.approx(1.0)
    assert ws[5.0].vd_mps == pytest.approx(1 - 2.6 / 7.1)
    assert ws[12.0].vd_mps == pytest.approx(0.0)


def test_stored_trajectory_is_edited_in_place():
    pr = predictor()
    stored = pr.oppwpnts
    pred = pr.on_proj_trajectory(1.0, False, detections(1.0), to_xy)
    assert pred.oppwpnts is stored and stored[3].vs_var == PREDICTED


def test_detections_are_unwrapped_in_the_message():
    pr = predictor()
    dets = detections(19.5, n=10)                       # 19.5 .. 20.4 -> wraps to 0.0 .. 0.4
    pr.on_proj_trajectory(1.0, False, dets, to_xy)
    assert dets[-1].s == pytest.approx(20.4)


def test_a_return_past_the_start_line_lands_at_the_start():
    """Detections 17.0 .. 18.4 merge second-lap points 25.5 .. 32.5, i.e. 5.5 .. 12.5."""
    pr = predictor()
    ws = by_s(pr.on_proj_trajectory(1.0, False, detections(17.0), to_xy).oppwpnts)
    assert ws[5.0].vs_var == PREDICTED and ws[12.5].vs_var == PREDICTED
    assert ws[13.0].vs_var == 0.3 and ws[16.5].vs_var == 0.3


def test_waypoint_not_on_a_global_s_is_left_alone():
    traj = trajectory()
    traj[4].s_m += 1e-6
    pr = predictor(traj)
    pr.on_proj_trajectory(1.0, False, detections(1.0), to_xy)
    assert traj[4].vs_var == 0.3 and traj[5].vs_var == PREDICTED


def test_on_trajectory_and_empty_half_laps_are_ignored():
    pr = predictor()
    assert pr.on_proj_trajectory(1.0, True, detections(1.0), to_xy) is None
    assert pr.on_proj_trajectory(1.0, False, [], to_xy) is None


def test_nothing_before_waypoints_and_a_trajectory():
    pr = ReturnPredictor()
    assert pr.on_proj_trajectory(1.0, False, detections(1.0), to_xy) is None
    pr.set_global_waypoints([FakeWpnt(i * SPACING) for i in range(N + 1)])
    assert pr.on_proj_trajectory(1.0, False, detections(1.0), to_xy) is None
    pr.set_opponent_trajectory([])
    assert pr.on_proj_trajectory(1.0, False, detections(1.0), to_xy) is not None


def test_span_with_no_global_s_is_skipped():
    """One detection between waypoints and nothing to merge: upstream raised ValueError."""
    pr = predictor(traj=[])
    assert pr.on_proj_trajectory(1.0, False, [FakePoint(1.2)], to_xy) is None


def test_waypoints_fixed_from_the_first_message_but_lap_length_follows():
    pr = predictor()
    pr.set_global_waypoints([FakeWpnt(i * 0.25, 1.0) for i in range(N + 1)])
    assert pr.track_length == 10.0
    assert pr.max_velocity == TOP and pr.ego_s_doublelap[-1] == pytest.approx(2 * L - SPACING)


# ------------------------------------------------------------------ markers and kernel

def test_markers_follow_speed_and_variance():
    ws = trajectory()[:3]
    ws[0].d_var, ws[1].d_var, ws[2].d_var = 0.001, 0.1, 1.0
    ws[1].vs_var = PREDICTED
    recs = marker_records(ws, TOP)
    assert [r.width for r in recs] == pytest.approx([0.07, 0.5, 0.7])
    assert recs[0].height == pytest.approx(5.0 / TOP)
    assert [r.predicted for r in recs] == [False, True, False]


def test_marker_height_is_zero_without_a_top_speed():
    """Upstream divided by zero here."""
    assert marker_records(trajectory()[:1], 0.0)[0].height == 0.0


def test_kernel_is_upstreams():
    p = make_kernel().get_params()
    assert p['k1__k1__constant_value'] == 0.5 and p['k2__k1__constant_value'] == 0.2
    assert p['k1__k1__constant_value_bounds'] == (1e-6, 1e3)
    assert p['k1__k2__length_scale'] == 1.0 and p['k1__k2__nu'] == 1.5
    assert p['k2__k2__noise_level'] == 1
