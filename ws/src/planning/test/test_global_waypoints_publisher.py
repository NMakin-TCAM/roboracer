"""raceline.csv -> Wpnt conversion and the state stub's validation. No ROS: plain pytest."""
import os
import re

import numpy as np
import pytest

from planning.frenet import FrenetConverter
from planning.global_waypoints_publisher import (WPNT_FIELDS, load_global_waypoints,
                                                 provenance_warning)
from planning.raceline import write_raceline
from planning.state_machine_stub import KNOWN_STATES, validate_state

WPNT_MSG = os.path.join(os.path.dirname(__file__), '..', '..', 'f1t_msgs', 'msg', 'Wpnt.msg')

A, B = 6.0, 3.5                 # centerline ellipse semi-axes, m
N = 400
W_RIGHT, W_LEFT = 0.9, 0.8      # centerline widths
OFFSET = 0.2                    # the raceline runs this far left of the centerline
CLEAN = {'method': 'mintime', 'clean': True, 'notes': []}


def ellipse(n=N):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack((A * np.cos(th), B * np.sin(th)))   # anticlockwise


def left_normals(xy):
    t = np.roll(xy, -1, axis=0) - np.roll(xy, 1, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    return np.column_stack((-t[:, 1], t[:, 0]))


def yaw_of(xy):
    t = np.roll(xy, -1, axis=0) - np.roll(xy, 1, axis=0)
    return np.arctan2(t[:, 1], t[:, 0])


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def write_track(d, psi='tum', closed=True, vx=None, ax=None, kappa=None, offset=OFFSET,
                centerline=True, provenance=CLEAN):
    """An ellipse track as build_raceline writes it: TUM psi, closed, provenance line."""
    c = ellipse()
    if centerline:
        np.savetxt(d / 'centerline.csv',
                   np.column_stack((c, np.full(N, W_RIGHT), np.full(N, W_LEFT))),
                   delimiter=',', header='x_m,y_m,w_tr_right_m,w_tr_left_m')
    xy = c + offset * left_normals(c)
    seg = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg)[:-1]))
    yaw = yaw_of(xy)
    psi_col = {'tum': wrap(yaw - np.pi / 2), 'ros': yaw, 'garbage': wrap(yaw + 1.3)}[psi]
    if kappa is None:
        kappa = (A * B) / ((A * np.sin(np.linspace(0, 2 * np.pi, N, endpoint=False))) ** 2
                           + (B * np.cos(np.linspace(0, 2 * np.pi, N, endpoint=False))) ** 2) ** 1.5
    vx = 3.0 + np.cos(np.linspace(0, 4 * np.pi, N, endpoint=False)) if vx is None else vx
    ax = (np.roll(vx, -1) ** 2 - vx ** 2) / (2 * seg) if ax is None else ax
    traj = np.column_stack((s, xy, psi_col, np.broadcast_to(kappa, (N,)),
                            np.broadcast_to(vx, (N,)), np.broadcast_to(ax, (N,))))
    if closed:
        traj = np.vstack((traj, traj[0]))
        traj[-1, 0] = s[-1] + seg[-1]
    write_raceline(str(d / 'raceline.csv'), traj, provenance)
    return traj


def column(g, field):
    return np.array([getattr(w, field) for w in g.waypoints])


# ------------------------------------------------------------------ fields


def test_field_list_is_exactly_wpnt_msg():
    """Read from Wpnt.msg itself, not assumed."""
    with open(WPNT_MSG) as f:
        declared = [m.group(1) for line in f
                    if (m := re.match(r'^\s*\w+\s+(\w+)\s*$', line.split('#')[0]))]
    assert tuple(declared) == WPNT_FIELDS


def test_every_field_is_populated_from_a_build_raceline_file(tmp_path):
    traj = write_track(tmp_path)
    g = load_global_waypoints(str(tmp_path))
    assert g.warnings == []
    assert len(g.waypoints) == len(traj)            # closing row kept
    for f in WPNT_FIELDS:
        assert np.all(np.isfinite(column(g, f))), f
    assert list(column(g, 'id')) == list(range(len(traj)))
    np.testing.assert_allclose(column(g, 's_m'), traj[:, 0], atol=1e-6)
    np.testing.assert_allclose(column(g, 'x_m'), traj[:, 1], atol=1e-6)
    np.testing.assert_allclose(column(g, 'y_m'), traj[:, 2], atol=1e-6)
    np.testing.assert_allclose(column(g, 'kappa_radpm'), traj[:, 4], atol=1e-6)
    np.testing.assert_allclose(column(g, 'vx_mps'), traj[:, 5], atol=1e-6)
    np.testing.assert_allclose(column(g, 'ax_mps2'), traj[:, 6], atol=1e-6)


def test_closing_row_is_kept_so_the_last_s_is_the_lap_length(tmp_path):
    """opponent_trajectory and the predictor take wpnts[-1].s_m as the track length."""
    traj = write_track(tmp_path)
    g = load_global_waypoints(str(tmp_path))
    w0, wn = g.waypoints[0], g.waypoints[-1]
    assert wn.s_m == pytest.approx(traj[-1, 0])
    assert (wn.x_m, wn.y_m) == (w0.x_m, w0.y_m)
    for f in ('d_right', 'd_left', 'psi_rad', 'kappa_radpm', 'vx_mps', 'ax_mps2'):
        assert getattr(wn, f) == getattr(w0, f), f


def test_d_m_is_zero_because_the_waypoints_are_the_reference(tmp_path):
    write_track(tmp_path)
    assert np.all(column(load_global_waypoints(str(tmp_path)), 'd_m') == 0.0)


def test_widths_are_positive_distances_measured_from_the_raceline(tmp_path):
    """centerline.csv widths are about the centerline; the raceline runs OFFSET left of it."""
    write_track(tmp_path)
    g = load_global_waypoints(str(tmp_path))
    np.testing.assert_allclose(column(g, 'd_left'), W_LEFT - OFFSET, atol=0.02)
    np.testing.assert_allclose(column(g, 'd_right'), W_RIGHT + OFFSET, atol=0.02)


def test_psi_is_ros_yaw_not_tum(tmp_path):
    traj = write_track(tmp_path, psi='tum')
    g = load_global_waypoints(str(tmp_path))
    yaw = yaw_of(traj[:-1, 1:3])
    err = wrap(column(g, 'psi_rad')[:-1] - yaw)
    assert np.max(np.abs(err)) < 1e-6
    assert np.all(np.abs(column(g, 'psi_rad')) <= np.pi)


# ------------------------------------------------------------------ loud, not silent


def test_ros_yaw_file_is_accepted_with_a_warning(tmp_path):
    traj = write_track(tmp_path, psi='ros')
    g = load_global_waypoints(str(tmp_path))
    assert any('already ROS yaw' in w for w in g.warnings)
    np.testing.assert_allclose(wrap(column(g, 'psi_rad')[:-1] - yaw_of(traj[:-1, 1:3])), 0,
                               atol=1e-6)


def test_heading_in_neither_convention_is_refused(tmp_path):
    write_track(tmp_path, psi='garbage')
    with pytest.raises(ValueError, match='neither convention'):
        load_global_waypoints(str(tmp_path))


def test_unclosed_file_gets_its_closing_row_and_a_warning(tmp_path):
    traj = write_track(tmp_path, closed=False)
    g = load_global_waypoints(str(tmp_path))
    assert len(g.waypoints) == len(traj) + 1
    assert any('not closed' in w for w in g.warnings)
    seg = np.hypot(*(traj[0, 1:3] - traj[-1, 1:3]))
    assert g.lap_length == pytest.approx(traj[-1, 0] + seg)


def test_zero_curvature_is_computed_from_geometry_and_said(tmp_path):
    write_track(tmp_path, kappa=0.0, offset=0.0)
    g = load_global_waypoints(str(tmp_path))
    assert any('kappa_radpm is zero' in w for w in g.warnings)
    k = column(g, 'kappa_radpm')
    # ellipse curvature runs from B/A^2 (flat sides) to A/B^2 (tight ends)
    assert k.min() == pytest.approx(B / A ** 2, rel=0.1)
    assert k.max() == pytest.approx(A / B ** 2, rel=0.1)


@pytest.mark.parametrize('vx', [0.0, -1.0, np.nan])
def test_unusable_speeds_are_refused_not_zero_filled(tmp_path, vx):
    write_track(tmp_path, vx=vx, ax=0.0)
    with pytest.raises(ValueError, match='vx_mps'):
        load_global_waypoints(str(tmp_path))


def test_zero_acceleration_with_varying_speed_is_computed_and_said(tmp_path):
    traj = write_track(tmp_path)
    expect = traj[:-1, 6].copy()
    write_track(tmp_path, ax=0.0)
    g = load_global_waypoints(str(tmp_path))
    assert any('ax_mps2 is zero' in w for w in g.warnings)
    np.testing.assert_allclose(column(g, 'ax_mps2')[:-1], expect, atol=1e-5)


def test_zero_acceleration_at_constant_speed_is_correct_and_silent(tmp_path):
    write_track(tmp_path, vx=4.0, ax=0.0)
    g = load_global_waypoints(str(tmp_path))
    assert g.warnings == []
    assert np.all(column(g, 'ax_mps2') == 0.0)


def test_missing_centerline_is_an_error_naming_the_widths(tmp_path):
    write_track(tmp_path, centerline=False)
    with pytest.raises(FileNotFoundError, match='widths'):
        load_global_waypoints(str(tmp_path))


def test_missing_raceline_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match='build_raceline'):
        load_global_waypoints(str(tmp_path))


def test_raceline_outside_the_track_is_said(tmp_path):
    write_track(tmp_path, offset=W_LEFT + 0.1)       # past the left edge everywhere
    g = load_global_waypoints(str(tmp_path))
    assert any('outside the track' in w for w in g.warnings)


def test_provenance_warning():
    assert provenance_warning(CLEAN) is None
    assert 'not a clean' in provenance_warning({'method': 'mintime-retry', 'clean': False})
    assert 'no provenance' in provenance_warning(None)


def test_provenance_is_read_from_the_header(tmp_path):
    write_track(tmp_path, provenance={'method': 'mintime-retry', 'clean': False})
    assert load_global_waypoints(str(tmp_path)).provenance['method'] == 'mintime-retry'


# ------------------------------------------------------------------ state stub


@pytest.mark.parametrize('state', KNOWN_STATES)
def test_known_states_pass(state):
    assert validate_state(state) == state


@pytest.mark.parametrize('state', ['overtake', 'Trailing', '', 'FOLLOW'])
def test_unknown_states_are_refused(state):
    with pytest.raises(ValueError, match='unknown state'):
        validate_state(state)


def test_the_gated_states_are_the_ones_the_ports_compare_against():
    """By the ports' code: update_waypoints gates on GB_TRACK, opponent_trajectory on OVERTAKE."""
    from planning.opponent_trajectory import OpponentTrajectoryTracker
    from planning.update_waypoints import GB_TRACK
    assert GB_TRACK in KNOWN_STATES
    tr = OpponentTrajectoryTracker()
    tr.set_state('OVERTAKE')
    assert tr.overtake
    tr.set_state('TRAILING')
    assert not tr.overtake


# ------------------------------------------------------------------ converter agrees


def test_frenet_converter_and_waypoints_agree_on_heading(tmp_path):
    """FrenetConverter.from_track used to read TUM psi as ROS yaw (a quarter turn off); this
    was the strict xfail that pinned it. Both now convert with frenet.yaw_from_csv_psi."""
    traj = write_track(tmp_path, psi='tum')
    conv = FrenetConverter.from_track(str(tmp_path))
    err = wrap(conv.heading_at(traj[:-1, 0]) - yaw_of(traj[:-1, 1:3]))
    assert np.max(np.abs(err)) < 0.05
    g = load_global_waypoints(str(tmp_path))
    assert np.max(np.abs(wrap(conv.heading_at(column(g, 's_m')) - column(g, 'psi_rad')))) < 1e-6
