"""Centerline extraction on a real Cartographer map (fixtures/track_map_v3: ~16 x 9 m, ~30 m loop)."""
import os
from types import SimpleNamespace

import numpy as np
import pytest

from planning.centerline import (OccupancyMap, extract_centerline, fill_unobserved, first_lap, lap_length,
                                 normals_left, signed_area, write_centerline)

FIX = os.path.join(os.path.dirname(__file__), 'fixtures')
MAP = os.path.join(FIX, 'track_map_v3.yaml')


@pytest.fixture(scope='module')
def result():
    return extract_centerline(MAP, step=0.1)


def test_map_frame_roundtrip():
    m = OccupancyMap(MAP)
    rc = np.array([[0, 0], [m.h - 1, m.w - 1], [10, 20]])
    assert np.allclose(m.to_pixel(m.to_world(rc)), rc)
    # bottom-left pixel centre is half a cell from the origin
    assert np.allclose(m.to_world([[m.h - 1, 0]]), m.origin[:2] + 0.5 * m.res)


def test_closed_sane_loop(result):
    track, m, mask = result
    xy = track[:, :2]
    seg = np.hypot(*(np.roll(xy, -1, axis=0) - xy).T)
    length = seg.sum()
    assert 20.0 < length < 60.0
    assert seg.max() < 0.15 and seg.min() > 0.05          # uniform ~0.1 m, closing segment included
    assert np.all(track[:, 2] > 0.2) and np.all(track[:, 3] > 0.2)
    width = track[:, 2] + track[:, 3]
    assert 0.8 < np.median(width) < 3.0
    # every point is on free space of the selected region
    rc = np.round(m.to_pixel(xy)).astype(int)
    assert mask[rc[:, 0], rc[:, 1]].all()


def test_default_direction_and_start(result):
    track = result[0]
    assert signed_area(track[:, :2]) > 0                  # counter-clockwise without a trajectory
    assert np.hypot(*track[0, :2]) < 0.5                  # starts near the map origin


def test_follows_trajectory(tmp_path, result):
    track = result[0]
    # a clockwise "mapping lap" starting elsewhere on the loop
    lap = track[::-1][::5, :2]
    lap = np.roll(lap, 20, axis=0)
    t = np.arange(len(lap)) * 0.1
    traj_csv = tmp_path / 'trajectory.csv'
    np.savetxt(traj_csv, np.column_stack((t, lap, np.zeros(len(lap)))), delimiter=',',
               header='t,x_m,y_m,yaw_rad')
    cw, _, _ = extract_centerline(MAP, str(traj_csv), step=0.1)
    assert signed_area(cw[:, :2]) < 0
    assert np.hypot(*(cw[0, :2] - lap[0])) < 0.3


def test_lap_length_ignores_overrun_and_second_lap(tmp_path, result):
    track = result[0]
    lap = track[::5, :2]
    one = np.sum(np.hypot(*np.diff(np.vstack((lap, lap[:1])), axis=0).T))
    # map_track.sh runs drive on past the start line, or a second lap
    for drive in (np.vstack((lap, lap[:len(lap) // 4])), np.vstack((lap, lap, lap[:10]))):
        assert lap_length(drive) == pytest.approx(one, rel=0.02)
        t = np.arange(len(drive)) * 0.1
        traj_csv = tmp_path / 'trajectory.csv'
        np.savetxt(traj_csv, np.column_stack((t, drive, np.zeros(len(drive)))), delimiter=',',
                   header='t,x_m,y_m,yaw_rad')
        extract_centerline(MAP, str(traj_csv), step=0.1)   # raised 'does not follow the lap' before


def test_csv_format(tmp_path, result):
    path = tmp_path / 'centerline.csv'
    write_centerline(path, result[0])
    lines = path.read_text().splitlines()
    assert lines[0] == '# x_m,y_m,w_tr_right_m,w_tr_left_m'
    assert np.loadtxt(path, delimiter=',', comments='#').shape == result[0].shape


def test_widths_left_and_right(result):
    # w_left along the left normal, w_right along the right one: a swap leaves 16-25 % of the points outside
    track, m, mask = result
    xy, nl = track[:, :2], normals_left(track[:, :2])
    for p in (xy + 0.9 * track[:, 3:4] * nl, xy - 0.9 * track[:, 2:3] * nl):
        rc = np.round(m.to_pixel(p)).astype(int)
        assert mask[rc[:, 0], rc[:, 1]].all()


def test_first_lap_returns_off_its_start():
    # on a wide track the second lap passes 1.5 m outside the start point
    th = np.linspace(0, 2.5 * 2 * np.pi, 2000)
    r = 10 + 1.5 * th / (2 * np.pi)
    drive = np.column_stack((r * np.cos(th), r * np.sin(th)))
    lap = first_lap(drive)
    one = np.sum(np.hypot(*np.diff(drive[th <= 2 * np.pi], axis=0).T))
    assert np.sum(np.hypot(*np.diff(lap, axis=0).T)) == pytest.approx(one, rel=0.01)


def test_short_loop_without_trajectory_fails():
    with pytest.raises(RuntimeError, match='only'):
        extract_centerline(MAP, min_loop_m=100.0)


def test_large_sparse_infield_kept():
    # a 25 x 25 m unknown infield behind a 1-cell wall is only 0.8 % occupied, but its edge is all wall
    n = 600
    occupied = np.zeros((n, n), bool)
    free = np.zeros((n, n), bool)
    free[20:-20, 20:-20] = True
    free[50:-50, 50:-50] = False
    occupied[50:-50, 50:-50] = True
    occupied[51:-51, 51:-51] = False
    free[30:33, 30:33] = False                              # unobserved speck on the track
    out = fill_unobserved(SimpleNamespace(free=free, occupied=occupied))
    assert out[31, 31] and not out[n // 2, n // 2]
