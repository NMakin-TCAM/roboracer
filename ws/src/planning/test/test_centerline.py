"""Centerline extraction on a real Cartographer map (fixtures/track_map_v3: ~16 x 9 m, ~30 m loop)."""
import os

import numpy as np
import pytest

from planning.centerline import OccupancyMap, extract_centerline, signed_area, write_centerline

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


def test_csv_format(tmp_path, result):
    path = tmp_path / 'centerline.csv'
    write_centerline(path, result[0])
    lines = path.read_text().splitlines()
    assert lines[0] == '# x_m,y_m,w_tr_right_m,w_tr_left_m'
    assert np.loadtxt(path, delimiter=',', comments='#').shape == result[0].shape
