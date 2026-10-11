"""Regression: AutoDRIVE iros2026 mapping run (fixtures/iros2026*, a ~209 m lap).

The first version returned a 24 m loop around an unobserved (unknown) band on the start straight.
The lap also passes through mapped walls at (-13.9, 0.2) and (-0.1, 9.5), where a stretch is driven
in both directions: the map does not match the drive, which must fail loudly by default.
"""
import os

import numpy as np
import pytest

from planning.centerline import OccupancyMap, extract_centerline, fill_unobserved, load_trajectory

FIX = os.path.join(os.path.dirname(__file__), 'fixtures')
MAP = os.path.join(FIX, 'iros2026.yaml')
TRAJ = os.path.join(FIX, 'iros2026_trajectory.csv')


def free_at(m, free, xy):
    r, c = np.round(m.to_pixel(xy)[0]).astype(int)
    return bool(free[r, c])


def lap_length():
    t = load_trajectory(TRAJ)
    return np.sum(np.hypot(*np.diff(np.vstack((t, t[:1])), axis=0).T))


def test_unobserved_islands_filled_infields_kept():
    m = OccupancyMap(MAP)
    free = fill_unobserved(m)
    assert not free_at(m, m.free, (-3.8, 0.1))         # unknown band under the car at the start
    assert free_at(m, free, (-3.8, 0.1))
    assert not free_at(m, free, (4.0, 14.0))           # infields stay
    assert not free_at(m, free, (-5.0, 3.0))
    v3 = OccupancyMap(os.path.join(FIX, 'track_map_v3.yaml'))
    assert not free_at(v3, fill_unobserved(v3), (-4.0, -1.5))


def test_lap_through_walls_fails_loudly():
    with pytest.raises(RuntimeError, match='through walls'):
        extract_centerline(MAP, TRAJ)


def test_carved_centerline_follows_the_lap():
    track = extract_centerline(MAP, TRAJ, carve_radius_m=0.3)[0]
    xy = track[:, :2]
    length = np.sum(np.hypot(*(np.roll(xy, -1, axis=0) - xy).T))
    assert abs(length - lap_length()) < 0.05 * lap_length()
    assert np.hypot(*(xy[0] - load_trajectory(TRAJ)[0])) < 1.0
    assert np.all(track[:, 2] > 0.2) and np.all(track[:, 3] > 0.2)
