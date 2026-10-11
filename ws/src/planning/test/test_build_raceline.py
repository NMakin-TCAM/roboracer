"""End to end on the fixture map: centerline + TUM mintime + preview. Needs the TUM venv (Docker image)."""
import os
import shutil

import numpy as np
import pytest

from planning import build_raceline
from planning.raceline import HEADER, TUM_PYTHON, read_raceline

FIX = os.path.join(os.path.dirname(__file__), 'fixtures')
CONFIG = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'racecar.ini')


@pytest.mark.skipif(not os.path.isfile(TUM_PYTHON), reason='TUM venv not installed')
def test_build_raceline(tmp_path):
    d = tmp_path / 'fixture'
    d.mkdir()
    shutil.copy(os.path.join(FIX, 'track_map_v3.pgm'), d / 'map.pgm')
    yaml = open(os.path.join(FIX, 'track_map_v3.yaml')).read().replace('track_map_v3.pgm', 'map.pgm')
    (d / 'map.yaml').write_text(yaml)

    build_raceline.main(['fixture', '--tracks-dir', str(tmp_path), '--config', CONFIG])

    for f in ('centerline.csv', 'raceline.csv', 'preview.png'):
        assert (d / f).is_file()
    assert (d / 'raceline.csv').read_text().splitlines()[0] == '# ' + HEADER
    rl = read_raceline(d / 'raceline.csv')
    assert rl.shape[1] == 7
    assert np.allclose(rl[0, 1:], rl[-1, 1:])               # closed: last row repeats the first
    assert np.all(np.diff(rl[:, 0]) > 0)
    assert np.all(rl[:, 5] > 0.5) and rl[:, 5].max() < 15.0
    # same direction as the centerline (counter-clockwise here)
    x, y = rl[:-1, 1], rl[:-1, 2]
    assert np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) > 0
