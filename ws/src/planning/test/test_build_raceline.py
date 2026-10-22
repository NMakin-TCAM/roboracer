"""End to end on the fixture map: centerline + TUM mintime + preview. Needs the TUM venv (Docker image)."""
import os
import shutil

import numpy as np
import pytest
from PIL import Image

from planning import build_raceline
from planning.raceline import HEADER, TUM_PYTHON, OptimizationError, read_raceline, run_mintime

FIX = os.path.join(os.path.dirname(__file__), 'fixtures')
CONFIG = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'racecar.ini')


@pytest.mark.skipif(not os.path.isfile(TUM_PYTHON),
                    reason='TUM venv not installed: run the tests in the Docker container '
                           '(docker compose exec ros, then python3 -m pytest src/planning/test)')
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


def test_failed_optimizer_reported(tmp_path, capsys):
    # stand-in for the venv python: TUM's coloured ERROR line, then a malformed output file ($5)
    fake = tmp_path / 'python'
    fake.write_text('#!/bin/sh\nprintf \'\\033[91mERROR: Optimization did not succeed!\\033[0m\\n\'\n'
                    'echo garbage > "$5"\n')
    fake.chmod(0o755)
    with pytest.raises(OptimizationError, match='malformed'):
        run_mintime(str(tmp_path / 'centerline.csv'), CONFIG, python=str(fake))
    assert '[tum] ERROR: Optimization did not succeed!' in capsys.readouterr().out


def test_stale_preview_removed(tmp_path):
    d = tmp_path / 'blank'
    d.mkdir()
    Image.fromarray(np.full((20, 20), 205, dtype=np.uint8)).save(d / 'map.pgm')   # all unknown
    (d / 'map.yaml').write_text('image: map.pgm\nresolution: 0.05\norigin: [0.0, 0.0, 0.0]\n'
                                'negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n')
    (d / 'preview.png').write_bytes(b'old')
    with pytest.raises(SystemExit):
        build_raceline.main(['blank', '--tracks-dir', str(tmp_path), '--config', CONFIG])
    assert not (d / 'preview.png').exists()
