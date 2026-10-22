"""TUM minimum-time raceline: runs tum_mintime.py in the TUM venv and reads back its trajectory.

The optimizer lives in its own venv (/opt/tum_venv, see docker/install_deps.sh) so its numpy /
casadi / scipy pins never meet ROS's Python; we only exchange CSV files with it.

Raceline columns (TUM's export, map frame, closed: the last row repeats the first at s = lap length):
  s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2
psi is TUM's heading convention: 0 = +y (north), counter-clockwise positive, in [-pi, pi),
i.e. psi = yaw - pi/2 with yaw the usual ROS angle from +x.
"""
import os
import re
import subprocess
import sys
import tempfile
import time

import numpy as np
from scipy import ndimage

HEADER = 's_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2'
TUM_PYTHON = os.environ.get('TUM_PYTHON', '/opt/tum_venv/bin/python')
TUM_DIR = os.environ.get('TUM_DIR', '/opt/tum_optimizer')
DRIVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tum_mintime.py')


class OptimizationError(RuntimeError):
    pass


def run_mintime(centerline_csv, ini_path, python=TUM_PYTHON):
    """Optimize; returns (trajectory Nx7, runtime s). Raises OptimizationError on failure."""
    if not os.path.isfile(python):
        raise OptimizationError(f'TUM venv python not found at {python} (built by the Dockerfile, '
                                'see ws/src/planning/docker/install_deps.sh)')
    with tempfile.TemporaryDirectory(prefix='mintime_') as work:
        out = os.path.join(work, 'traj_race_cl.csv')
        # -I: ignore ROS's PYTHONPATH so only the venv's packages are used
        cmd = [python, '-I', DRIVER, centerline_csv, ini_path, out, work]
        env = dict(os.environ, MPLBACKEND='Agg', TUM_DIR=TUM_DIR)
        t0 = time.monotonic()
        proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        dt = time.monotonic() - t0
        log = proc.stdout
        # IPOPT's iteration table is long; keep the summary lines
        for line in log.splitlines():
            line = re.sub(r'\x1b\[[0-9;]*m', '', line)   # TUM colours its ERROR / WARNING lines
            if line.startswith(('INFO', 'WARNING', 'EXIT', 'Spline', 'Number of Iterations',
                                'Total seconds', 'ERROR')):
                print('  [tum] ' + line)
        if proc.returncode != 0 or not os.path.isfile(out):
            tail = '\n'.join(log.splitlines()[-25:])
            if 'EXIT: Optimal Solution Found' not in log and 'EXIT:' in log:
                why = 'IPOPT did not converge'
            else:
                why = f'optimizer failed (exit code {proc.returncode})'
            raise OptimizationError(f'mintime {why}. Last output:\n{tail}')
        try:
            traj = np.loadtxt(out, comments='#', delimiter=';', ndmin=2)
        except ValueError as e:
            raise OptimizationError(f'mintime output is malformed: {e}')
    if traj.shape[1] != 7:
        raise OptimizationError(f'mintime output has {traj.shape[1]} columns, expected 7 ({HEADER})')
    return traj, dt


def write_raceline(path, traj):
    np.savetxt(path, traj, delimiter='; ', fmt='%.7f', header=HEADER)


def read_raceline(path):
    return np.loadtxt(path, comments='#', delimiter=';', ndmin=2)


def lap_time(traj):
    ds = np.diff(traj[:, 0])
    v = 0.5 * (traj[:-1, 5] + traj[1:, 5])
    return float(np.sum(ds / np.maximum(v, 1e-3)))


def check_inside(traj, m, mask, half_width):
    """Smallest wall clearance of the raceline (centre to wall, m). Fails if the centre leaves the track."""
    dist = ndimage.distance_transform_edt(mask) * m.res - 0.5 * m.res
    rc = m.to_pixel(traj[:, 1:3])
    d = ndimage.map_coordinates(dist, rc.T, order=1, mode='constant', cval=-1.0)
    if d.min() < 0:
        raise OptimizationError('raceline leaves the free space (min clearance %.2f m)' % d.min())
    if d.min() < half_width:
        print('WARNING: raceline gets within %.2f m of a wall (car half width %.2f m)'
              % (d.min(), half_width), file=sys.stderr)
    return float(d.min())
