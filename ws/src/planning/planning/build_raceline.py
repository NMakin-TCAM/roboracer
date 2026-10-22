#!/usr/bin/env python3
"""ros2 run planning build_raceline <name>

Reads   <tracks>/<name>/map.pgm, map.yaml, trajectory.csv (optional: driving direction + start)
Writes  <tracks>/<name>/centerline.csv, raceline.csv, preview.png   (formats: docs/pipeline.md)

<tracks> = $TRACKS_DIR or ~/ws/tracks. The raceline is TUM's minimum-time optimization
(opt_type 'mintime'); there is no fallback: if IPOPT does not converge this exits with status 1.
"""
import argparse
import configparser
import json
import os
import sys
import time

import numpy as np

from planning.centerline import extract_centerline, write_centerline
from planning.preview import save_preview
from planning.raceline import (OptimizationError, TUM_PYTHON, check_inside, lap_time,
                               run_mintime, write_raceline)


def default_config():
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory('planning'), 'config', 'racecar.ini')
    except Exception:   # not installed: running from the source tree
        return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'config', 'racecar.ini')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('name', help='track name, i.e. directory under the tracks dir')
    ap.add_argument('--tracks-dir', default=os.environ.get('TRACKS_DIR', os.path.expanduser('~/ws/tracks')))
    ap.add_argument('--config', default=default_config(), help='TUM racecar.ini')
    ap.add_argument('--step', type=float, default=0.1, help='centerline point spacing [m]')
    ap.add_argument('--smooth', type=float, default=0.05, help='centerline spline smoothing [m rms]')
    ap.add_argument('--carve-trajectory', type=float, nargs='?', const=0.3, default=None, metavar='RADIUS',
                    help='treat cells within RADIUS m (default 0.3) of the mapping lap as free; only for '
                         'tracks whose map shows walls across the driven path (overpass)')
    ap.add_argument('--tum-python', default=TUM_PYTHON)
    args = ap.parse_args(argv)

    d = os.path.join(args.tracks_dir, args.name)
    map_yaml = os.path.join(d, 'map.yaml')
    if not os.path.isfile(map_yaml):
        sys.exit(f'build_raceline: {map_yaml} not found')
    traj_csv = os.path.join(d, 'trajectory.csv')
    has_traj = os.path.isfile(traj_csv)
    t0 = time.monotonic()
    # never leave stale outputs next to a failed run
    for f in ('centerline.csv', 'raceline.csv', 'preview.png'):
        if os.path.exists(os.path.join(d, f)):
            os.remove(os.path.join(d, f))

    how = ('following the mapping lap in trajectory.csv' if has_traj
           else 'no trajectory.csv: counter-clockwise, start near the map origin')
    print(f'[1/3] centerline from {map_yaml} ({how})')
    try:
        track, m, mask = extract_centerline(map_yaml, traj_csv if has_traj else None,
                                            step=args.step, smooth_m=args.smooth,
                                            carve_radius_m=args.carve_trajectory)
    except RuntimeError as e:
        sys.exit(f'build_raceline: centerline extraction failed: {e}')
    cl_csv = os.path.join(d, 'centerline.csv')
    write_centerline(cl_csv, track)
    seg = np.hypot(*(np.roll(track[:, :2], -1, axis=0) - track[:, :2]).T)
    width = track[:, 2] + track[:, 3]
    print(f'      {len(track)} points, {seg.sum():.1f} m, width {width.min():.2f}-{width.max():.2f} m -> {cl_csv}')

    print(f'[2/3] TUM mintime ({args.config})')
    rl_csv = os.path.join(d, 'raceline.csv')
    cfg = configparser.ConfigParser()
    cfg.read(args.config)
    width_veh = json.loads(cfg.get('GENERAL_OPTIONS', 'veh_params'))['width']
    width_opt = json.loads(cfg.get('OPTIMIZATION_OPTIONS', 'optim_opts_mintime'))['width_opt']
    try:
        narrow = width < width_opt
        if narrow.any():
            i = np.argmin(width)
            raise OptimizationError(
                'track is narrower than width_opt = %.2f m over %.1f m, down to %.2f m at (%.1f, %.1f)'
                % (width_opt, narrow.sum() * seg.mean(), width[i], *track[i, :2]))
        traj, dt = run_mintime(cl_csv, args.config, python=args.tum_python)
        clearance = check_inside(traj, m, mask, width_veh / 2)
    except OptimizationError as e:
        save_preview(os.path.join(d, 'preview.png'), m, track, None, f'{args.name}: mintime FAILED')
        sys.exit(f'build_raceline: {e}')
    write_raceline(rl_csv, traj)
    t_lap = lap_time(traj)
    print(f'      {len(traj)} points, {traj[-1, 0]:.1f} m, lap {t_lap:.2f} s, vx {traj[:, 5].min():.1f}-'
          f'{traj[:, 5].max():.1f} m/s, min wall clearance {clearance:.2f} m, optimizer {dt:.1f} s -> {rl_csv}')

    png = os.path.join(d, 'preview.png')
    save_preview(png, m, track, traj,
                 f'{args.name}: TUM mintime, lap {t_lap:.2f} s, {traj[-1, 0]:.1f} m, '
                 f'vx max {traj[:, 5].max():.1f} m/s')
    print(f'[3/3] preview -> {png}   (total {time.monotonic() - t0:.1f} s)')


if __name__ == '__main__':
    main()
