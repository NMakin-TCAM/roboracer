"""What the car was doing in the 1.5 s before each simulator collision, and after.

Usage: crash_detail.py TRACK_DIR RUN_DIR
"""
import bisect
import math
import os
import pickle
import sys

import numpy as np

from planning.frenet import FrenetConverter

THETA = math.radians(-92.693)
T = np.array([0.5065, -4.3836])
conv = FrenetConverter.from_track(sys.argv[1])
rec = pickle.load(open(os.path.join(sys.argv[2], 'record.pkl'), 'rb'))


def w2m(x, y):
    c, s = math.cos(THETA), math.sin(THETA)
    dx, dy = x - T[0], y - T[1]
    return c * dx + s * dy, -s * dx + c * dy


truth = [(t, *w2m(v[1], v[2]), v[4]) for t, v in rec['/autodrive/roboracer_1/odom']]
tt = [x[0] for x in truth]
loc = rec['/localized_pose']
lt = [x[0] for x in loc]
st = rec['/planner/avoidance/status']
stt = [x[0] for x in st]
drv = rec['/drive']
dt_ = [x[0] for x in drv]
for tc, why in rec['/autodrive/roboracer_1/sim_reset']:
    print(f'\n=== {why} at t {tc:.2f}')
    for k in np.arange(-1.5, 1.01, 0.25):
        t = tc + k
        i = min(bisect.bisect_left(tt, t), len(truth) - 1)
        _, x, y, v = truth[i]
        s, d = (float(a) for a in conv.to_frenet(x, y))
        hi, lo = (float(a) for a in conv.width_at(s))
        j = bisect.bisect_right(lt, t) - 1
        loc_s = (f'{float(conv.to_frenet(loc[j][1][1], loc[j][1][2])[0]):6.2f}'
                 if j >= 0 and t - lt[j] < 0.2 else '  lost')
        m = bisect.bisect_right(stt, t) - 1
        n = bisect.bisect_right(dt_, t) - 1
        cmd = drv[n][1] if n >= 0 else (0, 0, 0)
        print(f'  {k:+5.2f}s truth s {s:6.2f} d {d:+.2f} (track {lo:+.2f}..{hi:+.2f}) v {v:4.2f} '
              f'| loc s {loc_s} | {st[m][1][1] if m >= 0 else "-":9s} | cmd v {cmd[1]:.2f} '
              f'steer {math.degrees(cmd[2]):+5.1f}deg')
