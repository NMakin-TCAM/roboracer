"""How far the planned line is from the car just after a planner switch (the REJOIN fix).

Usage: step_analyze.py TRACK_DIR RUN_DIR...   (closed_loop.py records)

For every 20 ms car sample while the plan in force says clear or evading, |car d - plan d|
at the car's own s, split by whether the status changed within the last second. A plan
that starts where the car is keeps the first small; one that snaps to the raceline does
not.
"""
import bisect
import os
import pickle
import sys

import numpy as np

from planning.frenet import FrenetConverter

conv = FrenetConverter.from_track(sys.argv[1])
L = conv.length
for run in sys.argv[2:]:
    rec = pickle.load(open(os.path.join(run, 'record.pkl'), 'rb'))
    status = {round(x[3], 4): x[1] for x in rec['status']}
    plans = [(p[0], status.get(round(p[1], 4), '?'), p[3]) for p in rec['plan']]
    pt = [p[0] for p in plans]
    since, last, after, steady = None, None, [], []
    for t, x, y, yaw, v in rec['ego']:
        i = bisect.bisect_right(pt, t) - 1
        if i < 0 or v < 0.3:
            continue
        _, st, path = plans[i]
        if st != last:
            since, last = t, st
        if st not in ('clear', 'evading'):
            continue
        s, d = (float(a) for a in conv.to_frenet(x, y))
        dp = 0.0
        if st == 'evading' and len(path) > 1:
            u = np.concatenate(([0.0], np.cumsum(np.diff(path[:, 0]) % L)))
            dp = float(np.interp(float(conv.ds(path[0, 0], s)), u, path[:, 1]))
        (after if t - since < 1.0 else steady).append(abs(d - dp))
    for name, xs in (('< 1 s after a switch', after), ('steady', steady)):
        xs = np.array(xs)
        print(f'{os.path.basename(run):18s} {name:22s} n {len(xs):6d}  p50 {np.percentile(xs, 50) * 100:5.1f} cm  '
              f'p95 {np.percentile(xs, 95) * 100:5.1f} cm  max {xs.max() * 100:5.1f} cm')
