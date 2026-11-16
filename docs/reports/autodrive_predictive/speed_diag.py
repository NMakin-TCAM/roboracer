"""Why no evasion starts at full speed: planner statuses and notes, and the regions.

Usage: speed_diag.py TRACK_DIR RUN_DIR   (closed_loop.py record)
"""
import os
import pickle
import sys
from collections import Counter

import numpy as np

from planning.frenet import FrenetConverter
from planning.spliner import SplinerParams, choose_side

conv = FrenetConverter.from_track(sys.argv[1])
rec = pickle.load(open(os.path.join(sys.argv[2], 'record.pkl'), 'rb'))
p = SplinerParams(evasion_dist=0.25, bound_margin=float(rec['meta'].get('bound_margin', 0.15)))
c = Counter((x[1], x[2].split(';')[0].split(' (')[0]) for x in rec['status'])
print('statuses:')
for (s, n), k in c.most_common(12):
    print(f'  {k:6d}  {s:10s} {n}')
regs = [r[0] for t, r in rec['regions'] if r]
walls = sum(r[4] >= 10 for r in regs)
print(f'regions: {len(regs)} non-empty of {len(rec["regions"])}, walls {walls}')


class O:
    def __init__(self, r):
        self.s_start, self.s_end, self.s_center, self.d_right, self.d_left, self.d_center = r


ended = [O(r) for r in regs if r[4] < 10]
if ended:
    side = [choose_side(o, conv, p) for o in ended]
    room = np.array([[x.space_left, x.space_right] for x in side])
    need = p.car_width + p.evasion_dist
    print(f'ended regions: {len(ended)}; a side with room (>= {need:.2f} m): '
          f'{np.mean([x.side is not None for x in side]) * 100:.1f}%')
    print(f'  room left p5/50/95 {np.percentile(room[:, 0], [5, 50, 95]).round(2)}, '
          f'right {np.percentile(room[:, 1], [5, 50, 95]).round(2)}')
    print(f'  width d_left-d_right p5/50/95 '
          f'{np.percentile([o.d_left - o.d_right for o in ended], [5, 50, 95]).round(2)}')
