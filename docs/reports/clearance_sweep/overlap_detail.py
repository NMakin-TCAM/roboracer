"""The cycles where an EVADING plan overlaps the true opponent: what the region said.

Usage: overlap_detail.py TRACK_DIR RUN_DIR [N]   (a closed_loop.py record)
For the N worst planned gaps while evading: the car and plan d, the opponent's true
s and d, the region the plan was made around (at its commit) and the region now.
"""
import bisect
import math
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from closed_loop_geom import box, gap  # noqa: E402

from planning.frenet import FrenetConverter  # noqa: E402

conv = FrenetConverter.from_track(sys.argv[1])
rec = pickle.load(open(os.path.join(sys.argv[2], 'record.pkl'), 'rb'))
n_show = int(sys.argv[3]) if len(sys.argv) > 3 else 6
L = conv.length
LAM = L / max(1, round(L / 12.0))
t = np.array([o[0] for o in rec['opp_true']])
s = np.array([conv.to_frenet(o[1], o[2])[0] for o in rec['opp_true']])
v = rec['meta']['opp_v']
ph = np.angle(np.mean(np.exp(1j * 2 * np.pi * ((s - v * t) % L) / L))) * L / (2 * np.pi)
opp = lambda tt: ((ph + v * tt) % L, 0.4 * math.sin(2 * math.pi * ((ph + v * tt) % L) / LAM))

status = {round(x[3], 4): (x[1], x[2]) for x in rec['status']}
plans = [(p[0], status.get(round(p[1], 4), ('?', '?')), p[3]) for p in rec['plan']]
pt = [p[0] for p in plans]
rg, rgt = rec['regions'], [r[0] for r in rec['regions']]

# Where each committed path began: the first plan whose path starts at that s.
rows = []
for tt, x, y, yaw, vel in rec['ego']:
    i = bisect.bisect_right(pt, tt) - 1
    if i < 0 or plans[i][1][0] != 'evading' or len(plans[i][2]) < 2:
        continue
    path = plans[i][2]
    sc, dc = (float(a) for a in conv.to_frenet(x, y))
    u = np.concatenate(([0.0], np.cumsum(np.diff(path[:, 0]) % L)))
    uc = float(conv.ds(path[0, 0], sc))
    if not 0 <= uc <= u[-1]:
        continue
    dp = float(np.interp(uc, u, path[:, 1]))
    os_, od = opp(tt)
    ox, oy = conv.to_cartesian(os_, od)
    obox = box(float(ox), float(oy), float(conv.heading_at(os_)), 0.25, 0.25, 0.30)
    px, py = conv.to_cartesian(sc, dp)
    g = gap(box(float(px), float(py), float(conv.heading_at(sc)), 0.45, 0.10, 0.30), obox)
    j = bisect.bisect_right(rgt, tt) - 1
    now = rg[j][1][0] if j >= 0 and rg[j][1] else None
    # The commit: walk back while consecutive plans are the same committed path.
    k = i
    while k > 0 and plans[k - 1][1][0] == 'evading' and len(plans[k - 1][2]) and \
            abs(conv.ds(plans[k - 1][2][-1, 0], path[-1, 0])) < 1e-6:
        k -= 1
    jc = bisect.bisect_right(rgt, pt[k]) - 1
    at_commit = rg[jc][1][0] if jc >= 0 and rg[jc][1] else None
    rows.append((g, tt, sc, dc, dp, os_, od, now, at_commit, pt[k], plans[i][1][1]))

rows.sort(key=lambda r: r[0])
fmt = lambda r: (f's {r[0]:.2f}..{r[1]:.2f} d [{r[3]:+.2f}, {r[4]:+.2f}]' if r else 'none')
shown = 0
last_t = -10
for g, tt, sc, dc, dp, os_, od, now, at_commit, tc, note in rows:
    if g >= 0 or shown >= n_show:
        break
    if abs(tt - last_t) < 0.3:
        continue
    last_t, shown = tt, shown + 1
    print(f't {tt:6.2f}: planned gap {g:+.3f}  car s {sc:.2f} d {dc:+.2f}  plan d {dp:+.2f}  '
          f'opponent s {os_:.2f} d {od:+.2f} (car {conv.ds(os_, sc):+.2f} m ahead)  [{note}]')
    print(f'          committed at t {tc:.2f} on region {fmt(at_commit)}')
    print(f'          region now                {fmt(now)}')
if not shown:
    print('no evading cycle overlaps the opponent')
