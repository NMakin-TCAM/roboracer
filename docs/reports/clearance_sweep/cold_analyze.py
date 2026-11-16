"""The first laps of a cold-start run (closed_loop.py with CL_COLD=1).

Usage: cold_analyze.py TRACK_DIR RUN_DIR [LAPS]
"""
import math
import os
import pickle
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from closed_loop_geom import box, gap  # noqa: E402

from planning.frenet import FrenetConverter  # noqa: E402

conv = FrenetConverter.from_track(sys.argv[1])
run, laps = sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 2
rec = pickle.load(open(os.path.join(run, 'record.pkl'), 'rb'))
L = conv.length
LAM = L / max(1, round(L / 12.0))
off = rec['clock_offset']

t = np.array([o[0] for o in rec['opp_true']])
s = np.array([conv.to_frenet(o[1], o[2])[0] for o in rec['opp_true']])
v_opp = rec['meta']['opp_v']
ph = np.angle(np.mean(np.exp(1j * 2 * np.pi * ((s - v_opp * t) % L) / L))) * L / (2 * np.pi)
opp = lambda tt: (ph + v_opp * tt) % L

def opp_d_model(rec, conv, lam):
    """The opponent's d as a function of its s: the weave, or (constant_speed) its
    measured d, the median over its true marker positions."""
    if rec['meta'].get('opp_mode', 'weaving') == 'constant_speed':
        src = rec.get('opp_true') or rec['opp']
        d0 = float(np.median([conv.to_frenet(o[1], o[2])[1] for o in src]))
        return lambda s: d0
    return lambda s: 0.4 * math.sin(2 * math.pi * s / lam)


DM = opp_d_model(rec, conv, LAM)
ego = rec['ego']
dist = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff([e[1] for e in ego]),
                                                np.diff([e[2] for e in ego])))))
lap_t = [ego[int(np.searchsorted(dist, k * L))][0] if dist[-1] >= k * L else None
         for k in range(1, laps + 1)]
end = lap_t[-1] if lap_t[-1] is not None else ego[-1][0]
print(f'car laps ({L:.2f} m) completed at t = {[round(x, 1) if x else None for x in lap_t]} s '
      f'(harness time from launch; the car stands until the planner first says clear)')

events = []
for line in open(os.path.join(run, 'spliner.log')):
    m = re.search(r'\[(\d+\.\d+)\] \[spliner_node\]: (planning on .*|evasion released: .*|'
                  r'not planning: .*)', line)
    if m:
        events.append((float(m.group(1)) - off, m.group(2)))
for tt, msg in rec['gp']:
    events.append((tt, f'GP publishes lap_count {msg:g}'))
first_move = next((e[0] for e in ego if e[4] > 0.05), None)
events.append((first_move, 'car starts moving'))

st = rec['status']
prev = None
for x in st:
    if x[0] > end:
        break
    k = (x[1], x[2])
    if k != prev:
        events.append((x[0], f'status {x[1]}: {x[2]}'))
        prev = k
events.sort(key=lambda e: e[0])
print('\ntimeline to the end of lap', laps)
for tt, msg in events:
    if tt is not None and tt <= end:
        print(f'  {tt:6.2f}  {msg}')

print('\nper lap')
start = 0.0
for k, lt in enumerate(lap_t, 1):
    if lt is None:
        print(f'  lap {k}: not completed'); break
    sel = [e for e in ego if start < e[0] <= lt]
    g, rel = [], []
    for tt, x, y, yaw, v in sel:
        os_ = opp(tt)
        od = DM(os_)
        ox, oy = conv.to_cartesian(os_, od)
        g.append(gap(box(x, y, yaw, 0.45, 0.10, 0.30),
                     box(float(ox), float(oy), float(conv.heading_at(os_)), 0.25, 0.25, 0.30)))
        rel.append(float(conv.ds(os_, conv.to_frenet(x, y)[0])))
    g, rel = np.array(g), np.array(rel)
    v = np.array([e[4] for e in sel])
    passes, behind = 0, False
    for r in rel:
        if -3 < r < 0:
            behind = True
        elif r > 1.0 and behind:
            passes, behind = passes + 1, False
    stops = int(np.sum((v[1:] < 0.05) & (v[:-1] >= 0.05)))
    statuses = {}
    for x in st:
        if start < x[0] <= lt:
            statuses[x[1]] = statuses.get(x[1], 0) + 1
    tot = sum(statuses.values())
    behind_gap = -rel[(rel < 0) & (rel > -12)]
    print(f'  lap {k}: {lt - start:5.1f} s, mean {v.mean():.2f} m/s, max {v.max():.2f}, '
          f'{stops} stops ({np.sum(v < 0.05) * 0.02:.1f} s stopped), {passes} passes, '
          f'min gap {g.min():+.3f} m, contact {np.sum(g <= 0) * 0.02:.2f} s, '
          f'gap behind p10/p50/p90 {np.percentile(behind_gap, [10, 50, 90]).round(2) if len(behind_gap) else "-"} m')
    print('         status share: ' + ', '.join(f'{s_} {n * 100 / tot:.0f}%'
                                              for s_, n in sorted(statuses.items())))
    start = lt
