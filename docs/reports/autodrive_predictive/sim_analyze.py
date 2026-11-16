"""Analyse a sim_record.py record of a predictive_race.launch.py run in AutoDRIVE.

Usage: sim_analyze.py TRACK_DIR RUN_DIR

The car's ground truth (/autodrive/roboracer_1/odom, sim world frame) is put in the map
frame with porto_odom_trial's map->world transform as verified by SIGRobotics: rotation
-92.693 deg, translation (0.5065, -4.3836), to 4.3 cm at 7.4 m. The reference point of
the sim's odometry on the car is not documented, so localization error carries an
unknown constant offset; tracking and gaps use the localized pose's convention (rear
axle) where they can and say which.
"""
import bisect
import math
import os
import pickle
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                                'clearance_sweep'))
from closed_loop_geom import box, gap  # noqa: E402

from planning.frenet import FrenetConverter  # noqa: E402

THETA = math.radians(-92.693)        # map -> world rotation
T = np.array([0.5065, -4.3836])      # map -> world translation
EGO = (0.45, 0.10, 0.30)
OPP = (0.25, 0.25, 0.30)

conv = FrenetConverter.from_track(sys.argv[1])
run = sys.argv[2]
rec = pickle.load(open(os.path.join(run, 'record.pkl'), 'rb'))
L = conv.length


def world_to_map(x, y, yaw):
    c, s = math.cos(THETA), math.sin(THETA)
    dx, dy = x - T[0], y - T[1]
    return c * dx + s * dy, -s * dx + c * dy, yaw - THETA


def series(topic):
    return rec.get(topic, [])


def first(topic, pred=lambda v: True):
    return next((t for t, v in series(topic) if pred(v)), None)


print(f'run {run}: {rec["meta"]["end"]:.1f} s recorded')

# ------------------------------------------------------------------ timeline
truth = [(t, *world_to_map(v[1], v[2], v[3]), v[4]) for t, v in series('/autodrive/roboracer_1/odom')
         if not (isinstance(v, tuple) and v and v[0] == 'error')]
moving = next((t for t, x, y, yaw, v in truth if abs(v) > 0.05), None)
status = series('/planner/avoidance/status')
events = [
    ('first /scan', first('/scan')),
    ('first /tracked_pose (Cartographer)', first('/tracked_pose')),
    ('first /localized_pose (gate: localized)', first('/localized_pose')),
    ('first /perception/obstacles with the opponent', first('/perception/obstacles', lambda v: v[1])),
    ('first planner status', first('/planner/avoidance/status')),
    ('first collision-free plan', first('/planner/avoidance/status', lambda v: v[1] in ('clear', 'evading'))),
    ('car moves (truth speed > 0.05)', moving),
    ('first /opponent_trajectory (GP)', first('/opponent_trajectory')),
    ('GP lap_count >= 1', first('/opponent_trajectory', lambda v: v[1] >= 1)),
    ('first collision region', first('/collision_prediction/obstacles', lambda v: v[1])),
    ('first evasion', first('/planner/avoidance/status', lambda v: v[1] == 'evading')),
]
print('\n== timeline (s from recorder start; the launch starts within ~0.1 s of it)')
for name, t in sorted(events, key=lambda e: (e[1] is None, e[1] or 0)):
    print(f'  {"-" if t is None else f"{t:7.2f}"}  {name}')
for t, v in series('/autodrive/roboracer_1/sim_reset'):
    print(f'  {t:7.2f}  SIM RESET: {v}')
cc = [v for _, v in series('/autodrive/roboracer_1/collision_count')]
if cc:
    print(f'  collision_count {cc[0]} -> {cc[-1]}')

# ------------------------------------------------------------------ rates
print('\n== rates (arrival at the recorder; a gap is > 3x the median period)')
print(f'  {"topic":40s} {"n":>6s} {"Hz":>7s} {"med ms":>7s} {"max gap ms":>10s} {"gaps":>5s} '
      f'{"stamp dup":>9s} {"stamp back":>10s}')
for topic in rec:
    if topic == 'meta' or not series(topic):
        continue
    t = np.array([x[0] for x in series(topic)])
    if len(t) < 3:
        print(f'  {topic:40s} {len(t):6d}')
        continue
    dt = np.diff(t)
    med = np.median(dt)
    stamps = [x[1][0] if isinstance(x[1], tuple) and x[1] and isinstance(x[1][0], float)
              else (x[1] if isinstance(x[1], float) else None) for x in series(topic)]
    st = np.array([s for s in stamps if s is not None])
    dup = int(np.sum(np.diff(st) == 0)) if len(st) > 2 else 0
    back = int(np.sum(np.diff(st) < 0)) if len(st) > 2 else 0
    print(f'  {topic:40s} {len(t):6d} {len(t) / (t[-1] - t[0]):7.1f} {med * 1e3:7.1f} '
          f'{dt.max() * 1e3:10.1f} {int(np.sum(dt > 3 * med)):5d} {dup:9d} {back:10d}')

# ------------------------------------------------------------------ localization error
loc = series('/localized_pose')
tt = [x[0] for x in truth]
errs = []
for t, v in loc:
    i = bisect.bisect_left(tt, t)
    if 0 < i < len(truth):
        _, x, y, yaw, _ = truth[i]
        errs.append((t, v[1] - x, v[2] - y, (v[3] - yaw + math.pi) % (2 * math.pi) - math.pi))
if errs:
    e = np.array(errs)
    # In the car's frame: along and across.
    along, across = [], []
    for t, ex, ey, _ in errs:
        i = bisect.bisect_left(tt, t)
        yaw = truth[min(i, len(truth) - 1)][3]
        along.append(ex * math.cos(yaw) + ey * math.sin(yaw))
        across.append(-ex * math.sin(yaw) + ey * math.cos(yaw))
    along, across = np.array(along), np.array(across)
    print('\n== localization: /localized_pose minus truth (car frame)')
    print(f'  along  median {np.median(along):+.3f} m (a constant here is the truth reference '
          f'point), spread p5..p95 {np.percentile(along, 5):+.3f}..{np.percentile(along, 95):+.3f}')
    print(f'  across median {np.median(across):+.3f} m, p5..p95 '
          f'{np.percentile(across, 5):+.3f}..{np.percentile(across, 95):+.3f}, max |.| '
          f'{np.abs(across).max():.3f}')
    print(f'  yaw    median {math.degrees(np.median(e[:, 3])):+.2f} deg, max |.| '
          f'{math.degrees(np.abs(e[:, 3]).max()):.2f} deg')
    off_along = float(np.median(along))
else:
    off_along = 0.0

# ------------------------------------------------------------------ plan, tracking, gaps
stat_by = {round(v[0], 4): (v[1], v[2]) for _, v in status}
plans = [(t, stat_by.get(round(v[0], 4), ('?', '?')), v[1])
         for t, v in series('/planner/avoidance/otwpnts')]
pt = [p[0] for p in plans]
opp = series('/fake_opponent/truth')
ot = [x[0] for x in opp]


def plan_d(i, s_car):
    st, path = plans[i][1][0], plans[i][2]
    if st == 'evading' and len(path) > 1:
        u = np.concatenate(([0.0], np.cumsum(np.diff(path[:, 0]) % L)))
        uc = float(conv.ds(path[0, 0], s_car))
        d = float(np.interp(uc, u, path[:, 1]))
        slope = float(np.interp(uc, u[1:], np.diff(path[:, 1]) / np.maximum(np.diff(u), 1e-6)))
        return d, slope
    return 0.0, 0.0


rows = []
for t, x, y, yaw, v in truth:
    # Truth shifted to the rear axle by the along-track offset found above.
    xr, yr = x + off_along * math.cos(yaw), y + off_along * math.sin(yaw)
    i = bisect.bisect_right(pt, t) - 1
    j = bisect.bisect_right(ot, t) - 1
    if i < 0 or j < 0 or t - ot[j] > 0.2:
        continue
    st = plans[i][1][0]
    s_car, d_car = (float(a) for a in conv.to_frenet(xr, yr))
    dp, slope = plan_d(i, s_car)
    _, ox, oy, oyaw = opp[j][1]
    obox = box(ox, oy, oyaw, *OPP)
    achieved = gap(box(xr, yr, yaw, *EGO), obox)
    planned = None
    if st in ('clear', 'evading'):
        px, py = conv.to_cartesian(s_car, dp)
        planned = gap(box(float(px), float(py), float(conv.heading_at(s_car)) + math.atan(slope),
                          *EGO), obox)
    rel = float(conv.ds(conv.to_frenet(ox, oy)[0], s_car))
    rows.append(dict(t=t, v=v, st=st, track=d_car - dp, planned=planned, achieved=achieved,
                     rel=rel, s=s_car))

moving_rows = [r for r in rows if r['v'] > 0.3 and r['st'] in ('clear', 'evading')]
if moving_rows:
    tr = np.abs([r['track'] for r in moving_rows])
    ev = np.abs([r['track'] for r in moving_rows if r['st'] == 'evading'])
    print('\n== tracking: car d (truth, rear axle) minus the plan\'s d at its s, moving, clear or evading')
    print(f'  all      n {len(tr)}: p50 {np.percentile(tr, 50) * 100:.1f} cm, p95 '
          f'{np.percentile(tr, 95) * 100:.1f} cm, max {tr.max() * 100:.1f} cm')
    if len(ev):
        print(f'  evading  n {len(ev)}: p50 {np.percentile(ev, 50) * 100:.1f} cm, p95 '
              f'{np.percentile(ev, 95) * 100:.1f} cm, max {ev.max() * 100:.1f} cm')
    # "clear" right after an evasion is a step in the reference (the plan jumps back to the
    # raceline while the car is still off it), not tracking. Split it off.
    since, last, steady, step = None, None, [], []
    for r in rows:
        if r['st'] != last:
            since, last = r['t'], r['st']
        if r['st'] == 'clear' and r['v'] > 0.3:
            (steady if r['t'] - since >= 1.0 else step).append(abs(r['track']))
    for name, xs in (('clear >= 1 s (raceline)', steady), ('clear < 1 s (after a switch)', step)):
        if xs:
            xs = np.array(xs)
            print(f'  {name:28s} n {len(xs)}: p50 {np.percentile(xs, 50) * 100:.1f} cm, p95 '
                  f'{np.percentile(xs, 95) * 100:.1f} cm, max {xs.max() * 100:.1f} cm')

if rows:
    pl = [r for r in rows if r['planned'] is not None]
    w = min(pl, key=lambda r: r['planned']) if pl else None
    a = min(rows, key=lambda r: r['achieved'])
    print('\n== gaps against the opponent\'s true pose')
    if w:
        print(f'  closest planned  {w["planned"]:+.3f} m at t {w["t"]:.1f} ({w["st"]}, car '
              f'{w["rel"]:+.2f} m ahead of the opponent); planned-through cycles '
              f'{sum(r["planned"] < 0 for r in pl)}')
    print(f'  closest achieved {a["achieved"]:+.3f} m at t {a["t"]:.1f} ({a["st"]}, car '
          f'{a["rel"]:+.2f} m ahead, tracking {a["track"]:+.3f}, v {a["v"]:.2f}); contact '
          f'{sum(r["achieved"] <= 0 for r in rows) * 0.01:.2f} s (approx.)')

# ------------------------------------------------------------------ laps and passes
if truth:
    xy = np.array([(x, y) for _, x, y, _, _ in truth])
    dist = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))))
    print(f'\n== driven {dist[-1]:.1f} m = {dist[-1] / L:.2f} laps of {L:.2f} m; mean speed while '
          f'moving {np.mean([v for *_, v in truth if v > 0.3]) if any(v > 0.3 for *_, v in truth) else 0:.2f} m/s')
    passes, behind = 0, False
    for r in rows:
        if -3 < r['rel'] < 0:
            behind = True
        elif r['rel'] > 1.0 and behind:
            passes, behind = passes + 1, False
    print(f'  passes {passes}')
c = Counter(v[1] for _, v in status)
print(f'  planner statuses {dict(c)}')
