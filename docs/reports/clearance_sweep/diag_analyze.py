"""Contact diagnosis: the gap achieved against the gap the plan would have given, per cycle.

Usage: diag_analyze.py TRACK_DIR RUN_DIR...   (records from closed_loop.py)

For every 20 ms car sample after the handover:
  achieved  the car as driven, against the opponent's true footprint
  plan      the car put exactly on the plan (the latest planner output at that time: the
            evasion's d at the car's s while evading, the raceline otherwise), same s,
            heading along the planned line, against the same true footprint
  region    the car on the plan against the collision region the plan was made around
            (Frenet boxes), i.e. what the planner believes its clearance is
  tracking  the car's d minus the plan's d at the car's s
achieved - plan is what tracking costs; plan - region is what the region gets wrong.
"""
import bisect
import math
import os
import pickle
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from closed_loop_geom import box, gap  # noqa: E402

from planning.frenet import FrenetConverter  # noqa: E402

conv = FrenetConverter.from_track(sys.argv[1])
L = conv.length
LAM = L / max(1, round(L / 12.0))
EGO = (0.45, 0.10, 0.30)
OPP_LEN, OPP_W = 0.50, 0.30


def truth(rec, v):
    t = np.array([o[0] for o in rec['opp_true']])
    s = np.array([conv.to_frenet(o[1], o[2])[0] for o in rec['opp_true']])
    ph = np.angle(np.mean(np.exp(1j * 2 * np.pi * ((s - v * t) % L) / L))) * L / (2 * np.pi)
    res = np.abs(conv.ds(ph % L, (s - v * t) % L))
    f = lambda tt: ((ph + v * tt) % L,
                    0.4 * math.sin(2 * math.pi * ((ph + v * tt) % L) / LAM))
    return f, float(res.max()), float(np.median(res))


def frenet_box_gap(a, b):
    """Distance between axis-aligned Frenet boxes (s0, s1, d0, d1); negative = overlap depth."""
    ds = max(b[0] - a[1], a[0] - b[1])
    dd = max(b[2] - a[3], a[2] - b[3])
    if ds < 0 and dd < 0:
        return max(ds, dd)
    return math.hypot(max(ds, 0), max(dd, 0))


for run in sys.argv[2:]:
    rec = pickle.load(open(os.path.join(run, 'record.pkl'), 'rb'))
    m = rec['meta']
    h, ev = m['handover'], m['evasion_dist']
    opp, worst_fit, med_fit = truth(rec, m['opp_v'])
    status = {round(x[3], 4): (x[1], x[2]) for x in rec['status']}
    plans = [(p[0], status.get(round(p[1], 4), ('?', '?')), p[3]) for p in rec['plan']]
    pt = [p[0] for p in plans]
    rg = rec['regions']
    rgt = [r[0] for r in rg]
    rows = []
    for t, x, y, yaw, v in rec['ego']:
        if t <= h:
            continue
        i = bisect.bisect_right(pt, t) - 1
        if i < 0:
            continue
        _, (st, note), path = plans[i]
        s_car, d_car = conv.to_frenet(x, y)
        s_car, d_car = float(s_car), float(d_car)
        if st == 'evading' and len(path) > 1:
            u = np.concatenate(([0.0], np.cumsum(np.diff(path[:, 0]) % L)))
            # The path starts at the first sample at or ahead of the car (up to one sample
            # in front of it); past its end it is back on the raceline. np.interp holds the
            # end values for both.
            uc = float(conv.ds(path[0, 0], s_car))
            d_plan = float(np.interp(uc, u, path[:, 1]))
            slope = float(np.interp(uc, u[1:], np.diff(path[:, 1]) / np.maximum(np.diff(u), 1e-6)))
        else:
            d_plan, slope = 0.0, 0.0
        os_, od = opp(t)
        ox, oy = conv.to_cartesian(os_, od)
        obox = box(float(ox), float(oy), float(conv.heading_at(os_)), OPP_LEN / 2, OPP_LEN / 2, OPP_W)
        g_act = gap(box(x, y, yaw, *EGO), obox)
        px, py = conv.to_cartesian(s_car, d_plan)
        pyaw = float(conv.heading_at(s_car)) + math.atan(slope)
        g_plan = gap(box(float(px), float(py), pyaw, *EGO), obox)
        j = bisect.bisect_right(rgt, pt[i]) - 1
        g_reg, reg = math.nan, None
        if j >= 0 and rg[j][1]:
            reg = rg[j][1][0]                              # s_start, s_end, s_c, d_r, d_l, d_c
            u0 = float(conv.ds(s_car, reg[0]))
            car = (-EGO[1], EGO[0], d_plan - EGO[2] / 2, d_plan + EGO[2] / 2)
            g_reg = frenet_box_gap(car, (u0, u0 + (reg[1] - reg[0]), reg[3], reg[4]))
        rel = float(conv.ds(os_, s_car))
        rows.append(dict(t=t, v=v, st=st, note=note, g_act=g_act, g_plan=g_plan, g_reg=g_reg,
                         track=d_car - d_plan, d_car=d_car, d_plan=d_plan, rel=rel,
                         opp_d=od, reg=reg))

    print(f'\n===== evasion_dist {ev}: {len(rows)} samples; opponent truth fit to stamped '
          f'markers: median {med_fit * 100:.1f} cm, worst {worst_fit * 100:.1f} cm')
    # Contact events: contiguous runs with g_act <= 0.
    events, cur = [], []
    for r in rows:
        if r['g_act'] <= 0:
            cur.append(r)
        elif cur:
            events.append(cur)
            cur = []
    if cur:
        events.append(cur)
    for e in events:
        w = min(e, key=lambda r: r['g_act'])
        reg = w['reg']
        print(f"  contact t {e[0]['t']:.2f}-{e[-1]['t']:.2f} ({len(e) * 0.02:.2f} s) worst: "
              f"achieved {w['g_act']:+.3f} plan {w['g_plan']:+.3f} region {w['g_reg']:+.3f} "
              f"tracking {w['track']:+.3f} (car d {w['d_car']:+.2f}, plan d {w['d_plan']:+.2f}) "
              f"v {w['v']:.2f} status {w['st']!r} ({w['note']})")
        if reg:
            print(f"      region d band [{reg[3]:+.2f}, {reg[4]:+.2f}] s {reg[0]:.2f}..{reg[1]:.2f}; "
                  f"opponent true d {w['opp_d']:+.2f}; car {w['rel']:+.2f} m ahead of opponent")
        statuses = Counter((r['st'], r['note']) for r in e)
        print(f'      statuses during the contact: {dict(statuses)}')
        # What the planner said in the second before.
        before = [r for r in rows if e[0]['t'] - 1.0 <= r['t'] < e[0]['t']]
        seq = []
        for r in before:
            k = (r['st'], r['note'])
            if not seq or seq[-1][0] != k:
                seq.append((k, r['t']))
        print('      1 s before: ' + ' -> '.join(f"{k[0]} @{t:.2f}" for k, t in seq))
    # Distributions while alongside (car within 1.5 m of the opponent in s) and evading.
    side = [r for r in rows if abs(r['rel']) < 1.5 and r['st'] == 'evading']
    if side:
        tr = np.array([r['track'] for r in side])
        cost = np.array([r['g_plan'] - r['g_act'] for r in side])
        gr = np.array([r['g_reg'] for r in side if not math.isnan(r['g_reg'])])
        gp = np.array([r['g_plan'] for r in side])
        print(f'  alongside & evading ({len(side)} samples): |tracking| p50 {np.percentile(np.abs(tr), 50):.3f} '
              f'p95 {np.percentile(np.abs(tr), 95):.3f} max {np.abs(tr).max():.3f}; '
              f'plan-achieved p50 {np.percentile(cost, 50):+.3f} max {cost.max():+.3f}; '
              f'plan gap min {gp.min():+.3f}; region gap min {gr.min() if len(gr) else math.nan:+.3f}')
    planned = [r for r in rows if r['st'] in ('clear', 'evading')]
    if planned:
        w = min(planned, key=lambda r: r['g_plan'])
        neg = sum(r['g_plan'] < 0 for r in planned)
        a = min(rows, key=lambda r: r['g_act'])
        print(f'  closest achieved at t {a["t"]:.1f}: {a["g_act"]:+.3f} m, status {a["st"]}, '
              f'planned {a["g_plan"]:+.3f}, tracking {a["track"]:+.3f}, car '
              f'{a["rel"]:+.2f} m ahead of the opponent, v {a["v"]:.2f}')
        print(f'  SUMMARY closest planned {w["g_plan"]:+.3f} m (t {w["t"]:.1f}, {w["st"]}), '
              f'{neg} planned-through cycles; closest achieved '
              f'{min(r["g_act"] for r in rows):+.3f} m; contact '
              f'{sum(r["g_act"] <= 0 for r in rows) * 0.02:.2f} s')
    allside = [r for r in rows if abs(r['rel']) < 1.5]
    c = Counter(r['st'] for r in allside)
    print(f'  alongside, any status: {dict(c)}; min achieved {min(r["g_act"] for r in allside):+.3f}')
