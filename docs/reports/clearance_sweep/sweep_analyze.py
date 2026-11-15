"""The clearance curve from the sweep records. Usage: sweep_analyze.py TRACK_DIR RUN_DIR..."""
import math
import os
import pickle
import sys
from collections import Counter

import numpy as np

from planning.frenet import FrenetConverter
from planning.spliner import SplinerParams, choose_side

conv = FrenetConverter.from_track(sys.argv[1])


class Obs:
    def __init__(self, r):
        self.s_start, self.s_end, self.s_center, self.d_right, self.d_left, self.d_center = r


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from closed_loop_geom import box, gap  # noqa: E402

L = conv.length
LAM = L / max(1, round(L / 12.0))       # fake_opponent's Weaving snaps the wavelength
OPP_LEN, OPP_W = 0.50, 0.30
EGO_BOX = (0.45, 0.10, 0.30)


def opponent_truth(rec, v):
    """Weaving truth from its own model, phase fitted to the marker samples.

    The marker exists only while the opponent is in sensor range ahead, which loses it
    exactly when the ego is alongside; the model does not.
    """
    t = np.array([o[0] for o in rec['opp']])
    s = np.array([conv.to_frenet(o[1], o[2])[0] for o in rec['opp']])
    ph = np.angle(np.mean(np.exp(1j * 2 * np.pi * ((s - v * t) % L) / L))) * L / (2 * np.pi)
    resid = conv.ds(ph % L, (s - v * t) % L)
    return lambda tt: (ph + v * tt) % L, float(np.max(np.abs(resid)))


rows = []
for run in sys.argv[2:]:
    rec = pickle.load(open(os.path.join(run, 'record.pkl'), 'rb'))
    m = rec['meta']
    h, ev = m['handover'], m['evasion_dist']
    if h is None:
        print(f'{run}: never handed over'); continue
    p = SplinerParams(evasion_dist=ev)
    regs = [Obs(r[0]) for t, r in rec['regions'] if t > h and r]
    walls = sum(o.d_left >= 10 for o in regs)
    passable = sum(choose_side(o, conv, p).side is not None for o in regs)
    ended = [o for o in regs if o.d_left < 10]
    passable_ended = sum(choose_side(o, conv, p).side is not None for o in ended)
    st = [(t, s) for t, s, _ in rec['status'] if t > h]
    blocked = [s for _, s in st if s != 'clear']
    evading = sum(s == 'evading' for s in blocked)
    starts = sum(1 for (_, a), (_, b) in zip(st, st[1:]) if a != 'evading' and b == 'evading')
    ego = np.array([e for e in rec['ego'] if e[0] > h])
    opp_s, fit_err = opponent_truth(rec, m['opp_v'])
    gaps, rel = [], []
    for t, x, y, yaw, _ in ego:
        os_ = opp_s(t)
        od = 0.4 * math.sin(2 * math.pi * os_ / LAM)
        ox, oy = conv.to_cartesian(os_, od)
        oyaw = float(conv.heading_at(os_))
        gaps.append(gap(box(x, y, yaw, *EGO_BOX),
                        box(float(ox), float(oy), oyaw, OPP_LEN / 2, OPP_LEN / 2, OPP_W)))
        rel.append(float(conv.ds(os_, conv.to_frenet(x, y)[0])))
    gaps = np.array(gaps)
    dist = float(np.sum(np.hypot(*np.diff(ego[:, 1:3], axis=0).T)))
    stopped = float(np.sum(ego[1:, 4] < 0.1) * np.median(np.diff(ego[:, 0])))
    # Passes: the ego's rear axle goes from behind the opponent to 1 m ahead of it.
    passes, behind = 0, False
    for r in rel:
        if -3 < r < 0:
            behind = True
        elif r > 1.0 and behind:
            passes, behind = passes + 1, False
    contact = float(np.sum(gaps <= 0) * 0.02)
    print(f'ev {ev}: opponent phase fit, worst marker residual {fit_err * 100:.1f} cm')
    rows.append((ev, len(regs), walls, passable / max(len(regs), 1),
                 passable_ended / max(len(ended), 1), evading / max(len(blocked), 1), starts,
                 passes, gaps.min() if len(gaps) else math.nan, contact,
                 np.percentile(gaps, 1) if len(gaps) else math.nan, dist, dist / (ego[-1, 0] - h),
                 stopped, Counter(blocked).most_common(3)))

print(f"{'clear':>6} {'regions':>7} {'walls':>5} {'pass%':>6} {'pass%end':>8} {'evade%':>6} "
      f"{'starts':>6} {'passes':>6} {'gap_min':>7} {'contact_s':>9} {'gap_p1':>6} "
      f"{'dist_m':>6} {'v_mean':>6} {'stop_s':>6}")
for r in sorted(rows, key=lambda r: -r[0]):
    print(f'{r[0]:6.2f} {r[1]:7d} {r[2]:5d} {r[3] * 100:6.1f} {r[4] * 100:8.1f} {r[5] * 100:6.1f} '
          f'{r[6]:6d} {r[7]:6d} {r[8]:7.3f} {r[9]:9.2f} {r[10]:6.3f} {r[11]:6.0f} {r[12]:6.2f} '
          f'{r[13]:6.1f}')
for r in sorted(rows, key=lambda r: -r[0]):
    print(f'{r[0]:.2f} blocked-cycle statuses: {r[14]}')
