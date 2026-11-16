"""How much room an evasion needs on porto, against how much there is.

Usage: room_needed.py TRACK_DIR BOUND_MARGIN RUN_DIR

Width: a region's band plus car_width + evasion_dist must fit inside the usable track
(the track less bound_margin each side). Length: one evasion is the lead-in (a
transition out to the apex), the region itself, the apex hold past it, and the rejoin;
with the spliner's defaults and the region lengths seen in the run. Then the stretches
of porto where the usable width is enough, and how long they are.
"""
import os
import pickle
import sys

import numpy as np

from planning.frenet import FrenetConverter
from planning.spliner import SplinerParams, transition_length

conv = FrenetConverter.from_track(sys.argv[1])
margin = float(sys.argv[2])
rec = pickle.load(open(os.path.join(sys.argv[3], 'record.pkl'), 'rb'))
p = SplinerParams(evasion_dist=0.25, bound_margin=margin)
regs = [r[0] for t, r in rec['regions'] if r and r[0][4] < 10]
band = float(np.median([r[4] - r[3] for r in regs]))
length = np.array([r[1] - r[0] for r in regs])
need_w = band + p.car_width + p.evasion_dist
apex = band / 2 + p.car_width / 2 + p.evasion_dist          # region centred on the raceline
lead = transition_length(apex, p) + p.car_front + p.apex_margin
hold = p.car_rear + p.apex_margin
rejoin = max(p.back_to_raceline_after, transition_length(apex, p))
need_l = lead + np.median(length) + hold + rejoin
print(f'width needed: band {band:.2f} + car {p.car_width:.2f} + clearance {p.evasion_dist:.2f} '
      f'= {need_w:.2f} m usable ({need_w + 2 * margin:.2f} m wall to wall at margin {margin})')
print(f'length needed: lead-in {lead:.2f} + region (median {np.median(length):.2f}, p90 '
      f'{np.percentile(length, 90):.2f}) + hold {hold:.2f} + rejoin {rejoin:.2f} = '
      f'{need_l:.2f} m (only the region itself must be at full width; the lead-in and '
      f'rejoin narrow towards the raceline)')
s = np.linspace(0, conv.length, 2000, endpoint=False)
hi, lo = conv.width_at(s)
usable = (np.asarray(hi) - margin) - (np.asarray(lo) + margin)
ok = usable >= need_w
# Contiguous stretches, wrapping across the start line.
k = int(np.argmin(ok)) if not ok.all() else 0
okr, sr = np.roll(ok, -k), np.roll(s, -k)
runs, start = [], None
for i, v in enumerate(okr):
    if v and start is None:
        start = i
    if (not v or i == len(okr) - 1) and start is not None:
        end = i if not v else i + 1
        runs.append((sr[start], (end - start) * conv.length / len(s)))
        start = None
print(f'porto: usable width >= {need_w:.2f} m on {ok.mean() * 100:.0f}% of the lap, in stretches:')
for st, ln in sorted(runs, key=lambda r: -r[1]):
    print(f'  from s {st:5.2f} m: {ln:4.2f} m long')
print(f'usable width over the lap: min {usable.min():.2f}, median {np.median(usable):.2f}, '
      f'max {usable.max():.2f} m')
