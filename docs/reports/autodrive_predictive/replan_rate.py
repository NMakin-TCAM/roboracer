"""How often the spliner re-planned before each contact, and how far out the car got.

Usage: replan_rate.py TRACK_DIR RUN_DIR CONTACT_T...   (closed_loop.py record + spliner.log)
"""
import bisect
import os
import pickle
import re
import sys

import numpy as np

from planning.frenet import FrenetConverter

conv = FrenetConverter.from_track(sys.argv[1])
run = sys.argv[2]
rec = pickle.load(open(os.path.join(run, 'record.pkl'), 'rb'))
off = rec['clock_offset']
replans = []
for line in open(os.path.join(run, 'spliner.log')):
    m = re.search(r'\[(\d+\.\d+)\] \[spliner_node\]: evasion re-planned', line)
    if m:
        replans.append(float(m.group(1)) - off)
print(f'{len(replans)} re-plans in the run')
et = [e[0] for e in rec['ego']]
for tc in map(float, sys.argv[3:]):
    w = [t for t in replans if tc - 2.0 <= t <= tc]
    d = [float(conv.to_frenet(e[1], e[2])[1]) for e in rec['ego'] if tc - 2.0 <= e[0] <= tc]
    print(f'contact at {tc:.1f}: {len(w)} re-plans in the 2 s before '
          f'({len(w) / 2.0 / 20 * 100:.0f}% of 20 Hz cycles); car d over those 2 s: '
          f'min {min(d):+.2f} max {max(d):+.2f} (apex needed about +0.65)')
