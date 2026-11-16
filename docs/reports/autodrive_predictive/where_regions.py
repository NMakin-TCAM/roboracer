"""Where on the track the collision regions are, and how much track is usable there.

Usage: where_regions.py TRACK_DIR BOUND_MARGIN RUN_DIR...   (closed_loop.py records)

For each ended region (not a wall): its s_center, its d band width, and the usable
track width at s_center (d_left - d_right of the track, less bound_margin each side).
The spliner needs the region's band plus car_width + evasion_dist (0.55 m) on one side
inside that. Binned by s, so the narrow places are visible.
"""
import os
import pickle
import sys

import numpy as np

from planning.frenet import FrenetConverter

conv = FrenetConverter.from_track(sys.argv[1])
margin = float(sys.argv[2])
NEED = 0.30 + 0.25
L = conv.length
bins = np.arange(0, L + 2, 2.0)
for run in sys.argv[3:]:
    rec = pickle.load(open(os.path.join(run, 'record.pkl'), 'rb'))
    regs = [r[0] for t, r in rec['regions'] if r and r[0][4] < 10]
    sc = np.array([r[2] % L for r in regs])
    band = np.array([r[4] - r[3] for r in regs])
    hi, lo = conv.width_at(sc)
    usable = (np.asarray(hi) - margin) - (np.asarray(lo) + margin)
    fits = usable >= band + NEED
    print(f'\n== {os.path.basename(run)}: {len(regs)} ended regions; band p50 {np.median(band):.2f} m, '
          f'usable width p50 {np.median(usable):.2f} m; band + {NEED} fits the usable width in '
          f'{fits.mean() * 100:.1f}% (ignoring where the band sits)')
    print('   s bin    share   usable   band   fits')
    idx = np.digitize(sc, bins) - 1
    for k in range(len(bins) - 1):
        m = idx == k
        if m.sum() < 0.02 * len(sc):
            continue
        print(f'  {bins[k]:4.0f}-{bins[k + 1]:<4.0f} {m.mean() * 100:5.1f}%   {np.median(usable[m]):5.2f}  '
              f'{np.median(band[m]):5.2f}  {fits[m].mean() * 100:5.1f}%')
