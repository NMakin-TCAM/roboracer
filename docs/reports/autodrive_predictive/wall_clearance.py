"""The car footprint's distance to the track edges (centerline.csv widths) before each sim collision.

Usage: wall_clearance.py TRACK_DIR RUN_DIR
Footprint corners (rear axle + 0.45 / - 0.10, +/- 0.15) are put in Frenet; clearance is the
smallest of d_left - d and d - d_right over them. The truth point is taken 0.043 m ahead of
the rear axle (sim_analyze's median along-track offset). Edges are the map-extracted
widths, so treat a few centimetres either way as noise.
"""
import math
import os
import pickle
import sys

import numpy as np

from planning.frenet import FrenetConverter

THETA = math.radians(-92.693)
T = np.array([0.5065, -4.3836])
conv = FrenetConverter.from_track(sys.argv[1])
rec = pickle.load(open(os.path.join(sys.argv[2], 'record.pkl'), 'rb'))


def w2m(x, y, yaw):
    c, s = math.cos(THETA), math.sin(THETA)
    dx, dy = x - T[0], y - T[1]
    return c * dx + s * dy, -s * dx + c * dy, yaw - THETA


truth = [(t, *w2m(v[1], v[2], v[3]), v[4]) for t, v in rec['/autodrive/roboracer_1/odom']]


def clearance(x, y, yaw):
    worst = math.inf
    for u, w in ((0.45, 0.15), (0.45, -0.15), (-0.10, 0.15), (-0.10, -0.15)):
        cx = x - 0.043 * math.cos(yaw) + u * math.cos(yaw) - w * math.sin(yaw)
        cy = y - 0.043 * math.sin(yaw) + u * math.sin(yaw) + w * math.cos(yaw)
        s, d = (float(a) for a in conv.to_frenet(cx, cy))
        hi, lo = (float(a) for a in conv.width_at(s))
        worst = min(worst, hi - d, d - lo)
    return worst


for tc, why in rec['/autodrive/roboracer_1/sim_reset']:
    pre = [r for r in truth if tc - 2.0 <= r[0] < tc - 0.02]
    c = [(clearance(x, y, yaw), t) for t, x, y, yaw, v in pre]
    w = min(c)
    print(f'{why} at {tc:.2f}: closest footprint-to-edge {w[0]:+.3f} m at {w[1] - tc:+.2f} s')
moving = [r for r in truth if r[4] > 0.3]
alls = np.array([clearance(x, y, yaw) for t, x, y, yaw, v in moving[::5]])
print(f'whole run while moving: footprint-to-edge p1 {np.percentile(alls, 1):+.3f} m, '
      f'p5 {np.percentile(alls, 5):+.3f}, median {np.median(alls):+.3f}')
