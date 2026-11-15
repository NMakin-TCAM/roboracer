"""Footprint geometry from closed_loop.py."""
import math

import numpy as np


def box(cx, cy, yaw, front, rear, width):
    c, s = math.cos(yaw), math.sin(yaw)
    pts = [(front, width / 2), (front, -width / 2), (-rear, -width / 2), (-rear, width / 2)]
    return np.array([(cx + c * x - s * y, cy + s * x + c * y) for x, y in pts])


def gap(a, b):
    """Signed distance between convex polygons: > 0 apart, < 0 penetration depth (SAT)."""
    overlap = math.inf
    for poly in (a, b):
        for i in range(len(poly)):
            e = poly[(i + 1) % len(poly)] - poly[i]
            n = np.array([-e[1], e[0]]) / np.linalg.norm(e)
            pa, pb = a @ n, b @ n
            o = min(pa.max(), pb.max()) - max(pa.min(), pb.min())
            if o <= 0:
                return min(seg_dist(p, q, r) for poly1, poly2 in ((a, b), (b, a))
                           for p in poly1 for q, r in zip(poly2, np.roll(poly2, -1, 0)))
            overlap = min(overlap, o)
    return -overlap


def seg_dist(p, q, r):
    d = r - q
    t = np.clip(np.dot(p - q, d) / np.dot(d, d), 0, 1)
    return float(np.linalg.norm(p - (q + t * d)))
