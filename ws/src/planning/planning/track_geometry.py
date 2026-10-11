"""Track geometry helpers shared by centerline.py (ROS Python) and tum_mintime.py (TUM venv): numpy only."""
import numpy as np


def uncross_normals(xy, nl, w_right, w_left, horizon_m=3.0, margin=0.85):
    """Shrink widths so no two normal segments (right bound -> left bound) within horizon_m cross.

    TUM rejects tracks with crossing normals and its curvilinear model needs 1 - n * kappa > 0;
    both fail on the inside of tight bends when a width exceeds the local radius.
    """
    w_right, w_left = w_right.copy(), w_left.copy()
    seg = np.hypot(*(np.roll(xy, -1, axis=0) - xy).T).mean()
    for k in range(1, max(2, int(horizon_m / seg)) + 1):
        j = np.roll(np.arange(len(xy)), -k)
        nj, dp = nl[j], xy[j] - xy
        den = np.cross(nl, nj)
        ok = np.abs(den) > 1e-9
        den = np.where(ok, den, 1.0)
        # xy_i + a * n_i = xy_j + b * n_j   (a, b > 0: left of the line)
        a = np.cross(dp, nj) / den
        b = np.cross(dp, nl) / den
        wa = np.where(a > 0, w_left, w_right)
        wb = np.where(b > 0, w_left[j], w_right[j])
        # near misses count too: TUM re-smooths and resamples the line, which moves the crossings
        hit = ok & (margin * np.abs(a) < wa) & (margin * np.abs(b) < wb)
        cap_a = np.where(hit, margin * np.abs(a), np.inf)
        cap_b = np.where(hit, margin * np.abs(b), np.inf)
        w_left = np.where(a > 0, np.minimum(w_left, cap_a), w_left)
        w_right = np.where(a <= 0, np.minimum(w_right, cap_a), w_right)
        w_left[j] = np.where(b > 0, np.minimum(w_left[j], cap_b), w_left[j])
        w_right[j] = np.where(b <= 0, np.minimum(w_right[j], cap_b), w_right[j])
    return w_right, w_left
