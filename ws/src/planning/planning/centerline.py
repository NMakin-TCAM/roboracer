"""Centerline + track widths from a Cartographer occupancy map (map.pgm / map.yaml).

  free space  -> fill unobserved (unknown-only) islands -> keep the region the car drove in
              -> morphological cleanup
  skeletonize -> prune spurs -> the loop: with trajectory.csv the skeleton path that follows the
                 mapping lap (its length must match the lap's), else the cycle through the map
                 origin -> periodic spline, uniform step
  widths      -> ray cast along the normal to the first non-free cell (occupied or unknown),
                 capped by the distance transform

Output is TUM's reference-track format in the map frame: x_m, y_m, w_tr_right_m, w_tr_left_m,
one row per point in driving direction, starting near the start pose. The loop is closed
implicitly (the last row is not a copy of the first), as TUM expects.
"""
import os

import numpy as np
import yaml
from PIL import Image
from scipy import interpolate, ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from skimage.morphology import disk, remove_small_holes, skeletonize

from planning.track_geometry import uncross_normals

# 8-neighbourhood offsets (row, col)
NB8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


class OccupancyMap:
    """map_server map: free / occupied / unknown masks plus pixel <-> map-frame conversion."""

    def __init__(self, yaml_path):
        with open(yaml_path) as f:
            meta = yaml.safe_load(f)
        img_path = meta['image']
        if not os.path.isabs(img_path):
            img_path = os.path.join(os.path.dirname(os.path.abspath(yaml_path)), img_path)
        img = np.asarray(Image.open(img_path).convert('L'), dtype=np.uint8)

        self.img = img
        self.res = float(meta['resolution'])
        self.origin = np.array(meta['origin'][:3], dtype=float)   # x, y, yaw of pixel (0, H-1)
        mode = meta.get('mode', 'trinary')
        negate = int(meta.get('negate', 0))
        occ_th = float(meta.get('occupied_thresh', 0.65))
        free_th = float(meta.get('free_thresh', 0.25))

        # Same rule as map_server: occupancy probability from the pixel shade.
        p = img / 255.0 if negate else (255.0 - img) / 255.0
        self.occupied = p > occ_th
        self.free = p < free_th
        if mode == 'trinary' and not negate:
            # map_saver writes unknown as 205 (p = 0.196), below the usual 0.25 free_thresh
            self.free &= img != 205
        self.h, self.w = img.shape

    def to_world(self, rc):
        """(row, col) pixel centres -> (x, y) in the map frame (image row 0 is the top)."""
        rc = np.atleast_2d(rc).astype(float)
        u = (rc[:, 1] + 0.5) * self.res
        v = (self.h - 1 - rc[:, 0] + 0.5) * self.res
        c, s = np.cos(self.origin[2]), np.sin(self.origin[2])
        return np.column_stack((self.origin[0] + c * u - s * v, self.origin[1] + s * u + c * v))

    def to_pixel(self, xy):
        """(x, y) map frame -> fractional (row, col)."""
        xy = np.atleast_2d(xy).astype(float) - self.origin[:2]
        c, s = np.cos(self.origin[2]), np.sin(self.origin[2])
        u = c * xy[:, 0] + s * xy[:, 1]
        v = -s * xy[:, 0] + c * xy[:, 1]
        return np.column_stack((self.h - 1 - (v / self.res - 0.5), u / self.res - 0.5))


def load_trajectory(path):
    """trajectory.csv (# t,x_m,y_m,yaw_rad) -> Nx2 xy, or None if missing/empty."""
    if not path or not os.path.isfile(path):
        return None
    data = np.loadtxt(path, delimiter=',', comments='#', ndmin=2)
    if data.shape[0] < 2:
        return None
    return data[:, 1:3]


def fill_unobserved(m, max_occ_frac=0.01):
    """Free space plus enclosed non-free islands that are (almost) all unknown.

    Cells the lidar never saw, e.g. under the car at the start of the mapping lap, form unknown
    islands inside the track of any size. A real infield is enclosed by occupied walls: on the
    maps we have its non-free component is 12-16 % occupied cells, an unobserved island 0 %.
    Islands touching the image border (the outside) are never filled.
    """
    lab, _ = ndimage.label(~m.free, structure=np.ones((3, 3)))
    size = np.bincount(lab.ravel())
    occ = np.bincount(lab.ravel(), weights=m.occupied.ravel(), minlength=size.size)
    fill = occ <= max_occ_frac * size
    fill[0] = False
    fill[np.unique(np.concatenate((lab[0], lab[-1], lab[:, 0], lab[:, -1])))] = False
    return m.free | fill[lab]


def carve_trajectory(m, free, traj, radius_m):
    """Mark the cells within radius_m of the driven path as free (the car was there)."""
    path = np.zeros_like(free)
    dense = resample_polyline(traj, m.res / 2)
    rc = np.round(m.to_pixel(dense)).astype(int)
    ok = (rc[:, 0] >= 0) & (rc[:, 0] < m.h) & (rc[:, 1] >= 0) & (rc[:, 1] < m.w)
    path[rc[ok, 0], rc[ok, 1]] = True
    r = max(1, int(round(radius_m / m.res)))
    return free | ndimage.binary_dilation(path, structure=disk(r))


def resample_polyline(xy, step):
    """Points every `step` metres along an open polyline (first point kept)."""
    seg = np.hypot(*np.diff(xy, axis=0).T)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    keep = np.concatenate(([True], seg > 1e-9))
    s, xy = s[keep], xy[keep]
    t = np.arange(0.0, s[-1], step)
    return np.column_stack((np.interp(t, s, xy[:, 0]), np.interp(t, s, xy[:, 1])))


def track_mask(m, seeds_xy, open_radius_m=0.1, hole_area_m2=0.5, free=None):
    """Free region the car drove in, with thin leaks through wall gaps cut and small holes filled."""
    free = fill_unobserved(m) if free is None else free
    r = max(1, int(round(open_radius_m / m.res)))
    opened = ndimage.binary_opening(free, structure=disk(r))
    labels, n = ndimage.label(opened)                 # 4-connected: no leaks through diagonal gaps
    if n == 0:
        raise RuntimeError('map has no free space')

    def label_at(xy):
        rc = np.round(m.to_pixel(xy)).astype(int)
        ok = (rc[:, 0] >= 0) & (rc[:, 0] < m.h) & (rc[:, 1] >= 0) & (rc[:, 1] < m.w)
        lab = labels[rc[ok, 0], rc[ok, 1]]
        return lab[lab > 0]

    keep = 0
    if seeds_xy is not None:
        lab = label_at(seeds_xy)
        if lab.size:
            keep = np.bincount(lab).argmax()
    if keep == 0:
        lab = label_at(np.zeros((1, 2)))               # map origin = where mapping started
        keep = lab[0] if lab.size else 0
    if keep == 0:
        keep = np.bincount(labels.ravel())[1:].argmax() + 1

    # re-grow into the free cells the opening shaved off the walls
    mask = ndimage.binary_dilation(labels == keep, structure=disk(r)) & free
    # specks of noise inside the track, but never the infield
    mask = remove_small_holes(mask, area_threshold=max(1, int(hole_area_m2 / m.res ** 2)))
    return mask


def prune_spurs(skel):
    """Repeatedly delete end points (<= 1 neighbour); only closed loops survive."""
    skel = skel.copy()
    k = np.ones((3, 3), dtype=int)
    k[1, 1] = 0
    while True:
        nb = ndimage.convolve(skel.astype(int), k, mode='constant')
        ends = skel & (nb <= 1)
        if not ends.any():
            return skel
        skel &= ~ends


def skeleton_graph(skel):
    """Skeleton pixels (rr, cc) and their 8-connected graph as (rows, cols, weights) in pixels."""
    rr, cc = np.nonzero(skel)
    if rr.size < 10:
        raise RuntimeError('skeleton has no closed loop: is the track closed in the map?')
    idx = -np.ones(skel.shape, dtype=int)
    idx[rr, cc] = np.arange(rr.size)

    # 8-connected pixel graph
    rows, cols, wts = [], [], []
    for dr, dc in NB8:
        r2, c2 = rr + dr, cc + dc
        ok = (r2 >= 0) & (r2 < skel.shape[0]) & (c2 >= 0) & (c2 < skel.shape[1])
        j = np.full(rr.size, -1)
        j[ok] = idx[r2[ok], c2[ok]]
        ok = j >= 0
        rows.append(np.nonzero(ok)[0])
        cols.append(j[ok])
        wts.append(np.full(ok.sum(), np.hypot(dr, dc)))
    rows, cols, wts = map(np.concatenate, (rows, cols, wts))
    return rr, cc, rows, cols, wts


def skeleton_loop(skel, start_rc):
    """Ordered pixels of the loop through the skeleton pixel nearest start_rc (no trajectory)."""
    rr, cc, rows, cols, wts = skeleton_graph(skel)
    order = np.argsort(np.hypot(rr - start_rc[0], cc - start_rc[1]))
    for p0 in order[:200]:
        nbrs = cols[rows == p0]
        # need two neighbours that are not adjacent to each other: one on each side of p0
        pair = None
        for i, a in enumerate(nbrs):
            for b in nbrs[i + 1:]:
                if max(abs(rr[a] - rr[b]), abs(cc[a] - cc[b])) > 1:
                    pair = (a, b)
                    break
            if pair:
                break
        if pair is None:
            continue
        # cut p0 out, then the shortest way from one side to the other is the loop
        keep = (rows != p0) & (cols != p0)
        g = coo_matrix((wts[keep], (rows[keep], cols[keep])), shape=(rr.size, rr.size)).tocsr()
        dist, pred = dijkstra(g, indices=pair[0], return_predecessors=True)
        if not np.isfinite(dist[pair[1]]) or dist[pair[1]] < 20:
            continue
        path = [pair[1]]
        while path[-1] != pair[0]:
            path.append(pred[path[-1]])
        loop = [p0] + path[::-1]
        return np.column_stack((rr[loop], cc[loop]))
    raise RuntimeError('could not find a closed loop in the skeleton')


def off_track(m, mask, xy, min_gap_m=1.0):
    """Places (one per cluster) where points of xy are not on the track mask."""
    rc = np.round(m.to_pixel(xy)).astype(int).clip(0, [m.h - 1, m.w - 1])
    bad = xy[~mask[rc[:, 0], rc[:, 1]]]
    spots = []
    for p in bad:
        if all(np.hypot(*(p - q)) > min_gap_m for q in spots):
            spots.append(p)
    return spots


def trajectory_loop(skel, m, mask, traj, step_m=0.5, max_detour=3.0):
    """Ordered skeleton pixels following the mapping lap: the lap is snapped to the skeleton every
    step_m and consecutive snaps are joined by shortest skeleton paths. Unlike skeleton_loop this
    never takes another cycle (an island, a short-cut between corridors) than the one driven.
    """
    rr, cc, rows, cols, wts = skeleton_graph(skel)
    sxy = m.to_world(np.column_stack((rr, cc)))
    pts = resample_polyline(np.vstack((traj, traj[:1])), step_m)

    # snap each lap point to the nearest skeleton pixel it can see through the track (no wall between)
    tree = cKDTree(sxy)
    _, cand = tree.query(pts, k=min(16, len(sxy)))
    snaps = []
    for p, cs in zip(pts, cand):
        for c in cs:
            seg = p + np.linspace(0, 1, 20)[:, None] * (sxy[c] - p)
            rc = np.round(m.to_pixel(seg)).astype(int).clip(0, [m.h - 1, m.w - 1])
            if mask[rc[:, 0], rc[:, 1]].all():
                snaps.append(c)
                break
        else:
            snaps.append(-1)
    snaps = np.array(snaps)
    if (snaps < 0).any():
        raise RuntimeError('mapping lap leaves the track at ' + fmt_spots(off_track(m, mask, pts[snaps < 0])))
    keep = np.concatenate(([True], snaps[1:] != snaps[:-1]))
    snaps = snaps[keep]
    pts = pts[keep]

    g = coo_matrix((wts, (rows, cols)), shape=(rr.size, rr.size)).tocsr()
    src = np.unique(snaps)
    limit = max_detour * 10 * step_m / m.res + 20
    dist, pred = dijkstra(g, indices=src, return_predecessors=True, limit=limit)
    row = {s: i for i, s in enumerate(src)}

    loop = []
    n = len(snaps)
    for k in range(n):
        a, b = snaps[k], snaps[(k + 1) % n]
        i = row[a]
        direct = np.hypot(*(sxy[a] - sxy[b])) / m.res
        if not np.isfinite(dist[i, b]) or dist[i, b] > max_detour * direct + 20:
            raise RuntimeError('mapping lap leaves the track between (%.1f, %.1f) and (%.1f, %.1f)'
                               % (*pts[k], *pts[(k + 1) % n]))
        seg = [b]
        while seg[-1] != a:
            seg.append(pred[i, seg[-1]])
        loop.extend(seg[::-1][:-1])                 # a .. (b excluded, it starts the next piece)

    # drop exact retraces (in and out of a dead end) left by snapping
    stack = []
    for p in loop:
        if len(stack) >= 2 and stack[-2] == p:
            stack.pop()
        elif not stack or stack[-1] != p:
            stack.append(p)
    return np.column_stack((rr[stack], cc[stack]))


def first_lap(traj, leave_m=3.0, back_m=1.0):
    """The first lap of a driven trajectory: up to where it first comes back within back_m of its
    start after having been leave_m away. map_track.sh runs drive on past the start line (or a
    second lap). The whole trajectory if it never comes back."""
    d0 = np.hypot(*(traj - traj[0]).T)
    away = np.flatnonzero(d0 > leave_m)
    if len(away):
        back = np.flatnonzero(d0[away[0]:] < back_m)
        if len(back):
            return traj[:away[0] + back[0] + 1]
    return traj


def lap_length(traj):
    """Closed length of the first lap."""
    lap = first_lap(traj)
    return np.sum(np.hypot(*np.diff(np.vstack((lap, lap[:1])), axis=0).T))


def fmt_spots(spots, n=6):
    s = ', '.join('(%.1f, %.1f)' % tuple(p) for p in spots[:n])
    return s + (' ...' if len(spots) > n else '')


def smooth_loop(xy, step, smooth_m):
    """Periodic smoothing spline through a closed polyline, resampled every `step` metres."""
    # drop duplicates, the spline fit dislikes them
    d = np.hypot(*np.diff(np.vstack((xy, xy[:1])), axis=0).T)
    xy = xy[d > 1e-6]
    pts = np.vstack((xy, xy[:1]))
    s = len(xy) * smooth_m ** 2
    tck, _ = interpolate.splprep(pts.T, s=s, per=1)
    u = np.linspace(0, 1, 20 * len(xy), endpoint=False)
    dense = np.column_stack(interpolate.splev(u, tck))
    seg = np.hypot(*np.diff(np.vstack((dense, dense[:1])), axis=0).T)
    s_dense = np.concatenate(([0.0], np.cumsum(seg)))
    length = s_dense[-1]
    n = max(10, int(round(length / step)))
    s_new = np.linspace(0, length, n, endpoint=False)
    u_new = np.interp(s_new, s_dense, np.append(u, 1.0))
    out = np.column_stack(interpolate.splev(u_new, tck))
    return out, length


def normals_left(xy):
    """Unit normals pointing left of the direction of travel, closed loop."""
    t = np.roll(xy, -1, axis=0) - np.roll(xy, 1, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    return np.column_stack((-t[:, 1], t[:, 0]))


def ray_widths(m, mask, xy, normal, max_m=5.0):
    """Distance from each point along +normal to the first cell outside the track mask."""
    ds = m.res / 4
    t = np.arange(ds, max_m, ds)
    out = np.full(len(xy), max_m)
    for i, (p, n) in enumerate(zip(xy, normal)):
        rc = np.round(m.to_pixel(p + t[:, None] * n)).astype(int)
        inside = ((rc[:, 0] >= 0) & (rc[:, 0] < m.h) & (rc[:, 1] >= 0) & (rc[:, 1] < m.w))
        hit = ~inside
        hit[inside] = ~mask[rc[inside, 0], rc[inside, 1]]
        j = np.argmax(hit)
        if hit[j]:
            out[i] = t[j] - ds / 2
    return out


def signed_area(xy):
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def orient(xy, traj):
    """Reverse the loop if needed: follow the mapping lap if given, else counter-clockwise."""
    if traj is not None and len(traj) > 10:
        traj = traj[::max(1, len(traj) // 2000)]
        near = np.argmin(((traj[:, None, :] - xy[None, :, :]) ** 2).sum(-1), axis=1)
        di = np.diff(near)
        n = len(xy)
        di = (di + n // 2) % n - n // 2                # wrap around the start
        if di.sum() != 0:
            return xy if di.sum() > 0 else xy[::-1]
    return xy if signed_area(xy) > 0 else xy[::-1]


def extract_centerline(yaml_path, trajectory_path=None, step=0.1, smooth_m=0.05,
                       open_radius_m=0.1, hole_area_m2=0.5, width_slack_m=0.2,
                       length_tol=0.15, carve_radius_m=None):
    """Returns (track Nx4 [x, y, w_right, w_left], OccupancyMap, track mask).

    With a trajectory the loop follows the mapping lap, and its length must be within length_tol
    of the lap's, else RuntimeError. carve_radius_m: treat cells within that distance of the lap
    as free (only for tracks whose map shows walls across the driven path, e.g. an overpass).
    """
    m = OccupancyMap(yaml_path)
    traj = load_trajectory(trajectory_path)
    free = fill_unobserved(m)
    if traj is not None and carve_radius_m:
        free = carve_trajectory(m, free, traj, carve_radius_m)
    mask = track_mask(m, traj, open_radius_m, hole_area_m2, free=free)
    skel = prune_spurs(skeletonize(mask))

    if traj is None:
        start_xy = np.zeros(2)
        loop_rc = skeleton_loop(skel, m.to_pixel(start_xy)[0])
        xy, length = smooth_loop(m.to_world(loop_rc), step, smooth_m)
        xy = orient(xy, None)
        xy = np.roll(xy, -np.argmin(np.hypot(*(xy - start_xy).T)), axis=0)
    else:
        spots = off_track(m, mask, traj)
        if spots:
            walls = off_track(m, ~m.occupied, traj)
            raise RuntimeError(
                'the mapping lap leaves the free space at %s%s. The map does not match the drive '
                '(overlapping SLAM submaps?) or the track has an overpass. Remap; only if the track '
                'really crosses itself, rerun with --carve-trajectory.'
                % (fmt_spots(spots), ' (through walls at %s)' % fmt_spots(walls) if walls else ''))
        loop_rc = trajectory_loop(skel, m, mask, first_lap(traj))
        xy, length = smooth_loop(m.to_world(loop_rc), step, smooth_m)
        lap = lap_length(traj)
        if abs(length - lap) > length_tol * lap:
            raise RuntimeError('centerline is %.1f m but the mapping lap is %.1f m (tolerance %d %%): '
                               'the loop does not follow the lap' % (length, lap, 100 * length_tol))

    # Widths along the normal, capped at the distance transform (nearest wall in any direction)
    # plus width_slack_m: the cap keeps a gap in a wall or a free strip behind it from making the
    # track look wider; the slack leaves room for the spline being slightly off the medial axis.
    nl = normals_left(xy)
    edt = ndimage.distance_transform_edt(mask) * m.res - 0.5 * m.res
    d = ndimage.map_coordinates(edt, m.to_pixel(xy).T, order=1)
    if np.any(d <= 0):
        raise RuntimeError('centerline leaves the free space; check the map')
    w_left = np.minimum(ray_widths(m, mask, xy, nl), d + width_slack_m)
    w_right = np.minimum(ray_widths(m, mask, xy, -nl), d + width_slack_m)
    w_right, w_left = uncross_normals(xy, nl, w_right, w_left)
    return np.column_stack((xy, w_right, w_left)), m, mask


def write_centerline(path, track):
    np.savetxt(path, track, delimiter=',', fmt='%.4f',
               header='x_m,y_m,w_tr_right_m,w_tr_left_m')
