"""Monte Carlo localization (particle filter) with a likelihood-field sensor model, in numpy.

No ROS imports: mcl_node.py wires this to topics, test_mcl.py tests it on synthetic tracks.

Why a likelihood field and not ray casting: the weight of a particle comes from where the lidar
endpoints would land if the car were at that pose, looked up in a distance-to-nearest-wall grid.
That needs no range_libc (no Cython/CUDA build), runs on amd64 and arm64, and is a few vectorized
numpy operations per scan.

The filter tracks a pose; ScanMatcher finds one. Its global search scores every pose in the drivable area
(coarse to fine) against one scan, so the car can find itself standing still: at the start, after a crash, or
when the filter is lost. It reports a second, distinct pose that fits almost as well (a track with
similar-looking corridors), so the caller can refuse to guess. Its local search re-anchors a filter that the
odometry has thrown off (wheel slip, a stalled sensor stream) faster than the cloud can diffuse back.

Poses are (x, y, yaw) of base_link (the rear axle) in the map frame. The map is the map.pgm/map.yaml
that save_map writes (ROS map_server format, trinary).
"""
import math
import os

import numpy as np
import yaml
from PIL import Image
from scipy import ndimage


def wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def compose(a, b):
    """SE(2) a * b: pose b, given in the frame of pose a, expressed in a's parent frame."""
    c, s = math.cos(a[2]), math.sin(a[2])
    return (a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], float(wrap(a[2] + b[2])))


def inverse(a):
    c, s = math.cos(a[2]), math.sin(a[2])
    return (-(c * a[0] + s * a[1]), s * a[0] - c * a[1], float(wrap(-a[2])))


def relative(a, b):
    """Pose b expressed in the frame of pose a (a^-1 * b)."""
    return compose(inverse(a), b)


class LikelihoodMap:
    """Occupancy grid plus its distance-to-nearest-occupied-cell field (metres, capped at max_dist)."""

    def __init__(self, occupied, free, resolution, origin_xy, max_dist=2.0):
        """occupied, free: bool arrays in image order (row 0 = max y); origin = lower-left corner."""
        if not occupied.any():
            raise ValueError('map has no occupied cells')
        self.h, self.w = occupied.shape
        self.res = float(resolution)
        self.ox, self.oy = float(origin_xy[0]), float(origin_xy[1])
        self.max_dist = float(max_dist)
        self.occupied = occupied
        self.free = free
        self.dist = np.minimum(ndimage.distance_transform_edt(~occupied) * self.res,
                               self.max_dist).astype(np.float32)

    @classmethod
    def from_yaml(cls, path, max_dist=2.0):
        with open(path) as f:
            meta = yaml.safe_load(f)
        image = meta['image']
        if not os.path.isabs(image):
            image = os.path.join(os.path.dirname(os.path.abspath(path)), image)
        origin = meta['origin']
        if abs(origin[2]) > 1e-6:
            raise ValueError(f'{path}: a rotated map origin (yaw {origin[2]}) is not supported')
        pix = np.asarray(Image.open(image).convert('L'), dtype=np.float64)
        p = pix / 255.0 if meta.get('negate', 0) else (255.0 - pix) / 255.0   # occupancy probability
        occupied = p > float(meta.get('occupied_thresh', 0.65))
        free = p < float(meta.get('free_thresh', 0.196))
        return cls(occupied, free, meta['resolution'], origin[:2], max_dist)

    def occupancy_grid(self):
        """int8 values for nav_msgs/OccupancyGrid.data (row 0 = min y): 100 occupied, 0 free, -1 unknown."""
        grid = np.full((self.h, self.w), -1, dtype=np.int8)
        grid[self.free] = 0
        grid[self.occupied] = 100
        return np.flipud(grid).ravel()

    def distance(self, x, y):
        """Distance to the nearest wall at map points (any shape); max_dist outside the map."""
        col = np.floor((x - self.ox) / self.res).astype(np.int64)
        row = self.h - 1 - np.floor((y - self.oy) / self.res).astype(np.int64)
        inside = (col >= 0) & (col < self.w) & (row >= 0) & (row < self.h)
        d = self.dist[np.clip(row, 0, self.h - 1), np.clip(col, 0, self.w - 1)]
        return np.where(inside, d, self.max_dist)


def select_beams(ranges, angle_min, angle_increment, n_beams, range_min, range_max):
    """About n_beams evenly spaced beams of a scan, without invalid ones and max-range (no-hit) returns."""
    ranges = np.asarray(ranges, dtype=np.float64)
    step = max(1, int(round(len(ranges) / max(1, n_beams))))
    idx = np.arange(0, len(ranges), step)
    r = ranges[idx]
    ok = np.isfinite(r) & (r > range_min) & (r < range_max * 0.999)
    return r[ok], angle_min + idx[ok] * angle_increment


class ParticleFilter:
    def __init__(self, n=1000, seed=None, sigma_hit=0.15, z_hit=0.8, z_rand=0.2, likelihood_scale=0.4,
                 noise_x=(0.10, 0.02), noise_y=(0.05, 0.05), noise_yaw=(0.05, 0.10), resample_ratio=0.5,
                 wall_penalty=1e-3):
        """noise_*: standard deviation of the motion per metre travelled and per radian turned."""
        self.n = int(n)
        self.rng = np.random.default_rng(None if seed is None or seed < 0 else seed)
        self.sigma_hit, self.z_hit, self.z_rand = sigma_hit, z_hit, z_rand
        self.likelihood_scale = likelihood_scale   # <1 tempers the beams' correlation: keeps the cloud diverse
        self.noise_x, self.noise_y, self.noise_yaw = noise_x, noise_y, noise_yaw
        self.resample_ratio = resample_ratio
        self.wall_penalty = wall_penalty
        self.p = np.zeros((self.n, 3))
        self.w = np.full(self.n, 1.0 / self.n)

    # ---------------- initialization ----------------
    def init_gaussian(self, pose, std_xy, std_yaw):
        self.p[:, 0] = pose[0] + self.rng.normal(0.0, std_xy, self.n)
        self.p[:, 1] = pose[1] + self.rng.normal(0.0, std_xy, self.n)
        self.p[:, 2] = wrap(pose[2] + self.rng.normal(0.0, std_yaw, self.n))
        self.w[:] = 1.0 / self.n

    # ---------------- motion ----------------
    def predict(self, delta):
        """Move every particle by the odometry delta (x, y, yaw in the previous base_link frame) plus noise
        that grows with the distance travelled and the angle turned (no motion, no noise)."""
        dx, dy, dyaw = delta
        trans, rot = math.hypot(dx, dy), abs(dyaw)
        if trans < 1e-9 and rot < 1e-9:
            return
        n = self.n
        ndx = dx + self.rng.normal(0.0, self.noise_x[0] * trans + self.noise_x[1] * rot, n)
        ndy = dy + self.rng.normal(0.0, self.noise_y[0] * trans + self.noise_y[1] * rot, n)
        ndyaw = dyaw + self.rng.normal(0.0, self.noise_yaw[0] * trans + self.noise_yaw[1] * rot, n)
        c, s = np.cos(self.p[:, 2]), np.sin(self.p[:, 2])
        self.p[:, 0] += c * ndx - s * ndy
        self.p[:, 1] += s * ndx + c * ndy
        self.p[:, 2] = wrap(self.p[:, 2] + ndyaw)

    # ---------------- sensor ----------------
    def update_sensor(self, ranges, angles, sensor_pose, lmap):
        """Reweight by how well each particle's lidar endpoints sit on walls. sensor_pose = laser in base_link."""
        if not len(ranges):
            return
        lx, ly, lyaw = sensor_pose
        x, y, th = self.p[:, 0], self.p[:, 1], self.p[:, 2]
        c, s = np.cos(th), np.sin(th)
        sx, sy, sth = x + c * lx - s * ly, y + s * lx + c * ly, th + lyaw
        a = sth[:, None] + np.asarray(angles)[None, :]
        ex = sx[:, None] + np.asarray(ranges)[None, :] * np.cos(a)
        ey = sy[:, None] + np.asarray(ranges)[None, :] * np.sin(a)
        d = lmap.distance(ex, ey)
        pz = self.z_hit * np.exp(-0.5 * (d / self.sigma_hit) ** 2) + self.z_rand
        loglik = self.likelihood_scale * np.log(pz).sum(axis=1)
        w = self.w * np.exp(loglik - loglik.max())
        w[lmap.distance(x, y) <= 0.0] *= self.wall_penalty   # the car itself cannot be inside a wall
        total = w.sum()
        self.w = w / total if total > 0.0 and np.isfinite(total) else np.full(self.n, 1.0 / self.n)

    def neff(self):
        return 1.0 / float(np.sum(self.w ** 2))

    def resample(self, force=False):
        """Low-variance resampling when the effective sample size drops below resample_ratio * n."""
        if not force and self.neff() >= self.resample_ratio * self.n:
            return False
        positions = (self.rng.random() + np.arange(self.n)) / self.n
        cumulative = np.cumsum(self.w)
        cumulative[-1] = 1.0
        self.p = self.p[np.searchsorted(cumulative, positions)]
        self.w[:] = 1.0 / self.n
        return True

    # ---------------- estimate ----------------
    def estimate(self):
        w, p = self.w, self.p
        return (float(w @ p[:, 0]), float(w @ p[:, 1]),
                float(math.atan2(w @ np.sin(p[:, 2]), w @ np.cos(p[:, 2]))))

    def spread(self):
        """(position std [m], yaw circular std [rad]) of the weighted cloud."""
        mx, my, _ = self.estimate()
        pos = math.sqrt(float(self.w @ ((self.p[:, 0] - mx) ** 2 + (self.p[:, 1] - my) ** 2)))
        r = min(1.0, math.hypot(float(self.w @ np.sin(self.p[:, 2])), float(self.w @ np.cos(self.p[:, 2]))))
        return pos, math.sqrt(-2.0 * math.log(max(r, 1e-12)))


class ScanMatcher:
    """Correlative scan matching on the likelihood field: the fit of a pose is the mean over the beams of
    exp(-d^2 / 2 sigma^2), d = distance from the beam's endpoint to the nearest wall (1 = every endpoint on a
    wall, 0 = none near one). Brute force over a pose grid, vectorized."""
    PASS_BACK = 0.3   # m before a beam's endpoint that must be free space

    def __init__(self, lmap, sigma=0.10, coarse_step=0.2, coarse_sigma=0.3, yaw_step=math.radians(5.0),
                 clearance=0.1):
        """coarse_step: m between the global search's candidate positions, drivable cells only (free and at
        least clearance from a wall); coarse_sigma: forgiving fit for that coarse grid."""
        self.lmap = lmap
        self.fine = np.exp(-0.5 * (lmap.dist / sigma) ** 2).astype(np.float32)
        self.coarse = np.exp(-0.5 * (lmap.dist / coarse_sigma) ** 2).astype(np.float32)
        self.passable = lmap.free.astype(np.float32)
        self.coarse_step = coarse_step
        self.yaw_step = yaw_step
        k = max(1, int(round(coarse_step / lmap.res)))
        rows, cols = np.nonzero(lmap.free & (lmap.dist >= clearance))
        keep = (rows % k == 0) & (cols % k == 0)
        rows, cols = rows[keep], cols[keep]
        self.candidates = np.c_[lmap.ox + (cols + 0.5) * lmap.res, lmap.oy + (lmap.h - rows - 0.5) * lmap.res]

    def _lookup(self, grid, x, y):
        m = self.lmap
        col = np.floor((x - m.ox) / m.res).astype(np.int64)
        row = m.h - 1 - np.floor((y - m.oy) / m.res).astype(np.int64)
        inside = (col >= 0) & (col < m.w) & (row >= 0) & (row < m.h)
        return np.where(inside, grid[np.clip(row, 0, m.h - 1), np.clip(col, 0, m.w - 1)], 0.0)

    def _scores(self, grid, xy, yaws, ranges, angles, sensor_pose):
        """Fit of every (position, yaw) pair: array (len(xy), len(yaws)). A beam counts only if the space it
        crossed is free in the map (sampled halfway and PASS_BACK before its endpoint): an endpoint that lands
        inside a wall block, or a beam that passes through a wall, does not fit."""
        lx, ly, lyaw = sensor_pose
        ranges, angles = np.asarray(ranges), np.asarray(angles)
        c, s = np.cos(yaws), np.sin(yaws)
        dirx, diry = np.cos(lyaw + angles), np.sin(lyaw + angles)

        def offsets(r):   # points at range r along each beam, relative to base_link's position: (yaws, beams)
            bx, by = lx + r * dirx, ly + r * diry
            return c[:, None] * bx[None, :] - s[:, None] * by[None, :], s[:, None] * bx[None, :] + c[:, None] * by[None, :]

        ends = offsets(ranges)
        passes = [offsets(0.5 * ranges), offsets(np.maximum(ranges - self.PASS_BACK, 0.0))]
        out = np.empty((len(xy), len(yaws)), dtype=np.float32)
        chunk = max(1, 600_000 // max(1, ends[0].size))
        for i in range(0, len(xy), chunk):
            px, py = xy[i:i + chunk, 0, None, None], xy[i:i + chunk, 1, None, None]
            v = self._lookup(grid, px + ends[0][None], py + ends[1][None])
            for ox, oy in passes:
                v = v * self._lookup(self.passable, px + ox[None], py + oy[None])
            out[i:i + chunk] = v.mean(axis=2)
        return out

    def fit(self, pose, ranges, angles, sensor_pose):
        """Fit of one pose (fine sigma)."""
        if not len(ranges):
            return 0.0
        return float(self._scores(self.fine, np.array([pose[:2]]), np.array([pose[2]]),
                                  ranges, angles, sensor_pose)[0, 0])

    def search_local(self, pose, ranges, angles, sensor_pose, radius=0.8, yaw_range=math.radians(15.0),
                     step=None, yaw_step=math.radians(1.5)):
        """Best (pose, fit) within radius / yaw_range of pose, on a grid of step (default: map resolution)."""
        step = step or self.lmap.res
        d = np.arange(-radius, radius + 1e-9, step)
        gx, gy = np.meshgrid(pose[0] + d, pose[1] + d)
        xy = np.c_[gx.ravel(), gy.ravel()]
        yaws = pose[2] + np.arange(-yaw_range, yaw_range + 1e-9, yaw_step)
        sc = self._scores(self.fine, xy, yaws, ranges, angles, sensor_pose)
        i, j = np.unravel_index(int(np.argmax(sc)), sc.shape)
        return (float(xy[i, 0]), float(xy[i, 1]), float(wrap(yaws[j]))), float(sc[i, j])

    def search_global(self, ranges, angles, sensor_pose, seeds=40, separation=1.0,
                      separation_yaw=math.radians(30.0)):
        """Distinct pose hypotheses [(pose, fit), ...], best first: every candidate position at every yaw_step
        on the coarse field, then the best seeds refined on the fine field. Two hypotheses are distinct when
        more than separation m or separation_yaw rad apart."""
        if not len(ranges) or not len(self.candidates):
            return []
        yaws = np.arange(-math.pi, math.pi, self.yaw_step)
        sc = self._scores(self.coarse, self.candidates, yaws, ranges, angles, sensor_pose)
        flat = np.argsort(sc, axis=None)[::-1]
        picked = []
        for f in flat:   # the best seeds, at most one per coarse neighbourhood
            i, j = np.unravel_index(int(f), sc.shape)
            seed = (self.candidates[i, 0], self.candidates[i, 1], yaws[j])
            if all(math.hypot(seed[0] - q[0], seed[1] - q[1]) > self.coarse_step * 1.5
                   or abs(float(wrap(seed[2] - q[2]))) > self.yaw_step * 1.5 for q in picked):
                picked.append(seed)
                if len(picked) >= seeds:
                    break
        refined = [self.search_local(q, ranges, angles, sensor_pose, radius=self.coarse_step,
                                     yaw_range=self.yaw_step, yaw_step=math.radians(1.0)) for q in picked]
        refined.sort(key=lambda r: -r[1])
        out = []
        for pose, score in refined:
            if all(math.hypot(pose[0] - q[0][0], pose[1] - q[0][1]) > separation
                   or abs(float(wrap(pose[2] - q[0][2]))) > separation_yaw for q in out):
                out.append((pose, score))
        return out
