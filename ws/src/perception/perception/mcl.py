"""Monte Carlo localization (particle filter) with a likelihood-field sensor model, in numpy.

No ROS imports: mcl_node.py wires this to topics, test_mcl.py tests it on synthetic tracks.

Why a likelihood field and not ray casting: the weight of a particle comes from where the lidar
endpoints would land if the car were at that pose, looked up in a distance-to-nearest-wall grid.
That needs no range_libc (no Cython/CUDA build), runs on amd64 and arm64, and is a few vectorized
numpy operations per scan.

The filter tracks from a known start pose (the map origin after a sim reset, or an /initialpose); it has no
global localization: on a track with similar-looking corridors a converged-looking cloud can sit on the wrong
copy of the track, so it cannot be trusted to find its own start.

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
