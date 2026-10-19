"""Stateful LiDAR processing for follow the gap, shared with tools/lidar_plot.py."""
import numpy as np


class LidarProcessor:
    def __init__(
        self,
        max_range=5.0,             # m, ranges are capped here
        disparity_threshold=2.0,   # m, a jump this large away from the obstacle ends it
        smooth_threshold=0.05,     # m, a drop this large toward the car starts another obstacle
        avoid_threshold=0.5,       # m, closer ranges always belong to the obstacle
        alpha=0.7,                 # exponential smoothing weight of the new scan
        car_width=0.27,            # m
        bubble_scale=5.0,          # bubble half-angle = bubble_scale * atan(car_width / 2 / range):
                                   # scaled up empirically to avoid crashes
    ):
        self.max_range = max_range
        self.disparity_threshold = disparity_threshold
        self.smooth_threshold = smooth_threshold
        self.avoid_threshold = avoid_threshold
        self.alpha = alpha
        self.car_width = car_width
        self.bubble_scale = bubble_scale
        self.ranges_window = None   # previous smoothed scan

    def process(self, scan):
        """Capped, smoothed ranges with the closest obstacle zeroed. scan is not modified, and the
        smoothing history keeps the ranges from before the obstacle was zeroed."""
        ranges = self.preprocess_lidar(scan.ranges)
        valid_idx = np.flatnonzero(ranges)
        if valid_idx.size:
            closest_idx = valid_idx[np.argmin(ranges[valid_idx])]
            self.remove_obstacles(ranges, scan, closest_idx)
        return ranges

    def preprocess_lidar(self, ranges):
        """Copy, NaN -> 0, cap at max_range, and exponentially smooth with the previous scan."""
        proc_ranges = np.array(ranges, dtype=np.float32, copy=True)
        np.nan_to_num(proc_ranges, copy=False)
        np.clip(proc_ranges, 0, self.max_range, out=proc_ranges)
        if self.ranges_window is not None and self.ranges_window.shape == proc_ranges.shape:
            proc_ranges = self.alpha * proc_ranges + (1 - self.alpha) * self.ranges_window
        self.ranges_window = proc_ranges.copy()
        return proc_ranges

    def remove_obstacles(self, ranges, scan, closest_idx: int):
        """Zero, in place, the obstacle around closest_idx plus a car-width bubble on each side."""
        lower = self._obstacle_edge(ranges, closest_idx, -1)
        upper = self._obstacle_edge(ranges, closest_idx, +1)

        def bubble(i):
            return np.ceil(self.bubble_scale * np.arctan2(self.car_width / 2, ranges[i]) / scan.angle_increment)

        start = int(max(0, lower - bubble(lower)))
        end = int(min(len(ranges), upper + bubble(upper) + 1))
        ranges[start:end] = 0.

    def _obstacle_edge(self, ranges, closest_idx, step):
        """Last index of the obstacle walking from closest_idx in direction step (-1 or +1).

        Walks until the range drops by more than smooth_threshold (another, nearer obstacle). The edge
        is the point before the farthest disparity (or range under avoid_threshold) met on the way.
        """
        edge = closest_idx
        i = closest_idx + step
        while 0 <= i < len(ranges):
            jump = ranges[i] - ranges[i - step]
            if jump > self.disparity_threshold or ranges[i] < self.avoid_threshold:
                edge = i - step
            elif jump < -self.smooth_threshold:
                break
            i += step
        return edge
