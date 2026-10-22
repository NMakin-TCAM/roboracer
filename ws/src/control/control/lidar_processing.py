"""Stateful LiDAR processing shared by the controller and diagnostic tools."""

import numpy as np


class LidarProcessor:
    def __init__(
        self, 
        max_range=5.0, 
        disparity_threshold=2.0, 
        smooth_threshold=0.05, 
        avoid_threshold=0.5,
        alpha=0.7,
        car_width=0.27
    ):
        self.max_range = max_range
        self.disparity_threshold = disparity_threshold
        self.smooth_threshold = smooth_threshold
        self.avoid_threshold = avoid_threshold
        self.alpha = alpha
        self.car_width = car_width
        self.ranges_window = None

    def process(self, scan):
        """Return capped, smoothed ranges with the obstacle region removed.

        The input scan is unchanged. Smoothing history precedes obstacle removal.
        """
        ranges = self.preprocess_lidar(scan.ranges)
        valid_idx = np.flatnonzero(ranges)
        if valid_idx.size:
            closest_idx = valid_idx[np.argmin(ranges[valid_idx])]
            self.remove_obstacles(ranges, scan, closest_idx)
        return ranges

    def preprocess_lidar(self, ranges):
        """Copy, sanitize, cap, and exponentially smooth a scan's ranges."""
        proc_ranges = np.array(ranges, dtype=np.float32, copy=True)

        np.nan_to_num(proc_ranges, copy=False)
        np.clip(proc_ranges, 0, self.max_range, out=proc_ranges)
        invalid = proc_ranges == 0  # NaN / -inf: no reading, keep it out of the smoothing

        if self.ranges_window is not None and self.ranges_window.shape == proc_ranges.shape:
            proc_ranges = self.alpha * proc_ranges + (1  - self.alpha) * self.ranges_window
        proc_ranges[invalid] = 0.

        self.ranges_window = proc_ranges.copy()

        return proc_ranges

    def remove_obstacles(self, ranges, data, closest_idx: int):
        # Find disparity around closest point
        i = closest_idx - 1
        lower_disp_idx = closest_idx
        while i >= 0:
            if ranges[i] == 0:  # no reading: stop the walk, a bubble edge needs a real range
                break
            # print(np.degrees(data.angle_min + i * data.angle_increment), ranges[i], ranges[i + 1])
            if ranges[i] - ranges[i + 1] > self.disparity_threshold or ranges[i] < self.avoid_threshold:
                lower_disp_idx = i + 1
                # break
            elif ranges[i] - ranges[i + 1] < -self.smooth_threshold: 
                # entering other obstacle
                break
            i -= 1
        upper_disp_idx = closest_idx
        i = closest_idx + 1
        while i < len(ranges):
            if ranges[i] == 0:  # no reading: stop the walk, a bubble edge needs a real range
                break
            # print(np.degrees(data.angle_min + i * data.angle_increment), ranges[i], ranges[i - 1])
            if ranges[i] - ranges[i - 1] > self.disparity_threshold or ranges[i] < self.avoid_threshold:
                upper_disp_idx = i - 1
                # break
            elif ranges[i] - ranges[i - 1] < -self.smooth_threshold:
                # entering other obstacle
                break
            i += 1

        # print(lower_disp_idx, upper_disp_idx)
        # print(ranges[lower_disp_idx], ranges[upper_disp_idx])

        # Eliminate all points within disparities + car width (set them to zero)
        # lol randomly scaled rn to avoid crashes 
        lower_halfwidth_angle = 5 * np.arctan2(self.car_width / 2, ranges[lower_disp_idx])
        upper_halfwidth_angle = 5 * np.arctan2(self.car_width / 2, ranges[upper_disp_idx])
        lower_halfwidth_idx = np.ceil(lower_halfwidth_angle / data.angle_increment)
        upper_halfwidth_idx = np.ceil(upper_halfwidth_angle / data.angle_increment)
        start = int(max(0, lower_disp_idx - lower_halfwidth_idx))
        end = int(min(len(ranges), upper_disp_idx + upper_halfwidth_idx + 1))
        ranges[start:end] = 0.

        # print(start, end)

