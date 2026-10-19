#!/usr/bin/env python3
"""Reactive follow the gap on /scan -> /drive (no map or localization needed).

Each scan: LidarProcessor caps and smooths the ranges and zeroes a safety bubble around the
closest obstacle; the node then drives at a constant speed toward the farthest point of the
widest run of non-zero ranges. Stops if no free space is left.

  ros2 run control follow_the_gap_node [--ros-args -p speed:=1.0 ...]

From the F1TENTH Lab 4 template:
https://github.com/f1tenth/f1tenth_lab4_template/tree/68bda6b1346b741e3d10da68736ae47dbe7c6743
"""
import numpy as np
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan

from control.lidar_processing import LidarProcessor


def find_max_gap(ranges):
    """(start, end) indices, both inclusive, of the longest run of non-zero ranges (first if tied)."""
    edges = np.diff(np.concatenate(([0], (np.asarray(ranges) != 0).astype(np.int8), [0])))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    if not starts.size:
        return 0, 0
    k = int(np.argmax(ends - starts))
    return int(starts[k]), int(ends[k])


class ReactiveFollowGap(Node):
    def __init__(self, parameter_overrides=None):
        super().__init__('reactive_node', parameter_overrides=parameter_overrides)
        p = self.declare_parameter
        self.MAX_RANGE = p('max_range', 5.0).value                       # m, ranges are capped here
        self.DISPARITY_THRESHOLD = p('disparity_threshold', 2.0).value   # m, range jump = obstacle edge
        self.ALPHA = p('alpha', 0.7).value                               # smoothing weight of the new scan
        self.SPEED = p('speed', 1.0).value                               # m/s, constant drive speed
        self.CAR_WIDTH = p('car_width', 0.27).value                      # m, sizes the safety bubble

        self.lidar_processor = LidarProcessor(
            max_range=self.MAX_RANGE,
            disparity_threshold=self.DISPARITY_THRESHOLD,
            alpha=self.ALPHA,
            car_width=self.CAR_WIDTH,
        )
        self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)

    def preprocess_lidar(self, ranges):
        return self.lidar_processor.preprocess_lidar(ranges)

    @staticmethod
    def find_max_gap(ranges):
        return find_max_gap(ranges)

    def publish(self, speed, steering_angle):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering_angle)
        self.drive_pub.publish(msg)

    def lidar_callback(self, scan: LaserScan):
        ranges = self.lidar_processor.process(scan)
        if len(ranges) == 0:
            self.get_logger().warning('Empty lidar scan')
            return
        if not np.any(ranges):
            self.publish(0.0, 0.0)
            return

        start, end = find_max_gap(ranges)
        best = start + int(np.argmax(ranges[start:end + 1]))   # farthest point in the gap
        steering_angle = scan.angle_min + best * scan.angle_increment
        self.publish(self.SPEED, steering_angle)
        self.get_logger().debug(f'Steering angle: {np.degrees(steering_angle):.2f} deg')


def main():
    rclpy.init()
    node = ReactiveFollowGap()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
