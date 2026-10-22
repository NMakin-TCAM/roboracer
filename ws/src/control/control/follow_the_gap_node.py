#!/usr/bin/env python3

# From Roboracer Lab 4: https://github.com/f1tenth/f1tenth_lab4_template/tree/68bda6b1346b741e3d10da68736ae47dbe7c6743

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor

from control.lidar_processing import LidarProcessor
from copy import deepcopy
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive

class ReactiveFollowGap(Node):
    """ 
    Follow the gap
    """
    def __init__(self, parameter_overrides=None):
        super().__init__('reactive_node', parameter_overrides=parameter_overrides)
        # Topics & Subs, Pubs
        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        params = self.declare_parameters(
            namespace='',
            parameters=[
                ('max_range', 5.0, ParameterDescriptor()),            # cap for lidar reading
                ('disparity_threshold', 2.0, ParameterDescriptor()),  # diff necessary for disparity
                ('alpha', 0.7, ParameterDescriptor()),                # exp moving average: weight of new data
                ('speed', 1., ParameterDescriptor()),                 # speed to drive at
                ('car_width', 0.27, ParameterDescriptor()),           # car width for collision bubble
            ],
        )

        self.MAX_RANGE, self.DISPARITY_THRESHOLD, self.ALPHA, self.SPEED, self.CAR_WIDTH = (
            p.get_parameter_value().double_value for p in params
        )
        # Steering lock (rad); default mirrors config/vehicle.yaml, which follow_the_gap.launch.py loads
        self.MAX_STEERING_ANGLE = self.declare_parameter('max_steering_angle', 0.524).value
        # Gaps are searched only within +-fov (rad) of straight ahead, not beside or behind the car
        self.FOV = self.declare_parameter('fov', np.pi / 2).value

        self.lidar_processor = LidarProcessor(
            max_range=self.MAX_RANGE,
            disparity_threshold=self.DISPARITY_THRESHOLD,
            alpha=self.ALPHA,
            car_width=self.CAR_WIDTH,
        )

        # Subscribe to LIDAR
        self.lidar_sub = self.create_subscription(LaserScan, lidarscan_topic, self.lidar_callback, 10)

        # Publish to drive
        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)

    def preprocess_lidar(self, ranges):
        return self.lidar_processor.preprocess_lidar(ranges)

    def remove_obstacles(self, ranges, data: LaserScan, closest_idx: int):
        self.lidar_processor.remove_obstacles(ranges, data, closest_idx)

    def find_max_gap(self, free_space_ranges) -> tuple[int, int]:
        """ Return the start index & end index of the max gap in free_space_ranges
        """

        max_gap = 0
        i = np.argmax(free_space_ranges != 0)
        j = i + 1
        max_i, max_j = i, j

        while j <= len(free_space_ranges):
            if j < len(free_space_ranges) and free_space_ranges[j] != 0:
                j += 1
            else:
                gap = j - i
                if gap > max_gap:
                    max_gap = gap
                    max_i, max_j = i, j - 1
                try:
                    i = j + np.argmax(free_space_ranges[j:] != 0)
                except ValueError:
                    i = j
                j = i + 1

        return int(max_i), int(max_j)
    
    def find_best_point(self, start_i, end_i, ranges):
        """Start_i & end_i are start and end indicies of max-gap range, respectively
        Return index of best point in ranges
	    Naive: Choose the furthest point within ranges and go there
        """
        # choose center of gap
        # best_idx = (start_i + end_i) / 2

        # choose farthest distance within gap: the middle of the run of farthest beams, since
        # ranges are capped at max_range and argmax alone returns the run's first (rightmost) beam
        gap = ranges[start_i:end_i + 1]
        far = np.flatnonzero(gap == gap.max())
        run_end = far[0]
        while run_end + 1 < len(gap) and gap[run_end + 1] == gap[far[0]]:
            run_end += 1
        best_idx = start_i + (far[0] + run_end) // 2

        # TODO: idea - try to stay in midpoint

        return best_idx

    def lidar_callback(self, data: LaserScan):
        """ Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message
        """
        proc_ranges = self.lidar_processor.process(data)
        angles = data.angle_min + np.arange(len(proc_ranges)) * data.angle_increment
        proc_ranges[np.abs(angles) > self.FOV] = 0.

        if len(proc_ranges) == 0:
            self.get_logger().warning(f"No ranges found from lidar scan")
            return
        
        if not np.any(proc_ranges):
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.speed = 0.0
            drive_msg.drive.steering_angle = 0.0
            self.drive_pub.publish(drive_msg)
            return

        # Find max length gap 
        max_start, max_end = self.find_max_gap(proc_ranges)

        # Find the best point in the gap 
        best_idx = self.find_best_point(max_start, max_end, proc_ranges)
        
        steering_angle = float(np.clip(self.get_angle(data, best_idx), -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE))

        # Publish Drive message
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = self.SPEED
        drive_msg.drive.steering_angle = steering_angle
        self.drive_pub.publish(drive_msg)

        self.get_logger().debug(f"Steering angle: {steering_angle * 180 / np.pi:.2f}")

    @staticmethod
    def get_angle(data: LaserScan, idx):
        return data.angle_min + idx * data.angle_increment

def main(args=None):
    rclpy.init(args=args)
    print("ReactiveFollowGap Initialized")
    reactive_node = ReactiveFollowGap()
    rclpy.spin(reactive_node)

    reactive_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()