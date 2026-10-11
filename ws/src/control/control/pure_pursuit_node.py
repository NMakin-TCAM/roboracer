#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

import numpy as np
import os
from scipy.spatial.transform import Rotation as R

class PurePursuit(Node):
    """ 
    Implement Pure Pursuit on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('pure_pursuit_node')
        # launch/levine_launch.py sets track:=levine, launch/spielberg_launch.py track:=spielberg
        self.declare_parameter('track', 'Austin')
        track = self.get_parameter('track').value

        lookahead_param = self.declare_parameter('lookahead_distance', 3)
        assert lookahead_param.value is not None
        self.lookahead_distance: float = lookahead_param.value

        # TODO: figure out how to get from xacro file in f1tenth_gym_ros/launch/
        wheelbase_param = self.declare_parameter('wheelbase', 0.3302)
        assert wheelbase_param.value is not None
        self.wheelbase = wheelbase_param.value

        assert self.lookahead_distance is not None

        self.get_logger().info(f'Initialing pure pursuit with lookahead distance = {self.lookahead_distance}')

        # TODO: load the waypoints of `track` from the package's installed waypoints/ folder:
        #       os.path.join(get_package_share_directory('pure_pursuit'), 'waypoints', <your csv>)
        #       (ament_index_python.packages). A path like /home/you/... only exists on your laptop.
        track_path = os.path.join(get_package_share_directory('control'), "waypoints", f"{track}_raceline.csv")
        self.waypoints = np.loadtxt(track_path, delimiter=';')

        self.get_logger().info(f'Loaded {track} raceline. Waypoints shape: {self.waypoints.shape}')

        # create ROS subscribers and publishers
        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.pose_callback, 10)

        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        self.last_pos_idx = 0

    def pose_callback(self, pose_msg: Odometry):
        # TODO: find the current waypoint to track using methods mentioned in lecture
        self.get_logger().debug(f'Position: {pose_msg.pose.pose.position}')
        position = pose_msg.pose.pose.position
        x = position.x
        y = position.y
        q = pose_msg.pose.pose.orientation
        q_list = [q.x, q.y, q.z, q.w]
        if np.any(np.isnan(q_list)):
            # if quarternion invalid
            self.get_logger().warning("Received invalid quarternion")
            return
        r = R.from_quat(q_list)
        yaw = r.as_euler('zyx')[0]

        # TODO: Try using arc length instead

        # Obtain closest waypoint to vehicle
        window_size = len(self.waypoints) # TODO: optimize by reducing window... remember to check if window exceeded

        #     self.get_logger().warning(f"exceeded window size for closest waypoint search")

        indices = np.arange(self.last_pos_idx, self.last_pos_idx + window_size) % len(self.waypoints)
        dists = np.hypot(self.waypoints[indices, 1] - x, self.waypoints[indices, 2] - y)
        min_local_idx = None
        min_local_idx = np.argmin(dists)
        self.last_pos_idx = indices[min_local_idx]

        # self.get_logger().info(f"last_pox_idx = {self.last_pos_idx} | dist to car = {dists[min_local_idx]}")
        # self.get_logger().info(f"current pos = {(x, y)}")
        # self.get_logger().info(f"closest_waypoint = {self.waypoints[self.last_pos_idx][1:3]}")

        # find waypoint at lookahead distance away
        # TODO: more efficient way then looping
        i = self.last_pos_idx
        while (target_dist := (np.hypot(self.waypoints[i][1] - x, self.waypoints[i][2] - y))) < self.lookahead_distance:
            i = (i + 1) % len(self.waypoints)
        target_waypoint = self.waypoints[i]

        # TODO: use curvature to adjust lookahead
        self.lookahead_distance = min(3, 1/7 * 1 / abs(target_waypoint[4]))
        self.get_logger().info(f"Track curvature: {target_waypoint[4]} lookahead: {self.lookahead_distance}")

        # TODO: interpolate to get exact lookahead distance

        # self.get_logger().info(f"closest_waypoint_idx = {self.last_pos_idx}, target_waypoint_idx = {i} | dist to car = {target_dist}")

        # transform goal point to vehicle frame of reference 
        dx = target_waypoint[1] - x
        dy = target_waypoint[2] - y

        target_x = dx * np.cos(yaw) + dy * np.sin(yaw)
        target_y = -dx * np.sin(yaw) + dy * np.cos(yaw)

        # self.get_logger().info(f"x {x} y {y} yaw {yaw} tx {target_waypoint[1]} ty {target_waypoint[2]}")
        # self.get_logger().info(f"dx {dx}, dy {dy}, yaw {yaw}, target_x {target_x} target_y {target_y}")

        # calculate curvature/steering angle
        curvature = 2 * abs(target_y) / target_dist ** 2
        steering_angle = np.sign(target_y) * np.arctan(self.wheelbase * curvature)
        # MAX_STEERING_ANGLE = 0.4189 # TODO: find where to get this
        # steering_angle = np.clip(steering_angle, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE)

        speed = target_waypoint[5] / 8 # scale down for now
        MAX_SPEED = 1. # temp rate limit
        speed = min(speed, MAX_SPEED)

        # TODO: use curvature and target acceleration to adjust output

        # TODO: publish drive message, don't forget to limit the steering angle.
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = 'pure_pursuit'

        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = speed

        # self.get_logger().info(f"Speed: {speed}, Steering Angle: {steering_angle}")

        self.drive_pub.publish(drive_msg)

def main(args=None):
    rclpy.init(args=args)
    print("PurePursuit Initialized")
    pure_pursuit_node = PurePursuit()
    rclpy.spin(pure_pursuit_node)

    pure_pursuit_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
