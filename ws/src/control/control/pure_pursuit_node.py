#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from geometry_msgs.msg import PoseStamped

import numpy as np
import os

class PurePursuit(Node):
    """ 
    Implement Pure Pursuit on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('pure_pursuit_node')
        # launch/pure_pursuit.launch.py sets track:=<name> (ws/tracks/<name>/raceline.csv)
        self.declare_parameter('track', '')
        track = self.get_parameter('track').value
        self.declare_parameter('raceline_path', '')  # overrides track
        self.declare_parameter('pose_topic', '/tracked_pose')  # /ground_truth/pose for debugging
        pose_topic = self.get_parameter('pose_topic').value
        self.declare_parameter('speed_scale', 0.5)
        self.speed_scale = self.get_parameter('speed_scale').value
        self.declare_parameter('max_speed', 0.0)  # m/s cap on the drive speed, 0 = no cap
        self.max_speed = self.get_parameter('max_speed').value
        # vehicle geometry: config/vehicle.yaml only (pure_pursuit.launch.py loads it)
        self.max_steering_angle = self.declare_parameter('max_steering_angle', Parameter.Type.DOUBLE).value

        # lookahead = clip(lookahead_time * drive speed, lookahead_min, lookahead_distance)
        lookahead_param = self.declare_parameter('lookahead_distance', 3.0)  # max [m]
        assert lookahead_param.value is not None
        self.lookahead_distance: float = lookahead_param.value
        self.lookahead_min = self.declare_parameter('lookahead_min', 0.6).value  # m, ~2x the wheelbase
        self.lookahead_time = self.declare_parameter('lookahead_time', 0.4).value  # s

        self.wheelbase = self.declare_parameter('wheelbase', Parameter.Type.DOUBLE).value
        if self.max_steering_angle is None or self.wheelbase is None:
            raise RuntimeError('max_steering_angle / wheelbase unset: load control/config/vehicle.yaml')

        assert self.lookahead_distance is not None

        self.get_logger().info(f'Initialing pure pursuit with lookahead {self.lookahead_time} s x speed, '
                               f'{self.lookahead_min}-{self.lookahead_distance} m')

        # racelines are generated per track into ws/tracks/<track>/raceline.csv (TUM output format)
        tracks_dir = os.environ.get('TRACKS_DIR', os.path.expanduser('~/ws/tracks'))
        track_path = self.get_parameter('raceline_path').value or os.path.join(tracks_dir, track, 'raceline.csv')
        if not os.path.isfile(track_path):
            self.get_logger().fatal(f'Raceline not found: {track_path} (set track or raceline_path)')
            raise FileNotFoundError(track_path)
        self.waypoints = np.loadtxt(track_path, delimiter=';', ndmin=2)

        self.get_logger().info(f'Loaded {track} raceline. Waypoints shape: {self.waypoints.shape}')

        # create ROS subscribers and publishers
        self.pose_sub = self.create_subscription(PoseStamped, pose_topic, self.pose_callback, 10)

        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        self.last_pos_idx = 0

    def pose_callback(self, pose_msg: PoseStamped):
        # TODO: find the current waypoint to track using methods mentioned in lecture
        self.get_logger().debug(f'Position: {pose_msg.pose.position}')
        position = pose_msg.pose.position
        x = position.x
        y = position.y
        q = pose_msg.pose.orientation
        q_list = [q.x, q.y, q.z, q.w]
        if np.any(np.isnan(q_list)):
            # if quarternion invalid
            self.get_logger().warning("Received invalid quarternion")
            return
        yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))

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
        # off the raceline (e.g. starting off the loop), add the offset so the target stays ahead along
        # the line instead of being the closest point, which would cut straight across corners
        # Lookahead for the drive speed at the car. (It used to be min(3, 1/(7 kappa)) of the last
        # target, overwriting the parameter: 0.13 m, under the 0.33 m wheelbase, in tight corners,
        # where pure pursuit then saturated the steering and ran wide.)
        v = self.waypoints[self.last_pos_idx][5] * self.speed_scale
        if self.max_speed > 0:
            v = min(v, self.max_speed)
        lookahead = min(max(self.lookahead_time * v, self.lookahead_min), self.lookahead_distance)
        search_distance = lookahead + dists[min_local_idx]
        # TODO: more efficient way then looping
        i = self.last_pos_idx
        n_checked = 0  # at most one lap: every waypoint may be within lookahead on a tiny track
        while (target_dist := (np.hypot(self.waypoints[i][1] - x, self.waypoints[i][2] - y))) < search_distance \
                and n_checked < len(self.waypoints) - 1:
            i = (i + 1) % len(self.waypoints)
            n_checked += 1
        target_waypoint = self.waypoints[i]
        self.get_logger().debug(f"lookahead: {lookahead:.2f} m")

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
        steering_angle = np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle)

        speed = target_waypoint[5] * self.speed_scale # raceline vx, scaled down for now
        if self.max_speed > 0:
            speed = min(speed, self.max_speed)

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
