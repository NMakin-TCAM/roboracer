#!/usr/bin/env python3
"""Pure pursuit on a generated raceline (tracks/<track>/raceline.csv, TUM output format).

On every pose (pose_topic, map frame): find the closest raceline point, pick the first point at
least one lookahead ahead of the car, steer on the arc through it, drive at the raceline speed
* speed_scale (capped at max_speed). One /drive per pose.

  lookahead = clip(lookahead_time * drive speed, lookahead_min, lookahead_distance)
"""
import os

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped

from perception.tracks import track_dir

# raceline.csv columns: s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2
X, Y, VX = 1, 2, 5


class PurePursuit(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')
        p = self.declare_parameter
        track = p('track', '').value                      # set by pure_pursuit.launch.py
        raceline_path = p('raceline_path', '').value      # overrides track
        pose_topic = p('pose_topic', '/tracked_pose').value   # /ground_truth/pose for debugging
        self.speed_scale = p('speed_scale', 0.5).value
        self.max_speed = p('max_speed', 0.0).value        # m/s cap on the drive speed, 0 = no cap
        self.lookahead_distance = p('lookahead_distance', 3.0).value   # m, max
        self.lookahead_min = p('lookahead_min', 0.6).value             # m, ~2x the wheelbase
        self.lookahead_time = p('lookahead_time', 0.4).value           # s
        # vehicle geometry comes only from config/vehicle.yaml (pure_pursuit.launch.py loads it)
        self.max_steering_angle = p('max_steering_angle', Parameter.Type.DOUBLE).value
        self.wheelbase = p('wheelbase', Parameter.Type.DOUBLE).value
        if self.max_steering_angle is None or self.wheelbase is None:
            raise RuntimeError('max_steering_angle / wheelbase unset: load control/config/vehicle.yaml')

        path = raceline_path or os.path.join(track_dir(track), 'raceline.csv')
        if not os.path.isfile(path):
            self.get_logger().fatal(f'Raceline not found: {path} (set track or raceline_path)')
            raise FileNotFoundError(path)
        self.waypoints = np.loadtxt(path, delimiter=';', ndmin=2)
        self.xy = self.waypoints[:, [X, Y]]
        self.get_logger().info(f'Loaded {path}: {len(self.waypoints)} points; lookahead '
                               f'{self.lookahead_time} s x speed, {self.lookahead_min}-{self.lookahead_distance} m')

        self.create_subscription(PoseStamped, pose_topic, self.pose_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)

    def drive_speed(self, i):
        v = self.waypoints[i, VX] * self.speed_scale
        return min(v, self.max_speed) if self.max_speed > 0 else v

    def pose_callback(self, msg: PoseStamped):
        x, y = msg.pose.position.x, msg.pose.position.y
        q = msg.pose.orientation
        if np.isnan([q.x, q.y, q.z, q.w]).any():
            self.get_logger().warning('Received invalid quaternion')
            return
        yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))

        n = len(self.xy)
        dists = np.hypot(self.xy[:, 0] - x, self.xy[:, 1] - y)
        closest = int(np.argmin(dists))

        # Target: first point after the closest whose distance from the car reaches the lookahead.
        # Off the line, the car's offset is added so the target stays ahead along the line instead
        # of being the closest point, which would cut straight across corners. At most one lap is
        # searched: on a tiny track every point may be within the lookahead.
        lookahead = np.clip(self.lookahead_time * self.drive_speed(closest),
                            self.lookahead_min, self.lookahead_distance)
        ahead = (closest + np.arange(n)) % n
        far = np.flatnonzero(dists[ahead] >= lookahead + dists[closest])
        target = ahead[far[0]] if far.size else ahead[-1]
        target_dist = dists[target]
        if target_dist < 1e-6:
            return

        # target in the vehicle frame, then the arc through it: curvature = 2 y / d^2
        dx, dy = self.xy[target, 0] - x, self.xy[target, 1] - y
        target_y = -dx * np.sin(yaw) + dy * np.cos(yaw)
        steering = np.arctan(self.wheelbase * 2 * target_y / target_dist ** 2)

        drive = AckermannDriveStamped()
        drive.header.stamp = self.get_clock().now().to_msg()
        drive.header.frame_id = 'base_link'
        drive.drive.steering_angle = float(np.clip(steering, -self.max_steering_angle, self.max_steering_angle))
        drive.drive.speed = float(self.drive_speed(target))
        self.drive_pub.publish(drive)


def main():
    rclpy.init()
    node = PurePursuit()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
