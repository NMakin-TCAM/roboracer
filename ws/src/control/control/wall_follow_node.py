"""Centring wall follower: drives the mapping lap(s) without a map, localization or ground truth.

Measures the distance to the left and right walls, projected a short distance ahead of the car, and
steers with a PD controller to keep the two equal (i.e. stay on the track centre). Optionally counts
laps on pose_topic and stops the car after `laps` of them.

    ros2 launch control wall_follow.launch.py [speed:=1.0] [laps:=2] [debug:=true]

Subscribes /scan (sensor_msgs/LaserScan) and pose_topic (geometry_msgs/PoseStamped, for laps only),
publishes /drive (ackermann_msgs/AckermannDriveStamped). Used by scripts/map_track.sh --wall-follow.
"""
import math
import time

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import LaserScan


class LapCounter:
    """Counts returns to the start position.

    A lap is complete when the car has been more than leave_radius from the start, comes back within
    return_radius of it, and has driven at least min_distance since the last lap.
    """

    def __init__(self, leave_radius=3.0, return_radius=1.5, min_distance=10.0):
        self.leave_radius = leave_radius
        self.return_radius = return_radius
        self.min_distance = min_distance
        self.start = None
        self.last = None
        self.distance = 0.0   # driven since the start or the last lap
        self.away = False
        self.laps = 0

    def update(self, x, y):
        """Feed one position; returns True when it completes a lap."""
        if self.start is None:
            self.start = self.last = (x, y)
            return False
        self.distance += math.hypot(x - self.last[0], y - self.last[1])
        self.last = (x, y)
        r = math.hypot(x - self.start[0], y - self.start[1])
        if r > self.leave_radius:
            self.away = True
        elif self.away and r < self.return_radius and self.distance >= self.min_distance:
            self.laps += 1
            self.away = False
            self.distance = 0.0
            return True
        return False


class WallFollower(Node):

    def __init__(self):
        super().__init__('wall_follow_node')
        p = self.declare_parameter
        self.speed = p('speed', 1.0).value                        # m/s on straights
        self.kp = p('kp', 1.05).value                             # rad of steering per m off centre
        self.kd = p('kd', 0.10).value                             # rad per (m/s) of drift towards a wall
        self.lookahead = p('lookahead', 0.8).value                # m ahead at which wall distances are projected
        self.beam_angle = math.radians(p('beam_angle_deg', 45.0).value)  # forward beam, from the side beam
        self.max_range = p('max_range', 5.0).value                # beyond this a side counts as "no wall"
        self.track_width = p('track_width', 1.2).value            # m, used when only one wall is visible
        self.corner_slowdown = p('corner_slowdown', 0.5).value    # fraction of speed shed at full steering
        self.max_steer = p('max_steering_angle', 0.524).value     # rad, steering is clipped to this
        self.laps = p('laps', 0).value                            # stop after this many laps, 0 = never
        pose_topic = p('pose_topic', '/tracked_pose').value       # for lap counting only
        self.debug = p('debug', False).value                      # log wall distances and steering

        self.prev_error = None
        self.prev_time = None
        self.prev_walls = None   # which walls were visible last scan (left, right)
        self.lap_counter = LapCounter(p('lap_leave_radius', 3.0).value, p('lap_return_radius', 1.5).value,
                                      p('lap_min_distance', 10.0).value)
        self.done = False
        self.started = time.monotonic()
        self.pose_warned = False

        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        if self.laps > 0:
            self.create_subscription(PoseStamped, pose_topic, self.pose_callback, 10)
            self.get_logger().info(f'Wall following at {self.speed:.1f} m/s, stopping after {self.laps} '
                                   f'lap(s) counted on {pose_topic}')
        else:
            self.get_logger().info(f'Wall following at {self.speed:.1f} m/s until stopped')

    def range_at(self, msg, ranges, angle):
        """Median range over +/-1 deg around angle (rad, positive = left); max_range if no return."""
        i = int(round((angle - msg.angle_min) / msg.angle_increment))
        half = max(1, int(round(math.radians(1.0) / msg.angle_increment)))
        window = ranges[max(0, i - half):i + half + 1]
        return float(np.median(window)) if window.size else self.max_range

    def wall_distance(self, msg, ranges, side):
        """Distance to the wall on one side (side = +1 left, -1 right), projected lookahead metres ahead.

        Classic two-beam method: b is perpendicular to the car, a is beam_angle further forward.
        alpha is the wall's angle relative to the car; positive when the wall falls away ahead.
        Returns None if either beam sees no wall.
        """
        theta = self.beam_angle
        b = self.range_at(msg, ranges, side * math.pi / 2)
        a = self.range_at(msg, ranges, side * (math.pi / 2 - theta))
        if a >= self.max_range or b >= self.max_range:
            return None
        alpha = math.atan2(a * math.cos(theta) - b, a * math.sin(theta))
        return b * math.cos(alpha) + self.lookahead * math.sin(alpha)

    def scan_callback(self, msg):
        if self.done:
            return
        ranges = np.array(msg.ranges, dtype=float)
        ranges[~np.isfinite(ranges)] = self.max_range       # no return -> "far"
        ranges = np.clip(ranges, 0.0, self.max_range)

        d_left = self.wall_distance(msg, ranges, +1)
        d_right = self.wall_distance(msg, ranges, -1)

        # Error: how far the car is right of centre (positive -> steer left, which is positive steering)
        half = self.track_width / 2.0
        if d_left is not None and d_right is not None:
            error = (d_left - d_right) / 2.0
        elif d_left is not None:
            error = d_left - half                           # only the left wall: hold half a track from it
        elif d_right is not None:
            error = half - d_right
        else:
            error = 0.0                                     # no walls in view: go straight

        # PD control, with the derivative from sensor timestamps. When a wall drops in or out of view
        # the error jumps; skip the derivative for that scan, or it kicks the steering to full lock.
        walls = (d_left is not None, d_right is not None)
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        d_error = 0.0
        if self.prev_time is not None and t > self.prev_time and walls == self.prev_walls:
            d_error = float(np.clip((error - self.prev_error) / (t - self.prev_time), -1.0, 1.0))
        self.prev_error, self.prev_time, self.prev_walls = error, t, walls

        steering = float(np.clip(self.kp * error + self.kd * d_error, -self.max_steer, self.max_steer))
        speed = self.speed * (1.0 - self.corner_slowdown * abs(steering) / self.max_steer)
        self.publish(speed, steering)

        if self.laps > 0 and self.lap_counter.start is None and not self.pose_warned \
                and time.monotonic() - self.started > 5.0:
            self.get_logger().warn('No pose received yet: laps are not being counted')
            self.pose_warned = True

        if self.debug:
            fmt = lambda d: f'{d:.2f}' if d is not None else ' -- '  # noqa: E731
            self.get_logger().info(f'wall L {fmt(d_left)} R {fmt(d_right)}  error {error:+.2f} m  d_error {d_error:+.2f}'
                                   f'  steer {steering:+.3f} rad  speed {speed:.2f} m/s', throttle_duration_sec=0.5)

    def pose_callback(self, msg):
        if self.done or not self.lap_counter.update(msg.pose.position.x, msg.pose.position.y):
            return
        self.get_logger().info(f'Lap {self.lap_counter.laps}/{self.laps} complete')
        if self.lap_counter.laps >= self.laps:
            self.done = True

    def publish(self, speed, steering):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering)
        self.drive_pub.publish(msg)

    def stop(self):
        self.get_logger().info('Stopping: sending zero speed and steering')
        for _ in range(3):   # repeat and wait, so the commands are delivered before the node exits
            self.publish(0.0, 0.0)
            time.sleep(0.05)


def main():
    # Handle Ctrl+C ourselves: rclpy's default handler shuts ROS down first, so stop() could
    # never publish and the car would coast until the adapter's command timeout
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = WallFollower()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)   # short waits so Ctrl+C is noticed even with no scans arriving
    except KeyboardInterrupt:
        pass
    node.stop()                 # always leave the car stopped
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
