#!/usr/bin/env python3
"""Wheel odometry from the AutoDRIVE rear-wheel encoders (simulation only).

Sim stand-in for f1tenth_system's vesc_to_odom: publishes /odom (odom -> base_link) with the
same kinematic bicycle model, so everything downstream is identical on the real car, where
vesc_to_odom publishes /odom from the VESC instead and this node is not launched.

  distance = mean rear-wheel angle change * wheel_radius   (integrated each frame, exact)
  speed    = distance over the last speed_window seconds  (sim frames arrive unevenly, so
             a single-frame difference is noisy)
  yaw rate = IMU gyro z (use_imu_yaw_rate, default), or the bicycle model
             speed * tan(steering angle) / wheelbase, which reads only ~55% of the sim's real
             turn rate (measured on Porto) and made Cartographer drift in hairpins

Subscribed (vehicle_ns = /autodrive/roboracer_1):
  left_encoder, right_encoder (JointState, cumulative wheel angle in rad), steering (Float32, -1..1),
  imu (Imu)
"""
import math
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float32
from tf2_ros import TransformBroadcaster


class WheelOdom(Node):
    def __init__(self):
        super().__init__('wheel_odom')
        p = self.declare_parameter
        ns = p('vehicle_ns', '/autodrive/roboracer_1').value
        self.wheel_radius = p('wheel_radius', 0.0585).value      # m; calibrated against /ground_truth/odom
        self.speed_window = p('speed_window', 0.1).value         # s
        self.wheelbase = p('wheelbase', 0.33).value              # m, rear axle -> front axle
        self.max_steer = p('max_steering_angle', 0.5236).value   # rad at steering = 1
        self.publish_tf = p('publish_tf', True).value            # TF odom -> base_link
        self.use_imu = p('use_imu_yaw_rate', True).value          # gyro yaw rate instead of the bicycle model
        self.max_speed = p('max_plausible_speed', 30.0).value     # m/s; faster wheel travel = sim reset

        # AutoDRIVE's bridge publishes RELIABLE / depth 1; match it.
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(JointState, ns + '/left_encoder', self.left_cb, qos)
        self.create_subscription(JointState, ns + '/right_encoder', self.right_cb, qos)
        self.create_subscription(Float32, ns + '/steering', self.steering_cb, qos)
        self.create_subscription(Imu, ns + '/imu', self.imu_cb, qos)

        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.tf = TransformBroadcaster(self)

        self.left = None
        self.last = None              # (t, left angle, right angle), last integrated
        self.pending = None           # (sample, stamp) waiting for the gyro of its frame
        self.history = deque()        # (t, distance travelled)
        self.dist = 0.0
        self.steer = 0.0
        self.gyro_z = None
        self.gyro_t = None
        self.x = self.y = self.yaw = 0.0

    def steering_cb(self, msg):
        self.steer = float(msg.data) * self.max_steer

    def imu_cb(self, msg):
        self.gyro_z = msg.angular_velocity.z
        self.gyro_t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.integrate()

    def left_cb(self, msg):
        self.left = msg.position[0]

    # fast_bridge publishes left then right in the same frame, so the right encoder closes a sample
    def right_cb(self, msg):
        if self.left is None:
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.pending = ((t, self.left, msg.position[0]), msg.header.stamp)
        self.integrate()

    def integrate(self):
        """Integrate the pending encoder sample once the gyro of its frame (or a later one) is in.

        Executors run callbacks in subscription order, not publish order, so the encoder callback
        usually runs before the IMU callback of the same sim frame.
        """
        if self.pending is None:
            return
        sample, stamp = self.pending
        t = sample[0]
        gyro = self.use_imu and self.gyro_z is not None
        if gyro and self.gyro_t < t:
            return
        self.pending = None
        last, self.last = self.last, sample
        if last is None or t <= last[0]:
            return

        ds = 0.5 * ((sample[1] - last[1]) + (sample[2] - last[2])) * self.wheel_radius
        if abs(ds) > self.max_speed * (t - last[0]) + 0.05:
            # A sim reset teleports the car and resets the encoders (one reset read as a 76 m step at
            # -700 m/s, and the adapter's speed loop answered with full throttle). Re-reference the
            # encoders and drop the speed history; the pose continues from where it was.
            self.get_logger().warn(f'encoder jump of {ds:.1f} m in one frame (sim reset?), skipped')
            self.history.clear()
            return
        curvature = math.tan(self.steer) / self.wheelbase
        dyaw = self.gyro_z * (t - last[0]) if gyro else ds * curvature
        self.x += ds * math.cos(self.yaw + 0.5 * dyaw)
        self.y += ds * math.sin(self.yaw + 0.5 * dyaw)
        self.yaw += dyaw
        self.dist += ds

        self.history.append((t, self.dist))
        while len(self.history) > 2 and t - self.history[1][0] >= self.speed_window:
            self.history.popleft()
        t0, d0 = self.history[0]
        speed = (self.dist - d0) / (t - t0) if t > t0 else 0.0
        yaw_rate = self.gyro_z if gyro else speed * curvature
        qz, qw = math.sin(self.yaw / 2.0), math.cos(self.yaw / 2.0)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = speed
        odom.twist.twist.angular.z = yaw_rate
        self.odom_pub.publish(odom)

        if self.publish_tf:
            tf = TransformStamped()
            tf.header = odom.header
            tf.child_frame_id = 'base_link'
            tf.transform.translation.x = self.x
            tf.transform.translation.y = self.y
            tf.transform.rotation.z = qz
            tf.transform.rotation.w = qw
            self.tf.sendTransform(tf)


def main():
    rclpy.init()
    node = WheelOdom()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
