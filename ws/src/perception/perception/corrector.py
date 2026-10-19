#!/usr/bin/env python3
"""Drift correction for the IMU-only EKF velocity (ekf.launch.py; sim only, experimental).

The EKF integrates forward acceleration, so its vx drifts. This node subtracts a bias that it
re-estimates from the wheel encoders and the IMU:
  1. zero-velocity update: both wheels stopped and the chassis not accelerating for 2 EKF
     updates -> bias = EKF vx (the car is at rest)
  2. upper clamp: accelerating forward, the corrected vx may not exceed the slip-scaled wheel speed
  3. floor: wheels not turning backwards, the corrected vx may not go negative
Publishes the EKF odometry with vx corrected (and vy = 0) on output_odom_topic, and optionally the
bias and the error against sim ground truth.
"""
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import JointState, Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64


def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class OdomDriftCorrector(Node):
    def __init__(self):
        super().__init__('odom_drift_corrector')
        p = self.declare_parameter
        ekf_odom_topic = p('ekf_odom_topic', '/odometry/filtered_accl').value
        left_topic = p('left_encoder_topic', '/autodrive/roboracer_1/left_encoder').value
        right_topic = p('right_encoder_topic', '/autodrive/roboracer_1/right_encoder').value
        imu_topic = p('imu_topic', '/autodrive/roboracer_1/imu').value
        gt_odom_topic = p('gt_odom_topic', '/autodrive/roboracer_1/odom').value
        output_topic = p('output_odom_topic', '/odometry/corrected').value
        self.publish_error = p('publish_error', True).value
        error_topic = p('error_topic', '/odometry/velocity_error').value
        bias_topic = p('bias_topic', '/odometry/velocity_bias').value
        self.wheel_radius = p('wheel_radius', 0.059).value   # m (wheel_odom uses the calibrated 0.0585)
        self.v_max = p('v_max', 22.88).value                 # m/s, sim top speed
        self.slip_scale = p('slip_scale', 0.9152).value      # 22.88 / 25.0

        self.wheels = {'left': [None, None, 0.0], 'right': [None, None, 0.0]}   # [last angle, last t, speed]
        self.imu_ax = 0.0
        self.imu_ay = 0.0
        self.gt_vx = None
        self.vel_bias = 0.0
        self.stopped_frames = 0

        self.create_subscription(JointState, left_topic, lambda m: self.wheel_cb('left', m), qos_profile_sensor_data)
        self.create_subscription(JointState, right_topic, lambda m: self.wheel_cb('right', m), qos_profile_sensor_data)
        self.create_subscription(Imu, imu_topic, self.imu_cb, qos_profile_sensor_data)
        self.create_subscription(Odometry, ekf_odom_topic, self.ekf_cb, 10)
        self.odom_pub = self.create_publisher(Odometry, output_topic, 10)

        if self.publish_error:
            # AutoDRIVE's bridge publishes RELIABLE / depth 1; match it.
            gt_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1)
            self.create_subscription(Odometry, gt_odom_topic, self.gt_cb, gt_qos)
            self.error_pub = self.create_publisher(Float64, error_topic, 10)
            self.bias_pub = self.create_publisher(Float64, bias_topic, 10)

        self.get_logger().info(f'Post-EKF drift corrector started (publish_error={self.publish_error})')

    def wheel_cb(self, side, msg: JointState):
        """Wheel surface speed from consecutive encoder angles (single-frame difference)."""
        if not msg.position:
            return
        w = self.wheels[side]
        angle, t = float(msg.position[0]), stamp_to_sec(msg.header.stamp)
        last_angle, last_t = w[0], w[1]
        if last_t is None or t > last_t:
            w[0], w[1] = angle, t
        if last_t is not None and t - last_t > 1e-4:
            w[2] = self.wheel_radius * (angle - last_angle) / (t - last_t)

    def imu_cb(self, msg: Imu):
        self.imu_ax = msg.linear_acceleration.x
        self.imu_ay = msg.linear_acceleration.y

    def gt_cb(self, msg: Odometry):
        self.gt_vx = msg.twist.twist.linear.x

    def ekf_cb(self, msg: Odometry):
        ekf_vx = msg.twist.twist.linear.x
        left_vel, right_vel = self.wheels['left'][2], self.wheels['right'][2]
        raw_wheel_vx = 0.5 * (left_vel + right_vel)
        scaled_wheel_vx = self.slip_scale * raw_wheel_vx

        wheels_stopped = abs(left_vel) < 0.05 and abs(right_vel) < 0.05
        chassis_settled = math.hypot(self.imu_ax, self.imu_ay) < 0.20

        if wheels_stopped and chassis_settled:
            # 1. zero-velocity update
            self.stopped_frames += 1
            if self.stopped_frames >= 2:
                self.vel_bias = ekf_vx
        else:
            self.stopped_frames = 0
            corrected = ekf_vx - self.vel_bias
            # 2. upper clamp while accelerating forward
            if self.imu_ax > 0.25 and scaled_wheel_vx > 0.5 and corrected > scaled_wheel_vx:
                self.vel_bias = ekf_vx - scaled_wheel_vx
                corrected = scaled_wheel_vx
            # 3. non-negative floor while the wheels are not reversing
            if raw_wheel_vx >= -0.05 and corrected < 0.0:
                self.vel_bias = ekf_vx

        corrected_vx = max(-5.0, min(self.v_max, ekf_vx - self.vel_bias))

        out = Odometry()
        out.header = msg.header
        out.child_frame_id = msg.child_frame_id
        out.pose = msg.pose
        out.twist = msg.twist
        out.twist.twist.linear.x = float(corrected_vx)
        out.twist.twist.linear.y = 0.0
        self.odom_pub.publish(out)

        if self.publish_error:
            self.bias_pub.publish(Float64(data=float(self.vel_bias)))
            if self.gt_vx is not None:
                self.error_pub.publish(Float64(data=float(corrected_vx - self.gt_vx)))


def main():
    rclpy.init()
    node = OdomDriftCorrector()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
