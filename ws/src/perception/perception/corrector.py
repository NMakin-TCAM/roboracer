import math
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.executors import ExternalShutdownException
from builtin_interfaces.msg import Time
from sensor_msgs.msg import JointState, Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64


class OdomDriftCorrector(Node):

    def __init__(self):
        super().__init__('odom_drift_corrector')

        # Input & Output topics
        self.declare_parameter('ekf_odom_topic', '/odometry/filtered_accl')
        self.declare_parameter('left_encoder_topic', '/autodrive/roboracer_1/left_encoder')
        self.declare_parameter('right_encoder_topic', '/autodrive/roboracer_1/right_encoder')
        self.declare_parameter('imu_topic', '/autodrive/roboracer_1/imu')
        self.declare_parameter('gt_odom_topic', '/autodrive/roboracer_1/odom')
        self.declare_parameter('output_odom_topic', '/odometry/corrected')

        # Parameterized error publishing
        self.declare_parameter('publish_error', True)
        self.declare_parameter('error_topic', '/odometry/velocity_error')
        self.declare_parameter('bias_topic', '/odometry/velocity_bias')

        # Vehicle physical constants
        self.declare_parameter('wheel_radius', 0.059)
        self.declare_parameter('v_max', 22.88)
        self.declare_parameter('slip_scale', 0.9152)  # 22.88 / 25.0

        self.wheel_radius = self.get_parameter('wheel_radius').value
        self.v_max = self.get_parameter('v_max').value
        self.slip_scale = self.get_parameter('slip_scale').value
        self.publish_error = self.get_parameter('publish_error').value

        self.left_prev_pos = None
        self.right_prev_pos = None
        self.left_prev_t = None
        self.right_prev_t = None

        self.left_vel = 0.0
        self.right_vel = 0.0
        self.imu_ax = 0.0
        self.imu_ay = 0.0
        self.gt_vx = None

        self.vel_bias = 0.0
        self.stopped_frames = 0

        self.create_subscription(
            JointState,
            self.get_parameter('left_encoder_topic').value,
            self.left_cb,
            qos_profile_sensor_data
        )
        self.create_subscription(
            JointState,
            self.get_parameter('right_encoder_topic').value,
            self.right_cb,
            qos_profile_sensor_data
        )
        self.create_subscription(
            Imu,
            self.get_parameter('imu_topic').value,
            self.imu_cb,
            qos_profile_sensor_data
        )
        self.create_subscription(
            Odometry,
            self.get_parameter('ekf_odom_topic').value,
            self.ekf_cb,
            10
        )

        self.odom_pub = self.create_publisher(
            Odometry,
            self.get_parameter('output_odom_topic').value,
            10
        )

        if self.publish_error:
            gt_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=1
            )
            self.create_subscription(
                Odometry,
                self.get_parameter('gt_odom_topic').value,
                self.gt_cb,
                gt_qos
            )
            self.error_pub = self.create_publisher(
                Float64,
                self.get_parameter('error_topic').value,
                10
            )
            self.bias_pub = self.create_publisher(
                Float64,
                self.get_parameter('bias_topic').value,
                10
            )

        self.get_logger().info(
            f'Post-EKF Drift Corrector started (publish_error={self.publish_error})'
        )

    def stamp_to_sec(self, stamp: Time) -> float:
        return stamp.sec + stamp.nanosec * 1e-9

    def wheel_speed(self, pos: float, prev_pos: float, t: float, prev_t: float) -> float:
        if prev_pos is None or prev_t is None:
            return None
        dt = t - prev_t
        if dt <= 1e-4:
            return None
        return self.wheel_radius * (pos - prev_pos) / dt

    def left_cb(self, msg: JointState):
        if not msg.position:
            return
        pos = float(msg.position[0])
        t = self.stamp_to_sec(msg.header.stamp)
        v = self.wheel_speed(pos, self.left_prev_pos, t, self.left_prev_t)
        if self.left_prev_t is None or t > self.left_prev_t:
            self.left_prev_pos, self.left_prev_t = pos, t
        if v is not None:
            self.left_vel = v

    def right_cb(self, msg: JointState):
        if not msg.position:
            return
        pos = float(msg.position[0])
        t = self.stamp_to_sec(msg.header.stamp)
        v = self.wheel_speed(pos, self.right_prev_pos, t, self.right_prev_t)
        if self.right_prev_t is None or t > self.right_prev_t:
            self.right_prev_pos, self.right_prev_t = pos, t
        if v is not None:
            self.right_vel = v

    def imu_cb(self, msg: Imu):
        self.imu_ax = msg.linear_acceleration.x
        self.imu_ay = msg.linear_acceleration.y

    def gt_cb(self, msg: Odometry):
        self.gt_vx = msg.twist.twist.linear.x

    def ekf_cb(self, msg: Odometry):
        ekf_vx = msg.twist.twist.linear.x
        raw_wheel_vx = 0.5 * (self.left_vel + self.right_vel)
        scaled_wheel_vx = self.slip_scale * raw_wheel_vx

        wheels_stopped = (abs(self.left_vel) < 0.05 and abs(self.right_vel) < 0.05)
        chassis_settled = math.hypot(self.imu_ax, self.imu_ay) < 0.20

        # 1. HARD ZUPT: Both wheels stopped AND chassis acceleration settled
        if wheels_stopped and chassis_settled:
            self.stopped_frames += 1
            if self.stopped_frames >= 2:
                self.vel_bias = ekf_vx
        else:
            self.stopped_frames = 0
            current_corrected_vx = ekf_vx - self.vel_bias

            # 2. UPPER-BOUND WHEEL CLAMP (While Accelerating Forward)
            if self.imu_ax > 0.25 and scaled_wheel_vx > 0.5:
                if current_corrected_vx > scaled_wheel_vx:
                    self.vel_bias = ekf_vx - scaled_wheel_vx
                    current_corrected_vx = scaled_wheel_vx

            # 3. NON-NEGATIVE FLOOR (When Wheels Are Not Reversing)
            if raw_wheel_vx >= -0.05 and current_corrected_vx < 0.0:
                self.vel_bias = ekf_vx

        corrected_vx = max(-5.0, min(self.v_max, ekf_vx - self.vel_bias))

        out_msg = Odometry()
        out_msg.header = msg.header
        out_msg.child_frame_id = msg.child_frame_id
        out_msg.pose = msg.pose
        out_msg.twist = msg.twist
        out_msg.twist.twist.linear.x = float(corrected_vx)
        out_msg.twist.twist.linear.y = 0.0

        self.odom_pub.publish(out_msg)

        # Publish error & bias if enabled
        if self.publish_error:
            self.bias_pub.publish(Float64(data=float(self.vel_bias)))
            if self.gt_vx is not None:
                err = corrected_vx - self.gt_vx
                self.error_pub.publish(Float64(data=float(err)))


def main(args=None):
    rclpy.init(args=args)
    node = OdomDriftCorrector()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()