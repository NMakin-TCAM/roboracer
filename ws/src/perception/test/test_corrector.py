"""corrector: calibrated clamp, encoder reset jumps, stale encoders."""
import pytest
import rclpy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState

from perception.corrector import OdomDriftCorrector


@pytest.fixture
def node():
    rclpy.init()
    n = OdomDriftCorrector()
    n.out = []
    n.odom_pub.publish = n.out.append
    yield n
    n.destroy_node()
    rclpy.try_shutdown()


def encoder(node, side, angle, t):
    msg = JointState()
    msg.position = [float(angle)]
    msg.header.stamp.sec, msg.header.stamp.nanosec = int(t), int(round((t % 1) * 1e9))
    node.wheel_cb(side, msg)


def ekf(node, vx):
    msg = Odometry()
    msg.twist.twist.linear.x = float(vx)
    node.ekf_cb(msg)
    return node.out[-1].twist.twist.linear.x


def drive(node, speed, t0=10.0, dt=0.01, frames=3):
    omega = speed / 0.0585
    for i in range(frames):
        t = t0 + i * dt
        for side in ('left', 'right'):
            encoder(node, side, omega * t, t)


def test_clamp_uses_calibrated_wheel_speed(node):
    drive(node, 5.0)
    node.imu_ax = 1.0
    assert ekf(node, 6.0) == pytest.approx(5.0, abs=1e-3)


def test_encoder_reset_jump_skipped(node):
    drive(node, 5.0)
    encoder(node, 'left', 0.0, 10.03)   # encoders reset to 0 at a sim reset
    assert node.wheels['left'][2] == pytest.approx(5.0, abs=1e-3)


def test_stale_encoders_hold_bias(node):
    drive(node, 0.0)
    node.vel_bias = 0.3
    for w in node.wheels.values():
        w[3] -= 1.0                     # last encoder message 1 s ago
    assert ekf(node, 2.0) == pytest.approx(1.7)
    assert ekf(node, 2.0) == pytest.approx(1.7)   # no zero-velocity update from stale zeros
    assert node.vel_bias == 0.3
