"""Functional test: launch pure_pursuit.launch.py on synthetic racelines, publish poses, check /drive.

Needs a sourced ROS 2 environment with the control package built:
    python3 -m pytest ws/src/control/test
"""
import math
import os
import signal
import subprocess
import tempfile
import time

import numpy as np
import pytest

rclpy = pytest.importorskip('rclpy')
from ackermann_msgs.msg import AckermannDriveStamped  # noqa: E402
from geometry_msgs.msg import PoseStamped  # noqa: E402

SPEED_SCALE = 0.5
MAX_STEER = 0.524
HEADER = '# s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2'


def write_raceline(tracks_dir, name, xy, psi, kappa, vx):
    d = os.path.join(tracks_dir, name)
    os.makedirs(d, exist_ok=True)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    n = len(xy)
    rows = np.column_stack([s, xy, psi, np.full(n, kappa), np.full(n, vx), np.zeros(n)])
    np.savetxt(os.path.join(d, 'raceline.csv'), rows, delimiter=';', fmt='%.7f', header=HEADER[2:])


def circle(radius, n):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)  # counter-clockwise
    return np.column_stack([radius * np.cos(t), radius * np.sin(t)]), t + np.pi / 2


@pytest.fixture(scope='module')
def tracks_dir():
    with tempfile.TemporaryDirectory() as d:
        xy, psi = circle(4.0, 200)
        write_raceline(d, 'circle', xy, psi, 0.25, 4.0)
        x = np.arange(0.0, 20.0, 0.2)
        write_raceline(d, 'straight', np.column_stack([x, np.zeros_like(x)]), np.zeros_like(x), 0.0, 3.0)
        xy, psi = circle(0.2, 10)  # every waypoint within the lookahead distance
        write_raceline(d, 'tiny', xy, psi, 5.0, 1.0)
        yield d


@pytest.fixture(scope='module')
def ros():
    rclpy.init()
    node = rclpy.create_node('pure_pursuit_test')
    yield node
    node.destroy_node()
    rclpy.shutdown()


def launch(tracks_dir, track):
    env = dict(os.environ, TRACKS_DIR=tracks_dir)
    return subprocess.Popen(
        ['ros2', 'launch', 'control', 'pure_pursuit.launch.py', f'track:={track}',
         f'speed_scale:={SPEED_SCALE}'],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)


def stop(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGINT)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    return proc.stdout.read()


class Driver:
    def __init__(self, node):
        self.node = node
        self.pub = node.create_publisher(PoseStamped, '/tracked_pose', 10)
        self.drives = []
        self.sub = node.create_subscription(AckermannDriveStamped, '/drive', self.drives.append, 10)

    def wait_for_node(self, proc, timeout=20.0):
        end = time.time() + timeout
        while self.pub.get_subscription_count() == 0:
            assert proc.poll() is None, 'node exited:\n' + proc.stdout.read()
            assert time.time() < end, 'pure_pursuit_node did not subscribe to /tracked_pose'
            rclpy.spin_once(self.node, timeout_sec=0.1)

    def drive(self, x, y, yaw, n=10):
        """Publish the same pose n times (the lookahead adapts after the first) and return the last /drive."""
        self.drives.clear()
        msg = PoseStamped()
        msg.header.frame_id = 'map'
        msg.pose.position.x, msg.pose.position.y = x, y
        msg.pose.orientation.z, msg.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        for _ in range(n):
            self.pub.publish(msg)
            end = time.time() + 0.1
            while time.time() < end:
                rclpy.spin_once(self.node, timeout_sec=0.02)
        end = time.time() + 2.0
        while len(self.drives) < n and time.time() < end:
            rclpy.spin_once(self.node, timeout_sec=0.05)
        assert self.drives, f'no /drive for pose ({x}, {y}, {yaw})'
        d = self.drives[-1].drive
        assert math.isfinite(d.steering_angle) and abs(d.steering_angle) <= MAX_STEER + 1e-6
        return d

    def close(self):
        self.node.destroy_publisher(self.pub)
        self.node.destroy_subscription(self.sub)


@pytest.fixture
def run(ros, tracks_dir):
    procs, drivers = [], []

    def start(track):
        proc = launch(tracks_dir, track)
        driver = Driver(ros)
        procs.append(proc)
        drivers.append(driver)
        driver.wait_for_node(proc)
        return driver, proc

    yield start
    for d in drivers:
        d.close()
    for p in procs:
        out = stop(p)
        # the only traceback allowed is rclpy.spin interrupted by our SIGINT
        assert out.count('Traceback') == out.count('] KeyboardInterrupt'), out


def test_circle(run):
    driver, _ = run('circle')
    on = driver.drive(4.0, 0.0, math.pi / 2)  # on the path heading along it (counter-clockwise)
    assert 0.0 < on.steering_angle
    assert on.speed == pytest.approx(4.0 * SPEED_SCALE)
    inside = driver.drive(3.0, 0.0, math.pi / 2)  # path is to the right
    assert inside.steering_angle < 0.0
    outside = driver.drive(5.0, 0.0, math.pi / 2)  # path is to the left: merge toward it, ahead along the line
    assert 0.0 < outside.steering_angle <= MAX_STEER


def test_straight(run):
    driver, _ = run('straight')  # kappa = 0 everywhere
    left = driver.drive(5.0, 0.5, 0.0)
    assert left.steering_angle < 0.0
    assert left.speed == pytest.approx(3.0 * SPEED_SCALE)
    right = driver.drive(5.0, -0.5, 0.0)
    assert right.steering_angle > 0.0
    assert driver.drive(5.0, 0.0, 0.0).steering_angle == pytest.approx(0.0, abs=1e-6)


def test_tiny_raceline_within_lookahead(run):
    driver, _ = run('tiny')
    d = driver.drive(0.2, 0.0, math.pi / 2)
    assert d.speed == pytest.approx(1.0 * SPEED_SCALE)


def test_missing_raceline(tracks_dir):
    proc = launch(tracks_dir, 'no_such_track')
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        pass
    out = stop(proc)
    assert 'Raceline not found' in out
