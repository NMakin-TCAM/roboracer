"""avoid_graph.launch.py, launched for real down to control's pure pursuit, on ROS domain 60.

The test is the localizer (localizer:=external): it holds a car on the raceline 6 m
before a static fake obstacle, and reads what pure pursuit sends on /drive. Nothing
moves (there is no simulator), so this checks the decision and the wiring, not driving.

  evade    an obstacle the car can pass: the planner evades, /local_path is the spliced
           raceline and valid, and pure pursuit drives
  no room  an obstacle wall to wall: the planner says no room, /local_path/valid is false,
           and every /drive command after that is a stop. Before PlanResult the planner's
           empty trajectory read as "clear", and the car would have been sent on.

This publishes /drive (through pure pursuit). Keep it off a live stack's domain.
"""
import os
import shutil
import signal
import subprocess
import time

import numpy as np
import pytest

rclpy = pytest.importorskip('rclpy')
pytest.importorskip('f1t_msgs.msg')
ackermann = pytest.importorskip('ackermann_msgs.msg')
from ament_index_python.packages import (PackageNotFoundError,  # noqa: E402
                                         get_package_share_directory)
from geometry_msgs.msg import PoseStamped  # noqa: E402
from nav_msgs.msg import Odometry, Path  # noqa: E402
from std_msgs.msg import Bool  # noqa: E402

from f1t_msgs.msg import PlannerStatus  # noqa: E402
from planning.local_path import load_raceline_rows  # noqa: E402
from planning.raceline import write_raceline  # noqa: E402

DOMAIN = 60
A, B, N = 7.0, 4.0, 360           # oval semi-axes, points: lap about 35 m
TRACK_HALF_WIDTH = 1.0
EGO_S, OBSTACLE_AHEAD = 1.0, 6.0
TIMEOUT = 30.0


def write_oval(track_dir):
    th = np.linspace(0, 2 * np.pi, N, endpoint=False)
    xy = np.column_stack((A * np.cos(th), B * np.sin(th)))
    np.savetxt(track_dir / 'centerline.csv',
               np.column_stack((xy, np.full(N, TRACK_HALF_WIDTH), np.full(N, TRACK_HALF_WIDTH))),
               delimiter=',', header='x_m,y_m,w_tr_right_m,w_tr_left_m')
    seg = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg)[:-1]))
    t = np.roll(xy, -1, axis=0) - np.roll(xy, 1, axis=0)
    psi = (np.arctan2(t[:, 1], t[:, 0]) - np.pi / 2 + np.pi) % (2 * np.pi) - np.pi
    kappa = A * B / ((A * np.sin(th)) ** 2 + (B * np.cos(th)) ** 2) ** 1.5
    traj = np.column_stack((s, xy, psi, kappa, np.full(N, 3.0), np.zeros(N)))
    traj = np.vstack((traj, traj[0]))
    traj[-1, 0] = s[-1] + seg[-1]
    write_raceline(str(track_dir / 'raceline.csv'), traj,
                   {'method': 'mintime', 'clean': True, 'notes': ['synthetic test oval']})


def stop(proc):
    try:
        os.killpg(proc.pid, signal.SIGINT)
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass
    except ProcessLookupError:
        return ''
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.killpg(proc.pid, 0)
        except ProcessLookupError:
            return ''
        time.sleep(0.2)
    survivors = subprocess.run(['ps', '-o', 'pid,args', '-g', str(proc.pid)],
                               capture_output=True, text=True).stdout
    os.killpg(proc.pid, signal.SIGKILL)
    return survivors


@pytest.fixture
def launch(tmp_path):
    if shutil.which('ros2') is None:
        pytest.skip('ros2 CLI not on PATH')
    try:
        share = get_package_share_directory('planning')
        get_package_share_directory('control')
    except PackageNotFoundError:
        pytest.skip('planning and control are not built and sourced')
    if not os.path.isfile(os.path.join(share, 'launch', 'avoid_graph.launch.py')):
        pytest.skip('installed planning predates avoid_graph.launch.py; rebuild it')

    tracks = tmp_path / 'tracks'
    (tracks / 'oval').mkdir(parents=True)
    write_oval(tracks / 'oval')
    procs = []

    def start(**args):
        log_path = tmp_path / 'launch.log'
        log = log_path.open('w')
        cmd = ['ros2', 'launch', 'planning', 'avoid_graph.launch.py', 'track:=oval',
               'localizer:=external', 'use_sim_time:=false',
               f'obstacle_s:={EGO_S + OBSTACLE_AHEAD}'] + [f'{k}:={v}' for k, v in args.items()]
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                env=dict(os.environ, ROS_DOMAIN_ID=str(DOMAIN),
                                         TRACKS_DIR=str(tracks)))
        procs.append((proc, log))
        return proc, log_path, tracks / 'oval'

    yield start
    for proc, log in procs:
        leftover = stop(proc)
        log.close()
        assert not leftover, f'launch left processes behind:\n{leftover}'


class Harness:
    """Holds the car still on the raceline at EGO_S and records the planner, path and /drive."""

    def __init__(self, node, track_dir):
        self.node = node
        self.s, self.xy = load_raceline_rows(str(track_dir / 'raceline.csv'))
        seg = np.roll(self.xy, -1, axis=0) - self.xy
        self.yaw = np.arctan2(seg[:, 1], seg[:, 0])
        self.i = int(np.argmin(np.abs(self.s - EGO_S)))
        self.status, self.path, self.valid, self.drive = [], [], [], []
        t = time.monotonic
        self.notes = []
        node.create_subscription(PlannerStatus, '/planner/avoidance/status',
                                 lambda m: (self.status.append((t(), m.status)),
                                            self.notes.append(m.note)), 10)
        node.create_subscription(Path, '/local_path', lambda m: self.path.append((t(), m)), 10)
        node.create_subscription(Bool, '/local_path/valid',
                                 lambda m: self.valid.append((t(), m.data)), 10)
        node.create_subscription(ackermann.AckermannDriveStamped, '/drive',
                                 lambda m: self.drive.append((t(), m.drive.speed,
                                                              m.drive.steering_angle)), 10)
        self.pose_pub = node.create_publisher(PoseStamped, '/localized_pose', 10)
        self.odom_pub = node.create_publisher(Odometry, '/odom', 10)
        node.create_timer(0.05, self.publish)

    def publish(self):
        stamp = self.node.get_clock().now().to_msg()
        p = PoseStamped()
        p.header.stamp, p.header.frame_id = stamp, 'map'
        p.pose.position.x, p.pose.position.y = map(float, self.xy[self.i])
        yaw = float(self.yaw[self.i])
        p.pose.orientation.z, p.pose.orientation.w = np.sin(yaw / 2), np.cos(yaw / 2)
        self.pose_pub.publish(p)
        o = Odometry()
        o.header.stamp, o.header.frame_id, o.child_frame_id = stamp, 'odom', 'base_link'
        self.odom_pub.publish(o)

    def spin_until(self, done, timeout, proc, log_path):
        end = time.monotonic() + timeout
        while time.monotonic() < end and not done():
            rclpy.spin_once(self.node, timeout_sec=0.05)
            assert proc.poll() is None, log_path.read_text()[-4000:]
            log = log_path.read_text()
            assert 'process has died' not in log, log[-4000:]


def run(launch, **args):
    proc, log_path, track_dir = launch(**args)
    rclpy.init(domain_id=DOMAIN)
    node = rclpy.create_node('avoid_graph_test_harness')
    return proc, log_path, track_dir, node


def settled(h, status, extra=2.0):
    """The first time `status` was seen, plus `extra` s of everything after it."""
    hits = [t for t, s in h.status if s == status]
    return bool(hits) and time.monotonic() > hits[0] + extra


def test_an_obstacle_with_room_is_evaded_and_the_car_drives(launch):
    proc, log_path, track_dir, node = run(launch)
    try:
        h = Harness(node, track_dir)
        h.spin_until(lambda: settled(h, 'evading'), TIMEOUT, proc, log_path)
        log = log_path.read_text()
        assert 'process has died' not in log, log[-4000:]
        assert any(s == 'evading' for _, s in h.status), (h.status[-3:], log[-4000:])
        assert 'obstacle_spliner_node' not in log, 'their planner was started alongside ours'
        t0 = next(t for t, s in h.status if s == 'evading') + 0.5
        after = [v for t, v in h.valid if t > t0]
        assert after and all(after), (
            f'valid after evading: {len(after)} msgs, {sorted(set(after))}; statuses '
            f'{[s for t, s in h.status if t > t0][-5:]}\n' + '\n'.join(
                l for l in log.splitlines() if 'local_path' in l or 'spliner' in l)[-3000:])
        path = [m for t, m in h.path if t > t0][-1]
        assert path.header.frame_id == 'map' and len(path.poses) == len(h.xy)
        xy = np.array([[q.pose.position.x, q.pose.position.y] for q in path.poses])
        off = np.hypot(*(xy - h.xy).T)
        at_obs = np.abs(h.s - (EGO_S + OBSTACLE_AHEAD)) < 0.2
        assert off[at_obs].min() > 0.3, 'the path does not leave the raceline at the obstacle'
        far = (h.s > EGO_S + OBSTACLE_AHEAD + 4.0) & (h.s < EGO_S + 20.0)
        assert off[far].max() < 1e-6, 'the path was moved away from the evasion'
        speeds = [v for t, v, _ in h.drive if t > t0]
        assert speeds and min(speeds) > 0.0, f'pure pursuit did not drive: {speeds[:10]}'
        steer = float(np.median([a for t, _, a in h.drive if t > t0]))
        tangent = np.roll(h.xy, -1, axis=0) - h.xy
        rel = xy - h.xy
        signed = np.sign(tangent[:, 0] * rel[:, 1] - tangent[:, 1] * rel[:, 0]) * off   # + = left
        near = np.abs(h.s - (EGO_S + 1.2)) < 0.1      # about one lookahead ahead of the car
        print(f'planner: {h.notes[-1]}; spliced d at the obstacle {signed[at_obs].mean():+.2f} m, '
              f'1.2 m ahead of the car {signed[near].mean():+.3f} m; pure pursuit steering '
              f'{steer:+.3f} rad (+ = left)')
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def test_no_room_tells_the_car_to_stop(launch):
    """The case that would otherwise drive it into something."""
    proc, log_path, track_dir, node = run(launch, obstacle_width=2.5)
    try:
        h = Harness(node, track_dir)
        h.spin_until(lambda: settled(h, 'no room'), TIMEOUT, proc, log_path)
        log = log_path.read_text()
        assert 'process has died' not in log, log[-4000:]
        assert any(s == 'no room' for _, s in h.status), (h.status[-3:], log[-4000:])
        assert all(s == 'no room' for _, s in h.status), \
            f'the planner did not consistently refuse: {sorted(set(s for _, s in h.status))}'
        t0 = next(t for t, s in h.status if s == 'no room') + 0.3
        valid = [v for t, v in h.valid if t > t0]
        assert valid and not any(valid), '/local_path/valid was not held false'
        speeds = [v for t, v, _ in h.drive if t > t0]
        assert speeds, 'pure pursuit published nothing'
        assert all(v == 0.0 for v in speeds), f'a non-zero speed was commanded: {set(speeds)}'
        assert 'no collision-free /local_path' in log, 'pure pursuit did not stop for this reason'
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
