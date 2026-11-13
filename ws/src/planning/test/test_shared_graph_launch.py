"""shared_graph.launch.py, launched for real, on a synthetic oval and ROS domain 58.

The test is the localizer (localizer:=external): it publishes /localized_pose and
/odom for an ego that follows 3 m behind fake_opponent_node's weaving opponent.
It needs the built and sourced planning package and the ros2 CLI, and skips
without them. It publishes no /drive and no simulator topic, but still keep it
off a live stack's domain.

Phase 1 asserts what the graph can do by itself today: every node that does not
depend on the main GP publishes within a bounded time, the planners' obstacles
come from obstacle_tracker_node (one stable id, is_static False, vs near the
opponent's speed) and opponent_trajectory builds its half lap from them, the waypoints are latched
for a late subscriber, and the two nodes that need the GP's /opponent_trajectory
(predictor_opponent_trajectory, collision_prediction) stay silent. Phase 2 stands
in for the missing GP with a synthetic /opponent_trajectory along the raceline,
and asserts that both of them then leave their wait loops and publish.
collision_prediction publishes force_trailing only when it has taken a dynamic
obstacle as the opponent, so that message is the evidence it sees the tracked one.

This exercises transport, QoS and launch wiring only. It says nothing about
whether an overtake would work.
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
from ament_index_python.packages import (PackageNotFoundError,  # noqa: E402
                                         get_package_share_directory)
from geometry_msgs.msg import PoseStamped  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from std_msgs.msg import Bool, String  # noqa: E402

from f1t_msgs.msg import (ObstacleArray, OppWpnt, OpponentTrajectory,  # noqa: E402
                          ProjOppTraj, WpntArray)
from planning.global_waypoints_publisher import load_global_waypoints  # noqa: E402
from planning.raceline import write_raceline  # noqa: E402
from planning.waypoint_qos import WAYPOINTS_QOS  # noqa: E402

DOMAIN = 58
A, B, N = 7.0, 4.0, 360          # oval semi-axes and points, lap about 35 m
EGO_GAP = 3.0                    # m behind the opponent; opponent_trajectory looks 6 m ahead
SPEED = 3.0                      # m/s, opponent and ego
PHASE1_TIMEOUT = 40.0            # s, includes node startup and half a lap of tracking
PHASE2_TIMEOUT = 60.0            # s, the predictor needs a full opponent lap first

# topic -> (type, qos); qos None is depth 10 VOLATILE
WATCH = {
    '/global_waypoints': (WpntArray, WAYPOINTS_QOS),
    '/global_waypoints_scaled': (WpntArray, WAYPOINTS_QOS),
    '/state_machine': (String, None),
    '/car_state/odom_frenet': (Odometry, None),
    '/perception/detections': (ObstacleArray, None),
    '/perception/obstacles': (ObstacleArray, None),
    '/proj_opponent_trajectory': (ProjOppTraj, None),
    '/global_waypoints_updated': (WpntArray, None),
    '/opponent_trajectory': (OpponentTrajectory, None),
    '/collision_prediction/force_trailing': (Bool, None),
    '/collision_prediction/obstacles': (ObstacleArray, None),
}
# Published by the graph on its own, by node.
PHASE1 = {
    '/global_waypoints': 'global_waypoints_publisher',
    '/global_waypoints_scaled': 'global_waypoints_publisher',
    '/state_machine': 'state_machine_stub',
    '/car_state/odom_frenet': 'frenet_state_node',
    '/perception/detections': 'fake_opponent_node',
    '/perception/obstacles': 'obstacle_tracker_node',
    '/proj_opponent_trajectory': 'opponent_trajectory',
    '/global_waypoints_updated': 'waypoint_updater',
}
# Blocked on the main GP's /opponent_trajectory, which nothing in the repo publishes.
NEEDS_GP = {
    '/opponent_trajectory': 'predictor_opponent_trajectory',
    '/collision_prediction/force_trailing': 'collision_predictor',
}
PREDICTOR_FRAME = 'opponent_trajectory'     # header.frame_id the predictor stamps
SYNTHETIC_FRAME = 'synthetic_gp'


def write_oval(track_dir):
    """A closed TUM-format raceline on the centerline of an oval, as build_raceline writes."""
    th = np.linspace(0, 2 * np.pi, N, endpoint=False)
    xy = np.column_stack((A * np.cos(th), B * np.sin(th)))
    np.savetxt(track_dir / 'centerline.csv', np.column_stack((xy, np.full(N, 1.0), np.full(N, 1.0))),
               delimiter=',', header='x_m,y_m,w_tr_right_m,w_tr_left_m')
    seg = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg)[:-1]))
    t = np.roll(xy, -1, axis=0) - np.roll(xy, 1, axis=0)
    psi = (np.arctan2(t[:, 1], t[:, 0]) - np.pi / 2 + np.pi) % (2 * np.pi) - np.pi
    kappa = A * B / ((A * np.sin(th)) ** 2 + (B * np.cos(th)) ** 2) ** 1.5
    vx = np.full(N, 4.0)
    traj = np.column_stack((s, xy, psi, kappa, vx, np.zeros(N)))
    traj = np.vstack((traj, traj[0]))
    traj[-1, 0] = s[-1] + seg[-1]
    write_raceline(str(track_dir / 'raceline.csv'), traj,
                   {'method': 'mintime', 'clean': True, 'notes': ['synthetic test oval']})


@pytest.fixture
def graph(tmp_path):
    if shutil.which('ros2') is None:
        pytest.skip('ros2 CLI not on PATH')
    try:
        share = get_package_share_directory('planning')
    except PackageNotFoundError:
        pytest.skip('planning is not built and sourced')
    if not os.path.isfile(os.path.join(share, 'launch', 'shared_graph.launch.py')):
        pytest.skip('installed planning predates shared_graph.launch.py; rebuild it')

    tracks = tmp_path / 'tracks'
    (tracks / 'oval').mkdir(parents=True)
    write_oval(tracks / 'oval')
    env = dict(os.environ, ROS_DOMAIN_ID=str(DOMAIN), TRACKS_DIR=str(tracks))
    log_path = tmp_path / 'launch.log'
    log = log_path.open('w')
    proc = subprocess.Popen(
        ['ros2', 'launch', 'planning', 'shared_graph.launch.py', 'track:=oval',
         'localizer:=external', 'use_sim_time:=false', 'opponent_mode:=weaving',
         f'opponent_speed:={SPEED}', 'opponent_s0:=4.0', 'opponent_max_range:=100.0'],
        env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        yield proc, load_global_waypoints(str(tracks / 'oval')), log_path
    finally:
        leftover = stop(proc)
        log.close()
        assert not leftover, f'launch left processes behind:\n{leftover}'


def stop(proc):
    """SIGINT the launch's process group (AGENTS.md); return what survived it, killed."""
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


class Harness:
    """The localizer, the ego's odometry, and a recorder for every output topic."""

    def __init__(self, node, waypoints):
        self.node = node
        w = waypoints.waypoints
        self.s = np.array([p.s_m for p in w])
        self.x = np.array([p.x_m for p in w])
        self.y = np.array([p.y_m for p in w])
        self.yaw = np.unwrap([p.psi_rad for p in w])
        self.lap = waypoints.lap_length
        self.wpnts = w
        self.first = {}               # topic -> monotonic time of first message
        self.last = {}
        self.ego_s = 0.0
        self.opp_s = None
        self.opp_seen = 0.0
        self.t0 = time.monotonic()
        self.tracked = []             # (t, id, is_static, vs, is_visible) per tracked obstacle
        self.vs_var = []

        for topic, (kind, qos) in WATCH.items():
            node.create_subscription(kind, topic, lambda m, t=topic: self.record(t, m),
                                     qos if qos is not None else 10)
        self.pose_pub = node.create_publisher(PoseStamped, '/localized_pose', 10)
        self.odom_pub = node.create_publisher(Odometry, '/odom', 10)
        self.gp_pub = None
        node.create_timer(0.05, self.drive)

    def record(self, topic, msg):
        if topic == '/opponent_trajectory' and msg.header.frame_id != PREDICTOR_FRAME:
            return                    # our own synthetic GP output, not the predictor's
        if topic == '/perception/obstacles' and msg.obstacles:
            self.opp_s = float(msg.obstacles[0].s_center)
            self.opp_seen = time.monotonic()
            for o in msg.obstacles:
                self.tracked.append((time.monotonic() - self.t0, o.id, o.is_static, o.vs,
                                     o.is_visible))
                self.vs_var.append(o.vs_var)
        self.first.setdefault(topic, time.monotonic() - self.t0)
        self.last[topic] = msg

    def drive(self):
        if self.opp_s is not None and time.monotonic() - self.opp_seen < 0.5:
            self.ego_s = (self.opp_s - EGO_GAP) % self.lap
        else:
            self.ego_s = (self.ego_s + 1.0) % self.lap     # sweep until the opponent is in view
        stamp = self.node.get_clock().now().to_msg()
        yaw = float(np.interp(self.ego_s, self.s, self.yaw))
        pose = PoseStamped()
        pose.header.stamp, pose.header.frame_id = stamp, 'map'
        pose.pose.position.x = float(np.interp(self.ego_s, self.s, self.x))
        pose.pose.position.y = float(np.interp(self.ego_s, self.s, self.y))
        pose.pose.orientation.z, pose.pose.orientation.w = np.sin(yaw / 2), np.cos(yaw / 2)
        self.pose_pub.publish(pose)
        odom = Odometry()
        odom.header.stamp, odom.header.frame_id, odom.child_frame_id = stamp, 'odom', 'base_link'
        odom.twist.twist.linear.x = SPEED
        self.odom_pub.publish(odom)

    def start_synthetic_gp(self):
        """Stand in for the main GP: the opponent's line is the raceline, at SPEED."""
        self.gp_pub = self.node.create_publisher(OpponentTrajectory, '/opponent_trajectory', 10)
        msg = OpponentTrajectory(lap_count=2.0, opp_is_on_trajectory=True)
        msg.header.frame_id = SYNTHETIC_FRAME
        msg.oppwpnts = [OppWpnt(id=w.id, s_m=w.s_m, d_m=0.0, x_m=w.x_m, y_m=w.y_m,
                                proj_vs_mps=SPEED, vd_mps=0.0, d_var=0.01, vs_var=0.01)
                        for w in self.wpnts]

        def publish():
            msg.header.stamp = self.node.get_clock().now().to_msg()
            self.gp_pub.publish(msg)
        self.node.create_timer(1.0, publish)

    def spin_until(self, topics, timeout, proc, log_path):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not all(t in self.first for t in topics):
            rclpy.spin_once(self.node, timeout_sec=0.05)
            assert proc.poll() is None, 'launch exited:\n' + log_path.read_text()[-4000:]


def missing(h, topics):
    return {t: node for t, node in topics.items() if t not in h.first}


def test_graph_comes_up_and_only_the_gp_is_missing(graph):
    proc, waypoints, log_path = graph
    rclpy.init(domain_id=DOMAIN)
    node = rclpy.create_node('shared_graph_test_harness')
    try:
        h = Harness(node, waypoints)

        # ---- phase 1: everything that does not need the main GP
        h.spin_until(PHASE1, PHASE1_TIMEOUT, proc, log_path)
        log = log_path.read_text()
        assert not missing(h, PHASE1), (f'silent after {PHASE1_TIMEOUT:.0f} s: '
                                        f'{missing(h, PHASE1)}\n{log[-4000:]}')
        assert len(h.last['/global_waypoints'].wpnts) == len(waypoints.waypoints)
        assert h.last['/global_waypoints_scaled'].wpnts == h.last['/global_waypoints'].wpnts
        assert h.last['/car_state/odom_frenet'].header.frame_id == 'frenet/raceline'
        assert h.last['/state_machine'].data == 'GB_TRACK'
        assert not (set(NEEDS_GP) & set(h.first)), \
            'a GP-dependent node published without any /opponent_trajectory'
        assert 'Not falling back to /tracked_pose' not in log   # the pose did arrive

        # The planners' obstacles are the tracker's: one id, moving, at the opponent's speed,
        # nothing coasted. The fake's own vs and is_static never reach them.
        assert {i for _, i, _, _, _ in h.tracked} == {0}
        settled = [x for x in h.tracked if x[0] > h.tracked[0][0] + 1.0]
        assert settled and not any(st for _, _, st, _, _ in settled)
        assert all(vis for *_, vis in h.tracked)
        assert np.median([vs for _, _, _, vs, _ in settled]) == pytest.approx(SPEED, abs=0.2)
        # The fake here is an ideal sensor and reports vs_var at the 1e-6 floor; the filter's
        # is far above it, so these are the tracker's numbers, not the fake's passed through.
        assert min(h.vs_var) > 1e-4
        # opponent_trajectory picked the tracked opponent: its half lap is made of it.
        half_lap = h.last['/proj_opponent_trajectory'].detections
        assert half_lap and not any(p.is_static for p in half_lap)

        # A subscriber that joins now, long after the single publish, still gets them.
        late = []
        node.create_subscription(WpntArray, '/global_waypoints', late.append, WAYPOINTS_QOS)
        end = time.monotonic() + 5
        while not late and time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.05)
        assert len(late) == 1 and len(late[0].wpnts) == len(waypoints.waypoints)

        # ---- phase 2: a synthetic GP output unblocks the last two
        h.start_synthetic_gp()
        h.spin_until(NEEDS_GP, PHASE2_TIMEOUT, proc, log_path)
        # force_trailing is only published once collision_prediction has an opponent, the
        # first dynamic obstacle: the tracked one, since nothing else publishes obstacles.
        assert not missing(h, NEEDS_GP), (f'still silent with a synthetic /opponent_trajectory: '
                                          f'{missing(h, NEEDS_GP)}\n'
                                          f'{log_path.read_text()[-4000:]}')
        print('first message, s after harness start:',
              {t: round(v, 1) for t, v in sorted(h.first.items(), key=lambda kv: kv[1])})
        # The ego drives along the raceline (heading from its psi_rad, i.e. the geometry) at
        # SPEED. Before frenet.py converted TUM psi this read vs 0.00, vd 3.00.
        tw = h.last['/car_state/odom_frenet'].twist.twist.linear
        print(f'ego at {SPEED} m/s along the raceline -> odom_frenet vs {tw.x:.3f}, vd {tw.y:.3f}')
        assert tw.x == pytest.approx(SPEED, abs=0.1)
        assert tw.y == pytest.approx(0.0, abs=0.25)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
