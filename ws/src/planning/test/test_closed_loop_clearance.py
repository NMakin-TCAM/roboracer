"""The planner never plans a path through the opponent: a closed-loop test, from cold.

The full predictive chain, started by planning predictive_race.launch.py, on
porto_odom_trial's geometry (test/fixtures/porto_odom_trial:
a copy of its raceline.csv and centerline.csv, 26.9 m lap, 1.07 to 2.49 m wide), where
the clearance violation was found (a synthetic 1.6 m oval did not reproduce it):
shared_graph.launch.py gp:=dense (Predictive Spliner's GP, collision_prediction, the
tracker, a weaving fake opponent), spliner_node on /collision_prediction/obstacles with
trailing on the tracked obstacles, local_path_node, and control's pure pursuit. The car
is a kinematic bicycle in this test, integrating /drive; the test is its localizer
(localizer:=external). It starts from standing with the opponent ahead, so the cold
start is exercised too.

Every 20 ms, against the opponent's TRUE pose (fake_opponent_node's evaluation-only
truth topic, which nothing that drives reads):

  planned gap   the car's footprint put exactly on the plan in force at the car's own
                s (the evasion's d while EVADING, the raceline while CLEAR), heading
                along the planned line
  achieved gap  the car's footprint where it actually is

Either one negative fails the test: a plan through the opponent is a planner bug
whether or not contact follows. NO_ROOM and TOO_CLOSE mean stop, and are not plans,
so only the achieved gap is checked then. The run must also pass the opponent at
least once, or it says nothing about overtaking.

It needs ros2, the built and sourced planning AND control packages, and Linux
process groups, and skips without them. It publishes /drive on ROS domain 62: keep
it off a live stack's domain. About 2.5 minutes per clearance.

Footprints: the car is spliner's box (car_front 0.45, car_rear 0.10, width 0.30 about
the rear axle), the opponent fake_opponent_node's body (0.50 x 0.30, before its
sensor inflation). Gaps are signed distances between the rectangles (separating axis
test), negative being penetration depth.
"""
import math
import os
import shutil
import signal
import subprocess
import time

import numpy as np
import pytest

rclpy = pytest.importorskip('rclpy')
pytest.importorskip('f1t_msgs.msg')
pytest.importorskip('ackermann_msgs.msg')
from ackermann_msgs.msg import AckermannDriveStamped  # noqa: E402
from ament_index_python.packages import (PackageNotFoundError,  # noqa: E402
                                         get_package_share_directory)
from geometry_msgs.msg import PoseStamped  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402

from f1t_msgs.msg import OTWpntArray, PlannerStatus  # noqa: E402
from planning.frenet import FrenetConverter  # noqa: E402

DOMAIN = 62
TRACK = 'porto_odom_trial'
FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures', TRACK)
OPP_V = 2.0                      # m/s
# 0.25 is the clearance chosen for racing. 0.15 is a stress case: more evasions start and
# more are released mid-pass, which is where the violation lived (2 in 135 s there).
CLEARANCES = (0.25, 0.15)
DURATION = 130.0                 # s of driving
EGO_BOX = (0.45, 0.10, 0.30)     # front of rear axle, behind it, width
OPP_BOX = (0.50, 0.30)           # length, width, centred
WHEELBASE, MAX_STEER, STEER_RATE, ACC, DEC = 0.33, 0.524, 4.0, 4.0, 8.0


# ------------------------------------------------------------------ geometry

def footprint(x, y, yaw, front, rear, width):
    c, s = math.cos(yaw), math.sin(yaw)
    pts = [(front, width / 2), (front, -width / 2), (-rear, -width / 2), (-rear, width / 2)]
    return np.array([(x + c * u - s * v, y + s * u + c * v) for u, v in pts])


def _seg_dist(p, q, r):
    d = r - q
    t = np.clip(np.dot(p - q, d) / np.dot(d, d), 0.0, 1.0)
    return float(np.linalg.norm(p - (q + t * d)))


def signed_gap(a, b):
    """Distance between two convex polygons; negative is the penetration depth."""
    overlap = math.inf
    for poly in (a, b):
        for i in range(len(poly)):
            e = poly[(i + 1) % len(poly)] - poly[i]
            n = np.array([-e[1], e[0]]) / np.linalg.norm(e)
            pa, pb = a @ n, b @ n
            o = min(pa.max(), pb.max()) - max(pa.min(), pb.min())
            if o <= 0:
                return min(_seg_dist(p, q, r) for p1, p2 in ((a, b), (b, a))
                           for p in p1 for q, r in zip(p2, np.roll(p2, -1, 0)))
            overlap = min(overlap, o)
    return -overlap


def test_signed_gap():
    a = footprint(0, 0, 0, 1, 1, 1)
    assert signed_gap(a, footprint(3, 0, 0, 1, 1, 1)) == pytest.approx(1.0)
    assert signed_gap(a, footprint(1.5, 0, 0, 1, 1, 1)) == pytest.approx(-0.5)
    assert signed_gap(a, footprint(3, 3, 0, 1, 1, 1)) == pytest.approx(math.hypot(1, 2))


# ------------------------------------------------------------------ the run

def start(cmd, log_path, env):
    return subprocess.Popen(cmd, env=env, stdout=open(log_path, 'w'), stderr=subprocess.STDOUT,
                            start_new_session=True)


def stop(proc):
    """SIGINT the process group (AGENTS.md); return what survived it, killed."""
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


@pytest.fixture(params=CLEARANCES, ids=lambda c: f'evasion_dist={c}')
def chain(tmp_path, request):
    if shutil.which('ros2') is None:
        pytest.skip('ros2 CLI not on PATH')
    for pkg, f in (('planning', 'predictive_race.launch.py'), ('control', 'pure_pursuit.launch.py')):
        try:
            share = get_package_share_directory(pkg)
        except PackageNotFoundError:
            pytest.skip(f'{pkg} is not built and sourced')
        if not os.path.isfile(os.path.join(share, 'launch', f)):
            pytest.skip(f'installed {pkg} has no {f}; rebuild it')
    tracks = tmp_path / 'tracks'
    shutil.copytree(FIXTURE, tracks / TRACK)
    env = dict(os.environ, ROS_DOMAIN_ID=str(DOMAIN), TRACKS_DIR=str(tracks))
    # The whole chain through its launch (predictive_race.launch.py), so its wiring is
    # tested too; this test is the localizer and the car.
    procs = [
        start(['ros2', 'launch', 'planning', 'predictive_race.launch.py', f'track:={TRACK}',
               'localizer:=external', 'use_sim_time:=false', 'state:=TRAILING',
               f'evasion_dist:={request.param}', 'opponent_mode:=weaving',
               f'opponent_speed:={OPP_V}', 'opponent_s0:=6.0',
               'opponent_truth_topic:=/fake_opponent/truth', 'speed_scale:=1.0',
               'max_speed:=4.0'], tmp_path / 'graph.log', env),
    ]
    try:
        yield procs, FrenetConverter.from_track(str(tracks / TRACK)), tmp_path
    finally:
        leftovers = [stop(p) for p in procs]
        assert not any(leftovers), f'processes left behind:\n{"".join(leftovers)}'


class Car:
    """Kinematic bicycle on /drive; publishes /localized_pose and /odom; records gaps."""

    def __init__(self, node, conv):
        self.node, self.conv = node, conv
        x, y = conv.to_cartesian(0.0, 0.0)
        self.x, self.y, self.yaw = float(x), float(y), float(conv.heading_at(0.0))
        self.v = self.steer = self.cmd_v = self.cmd_steer = 0.0
        self.cmd_t = None
        self.truth = None
        self.statuses, self.plans = {}, {}
        self.plan = None             # (status, note, s, d) in force
        self.samples = []            # (t, planned gap or None, achieved gap, status, rel)
        self.t0 = time.monotonic()
        node.create_subscription(AckermannDriveStamped, '/drive', self.on_drive, 10)
        node.create_subscription(PoseStamped, '/fake_opponent/truth', self.on_truth, 10)
        node.create_subscription(PlannerStatus, '/planner/avoidance/status', self.on_status, 10)
        node.create_subscription(OTWpntArray, '/planner/avoidance/otwpnts', self.on_plan, 10)
        self.pose_pub = node.create_publisher(PoseStamped, '/localized_pose', 10)
        self.odom_pub = node.create_publisher(Odometry, '/odom', 10)
        node.create_timer(0.02, self.tick)

    def now(self):
        return time.monotonic() - self.t0

    def on_drive(self, m):
        self.cmd_v, self.cmd_steer, self.cmd_t = m.drive.speed, m.drive.steering_angle, self.now()

    def on_truth(self, m):
        q = m.pose.orientation
        self.truth = (m.pose.position.x, m.pose.position.y, 2 * math.atan2(q.z, q.w))

    def _pair(self, key):
        if key in self.statuses and key in self.plans:
            st, note = self.statuses.pop(key)
            self.plan = (st, note, *self.plans.pop(key))

    def on_status(self, m):
        key = (m.header.stamp.sec, m.header.stamp.nanosec)
        self.statuses[key] = (m.status, m.note)
        self._pair(key)

    def on_plan(self, m):
        key = (m.header.stamp.sec, m.header.stamp.nanosec)
        self.plans[key] = (np.array([w.s_m for w in m.wpnts]), np.array([w.d_m for w in m.wpnts]))
        self._pair(key)

    def step(self, dt):
        fresh = self.cmd_t is not None and self.now() - self.cmd_t < 0.5   # command timeout
        target = self.cmd_v if fresh else 0.0
        self.v = max(0.0, self.v + max(-DEC * dt, min(ACC * dt, target - self.v)))
        ds = self.cmd_steer - self.steer
        self.steer = max(-MAX_STEER, min(MAX_STEER, self.steer + max(-STEER_RATE * dt,
                                                                    min(STEER_RATE * dt, ds))))
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.yaw += self.v / WHEELBASE * math.tan(self.steer) * dt

    def planned_pose(self, s_car):
        st, _, s, d = self.plan
        if st == 'evading' and len(s) > 1:
            u = np.concatenate(([0.0], np.cumsum(np.diff(s) % self.conv.length)))
            # The published path starts at the first sample at or ahead of the car, so the
            # car can be up to a sample short of it; past its end it has rejoined (d = 0).
            # np.interp holds the end values in both cases.
            uc = float(self.conv.ds(s[0], s_car))
            dp = float(np.interp(uc, u, d))
            slope = float(np.interp(uc, u[1:], np.diff(d) / np.maximum(np.diff(u), 1e-6)))
        else:
            dp, slope = 0.0, 0.0
        x, y = self.conv.to_cartesian(s_car, dp)
        return float(x), float(y), float(self.conv.heading_at(s_car)) + math.atan(slope)

    def tick(self):
        self.step(0.01)
        self.step(0.01)
        stamp = self.node.get_clock().now().to_msg()
        p = PoseStamped()
        p.header.stamp, p.header.frame_id = stamp, 'map'
        p.pose.position.x, p.pose.position.y = self.x, self.y
        p.pose.orientation.z, p.pose.orientation.w = math.sin(self.yaw / 2), math.cos(self.yaw / 2)
        self.pose_pub.publish(p)
        o = Odometry()
        o.header.stamp, o.header.frame_id, o.child_frame_id = stamp, 'odom', 'base_link'
        o.twist.twist.linear.x = self.v
        o.twist.twist.angular.z = self.v / WHEELBASE * math.tan(self.steer)
        self.odom_pub.publish(o)
        if self.truth is None:
            return
        ox, oy, oyaw = self.truth
        opp = footprint(ox, oy, oyaw, OPP_BOX[0] / 2, OPP_BOX[0] / 2, OPP_BOX[1])
        achieved = signed_gap(footprint(self.x, self.y, self.yaw, *EGO_BOX), opp)
        s_car = float(self.conv.to_frenet(self.x, self.y)[0])
        rel = float(self.conv.ds(self.conv.to_frenet(ox, oy)[0], s_car))
        planned, st = None, None
        if self.plan is not None:
            st = self.plan[0]
            if st in ('clear', 'evading'):
                planned = signed_gap(footprint(*self.planned_pose(s_car), *EGO_BOX), opp)
        self.samples.append((self.now(), planned, achieved, st, rel))


def passes(samples):
    """The car's rear axle getting from under 3 m behind the opponent to 1 m ahead."""
    n, behind = 0, False
    for *_, rel in samples:
        if -3 < rel < 0:
            behind = True
        elif rel > 1.0 and behind:
            n, behind = n + 1, False
    return n


def test_the_plan_never_overlaps_the_opponent(chain):
    procs, conv, logs = chain
    rclpy.init(domain_id=DOMAIN)
    node = rclpy.create_node('closed_loop_clearance_car')
    try:
        car = Car(node, conv)
        end = time.monotonic() + DURATION
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.01)
            for p in procs:
                assert p.poll() is None, (f'{p.args[:4]} exited:\n'
                                          + (logs / 'graph.log').read_text()[-3000:])
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

    smp = car.samples
    assert smp, 'no truth received from fake_opponent_node'
    planned = [(t, g, st) for t, g, _, st, _ in smp if g is not None]
    worst_plan = min(planned, key=lambda x: x[1]) if planned else None
    worst = min(smp, key=lambda x: x[2])
    moved = sum(1 for x in smp if x[3] is not None)
    n_pass = passes(smp)
    evading = sum(1 for x in smp if x[3] == 'evading')
    print(f'\n{len(smp)} samples, {evading} evading, {n_pass} passes; closest planned '
          f'{worst_plan[1]:+.3f} m at t {worst_plan[0]:.1f} ({worst_plan[2]}); closest achieved '
          f'{worst[2]:+.3f} m at t {worst[0]:.1f}')
    assert moved, 'the planner never produced a plan: the chain did not come up from cold'
    bad = [x for x in planned if x[1] < 0]
    assert not bad, (f'{len(bad)} cycles planned through the opponent; worst '
                     f'{worst_plan[1]:+.3f} m at t {worst_plan[0]:.1f} while {worst_plan[2]}\n'
                     + (logs / 'graph.log').read_text()[-3000:])
    assert worst[2] > 0, f'contact: {worst[2]:+.3f} m at t {worst[0]:.1f}'
    assert n_pass >= 1, f'never got past the opponent in {DURATION:.0f} s ({evading} evading)'
