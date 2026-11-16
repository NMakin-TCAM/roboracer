"""Closed-loop clearance run: the full graph driving a kinematic car past a weaving opponent.

Usage (in the container, overlay sourced, TRACKS_DIR set):
    closed_loop.py TRACK EVASION_DIST DURATION OUTDIR

Starts shared_graph (gp:=dense, tracker, weaving fake opponent, localizer:=external),
spliner_node on /collision_prediction/obstacles with evasion_dist, local_path_node, and
control's pure_pursuit (avoid:=true). This script is the car: a kinematic bicycle that
integrates /drive and publishes /localized_pose and /odom. Footprints: the ego is the
spliner's box (car_front 0.45, car_rear 0.10, width 0.30 about the rear axle); the
opponent is fake_opponent_node's (0.50 x 0.30), at its marker's pose, which is its true
pose (noise-free, no latency). Everything goes to OUTDIR/record.pkl.

WARM-UP. Nothing in the graph lets the car move before collision_prediction publishes,
and that needs the dense GP, which needs the car following the opponent for half a lap.
So until the GP has published lapcount >= 1 and an obstacle array has arrived, this
script places the car WARMUP_GAP behind the tracked opponent on the raceline (ignoring
/drive), then hands over at the opponent's speed to the bicycle and pure pursuit.
"""
import math
import os
import pickle
import signal
import subprocess
import sys
import time

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker

from f1t_msgs.msg import ObstacleArray, OpponentTrajectory, OTWpntArray, PlannerStatus
from planning.global_waypoints_publisher import load_global_waypoints

TRACK, EV, DURATION, OUT = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
# Overridable, to reproduce a simulator run's settings on the stand-in.
SPEED_SCALE = os.environ.get('CL_SPEED_SCALE', '1.0')
MAX_SPEED = os.environ.get('CL_MAX_SPEED', '4.0')
OPP_V = float(os.environ.get('CL_OPP_V', '2.0'))
BOUND = os.environ.get('CL_BOUND_MARGIN', '0.15')   # spliner_node's default
OPP_MODE = os.environ.get('CL_OPP_MODE', 'weaving')   # or constant_speed (d = 0)
WHEELBASE, MAX_STEER, STEER_RATE, ACC, DEC = 0.33, 0.524, 4.0, 4.0, 8.0
EGO_BOX = (0.45, 0.10, 0.30)          # front of rear axle, behind it, width
OPP_BOX = (0.50, 0.30)                # length, width, centred
WARMUP_GAP = 3.0
# TRAILING: update_waypoints writes no measured speeds, so collision_prediction rolls the
# ego at raceline speed, what pure pursuit drives (GB_TRACK teaches it the ego's own past
# speed, 2 m/s after the warm-up, and it then predicts no catch at all).
STATE = os.environ.get('CL_STATE', 'TRAILING')
# COLD: no warm-up. The car stands at the raceline's start, the opponent 6 m ahead, and
# spliner_node trails on the tracked obstacles until collision_prediction can be planned on.
COLD = os.environ.get('CL_COLD', '') == '1'
TRAIL = ['-p', 'trail_obs_topic:=/perception/obstacles'] if COLD else []
os.makedirs(OUT, exist_ok=True)
tracks = os.environ['TRACKS_DIR']
tp = ['-p', f'tracks_dir:={tracks}', '-p', f'track_name:={TRACK}']


def launch(cmd, log):
    return subprocess.Popen(cmd, stdout=open(os.path.join(OUT, log), 'w'),
                            stderr=subprocess.STDOUT, start_new_session=True)


def stop(proc):
    try:
        os.killpg(proc.pid, signal.SIGINT)
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def box(cx, cy, yaw, front, rear, width):
    c, s = math.cos(yaw), math.sin(yaw)
    pts = [(front, width / 2), (front, -width / 2), (-rear, -width / 2), (-rear, width / 2)]
    return np.array([(cx + c * x - s * y, cy + s * x + c * y) for x, y in pts])


def gap(a, b):
    """Signed distance between convex polygons: > 0 apart, < 0 penetration depth (SAT)."""
    overlap = math.inf
    for poly in (a, b):
        for i in range(len(poly)):
            e = poly[(i + 1) % len(poly)] - poly[i]
            n = np.array([-e[1], e[0]]) / np.linalg.norm(e)
            pa, pb = a @ n, b @ n
            o = min(pa.max(), pb.max()) - max(pa.min(), pb.min())
            if o <= 0:
                return min(seg_dist(p, q, r) for poly1, poly2 in ((a, b), (b, a))
                           for p in poly1 for q, r in zip(poly2, np.roll(poly2, -1, 0)))
            overlap = min(overlap, o)
    return -overlap


def seg_dist(p, q, r):
    d = r - q
    t = np.clip(np.dot(p - q, d) / np.dot(d, d), 0, 1)
    return float(np.linalg.norm(p - (q + t * d)))


wp = load_global_waypoints(os.path.join(tracks, TRACK))
WS = np.array([p.s_m for p in wp.waypoints])
WX = np.array([p.x_m for p in wp.waypoints])
WY = np.array([p.y_m for p in wp.waypoints])
WYAW = np.unwrap([p.psi_rad for p in wp.waypoints])
LAP = wp.lap_length
x0, y0, yaw0 = WX[0], WY[0], WYAW[0]

procs = [
    launch(['ros2', 'launch', 'planning', 'shared_graph.launch.py', f'track:={TRACK}',
            'localizer:=external', 'use_sim_time:=false', f'opponent_mode:={OPP_MODE}',
            f'opponent_speed:={OPP_V}', 'opponent_s0:=6.0', 'opponent_max_range:=10.0',
            'gp:=dense', f'state:={STATE}'], 'graph.log'),
    launch(['ros2', 'run', 'planning', 'spliner_node', '--ros-args', *tp,
            '-p', 'obs_topic:=/collision_prediction/obstacles', '-p', f'evasion_dist:={EV}',
            '-p', f'bound_margin:={BOUND}', *TRAIL],
           'spliner.log'),
    launch(['ros2', 'run', 'planning', 'local_path_node', '--ros-args', *tp], 'local_path.log'),
    launch(['ros2', 'launch', 'control', 'pure_pursuit.launch.py', f'track:={TRACK}',
            'avoid:=true', 'pose_topic:=/localized_pose', f'speed_scale:={SPEED_SCALE}',
            f'max_speed:={MAX_SPEED}', 'use_sim_time:=false'], 'pure_pursuit.log'),
]

rec = {k: [] for k in ('ego', 'opp', 'gap', 'regions', 'status', 'gp', 'plan', 'opp_true')}
rclpy.init()
node = rclpy.create_node('closed_loop_car')
t0 = time.monotonic()
now = lambda: time.monotonic() - t0
# Node clock (wall time, use_sim_time false) minus harness time, to place message stamps.
CLOCK = node.get_clock().now().nanoseconds * 1e-9 - now()
rec['clock_offset'] = CLOCK
stamp_t = lambda st: st.sec + st.nanosec * 1e-9 - CLOCK
car = dict(x=x0, y=y0, yaw=yaw0, v=0.0, steer=0.0, cmd_v=0.0, cmd_steer=0.0, cmd_t=None)
opp = {'pose': None, 't': -1.0}
warm = {'on': not COLD, 'opp_s': None, 'opp_seen': -1.0, 'lap1': False, 'arrays': -1.0,
        'handover': None}
rec['handover'] = 0.0 if COLD else None


def on_tracked(m):
    if m.obstacles:
        warm['opp_s'], warm['opp_seen'] = float(m.obstacles[0].s_center), now()


def on_regions(m):
    warm['arrays'] = now()
    rec['regions'].append((now(), [(o.s_start, o.s_end, o.s_center, o.d_right, o.d_left,
                                    o.d_center) for o in m.obstacles]))


def on_gp(m):
    rec['gp'].append((now(), m.lap_count))
    warm['lap1'] = warm['lap1'] or m.lap_count >= 1


def on_drive(m):
    car['cmd_v'], car['cmd_steer'], car['cmd_t'] = (float(m.drive.speed),
                                                    float(m.drive.steering_angle), now())


def on_marker(m):
    if m.action == Marker.ADD:
        yaw = 2 * math.atan2(m.pose.orientation.z, m.pose.orientation.w)
        opp['pose'], opp['t'] = (m.pose.position.x, m.pose.position.y, yaw), now()
        # The fake is noise-free with no latency: this is its true pose at the stamp.
        rec['opp_true'].append((stamp_t(m.header.stamp), m.pose.position.x, m.pose.position.y,
                                yaw))
    else:
        opp['pose'] = None


node.create_subscription(AckermannDriveStamped, '/drive', on_drive, 10)
node.create_subscription(Marker, '/fake_opponent/marker', on_marker, 10)
node.create_subscription(ObstacleArray, '/collision_prediction/obstacles', on_regions, 10)
node.create_subscription(ObstacleArray, '/perception/obstacles', on_tracked, 10)
node.create_subscription(
    PlannerStatus, '/planner/avoidance/status',
    lambda m: rec['status'].append((now(), m.status, m.note, stamp_t(m.header.stamp))), 10)
node.create_subscription(
    OTWpntArray, '/planner/avoidance/otwpnts',
    lambda m: rec['plan'].append((now(), stamp_t(m.header.stamp), m.ot_side,
                                  np.array([[w.s_m, w.d_m] for w in m.wpnts]).reshape(-1, 2))),
    10)
node.create_subscription(OpponentTrajectory, '/opponent_trajectory', on_gp, 10)
pose_pub = node.create_publisher(PoseStamped, '/localized_pose', 10)
odom_pub = node.create_publisher(Odometry, '/odom', 10)
DT = 0.01


def warmup():
    """Scripted placement behind the opponent; True while still warming up."""
    t = now()
    if warm['lap1'] and t - warm['arrays'] < 0.3:
        warm['on'] = False
        car['v'], car['steer'] = OPP_V, 0.0
        rec['handover'] = t
        return False
    if warm['opp_s'] is not None and t - warm['opp_seen'] < 0.5:
        s = (warm['opp_s'] - WARMUP_GAP) % LAP
    else:
        s = (warm.get('s', 0.0) + 0.5) % LAP       # sweep until the opponent is seen
    warm['s'] = s
    car['x'], car['y'] = float(np.interp(s, WS, WX)), float(np.interp(s, WS, WY))
    car['yaw'] = float(np.interp(s, WS, WYAW))
    car['v'] = OPP_V
    return True


def physics():
    # A stale command (watchdog) means stop, as the bridge's command timeout does.
    cmd_v = car['cmd_v'] if car['cmd_t'] is not None and now() - car['cmd_t'] < 0.5 else 0.0
    dv = cmd_v - car['v']
    car['v'] += max(-DEC * DT, min(ACC * DT, dv))
    car['v'] = max(car['v'], 0.0)
    ds = car['cmd_steer'] - car['steer']
    car['steer'] += max(-STEER_RATE * DT, min(STEER_RATE * DT, ds))
    car['steer'] = max(-MAX_STEER, min(MAX_STEER, car['steer']))
    car['x'] += car['v'] * math.cos(car['yaw']) * DT
    car['y'] += car['v'] * math.sin(car['yaw']) * DT
    car['yaw'] += car['v'] / WHEELBASE * math.tan(car['steer']) * DT


def publish():
    t = now()
    stamp = node.get_clock().now().to_msg()
    p = PoseStamped()
    p.header.stamp, p.header.frame_id = stamp, 'map'
    p.pose.position.x, p.pose.position.y = car['x'], car['y']
    p.pose.orientation.z, p.pose.orientation.w = math.sin(car['yaw'] / 2), math.cos(car['yaw'] / 2)
    pose_pub.publish(p)
    o = Odometry()
    o.header.stamp, o.header.frame_id, o.child_frame_id = stamp, 'odom', 'base_link'
    o.twist.twist.linear.x = car['v']
    o.twist.twist.angular.z = car['v'] / WHEELBASE * math.tan(car['steer'])
    odom_pub.publish(o)
    rec['ego'].append((t, car['x'], car['y'], car['yaw'], car['v']))
    if opp['pose'] is not None and t - opp['t'] < 0.2:
        ox, oy, oyaw = opp['pose']
        g = gap(box(car['x'], car['y'], car['yaw'], *EGO_BOX),
                box(ox, oy, oyaw, OPP_BOX[0] / 2, OPP_BOX[0] / 2, OPP_BOX[1]))
        rec['gap'].append((t, g))
        rec['opp'].append((t, ox, oy, oyaw))


def tick():
    if not (warm['on'] and warmup()):
        physics()
        physics()
    publish()


node.create_timer(0.02, tick)
try:
    while now() < DURATION:
        rclpy.spin_once(node, timeout_sec=0.01)
        for p in procs:
            if p.poll() is not None:
                raise RuntimeError(f'{p.args[:4]} exited early')
finally:
    for p in procs:
        stop(p)
    rec['meta'] = dict(handover=rec['handover'], state=STATE, cold=COLD, bound_margin=BOUND, opp_mode=OPP_MODE, track=TRACK, evasion_dist=EV, speed_scale=SPEED_SCALE,
                       max_speed=MAX_SPEED, opp_v=OPP_V, duration=DURATION)
    with open(os.path.join(OUT, 'record.pkl'), 'wb') as f:
        pickle.dump(rec, f)
    node.destroy_node()
    rclpy.try_shutdown()
    left = subprocess.run(['ros2', 'node', 'list'], capture_output=True, text=True).stdout
    print('leftover nodes:', left.strip() or 'none')
    print({k: len(v) for k, v in rec.items() if isinstance(v, list)})
