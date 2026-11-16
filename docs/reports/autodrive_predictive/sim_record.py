"""Passive recorder for a predictive_race.launch.py run in AutoDRIVE. Publishes nothing.

Usage (sourced container shell, the live stack's domain): sim_record.py DURATION OUT

Records, with arrival time (wall, monotonic from start) and header stamp where there is
one, every message on the topics below; and for evaluation only the simulator's ground
truth (/autodrive/roboracer_1/odom, world frame) and the fake opponent's true pose
(/fake_opponent/truth). Neither reaches anything that drives.
"""
import os
import pickle
import sys
import time

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Bool, Int32, String

from f1t_msgs.msg import (ObstacleArray, OpponentTrajectory, OTWpntArray, PlannerStatus,
                          ProjOppTraj)

DURATION, OUT = float(sys.argv[1]), sys.argv[2]
os.makedirs(OUT, exist_ok=True)

# topic -> (type, qos, what to keep)
stamp = lambda m: m.header.stamp.sec + m.header.stamp.nanosec * 1e-9  # noqa: E731
pose = lambda p: (p.position.x, p.position.y, 2 * np.arctan2(p.orientation.z, p.orientation.w))  # noqa: E731
TOPICS = {
    '/clock': (Clock, qos_profile_sensor_data, lambda m: m.clock.sec + m.clock.nanosec * 1e-9),
    '/scan': (LaserScan, qos_profile_sensor_data, lambda m: stamp(m)),
    '/odom': (Odometry, 10, lambda m: (stamp(m), m.twist.twist.linear.x)),
    '/sensors/imu/raw': (Imu, qos_profile_sensor_data, lambda m: stamp(m)),
    '/tracked_pose': (PoseStamped, 10, lambda m: (stamp(m), *pose(m.pose))),
    '/localized_pose': (PoseStamped, 10, lambda m: (stamp(m), *pose(m.pose))),
    '/car_state/odom_frenet': (Odometry, 10, lambda m: (stamp(m), m.pose.pose.position.x,
                                                        m.pose.pose.position.y)),
    '/perception/detections': (ObstacleArray, 10, lambda m: (stamp(m), len(m.obstacles))),
    '/perception/obstacles': (ObstacleArray, 10, lambda m: (stamp(m), [
        (o.id, o.s_center, o.d_center, o.vs, o.is_static) for o in m.obstacles])),
    '/proj_opponent_trajectory': (ProjOppTraj, 10, lambda m: (m.lapcount, m.opp_is_on_trajectory,
                                                              len(m.detections))),
    '/opponent_trajectory': (OpponentTrajectory, 10, lambda m: (stamp(m), m.lap_count)),
    '/collision_prediction/obstacles': (ObstacleArray, 10, lambda m: (stamp(m), [
        (o.s_start, o.s_end, o.s_center, o.d_right, o.d_left, o.d_center) for o in m.obstacles])),
    '/collision_prediction/force_trailing': (Bool, 10, lambda m: m.data),
    '/planner/avoidance/status': (PlannerStatus, 10, lambda m: (stamp(m), m.status, m.note)),
    '/planner/avoidance/otwpnts': (OTWpntArray, 10, lambda m: (stamp(m), np.array(
        [[w.s_m, w.d_m] for w in m.wpnts]).reshape(-1, 2))),
    '/local_path/valid': (Bool, 10, lambda m: m.data),
    '/local_path': (Path, 10, lambda m: stamp(m)),
    '/drive': (AckermannDriveStamped, 10, lambda m: (stamp(m), m.drive.speed,
                                                     m.drive.steering_angle)),
    '/autodrive/roboracer_1/sim_reset': (String, 10, lambda m: m.data),
    '/autodrive/roboracer_1/collision_count': (Int32, 10, lambda m: m.data),
    # Evaluation only.
    '/autodrive/roboracer_1/odom': (Odometry, qos_profile_sensor_data,
                                    lambda m: (stamp(m), *pose(m.pose.pose),
                                               m.twist.twist.linear.x)),
    '/fake_opponent/truth': (PoseStamped, 10, lambda m: (stamp(m), *pose(m.pose))),
}

rclpy.init()
node = rclpy.create_node('predictive_sim_recorder')
t0 = time.monotonic()
rec = {t: [] for t in TOPICS}
rec['meta'] = dict(start_wall=time.time(), duration=DURATION)


def cb(topic, keep):
    def f(m):
        try:
            rec[topic].append((time.monotonic() - t0, keep(m)))
        except Exception as e:      # a recorder must not die on one odd message
            rec[topic].append((time.monotonic() - t0, ('error', repr(e))))
    return f


for topic, (typ, qos, keep) in TOPICS.items():
    node.create_subscription(typ, topic, cb(topic, keep), qos)
try:
    while time.monotonic() - t0 < DURATION:
        rclpy.spin_once(node, timeout_sec=0.01)
except KeyboardInterrupt:
    pass
finally:
    rec['meta']['end'] = time.monotonic() - t0
    with open(os.path.join(OUT, 'record.pkl'), 'wb') as f:
        pickle.dump(rec, f)
    print({t: len(v) for t, v in rec.items() if isinstance(v, list)})
    node.destroy_node()
    rclpy.try_shutdown()
