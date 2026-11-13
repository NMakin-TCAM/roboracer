"""Publish /local_path and /local_path/valid from spliner_node's evasion, for pure pursuit.

All the logic is in local_path.py (no ROS, unit tested): the splice, the
stamp pairing of /planner/avoidance/otwpnts with /planner/avoidance/status, and
the validity decision. This node reads those topics and /car_state/odom_frenet,
and publishes on a timer at `rate`:

    /local_path        nav_msgs/Path, map frame, one pose per raceline row: the
                       raceline, with the evasion spliced in while evading
    /local_path/valid  std_msgs/Bool, false whenever local_path.decide says so

When going invalid the Bool is published before the path, and when valid after
it, so pure pursuit never pairs a stale path with a fresh true.

Ages are measured from when a message ARRIVED, on this node's clock, not from
its header stamp: the planner, the localizer and this node need not share a
clock for staleness to be caught.

ONE /local_path PUBLISHER. control's race.launch.py avoid:=true starts
obstacle_spliner_node, which publishes /local_path too. While any other node
publishes it, this one publishes valid=false and logs an ERROR, so two
planners can never take turns steering the car.
"""

import os

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool

from f1t_msgs.msg import OTWpntArray, PlannerStatus
from planning.frenet import FrenetConverter
from planning.local_path import Decision, Pairer, decide, load_raceline_rows, splice


def stamp_key(stamp):
    return (stamp.sec, stamp.nanosec)


class LocalPathNode(Node):

    def __init__(self, **kwargs):
        super().__init__('local_path_node', **kwargs)
        p = lambda n, v: self.declare_parameter(n, v).value

        tracks_dir = p('tracks_dir', os.environ.get('TRACKS_DIR',
                                                    os.path.expanduser('~/ws/tracks')))
        track_name = p('track_name', '')
        otwpnts_topic = p('otwpnts_topic', '/planner/avoidance/otwpnts')
        status_topic = p('status_topic', '/planner/avoidance/status')
        ego_topic = p('ego_topic', '/car_state/odom_frenet')
        self.path_topic = p('path_topic', '/local_path')
        valid_topic = p('valid_topic', '/local_path/valid')
        self.map_frame = p('map_frame', 'map')
        self.max_plan_age = float(p('max_plan_age', 0.25))   # s; spliner_node runs at 20 Hz
        self.max_ego_age = float(p('max_ego_age', 0.25))     # s
        rate = float(p('rate', 20.0))

        if not track_name:
            raise RuntimeError('local_path_node needs the track_name parameter')
        track_dir = os.path.join(tracks_dir, track_name)
        self.s, self.xy = load_raceline_rows(os.path.join(track_dir, 'raceline.csv'))
        self.conv = FrenetConverter.from_track(track_dir)

        self.pairer = Pairer()
        self.ego_received = None
        self.last_reason = None
        self.last = Decision(False, 'not started')

        self.path_pub = self.create_publisher(Path, self.path_topic, 10)
        self.valid_pub = self.create_publisher(Bool, valid_topic, 10)
        self.create_subscription(OTWpntArray, otwpnts_topic, self.otwpnts_cb, 10)
        self.create_subscription(PlannerStatus, status_topic, self.status_cb, 10)
        self.create_subscription(Odometry, ego_topic, self.ego_cb, 10)
        self.create_timer(1.0 / rate, self.tick)
        self.get_logger().info(
            f'{len(self.xy)} raceline rows from {track_dir}/raceline.csv; publishing '
            f'{self.path_topic} and {valid_topic} at {rate:.0f} Hz')

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def otwpnts_cb(self, msg):
        self.pairer.add_trajectory(stamp_key(msg.header.stamp), [w.s_m for w in msg.wpnts],
                                   [w.d_m for w in msg.wpnts], self.now())

    def status_cb(self, msg):
        self.pairer.add_status(stamp_key(msg.header.stamp), msg.status, msg.note, self.now())

    def ego_cb(self, msg):
        self.ego_received = self.now()

    def decision(self):
        """local_path.decide, plus the single-publisher check that needs the graph."""
        others = self.count_publishers(self.path_topic) - 1
        if others > 0:
            return Decision(False, f'{others} other node(s) publish {self.path_topic}; '
                                   f'refusing to compete (obstacle_spliner_node running?)')
        return decide(self.now(), self.pairer.plan, self.ego_received,
                      self.max_plan_age, self.max_ego_age)

    def tick(self):
        dec = self.decision()
        self.last = dec
        if dec.reason != self.last_reason:
            # Two call sites: rclpy refuses to change severity at one ("Logger severity cannot
            # be changed between calls"), which killed this node on its first valid tick.
            if dec.valid:
                self.get_logger().info(f'/local_path valid: {dec.reason}')
            else:
                self.get_logger().error(f'/local_path INVALID: {dec.reason}')
            self.last_reason = dec.reason

        plan = self.pairer.plan
        if dec.spliced:
            xy = splice(self.s, self.xy, self.conv.length, plan.s, plan.d, self.conv.to_cartesian)
        else:
            xy = self.xy
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = self.map_frame
        for x, y in xy:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x, ps.pose.position.y = float(x), float(y)
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)

        if dec.valid:
            self.path_pub.publish(path)
            self.valid_pub.publish(Bool(data=True))
        else:
            self.valid_pub.publish(Bool(data=False))
            self.path_pub.publish(path)


def main(args=None):
    rclpy.init(args=args)
    node = LocalPathNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
