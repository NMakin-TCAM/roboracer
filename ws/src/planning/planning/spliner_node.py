"""Reactive overtaking planner. Publishes an evasion trajectory on /planner/avoidance/otwpnts.

Topic names and message shapes follow ForzaETH so their planners, and ours, are
interchangeable behind the same interface. Swapping this node for Predictive
Spliner or FSDP should require changing nothing downstream, which is the whole
point of fixing the interface before writing any planner.

All the planning lives in spliner.py, which imports no ROS and is unit tested.
This file reads topics, calls plan() on a timer, and publishes.

STATUS. Each cycle also publishes f1t_msgs/PlannerStatus on status_topic
(/planner/avoidance/status) with the same header stamp as that cycle's
OTWpntArray. An empty OTWpntArray means "clear" or "cannot get round" (no room,
too close); the status says which, and consumers must pair the two by stamp
(local_path_node does). OTWpntArray itself stays identical to ForzaETH's.

STALE INPUTS. Nothing is published while the ego state or the obstacle array
is older than max_input_age (time since arrival, node clock), so a dead
perception or localizer stack makes the planner output go stale downstream
rather than keep reporting the last obstacles, or "clear", forever.

TRAILING (trail_obs_topic set; off by default). For obs_topic =
/collision_prediction/obstacles. Until collision_prediction first publishes, and
while its force_trailing (force_trailing_topic) is fresh and True, the node
plans on the tracked obstacles from trail_obs_topic instead, and never starts an
evasion around a moving one (spliner.trail). The car can then start from
standing: it trails the opponent, which lets the opponent GP learn, which lets
collision_prediction publish. local_path_node is unchanged, and so is what
stops the car: NO_ROOM and TOO_CLOSE still mean stop, and every mode needs
fresh inputs (spliner.plan_inputs). An evasion committed on the regions is
kept to its end if force_trailing flickers on mid-way; it is still released by
the usual conditions against the regions.
"""

import os

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from visualization_msgs.msg import Marker

from std_msgs.msg import Bool

from f1t_msgs.msg import ObstacleArray, OTWpntArray, PlannerStatus, Wpnt
from planning.frenet import FrenetConverter
from planning.spliner import REGIONS, Spliner, SplinerParams, plan_inputs, trail


class SplinerNode(Node):

    def __init__(self):
        super().__init__('spliner_node')
        p = lambda n, v: self.declare_parameter(n, v).value

        tracks_dir = p('tracks_dir', os.environ.get('TRACKS_DIR',
                                                    os.path.expanduser('~/ws/tracks')))
        track_name = p('track_name', '')
        self.reference = p('reference', 'raceline')
        ego_topic = p('ego_topic', '/car_state/odom_frenet')
        obs_topic = p('obs_topic', '/perception/obstacles')
        out_topic = p('out_topic', '/planner/avoidance/otwpnts')
        status_topic = p('status_topic', '/planner/avoidance/status')
        self.max_input_age = float(p('max_input_age', 0.5))   # s since arrival
        marker_topic = p('marker_topic', '/planner/avoidance/markers')
        self.map_frame = p('map_frame', 'map')
        rate = float(p('rate', 20.0))
        trail_topic = p('trail_obs_topic', '')         # '' = no trailing mode (old behaviour)
        force_topic = p('force_trailing_topic', '/collision_prediction/force_trailing')
        self.trail_gap = float(p('trail_gap', 2.0))    # m, nose to a moving obstacle's s_start
        self.trailing = bool(trail_topic)

        if not track_name:
            raise RuntimeError('spliner_node needs the track_name parameter')
        track_dir = os.path.join(tracks_dir, track_name)
        if self.reference == 'raceline':
            self.conv = FrenetConverter.from_track(track_dir)
        else:
            self.conv = FrenetConverter.from_centerline(
                os.path.join(track_dir, 'centerline.csv'))

        self.params = SplinerParams(
            lookahead=float(p('lookahead', 10.0)),
            on_line_threshold=float(p('on_line_threshold', 1.0)),
            car_width=float(p('car_width', 0.30)),
            evasion_dist=float(p('evasion_dist', 0.35)),
            bound_margin=float(p('bound_margin', 0.15)),
            back_to_raceline_after=float(p('back_to_raceline_after', 2.0)),
            car_front=float(p('car_front', 0.45)),
            car_rear=float(p('car_rear', 0.10)),
            apex_margin=float(p('apex_margin', 0.15)),
            max_lateral_slope=float(p('max_lateral_slope', 0.5)),
            max_commit_offset=float(p('max_commit_offset', 0.30)),
            match_tolerance=float(p('match_tolerance', 0.5)),
            speed_scale=float(p('speed_scale', 0.8)),
            resolution=float(p('resolution', 0.10)),
            switch_hysteresis=float(p('switch_hysteresis', 0.25)),
        )

        self.ego_s = None
        self.ego_d = 0.0
        self.ego_received = None
        self.obstacles = []
        self.obstacles_received = None
        self.perceived = []
        self.perceived_received = None
        self.force = None
        self.force_received = None
        self.last_mode = None
        self.committed_on_regions = False
        self.last_stale = None
        self.planner = Spliner(self.conv, self.params)   # commits to an evasion once begun
        self.side = None
        self.last_switch = self.get_clock().now()
        self.last_note = None
        self.warned_frame = False

        self.pub = self.create_publisher(OTWpntArray, out_topic, 10)
        self.status_pub = self.create_publisher(PlannerStatus, status_topic, 10)
        self.marker_pub = self.create_publisher(Marker, marker_topic, 10)
        self.create_subscription(Odometry, ego_topic, self.ego_cb, 10)
        self.create_subscription(ObstacleArray, obs_topic, self.obs_cb, 10)
        if self.trailing:
            self.create_subscription(ObstacleArray, trail_topic, self.perceived_cb, 10)
            self.create_subscription(Bool, force_topic, self.force_cb, 10)
        self.create_timer(1.0 / rate, self.tick)

        self.get_logger().info(
            f'spliner on {track_name} against the {self.reference}, lookahead '
            f'{self.params.lookahead:.1f} m, publishing {out_topic} at {rate:.0f} Hz')

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def ego_cb(self, msg):
        self.ego_received = self.now()
        self.ego_s = float(msg.pose.pose.position.x)
        self.ego_d = float(msg.pose.pose.position.y)
        expect = f'frenet/{self.reference}'
        if msg.header.frame_id and msg.header.frame_id != expect and not self.warned_frame:
            self.warned_frame = True
            self.get_logger().error(
                f'ego state is measured against {msg.header.frame_id}, this node uses '
                f'{expect}. Every overtake will pick the wrong side. Fix the reference '
                f'parameter before trusting anything.')

    def obs_cb(self, msg):
        self.obstacles = list(msg.obstacles)
        self.obstacles_received = self.now()

    def perceived_cb(self, msg):
        self.perceived = list(msg.obstacles)
        self.perceived_received = self.now()

    def force_cb(self, msg):
        self.force = bool(msg.data)
        self.force_received = self.now()

    def tick(self):
        mode, stale = plan_inputs(self.now(), self.max_input_age, self.ego_received,
                                  self.obstacles_received, self.trailing,
                                  self.perceived_received, self.force, self.force_received)
        if stale != self.last_stale:
            if stale:
                self.get_logger().error(f'not planning: {stale}')
            elif self.last_stale is not None:
                self.get_logger().info('inputs fresh again, planning')
            self.last_stale = stale
        if stale:
            return

        if self.trailing and mode != self.last_mode:
            what = ('the collision regions' if mode == REGIONS
                    else f'the tracked obstacles, trailing ({mode})')
            self.get_logger().warn(f'planning on {what}')
            self.last_mode = mode
        if mode == REGIONS or (self.planner.committed is not None and self.committed_on_regions):
            result = self.planner.step(self.ego_s, self.ego_d, self.obstacles)
            self.committed_on_regions = self.planner.committed is not None
        else:
            statics = [o for o in self.perceived if o.is_static]
            result = trail(self.planner.step(self.ego_s, self.ego_d, statics), self.ego_s,
                           self.perceived, self.conv, self.params, self.trail_gap, mode)
            self.committed_on_regions = False
        s, d, v, side, note = result.s, result.d, result.v, result.side, result.note
        if result.released:
            self.get_logger().warn(f'evasion released: {result.released}; now {result.status}')

        switched = side is not None and self.side is not None and side != self.side
        if switched:
            self.last_switch = self.get_clock().now()
            self.get_logger().warn(f'switched overtaking side to {side}')
        self.side = side

        if note != self.last_note:
            self.get_logger().info(note)
            self.last_note = note

        msg = OTWpntArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        status = PlannerStatus(status=result.status, note=note)
        status.header.stamp = msg.header.stamp
        status.header.frame_id = self.map_frame
        msg.header.frame_id = self.map_frame
        msg.last_switch_time = self.last_switch.to_msg()
        msg.side_switch = switched
        msg.ot_side = side or ''
        msg.ot_line = 'spliner'
        msg.wpnts = self._waypoints(s, d, v)
        # An empty array published every cycle is the planner saying "no overtake",
        # which a consumer must be able to tell from the planner having died.
        self.pub.publish(msg)
        self.status_pub.publish(status)
        self._publish_marker(msg.wpnts)

    def _waypoints(self, s, d, v):
        if len(s) == 0:
            return []
        x, y = self.conv.to_cartesian(s, d)
        psi = self.conv.heading_at(s)
        kappa = self.conv.curvature_at(s)
        hi, lo = self.conv.width_at(s)
        out = []
        for i in range(len(s)):
            w = Wpnt()
            w.id = i
            w.s_m, w.d_m = float(s[i]), float(d[i])
            w.x_m, w.y_m = float(x[i]), float(y[i])
            w.d_left, w.d_right = float(np.asarray(hi)[i]), float(np.asarray(lo)[i])
            w.psi_rad = float(np.asarray(psi)[i])
            w.kappa_radpm = float(np.asarray(kappa)[i])
            w.vx_mps = float(v[i])
            w.ax_mps2 = 0.0
            out.append(w)
        return out

    def _publish_marker(self, wpnts):
        m = Marker()
        m.header.frame_id = self.map_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'evasion'
        m.id = 0
        m.type = Marker.LINE_STRIP
        if not wpnts:
            m.action = Marker.DELETE
            self.marker_pub.publish(m)
            return
        m.action = Marker.ADD
        m.scale.x = 0.05
        m.color.a = 1.0
        m.color.r, m.color.g, m.color.b = (0.2, 0.8, 1.0)
        m.pose.orientation.w = 1.0
        for w in wpnts:
            pt = Point()
            pt.x, pt.y, pt.z = w.x_m, w.y_m, 0.05
            m.points.append(pt)
        self.marker_pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = SplinerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
