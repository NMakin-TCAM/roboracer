"""Publish the opponent's predicted return to its line on /opponent_trajectory.

Port of the ROS 1 predictor_opponent_trajectory node shared by Predictive
Spliner and FSDP. All the logic lives in predictor_opponent_trajectory.py,
which imports no ROS and is unit tested. This file reads topics, hands them to
ReturnPredictor, and publishes. Topic names are upstream's.

ROS 1 to ROS 2 differences:

  * Upstream's main loop called rospy.wait_for_message on
    /proj_opponent_trajectory once per pass, so a message arriving while it was
    fitting, or before it subscribed again, was lost. Here every message is
    handled in the subscription callback, and messages arriving during a fit
    queue (depth 10) and are handled in order. Off-trajectory messages are
    rare (opponent_trajectory waits up to 3 s after each), so in practice the
    same messages are processed.
  * The rospy.wait_for_message calls on /global_waypoints (three) and then
    /opponent_trajectory became subscriptions. Nothing is processed until both
    have delivered a message, in either order.
    /global_waypoints is subscribed TRANSIENT_LOCAL, KEEP_LAST(1)
    (planning.waypoint_qos), matching global_waypoints_publisher_node, which
    publishes them once, latched. A VOLATILE subscription would never see a
    message published before it started, and the node would wait forever.
    /proj_opponent_trajectory and /opponent_trajectory keep default
    (VOLATILE) QoS.
  * Upstream's node was named gaussian_process_opp_traj, the same name as the
    main GP node, which only worked because both were anonymous. Here it is
    predictor_opponent_trajectory. Upstream's rospy.Rate(10) was never slept on
    and is dropped.
  * Upstream set header.seq = lap_count. ROS 2 headers have no seq, and rospy
    overwrote seq with its own counter on publish anyway, so it is dropped.
  * Upstream built a FrenetConverter from the /global_waypoints x, y. Here it
    is FrenetConverter.from_track, as in the other ports, so the node needs
    track_name. Unlike there, this is not only for markers: the x_m, y_m of
    predicted waypoints are published.
  * Markers are stamped at publish time; upstream left the stamp at zero.
"""

import os

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray

from f1t_msgs.msg import OpponentTrajectory, ProjOppTraj, WpntArray
from planning.frenet import FrenetConverter
from planning.waypoint_qos import WAYPOINTS_QOS
from planning.predictor_opponent_trajectory import ReturnPredictor


class PredictorOpponentTrajectoryNode(Node):

    def __init__(self):
        super().__init__('predictor_opponent_trajectory')
        p = lambda n, v: self.declare_parameter(n, v).value

        tracks_dir = p('tracks_dir', os.environ.get('TRACKS_DIR',
                                                    os.path.expanduser('~/ws/tracks')))
        track_name = p('track_name', '')
        reference = p('reference', 'raceline')
        self.map_frame = p('map_frame', 'map')

        if not track_name:
            raise RuntimeError('predictor_opponent_trajectory_node needs the track_name parameter')
        track_dir = os.path.join(tracks_dir, track_name)
        if reference == 'raceline':
            self.conv = FrenetConverter.from_track(track_dir)
        else:
            self.conv = FrenetConverter.from_centerline(
                os.path.join(track_dir, 'centerline.csv'))

        self.predictor = ReturnPredictor()
        self.announced = False

        self.pub_traj = self.create_publisher(OpponentTrajectory, '/opponent_trajectory', 10)
        self.pub_markers = self.create_publisher(MarkerArray, '/opponent_traj_markerarray', 10)

        self.create_subscription(ProjOppTraj, '/proj_opponent_trajectory', self.proj_cb, 10)
        self.create_subscription(WpntArray, '/global_waypoints', self.waypoints_cb,
                                 WAYPOINTS_QOS)
        # Includes this node's own output, as upstream's did.
        self.create_subscription(OpponentTrajectory, '/opponent_trajectory',
                                 self.opponent_trajectory_cb, 10)

    def waypoints_cb(self, msg):
        if self.predictor.set_global_waypoints(msg.wpnts) and not self.announced:
            self.announced = True
            self.get_logger().info('[Tracking] initialized FrenetConverter object')

    def opponent_trajectory_cb(self, msg):
        self.predictor.set_opponent_trajectory(msg.oppwpnts)

    def _to_xy(self, s, d):
        return self.conv.to_cartesian(s, d)

    def proj_cb(self, msg):
        pred = self.predictor.on_proj_trajectory(msg.lapcount, msg.opp_is_on_trajectory,
                                                 msg.detections, self._to_xy)
        if pred is None:
            return
        now = self.get_clock().now().to_msg()
        out = OpponentTrajectory(lap_count=float(pred.lap_count), oppwpnts=pred.oppwpnts)
        out.header.stamp, out.header.frame_id = now, 'opponent_trajectory'
        self.pub_traj.publish(out)
        self.pub_markers.publish(MarkerArray(markers=[self._marker(m, now)
                                                      for m in pred.markers]))

    def _marker(self, rec, now):
        m = Marker()
        m.header.stamp, m.header.frame_id = now, self.map_frame
        m.id = rec.id
        m.type = Marker.CYLINDER
        m.pose.position.x, m.pose.position.y = float(rec.x), float(rec.y)
        m.pose.position.z = float(rec.height) / 2
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = float(rec.width)
        m.scale.z = float(rec.height)
        if rec.predicted:
            m.color.a, m.color.r, m.color.g, m.color.b = 0.8, 1.0, 0.0, 0.0
        else:
            m.color.a, m.color.r, m.color.g, m.color.b = 1.0, 1.0, 1.0, 0.0
        return m


def main(args=None):
    rclpy.init(args=args)
    node = PredictorOpponentTrajectoryNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
