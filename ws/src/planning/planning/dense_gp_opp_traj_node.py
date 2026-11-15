"""Publish Predictive Spliner's dense-GP opponent trajectory on /opponent_trajectory.

Port of the ROS 1 gaussian_process_opp_traj node from ForzaETH's Predictive
Spliner. All the logic lives in dense_gp_opp_traj.py, which imports no ROS and
is unit tested. This file reads topics, hands them to DenseOpponentGP, and
publishes. Topic names are upstream's.

ROS 1 to ROS 2 differences:

  * Upstream's main loop called rospy.wait_for_message on
    /proj_opponent_trajectory once per pass, so a half lap arriving during a
    fit was lost. Here every message is handled in the subscription callback,
    and messages arriving during a fit queue (depth 10). opponent_trajectory
    publishes a half lap every few seconds, and a whole-lap fit takes well
    under that, so the same messages are normally processed.
  * The rospy.wait_for_message calls on /global_waypoints (three) became a
    subscription. Half laps arriving before the first waypoints are ignored;
    upstream was not subscribed yet. /global_waypoints is subscribed
    TRANSIENT_LOCAL, KEEP_LAST(1) (planning.waypoint_qos), matching
    global_waypoints_publisher_node's single latched publish.
    /proj_opponent_trajectory keeps default (VOLATILE) QoS.
  * Upstream's node was gaussian_process_opp_traj (anonymous). Here it is
    dense_gp_opp_traj, to keep it apart from FSDP's GP of the same upstream
    name. Its rospy.Rate(10) was never slept on and is dropped.
  * Upstream set header.seq = lap_count; ROS 2 headers have no seq.
  * Upstream built a FrenetConverter from the /global_waypoints x, y. Here it
    is FrenetConverter.from_track, as in the other ports, so the node needs
    track_name. Unlike there, it sets published values: x_m, y_m everywhere,
    and in whole-lap mode d_m too (CCMA smooths in x, y).
  * Both publishes at lapcount == 1 carry the first one's header stamp, since
    upstream republished the same message object. Markers are stamped at
    publish time; upstream left the stamp at zero.
"""

import os
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray

from f1t_msgs.msg import OppWpnt, OpponentTrajectory, ProjOppTraj, WpntArray
from planning.dense_gp_opp_traj import DenseOpponentGP
from planning.frenet import FrenetConverter
from planning.waypoint_qos import WAYPOINTS_QOS


class DenseGpOppTrajNode(Node):

    def __init__(self):
        super().__init__('dense_gp_opp_traj')
        p = lambda n, v: self.declare_parameter(n, v).value

        tracks_dir = p('tracks_dir', os.environ.get('TRACKS_DIR',
                                                    os.path.expanduser('~/ws/tracks')))
        track_name = p('track_name', '')
        reference = p('reference', 'raceline')
        self.map_frame = p('map_frame', 'map')

        if not track_name:
            raise RuntimeError('dense_gp_opp_traj_node needs the track_name parameter')
        track_dir = os.path.join(tracks_dir, track_name)
        if reference == 'raceline':
            self.conv = FrenetConverter.from_track(track_dir)
        else:
            self.conv = FrenetConverter.from_centerline(
                os.path.join(track_dir, 'centerline.csv'))

        self.gp = DenseOpponentGP(self.conv.to_cartesian, self.conv.to_frenet)
        self.announced = False

        self.pub_traj = self.create_publisher(OpponentTrajectory, '/opponent_trajectory', 10)
        self.pub_markers = self.create_publisher(MarkerArray, '/opponent_traj_markerarray', 10)
        self.create_subscription(ProjOppTraj, '/proj_opponent_trajectory', self.proj_cb, 10)
        self.create_subscription(WpntArray, '/global_waypoints', self.waypoints_cb,
                                 WAYPOINTS_QOS)

    def waypoints_cb(self, msg):
        if self.gp.set_global_waypoints(msg.wpnts) and not self.announced:
            self.announced = True
            self.get_logger().info('[Tracking] initialized FrenetConverter object')

    def proj_cb(self, msg):
        if not self.gp.ready:
            return
        t0 = time.monotonic()
        pubs = self.gp.on_proj_trajectory(msg.lapcount, msg.opp_is_on_trajectory,
                                          msg.detections)
        if not pubs:
            return
        stamps = []
        for pub in pubs:
            stamp = (stamps[pub.stamp_of] if pub.stamp_of is not None
                     else self.get_clock().now().to_msg())
            stamps.append(stamp)
            out = OpponentTrajectory(lap_count=float(pub.lap_count),
                                     oppwpnts=[self._wpnt(w) for w in pub.oppwpnts])
            out.header.stamp, out.header.frame_id = stamp, 'opponent_trajectory'
            self.pub_traj.publish(out)
            now = self.get_clock().now().to_msg()
            self.pub_markers.publish(MarkerArray(markers=[self._marker(m, now)
                                                          for m in pub.markers]))
        self.get_logger().info(
            f'lapcount {msg.lapcount:g}: {len(msg.detections)} detections, '
            f'{len(self.gp.sorted_detections)} accumulated, {len(pubs)} published in '
            f'{time.monotonic() - t0:.2f} s')

    @staticmethod
    def _wpnt(w):
        return OppWpnt(id=int(w.id), s_m=float(w.s_m), d_m=float(w.d_m), x_m=float(w.x_m),
                       y_m=float(w.y_m), proj_vs_mps=float(w.proj_vs_mps),
                       vd_mps=float(w.vd_mps), d_var=float(w.d_var), vs_var=float(w.vs_var))

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
        m.color.a, m.color.r, m.color.g, m.color.b = 1.0, 1.0, 1.0, 0.0
        return m


def main(args=None):
    rclpy.init(args=args)
    node = DenseGpOppTrajNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
