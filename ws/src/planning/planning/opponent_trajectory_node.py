"""Publish the opponent's accumulated detections on /proj_opponent_trajectory.

Port of the ROS 1 opponent_trajectory node shared by Predictive Spliner and
FSDP, the input stage of both GPs. All the logic lives in
opponent_trajectory.py, which imports no ROS and is unit tested. This file
reads topics, hands them to OpponentTrajectoryTracker, and publishes on a
timer. Topic names are upstream's.

ROS 1 to ROS 2 differences:

  * LOOP. Upstream's rate.sleep() is inside `if opponent is not None`, so with
    no opponent in range the loop spun without sleeping. Its off-trajectory
    wait was a blocking loop of time.sleep(0.02) for up to 3 s. Neither works
    under rclpy, where a callback that never returns starves every
    subscription on the executor (rospy ran subscriptions on threads of their
    own), and the wait in particular would never see the /opponent_trajectory
    it waits for. Both are one timer at loop_rate (upstream's ~loop_rate, 25
    Hz). The wait is a state the tracker is in: each cycle runs one pass of
    upstream's wait-loop body, then its exit test, so the wait polls at 25 Hz
    instead of 50 Hz. With no opponent, the stale check now runs every 40 ms
    instead of continuously, so a stale flush can come up to 40 ms later.
  * The rospy.wait_for_message calls became subscriptions. Nothing is
    computed until /global_waypoints and /perception/obstacles have both
    delivered a message, in either order (upstream waited for the waypoints
    three times, then the obstacles). Upstream did not wait for odometry and
    still does not: no opponent is found until it arrives.
    /global_waypoints is subscribed TRANSIENT_LOCAL, KEEP_LAST(1)
    (planning.waypoint_qos), matching global_waypoints_publisher_node, which
    publishes them once, latched. A VOLATILE subscription would never see a
    message published before it started, and the node would wait forever.
    The other inputs keep default (VOLATILE) QoS.
  * The waypoints, bounds and track length are fixed from the first
    /global_waypoints message, as upstream's loop fixed them from the one it
    started with. Later messages only update the second track length (see
    opponent_trajectory.py).
  * Upstream built a FrenetConverter from the /global_waypoints x, y to place
    its markers. Here it is FrenetConverter.from_track, as in
    collision_prediction_node.py, so the node needs track_name. Markers only;
    nothing published on /proj_opponent_trajectory goes through it.
  * Time: rospy.Time.now() became the node clock, compared with the
    /perception/obstacles header stamps for staleness and velocities. Both
    must be on the same clock: with use_sim_time on one side only, every
    detection after lapcount 1 is stale or none ever is.
  * rclpy rejects ints in float64 fields and in bools where rospy did not, so
    nrofpoints, vs and vd are converted to float, and is_static and
    is_visible to bool, at publish time.
  * Upstream's node was anonymous, so several could run side by side; here it
    is named opponent_trajectory.
  * Markers are stamped at publish time and get an identity orientation;
    upstream left the stamp at zero and the quaternion all zeros.
"""

import os

import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from f1t_msgs.msg import (ObstacleArray, OpponentTrajectory, ProjOppPoint, ProjOppTraj,
                          WpntArray)
from planning.frenet import FrenetConverter
from planning.waypoint_qos import WAYPOINTS_QOS
from planning.opponent_trajectory import OpponentTrajectoryTracker


def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class OpponentTrajectoryNode(Node):

    def __init__(self):
        super().__init__('opponent_trajectory')
        p = lambda n, v: self.declare_parameter(n, v).value

        tracks_dir = p('tracks_dir', os.environ.get('TRACKS_DIR',
                                                    os.path.expanduser('~/ws/tracks')))
        track_name = p('track_name', '')
        self.reference = p('reference', 'raceline')
        self.map_frame = p('map_frame', 'map')
        rate = float(p('loop_rate', 25.0))

        if not track_name:
            raise RuntimeError('opponent_trajectory_node needs the track_name parameter')
        track_dir = os.path.join(tracks_dir, track_name)
        if self.reference == 'raceline':
            self.conv = FrenetConverter.from_track(track_dir)
        else:
            self.conv = FrenetConverter.from_centerline(
                os.path.join(track_dir, 'centerline.csv'))

        self.tracker = OpponentTrajectoryTracker()
        self.announced = False
        self.warned_frame = False

        self.pub_traj = self.create_publisher(ProjOppTraj, '/proj_opponent_trajectory', 10)
        self.pub_markers = self.create_publisher(MarkerArray, '/opponent_marker', 10)

        self.create_subscription(ObstacleArray, '/perception/obstacles', self.obstacles_cb, 10)
        self.create_subscription(Odometry, '/car_state/odom_frenet', self.odom_cb, 10)
        self.create_subscription(WpntArray, '/global_waypoints', self.waypoints_cb,
                                 WAYPOINTS_QOS)
        self.create_subscription(OpponentTrajectory, '/opponent_trajectory',
                                 self.opponent_trajectory_cb, 10)
        self.create_subscription(String, '/state_machine', self.state_cb, 10)
        self.create_timer(1.0 / rate, self.tick)

    # ------------------------------------------------------------------ inputs

    def obstacles_cb(self, msg):
        self.tracker.set_obstacles(msg.obstacles, stamp_to_sec(msg.header.stamp))

    def odom_cb(self, msg):
        expect = f'frenet/{self.reference}'
        if msg.header.frame_id and msg.header.frame_id != expect and not self.warned_frame:
            self.warned_frame = True
            self.get_logger().error(
                f'ego state is measured against {msg.header.frame_id}, this node uses '
                f'{expect}. Opponent markers will be in the wrong place.')
        pos, vel = msg.pose.pose.position, msg.twist.twist.linear
        self.tracker.set_ego(pos.x, pos.y, vel.x, vel.y)

    def waypoints_cb(self, msg):
        first = not self.tracker.configured
        if self.tracker.set_global_waypoints(msg.wpnts) and first:
            self.get_logger().info('[Tracking] initialized FrenetConverter object')

    def opponent_trajectory_cb(self, msg):
        self.tracker.set_opponent_trajectory(msg.oppwpnts)

    def state_cb(self, msg):
        self.tracker.set_state(msg.data)

    # ------------------------------------------------------------------ loop

    def tick(self):
        if not self.tracker.ready:
            return
        now = self.get_clock().now()
        step = self.tracker.step(now.nanoseconds * 1e-9)
        for traj in step.trajectories:
            self.pub_traj.publish(self._traj_msg(traj))
        stamp = now.to_msg()
        for markers in step.markers:
            self.pub_markers.publish(MarkerArray(markers=[self._marker(m, stamp)
                                                          for m in markers]))

    @staticmethod
    def _traj_msg(traj):
        msg = ProjOppTraj(lapcount=float(traj.lapcount), nrofpoints=float(traj.nrofpoints),
                          opp_is_on_trajectory=bool(traj.opp_is_on_trajectory))
        msg.detections = [ProjOppPoint(
            s=float(p.s), d=float(p.d), vs=float(p.vs), vd=float(p.vd),
            is_static=bool(p.is_static), is_visible=bool(p.is_visible), time=float(p.time),
            s_var=float(p.s_var), d_var=float(p.d_var), vs_var=float(p.vs_var),
            vd_var=float(p.vd_var)) for p in traj.detections]
        return msg

    def _marker(self, rec, stamp):
        m = Marker()
        m.header.stamp, m.header.frame_id = stamp, self.map_frame
        m.id = rec.id
        m.type = Marker.CYLINDER
        m.action = Marker.DELETEALL if rec.delete_all else Marker.ADD
        x, y = self.conv.to_cartesian(rec.s, rec.d)
        m.pose.position.x, m.pose.position.y = float(x), float(y)
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = 0.3
        m.scale.z = 1.0
        m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 1.0, 1.0, 1.0
        return m


def main(args=None):
    rclpy.init(args=args)
    node = OpponentTrajectoryNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
