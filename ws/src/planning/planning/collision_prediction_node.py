"""Publish the predicted ego/opponent collision region on /collision_prediction/obstacles.

Port of the ROS 1 collision_predictor shared by Predictive Spliner and FSDP. All
the logic lives in collision_prediction.py, which imports no ROS and is unit
tested. This file reads topics, hands them to CollisionPredictor, and publishes
on a timer. Topic names are upstream's, including the mismatch between
/collision_predict/* for the markers and /collision_prediction/* for the rest,
and the relative collision_prediction/force_trailing.

ROS 1 to ROS 2 differences:

  * LOOP RATE. Upstream sets a 10 Hz rate but its rate.sleep() is indented
    outside the while loop, so the loop never sleeps and runs as fast as the
    CPU allows. That cannot be kept: rclpy runs callbacks on the same thread as
    a timer, so a loop that never yields would starve every subscription,
    where rospy ran them on threads of their own. This node runs at the 10 Hz
    upstream evidently intended, set by the rate parameter. Because the expire
    counter counts cycles, a region now lingers max_expire_counter / rate
    seconds (1 s at the defaults) after the last one was found, where
    upstream's lingered for however long 10 unthrottled cycles took.
  * The rospy.wait_for_message calls became subscriptions. Nothing is
    computed until /global_waypoints, /global_waypoints_scaled,
    /global_waypoints_updated, /opponent_trajectory and /perception/obstacles
    have each delivered a message. Upstream waited on the first two and never
    used them, and still does not here. Upstream also waited in sequence,
    each for a message after the previous one arrived; here any order counts.
    /global_waypoints and /global_waypoints_scaled are subscribed
    TRANSIENT_LOCAL, KEEP_LAST(1)
    (planning.waypoint_qos), matching global_waypoints_publisher_node, which
    publishes them once, latched. A VOLATILE subscription would never see a
    message published before it started, and the node would wait forever.
    The other inputs are published continuously and keep default (VOLATILE)
    QoS.
  * The dynamic_reconfigure Config subscription became ROS 2 parameters with
    upstream's names and defaults (n_time_steps, dt, save_distance_front,
    save_distance_back, max_v, min_v, max_a, min_a, max_expire_counter),
    changeable at runtime with ros2 param set.
  * convert_frenet2globarr_service became FrenetConverter.to_cartesian,
    in-process, so the node needs track_name to load the raceline.
  * rclpy rejects an int in a float64 field where rospy did not, so upstream's
    integer literals (vs = 0, d = +/-10) are floats here.
  * Upstream stamped its two markers once at startup and published that stamp
    forever. They are stamped at publish time here.
"""

import os

import rclpy
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import SetParametersResult
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, String
from visualization_msgs.msg import Marker

from f1t_msgs.msg import Obstacle, ObstacleArray, OpponentTrajectory, WpntArray
from planning.collision_prediction import CollisionParams, CollisionPredictor
from planning.frenet import FrenetConverter
from planning.waypoint_qos import WAYPOINTS_QOS

# Upstream parameter name -> CollisionParams field, with the type it must have.
PARAMS = {
    'n_time_steps': ('time_steps', int),
    'dt': ('dt', float),
    'save_distance_front': ('save_distance_front', float),
    'save_distance_back': ('save_distance_back', float),
    'max_v': ('max_v', float),
    'min_v': ('min_v', float),
    'max_a': ('max_a', float),
    'min_a': ('min_a', float),
    'max_expire_counter': ('max_expire_counter', int),
}
WAIT_FOR = ('global', 'scaled', 'updated', 'opponent_trajectory', 'obstacles')


class CollisionPredictionNode(Node):

    def __init__(self):
        super().__init__('collision_predictor')
        p = lambda n, v: self.declare_parameter(n, v).value

        tracks_dir = p('tracks_dir', os.environ.get('TRACKS_DIR',
                                                    os.path.expanduser('~/ws/tracks')))
        track_name = p('track_name', '')
        self.reference = p('reference', 'raceline')
        self.map_frame = p('map_frame', 'map')
        rate = float(p('rate', 10.0))

        if not track_name:
            raise RuntimeError('collision_prediction_node needs the track_name parameter')
        track_dir = os.path.join(tracks_dir, track_name)
        if self.reference == 'raceline':
            self.conv = FrenetConverter.from_track(track_dir)
        else:
            self.conv = FrenetConverter.from_centerline(
                os.path.join(track_dir, 'centerline.csv'))

        defaults = CollisionParams()
        params = CollisionParams(**{
            field: kind(p(name, kind(getattr(defaults, field))))
            for name, (field, kind) in PARAMS.items()})
        self.predictor = CollisionPredictor(params)
        self.add_on_set_parameters_callback(self.param_cb)

        self.seen = set()
        self.ready = False
        self.warned_frame = False

        self.pub_begin = self.create_publisher(Marker, '/collision_predict/beginn', 10)
        self.pub_end = self.create_publisher(Marker, '/collision_predict/end', 10)
        self.pub_obs = self.create_publisher(ObstacleArray, '/collision_prediction/obstacles', 10)
        self.pub_trailing = self.create_publisher(Bool, 'collision_prediction/force_trailing', 10)

        self.create_subscription(ObstacleArray, '/perception/obstacles', self.obstacles_cb, 10)
        self.create_subscription(Odometry, '/car_state/odom_frenet', self.odom_cb, 10)
        self.create_subscription(OpponentTrajectory, '/opponent_trajectory',
                                 self.opponent_trajectory_cb, 10)
        self.create_subscription(WpntArray, '/global_waypoints_updated', self.updated_cb, 10)
        self.create_subscription(WpntArray, '/global_waypoints',
                                 lambda _: self._saw('global'), WAYPOINTS_QOS)
        self.create_subscription(WpntArray, '/global_waypoints_scaled',
                                 lambda _: self._saw('scaled'), WAYPOINTS_QOS)
        self.create_subscription(String, '/state_machine', self.state_cb, 10)
        self.create_timer(1.0 / rate, self.tick)

        self.get_logger().info('[Coll. Pred.] Collision Predictor waiting...')

    # ------------------------------------------------------------------ inputs

    def _saw(self, what):
        if what in self.seen:
            return
        self.seen.add(what)
        if not self.ready and all(w in self.seen for w in WAIT_FOR):
            self.ready = True
            self.get_logger().info('[Coll. Pred.] Collision Predictor ready!')

    def param_cb(self, changed):
        for prm in changed:
            if prm.name in PARAMS:
                field, kind = PARAMS[prm.name]
                setattr(self.predictor.params, field, kind(prm.value))
        self.get_logger().info(f'[Coll. Pred.] new params: {self.predictor.params}')
        return SetParametersResult(successful=True)

    def obstacles_cb(self, msg):
        self.predictor.set_obstacles(msg.obstacles)
        self._saw('obstacles')

    def odom_cb(self, msg):
        expect = f'frenet/{self.reference}'
        if msg.header.frame_id and msg.header.frame_id != expect and not self.warned_frame:
            self.warned_frame = True
            self.get_logger().error(
                f'ego state is measured against {msg.header.frame_id}, this node uses '
                f'{expect}. Every collision region will be in the wrong place.')
        self.predictor.set_ego(msg.pose.pose.position.x, msg.twist.twist.linear.x)

    def opponent_trajectory_cb(self, msg):
        if self.predictor.set_opponent_trajectory(msg.oppwpnts, msg.lap_count):
            self._saw('opponent_trajectory')

    def updated_cb(self, msg):
        if self.predictor.set_updated_waypoints(msg.wpnts):
            self._saw('updated')

    def state_cb(self, msg):
        # Upstream stores the state and never reads it.
        self.state = msg.data

    # ------------------------------------------------------------------ loop

    def tick(self):
        if not self.ready:
            return
        step = self.predictor.step()
        now = self.get_clock().now().to_msg()

        if step.force_trailing is not None:
            self.pub_trailing.publish(Bool(data=step.force_trailing))

        if step.region is not None:
            r = step.region
            obs = Obstacle(id=0, d_left=r.d_left, d_right=r.d_right, d_center=r.d_center,
                           vs=0.0, vd=0.0, is_actually_a_gap=False, is_static=False)
            obs.s_start, obs.s_end, obs.s_center = r.s_start, r.s_end, r.s_center
            arr = ObstacleArray(obstacles=[obs])
            arr.header.stamp, arr.header.frame_id = now, self.map_frame
            self.pub_obs.publish(arr)

            (bx, by), (ex, ey) = self.predictor.marker_positions(r, self.conv)
            self.pub_begin.publish(self._marker(0, bx, by, now))
            self.pub_end.publish(self._marker(1, ex, ey, now))

        if step.delete:
            self._delete_all(now)

    def _marker(self, mid, x, y, now):
        m = Marker()
        m.header.stamp, m.header.frame_id = now, self.map_frame
        m.id = mid
        m.type = Marker.SPHERE
        m.pose.position.x, m.pose.position.y = x, y
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.4
        m.color.a, m.color.r, m.color.g, m.color.b = 0.5, 0.63, 0.13, 0.94
        return m

    def _delete_all(self, now):
        m = Marker()
        m.header.stamp, m.header.frame_id = now, self.map_frame
        m.id = 0
        m.action = Marker.DELETE
        self.pub_begin.publish(m)
        m.id = 1
        self.pub_end.publish(m)
        arr = ObstacleArray()
        arr.header.stamp, arr.header.frame_id = now, self.map_frame
        self.pub_obs.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = CollisionPredictionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
