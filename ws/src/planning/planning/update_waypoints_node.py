"""Republish the scaled raceline with measured speeds on /global_waypoints_updated.

Port of the ROS 1 waypoint_updater shared by Predictive Spliner and FSDP. All the
logic lives in update_waypoints.py, which imports no ROS and is unit tested. This
file reads topics, hands them to WaypointUpdater, and publishes on a timer.

ROS 1 to ROS 2 differences:

  * rospy.wait_for_message on the two waypoint topics became subscriptions;
    nothing is published until both have arrived. They use default (VOLATILE)
    QoS, which matches both VOLATILE and TRANSIENT_LOCAL publishers, whereas a
    TRANSIENT_LOCAL subscriber silently gets nothing from a VOLATILE publisher.
    Nothing in the Predictive Spliner source latches these topics, so there was
    no evidence for anything stronger.

    The catch: only the first message on each topic is used, and a VOLATILE
    subscriber never sees a message published before it started. If the
    waypoints are published once, a node that starts late waits forever.
    Whoever writes the /global_waypoints and /global_waypoints_scaled
    publishers should decide the durability, and set it on both sides: either
    publish TRANSIENT_LOCAL and change these two subscriptions to match, or
    keep publishing periodically.
  * The dynamic_reconfigure Config subscription became the ROS 2 parameters
    update_waypoints and speed_offset, which can be changed at runtime with
    ros2 param set. Upstream read speed_offset with no default; it is 0.0 here.
  * time.time() became the node clock, so the hysteresis follows sim time when
    use_sim_time is set. On the car the two are the same.
  * The unused convert_frenet2globarr_service proxy is dropped.
"""

import rclpy
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import SetParametersResult
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from f1t_msgs.msg import WpntArray
from planning.update_waypoints import WaypointUpdater


class UpdateWaypointsNode(Node):

    def __init__(self):
        super().__init__('waypoint_updater')
        p = lambda n, v: self.declare_parameter(n, v).value

        odom_topic = p('odom_topic', '/car_state/odom_frenet')
        state_topic = p('state_topic', '/state_machine')
        scaled_topic = p('scaled_topic', '/global_waypoints_scaled')
        global_topic = p('global_topic', '/global_waypoints')
        out_topic = p('out_topic', '/global_waypoints_updated')
        marker_topic = p('marker_topic', '/updated_waypoints_marker')
        self.map_frame = p('map_frame', 'map')
        self.expect_frame = p('expect_frame', 'frenet/raceline')
        rate = float(p('rate', 1.0))

        self.updater = WaypointUpdater(
            hysteresis_time=float(p('hysteresis_time', 2.0)),
            speed_offset=float(p('speed_offset', 0.0)),
            enabled=bool(p('update_waypoints', True)),
        )
        self.add_on_set_parameters_callback(self.param_cb)
        self.warned_frame = False

        self.pub = self.create_publisher(WpntArray, out_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, marker_topic, 10)
        self.create_subscription(Odometry, odom_topic, self.odom_cb, 10)
        self.create_subscription(String, state_topic, self.state_cb, 10)
        self.create_subscription(WpntArray, scaled_topic, self.scaled_cb, 10)
        self.create_subscription(WpntArray, global_topic, self.global_cb, 10)
        self.create_timer(1.0 / rate, self.tick)

        self.get_logger().info('[Update Wpnts] Update Wpnts waiting...')

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def param_cb(self, params):
        for prm in params:
            if prm.name == 'update_waypoints':
                self.updater.enabled = bool(prm.value)
            elif prm.name == 'speed_offset':
                self.updater.speed_offset = float(prm.value)
            elif prm.name == 'hysteresis_time':
                self.updater.hysteresis_time = float(prm.value)
        self.get_logger().info(
            f'[Update Wpnts] update_waypoints: {self.updater.enabled}, '
            f'speed offset: {self.updater.speed_offset}')
        return SetParametersResult(successful=True)

    def scaled_cb(self, msg):
        if self.updater.set_scaled(msg):
            self._announce_if_ready()

    def global_cb(self, msg):
        if self.updater.set_global(msg):
            self._announce_if_ready()

    def _announce_if_ready(self):
        if self.updater.ready:
            self.get_logger().info('[Update Wpnts] Update Wpnts ready!')

    def state_cb(self, msg):
        self.updater.on_state(msg.data, self.now())

    def odom_cb(self, msg):
        if (msg.header.frame_id and msg.header.frame_id != self.expect_frame
                and not self.warned_frame):
            self.warned_frame = True
            self.get_logger().error(
                f'ego state is measured against {msg.header.frame_id}, the waypoints '
                f'are indexed by {self.expect_frame} s. The wrong waypoints will be '
                f'overwritten.')
        self.updater.on_odom(msg.pose.pose.position.x, msg.twist.twist.linear.x, self.now())

    def tick(self):
        if not self.updater.ready:
            return
        msg = self.updater.wpnts_msg
        msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(msg)
        self._publish_markers(msg.wpnts)

    def _publish_markers(self, wpnts):
        markers = MarkerArray()
        for i, (w, h) in enumerate(zip(wpnts, self.updater.marker_heights())):
            m = Marker()
            m.header.frame_id = self.map_frame
            m.id = i
            m.type = Marker.CYLINDER
            m.pose.position.x = w.x_m
            m.pose.position.y = w.y_m
            m.pose.orientation.w = 1.0
            m.scale.x = 0.1
            m.scale.y = 0.1
            m.scale.z = h
            m.pose.position.z = h / 2
            m.color.a = 1.0
            m.color.g = 1.0
            markers.markers.append(m)
        self.marker_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = UpdateWaypointsNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
