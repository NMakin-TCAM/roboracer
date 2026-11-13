"""Publish the track's raceline on /global_waypoints and /global_waypoints_scaled.

Stands in for ForzaETH's global planner output, which every shared port waits on
and nothing else in this repo produced. The conversion lives in
global_waypoints_publisher.py, which imports no ROS and is unit tested; this file
loads it, logs what it had to compute or doubts, and publishes.

Both topics are published once, latched (waypoint_qos.WAYPOINTS_QOS), so nodes
that start later still receive them. The raceline is fixed for a run.

/global_waypoints_scaled IS /global_waypoints. Upstream it carries the speeds
multiplied by a velocity factor that the state machine owns; this repo has no
velocity scaler, so the factor is 1. See docs/forzaeth-quirks.md, section 6.
"""

import os

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from f1t_msgs.msg import Wpnt, WpntArray
from planning.global_waypoints_publisher import (WPNT_FIELDS, load_global_waypoints,
                                                 provenance_warning)
from planning.waypoint_qos import WAYPOINTS_QOS


class GlobalWaypointsPublisherNode(Node):

    def __init__(self, **kwargs):
        super().__init__('global_waypoints_publisher', **kwargs)
        p = lambda n, v: self.declare_parameter(n, v).value

        tracks_dir = p('tracks_dir', os.environ.get('TRACKS_DIR',
                                                    os.path.expanduser('~/ws/tracks')))
        track_name = p('track_name', '')
        reference = p('reference', 'raceline')
        global_topic = p('global_topic', '/global_waypoints')
        scaled_topic = p('scaled_topic', '/global_waypoints_scaled')
        frame_id = p('frame_id', 'map')

        if not track_name:
            raise RuntimeError('global_waypoints_publisher_node needs the track_name parameter')
        if reference != 'raceline':
            # d_m = 0 and s_m along the raceline only mean anything if the ego and
            # opponent Frenet states are measured against the raceline too.
            raise RuntimeError(
                f"reference must be 'raceline' (got {reference!r}): the waypoints are the "
                f"raceline, so their s and d are only consistent with a raceline-referenced "
                f"/car_state/odom_frenet")

        track_dir = os.path.join(tracks_dir, track_name)
        loaded = load_global_waypoints(track_dir)
        log = self.get_logger()
        for w in loaded.warnings:
            log.warn(w)
        prov = provenance_warning(loaded.provenance)
        if prov:
            log.warn(prov)
        log.warn('/global_waypoints_scaled carries the unscaled raceline speeds: there is no '
                 'velocity scaler (stand-in, see docs/forzaeth-quirks.md section 6).')

        msg = WpntArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.wpnts = [Wpnt(**{f: getattr(w, f) for f in WPNT_FIELDS}) for w in loaded.waypoints]
        self.msg = msg

        self.pub_global = self.create_publisher(WpntArray, global_topic, WAYPOINTS_QOS)
        self.pub_scaled = self.create_publisher(WpntArray, scaled_topic, WAYPOINTS_QOS)
        self.pub_global.publish(msg)
        self.pub_scaled.publish(msg)

        top = max(w.vx_mps for w in loaded.waypoints)
        log.info(f'{len(msg.wpnts)} waypoints from {track_dir}/raceline.csv, lap '
                 f'{loaded.lap_length:.2f} m, top speed {top:.2f} m/s, latched on '
                 f'{global_topic} and {scaled_topic}')


def main(args=None):
    rclpy.init(args=args)
    node = GlobalWaypointsPublisherNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
