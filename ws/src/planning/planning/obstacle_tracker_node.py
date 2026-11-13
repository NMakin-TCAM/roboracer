"""Track obstacles: ObstacleArray in (positions only), ObstacleArray out (ids, velocities,
variances, is_static).

All the tracking lives in obstacle_tracker.py, which imports no ROS and is unit tested.
This file reads topics, calls step() once per input array, and publishes.

Topics are parameters so the node can sit between any detector and the planners:
    in_topic   (default /perception/detections)  f1t_msgs/ObstacleArray, from the detector
    out_topic  (default /perception/obstacles)   f1t_msgs/ObstacleArray, to the planners
    ego_topic  (default /car_state/odom_frenet)  nav_msgs/Odometry, Frenet s in pose.x;
                                                  only orders the dynamic obstacles
Today, behind fake_opponent_node: run that with out_topic:=/perception/detections.

One output per input, stamped and framed as the input, so the consumers' staleness
checks see the detector's rate: if detection stops, so does this. The input's
header.frame_id names the reference line its s and d are measured against; it must
match the reference this node was given or the lap length is wrong.
"""

import os

import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from f1t_msgs.msg import Obstacle, ObstacleArray
from planning.frenet import FrenetConverter
from planning.obstacle_tracker import ObstacleTracker, TrackerParams

FIELDS = ('s_start', 's_end', 'd_right', 'd_left', 's_center', 'd_center', 'size',
          'vs', 'vd', 's_var', 'd_var', 'vs_var', 'vd_var')


class ObstacleTrackerNode(Node):

    def __init__(self, **kwargs):
        super().__init__('obstacle_tracker_node', **kwargs)
        p = lambda n, v: self.declare_parameter(n, v).value

        tracks_dir = p('tracks_dir', os.environ.get('TRACKS_DIR',
                                                    os.path.expanduser('~/ws/tracks')))
        track_name = p('track_name', '')
        self.reference = p('reference', 'raceline')
        in_topic = p('in_topic', '/perception/detections')
        out_topic = p('out_topic', '/perception/obstacles')
        ego_topic = p('ego_topic', '/car_state/odom_frenet')
        defaults = TrackerParams()
        params = TrackerParams(**{
            f: type(getattr(defaults, f))(p(f, getattr(defaults, f)))
            for f in defaults.__dataclass_fields__})

        if not track_name:
            raise RuntimeError('obstacle_tracker_node needs the track_name parameter')
        track_dir = os.path.join(tracks_dir, track_name)
        if self.reference == 'raceline':
            conv = FrenetConverter.from_track(track_dir)
        else:
            conv = FrenetConverter.from_centerline(os.path.join(track_dir, 'centerline.csv'))
        self.tracker = ObstacleTracker(conv, params)
        self.ego_s = None
        self.warned_frame = False

        self.pub = self.create_publisher(ObstacleArray, out_topic, 10)
        self.create_subscription(ObstacleArray, in_topic, self.detections_cb, 10)
        self.create_subscription(Odometry, ego_topic, self.ego_cb, 10)
        self.get_logger().info(
            f'tracking {in_topic} -> {out_topic} on {track_name} ({self.reference}, lap '
            f'{conv.length:.2f} m): confirm after {params.confirm_frames} frames, '
            f'coast {params.max_misses}')

    def ego_cb(self, msg):
        self.ego_s = float(msg.pose.pose.position.x)

    def detections_cb(self, msg):
        expect = f'frenet/{self.reference}'
        if msg.header.frame_id and msg.header.frame_id != expect and not self.warned_frame:
            self.warned_frame = True
            self.get_logger().error(
                f'detections are measured against {msg.header.frame_id}, this node uses '
                f'{expect}: s wraps at the wrong lap length.')
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        tracked = self.tracker.step(msg.obstacles, stamp, self.ego_s)
        out = ObstacleArray()
        out.header = msg.header
        out.obstacles = [self._to_msg(o) for o in tracked]
        self.pub.publish(out)

    @staticmethod
    def _to_msg(o):
        m = Obstacle()
        m.id = int(o.id)
        for f in FIELDS:
            setattr(m, f, float(getattr(o, f)))
        m.is_actually_a_gap = bool(o.is_actually_a_gap)
        m.is_static = bool(o.is_static)
        m.is_visible = bool(o.is_visible)
        return m


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleTrackerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
