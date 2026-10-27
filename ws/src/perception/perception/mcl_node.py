#!/usr/bin/env python3
"""Monte Carlo localization on the saved map: a drop-in for Cartographer localization + localization_gate.

Started by mcl.launch.py. Same outputs as localization.launch.py:
  TF map -> odom      the filter's pose in the map composed with the inverse of odom -> base_link
  /localized_pose     PoseStamped in 'map' (base_link = rear axle), published at /odom rate, and only while
                      localized, so pure pursuit does not drive before that (as with localization_gate)
Extra: /mcl/particles (PoseArray), /map (OccupancyGrid, latched) for Foxglove.

  start:     particles around the map origin (the car starts where mapping started: race.sh resets the sim
             car there). Until the cloud is tight (converged_*), scans are used even though the car stands
             still, so the car can start driving; afterwards a scan is used after update_min_d / update_min_a
             of odometry motion.
  sim reset: fast_bridge's /autodrive/roboracer_1/sim_reset event: the pose is withheld at once; after
             debounce s with the car still, the cloud restarts at the ground truth pose mapped through
             map_to_world.yaml (the map origin if there is none, as on the real car).
  lost:      the cloud spreading beyond lost_std_xy withdraws the pose, so the car stops instead of driving
             on a wrong pose.
  manual:    a PoseWithCovarianceStamped on /initialpose (RViz, Foxglove) restarts the cloud there.
"""
import math
import os
import time

import rclpy
import yaml
from geometry_msgs.msg import Pose, PoseArray, PoseStamped, PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener

from perception.map_origin import pose2d
from perception.mcl import LikelihoodMap, ParticleFilter, compose, inverse, relative, select_beams
from perception.tracks import track_dir

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
MAX_PARTICLES_SHOWN = 200
MIN_BEAMS = 10
MAP_ORIGIN = (0.0, 0.0, 0.0)   # where the car starts after a sim reset: the pose mapping started from


def world_to_map(origin, x, y, yaw):
    """Car pose in the sim world -> map frame; origin = (x, y, yaw) of the map origin in the world."""
    ox, oy, oyaw = origin
    c, s = math.cos(oyaw), math.sin(oyaw)
    dx, dy = x - ox, y - oy
    return c * dx + s * dy, -s * dx + c * dy, math.atan2(math.sin(yaw - oyaw), math.cos(yaw - oyaw))


def yaw_quaternion(yaw):
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)   # z, w


class MclNode(Node):
    def __init__(self):
        super().__init__('mcl')
        p = self.declare_parameter
        track = p('track', '').value
        self.beams = p('beams', 60).value
        self.update_min_d = p('update_min_d', 0.05).value           # m of odometry between scan updates
        self.update_min_a = p('update_min_a', 0.05).value           # rad
        self.max_jump = p('max_jump', 1.0).value                    # m: a bigger odometry step is a teleport
        self.init_std_xy = p('init_std_xy', 0.15).value
        self.init_std_yaw = p('init_std_yaw', 0.10).value
        self.conv_xy = p('converged_std_xy', 0.10).value
        self.conv_yaw = p('converged_std_yaw', 0.08).value
        self.conv_updates = p('converged_updates', 3).value
        self.lost_xy = p('lost_std_xy', 0.60).value
        self.debounce = p('debounce', 0.5).value                    # s after the last reset event
        self.still_speed = p('still_speed', 0.1).value              # m/s: the car has stopped after a reset
        self.timeout = p('timeout', 10.0).value
        self.publish_particles = p('publish_particles', True).value
        self.frame = 'map'

        self.lmap = LikelihoodMap.from_yaml(os.path.join(track_dir(track), 'map.yaml'),
                                            max_dist=p('max_dist', 2.0).value)
        self.pf = ParticleFilter(
            n=p('particles', 1000).value, seed=p('seed', -1).value, sigma_hit=p('sigma_hit', 0.15).value,
            z_hit=p('z_hit', 0.8).value, z_rand=p('z_rand', 0.2).value,
            likelihood_scale=p('likelihood_scale', 0.4).value,
            noise_x=tuple(p('noise_x', [0.10, 0.02]).value), noise_y=tuple(p('noise_y', [0.05, 0.05]).value),
            noise_yaw=tuple(p('noise_yaw', [0.05, 0.10]).value))

        self.origin = None   # map origin in the sim world, for relocalizing after a reset
        path = os.path.join(track_dir(track), 'map_to_world.yaml')
        if os.path.isfile(path):
            with open(path) as f:
                w = yaml.safe_load(f)
            self.origin = (w['x_m'], w['y_m'], w['yaw_rad'])

        self.odom = None              # (pose, stamp msg, speed)
        self.last_odom_pose = None    # odom pose of the last filter update
        self.map_T_odom = None
        self.sensor_pose = {}         # scan frame -> pose in base_link
        self.initial_pose = MAP_ORIGIN   # the start pose until an /initialpose
        self.initialized = False
        self.localized = False
        self.good = 0
        self.pending = None           # monotonic time of the last reset signal
        self.gt = None                # /ground_truth/odom: (x, y, yaw, speed) in the sim world
        self.updates = 0
        self.warn_t = 0.0

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.pose_pub = self.create_publisher(PoseStamped, '/localized_pose', 10)
        self.particles_pub = self.create_publisher(PoseArray, '/mcl/particles', 1)
        map_pub = self.create_publisher(OccupancyGrid, '/map', LATCHED)
        self.publish_map(map_pub)

        # first: rclpy runs ready subscriptions in creation order, so a reset closes the relay before a
        # pose that arrived with it is relayed
        self.create_subscription(String, '/autodrive/roboracer_1/sim_reset', self.reset_cb, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.initialpose_cb, 10)
        self.create_subscription(Odometry, '/ground_truth/odom', self.gt_cb, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, qos_profile_sensor_data)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, qos_profile_sensor_data)
        self.create_timer(0.05, self.tick)
        self.get_logger().info(f'{self.pf.n} particles on {self.lmap.w}x{self.lmap.h} map '
                               f'({self.lmap.res:.3f} m), waiting for /odom and /scan')

    # ---------------- setup ----------------
    def publish_map(self, pub):
        msg = OccupancyGrid()
        msg.header.frame_id = self.frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = self.lmap.res
        msg.info.width, msg.info.height = self.lmap.w, self.lmap.h
        msg.info.origin.position.x, msg.info.origin.position.y = self.lmap.ox, self.lmap.oy
        msg.info.origin.orientation.w = 1.0
        msg.data = self.lmap.occupancy_grid().tolist()
        pub.publish(msg)

    def init_filter(self, pose, reason):
        self.pf.init_gaussian(pose, self.init_std_xy, self.init_std_yaw)
        self.localized = False
        self.good = 0
        self.pending = None
        self.get_logger().info(f'{reason}: particles at x={pose[0]:.2f} y={pose[1]:.2f} yaw={pose[2]:.2f}')
        if self.odom is None:   # odom_cb finishes the start with this pose
            self.initial_pose = pose
            self.initialized = False
            return
        self.initialized = True
        self.last_odom_pose = self.odom[0]
        self.map_T_odom = compose(pose, inverse(self.odom[0]))

    # ---------------- odometry: TF and pose at the odom rate ----------------
    def odom_cb(self, msg):
        pose = pose2d(msg.pose.pose)
        v = msg.twist.twist.linear
        self.odom = (pose, msg.header.stamp, math.hypot(v.x, v.y))
        if not self.initialized:
            self.init_filter(self.initial_pose, 'start')
        if self.map_T_odom is None:
            return
        self.broadcast(msg.header.stamp)
        if self.localized:
            x, y, yaw = compose(self.map_T_odom, pose)
            out = PoseStamped()
            out.header.stamp, out.header.frame_id = msg.header.stamp, self.frame
            out.pose.position.x, out.pose.position.y = x, y
            out.pose.orientation.z, out.pose.orientation.w = yaw_quaternion(yaw)
            self.pose_pub.publish(out)

    def broadcast(self, stamp):
        x, y, yaw = self.map_T_odom
        t = TransformStamped()
        t.header.stamp, t.header.frame_id, t.child_frame_id = stamp, self.frame, 'odom'
        t.transform.translation.x, t.transform.translation.y = x, y
        t.transform.rotation.z, t.transform.rotation.w = yaw_quaternion(yaw)
        self.tf_broadcaster.sendTransform(t)

    # ---------------- scans: the filter update ----------------
    def get_sensor_pose(self, frame):
        if frame not in self.sensor_pose:
            try:
                t = self.tf_buffer.lookup_transform('base_link', frame, Time())
            except TransformException as e:
                if time.monotonic() - self.warn_t > 2.0:
                    self.warn_t = time.monotonic()
                    self.get_logger().warning(f'waiting for TF base_link -> {frame}: {e}')
                return None
            q = t.transform.rotation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            self.sensor_pose[frame] = (t.transform.translation.x, t.transform.translation.y, yaw)
        return self.sensor_pose[frame]

    def scan_cb(self, msg):
        if self.odom is None or not self.initialized or self.pending is not None:
            return
        sensor = self.get_sensor_pose(msg.header.frame_id)
        if sensor is None:
            return
        odom_pose = self.odom[0]
        delta = relative(self.last_odom_pose, odom_pose)
        trans = math.hypot(delta[0], delta[1])
        if trans > self.max_jump:
            self.get_logger().warning(f'odometry jumped {trans:.2f} m (reset or respawn?): not moving the cloud')
            self.last_odom_pose = odom_pose
            return
        moved = trans >= self.update_min_d or abs(delta[2]) >= self.update_min_a
        if not moved and self.localized:
            return
        if moved:   # before localizing, scans are used while standing still: the start pose is known
            self.pf.predict(delta)
            self.last_odom_pose = odom_pose
        ranges, angles = select_beams(msg.ranges, msg.angle_min, msg.angle_increment, self.beams,
                                      max(msg.range_min, 0.05), msg.range_max)
        if len(ranges) < MIN_BEAMS:
            return
        self.pf.update_sensor(ranges, angles, sensor, self.lmap)
        self.pf.resample()
        self.map_T_odom = compose(self.pf.estimate(), inverse(odom_pose))
        self.check_localized(*self.pf.spread())
        self.updates += 1
        if self.publish_particles and self.updates % 5 == 0:
            self.publish_cloud(msg.header.stamp)

    def check_localized(self, pos_std, yaw_std):
        if self.localized:
            if pos_std > self.lost_xy:
                self.localized, self.good = False, 0
                self.get_logger().error(f'lost: particles spread {pos_std:.2f} m, not driving')
        else:
            self.good = self.good + 1 if pos_std <= self.conv_xy and yaw_std <= self.conv_yaw else 0
            if self.good >= self.conv_updates:
                self.localized = True
                self.get_logger().info(f'localized (spread {pos_std:.3f} m, {yaw_std:.3f} rad), driving')

    def publish_cloud(self, stamp):
        msg = PoseArray()
        msg.header.stamp, msg.header.frame_id = stamp, self.frame
        step = max(1, self.pf.n // MAX_PARTICLES_SHOWN)
        for x, y, yaw in self.pf.p[::step]:
            pose = Pose()
            pose.position.x, pose.position.y = float(x), float(y)
            pose.orientation.z, pose.orientation.w = yaw_quaternion(float(yaw))
            msg.poses.append(pose)
        self.particles_pub.publish(msg)

    # ---------------- resets and manual initial poses ----------------
    def reset_cb(self, msg):
        if self.localized or self.pending is None:
            self.get_logger().warning(f'sim reset ({msg.data}): relocalizing, not driving')
        self.localized = False
        self.good = 0
        self.pending = time.monotonic()

    def initialpose_cb(self, msg):
        if msg.header.frame_id not in ('', self.frame):
            self.get_logger().warning(f"/initialpose in frame '{msg.header.frame_id}', needs '{self.frame}'")
            return
        self.init_filter(pose2d(msg.pose.pose), '/initialpose')

    def gt_cb(self, msg):
        x, y, yaw = pose2d(msg.pose.pose)
        v = msg.twist.twist.linear
        self.gt = (x, y, yaw, math.hypot(v.x, v.y))

    def tick(self):
        """After a reset: restart the cloud once the car has settled."""
        if self.pending is None or self.odom is None:
            return
        waited = time.monotonic() - self.pending
        speed = self.gt[3] if self.gt is not None else self.odom[2]
        if waited < self.debounce or (speed >= self.still_speed and waited < self.timeout):
            return
        if self.gt is not None and self.origin is not None:
            self.init_filter(world_to_map(self.origin, *self.gt[:3]), 'relocalizing')
        else:
            self.get_logger().warning('no ground truth or map_to_world.yaml: assuming the car is at the map origin')
            self.init_filter(MAP_ORIGIN, 'relocalizing')


def main():
    rclpy.init()
    node = MclNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
