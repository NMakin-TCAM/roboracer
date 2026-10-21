#!/usr/bin/env python3
"""Remembers where the map origin is in the sim world (simulation debugging only).

Started by mapping.launch.py. On the first /tracked_pose (base_link in map) it takes the latest
/ground_truth/pose (car in the sim world) and latches the map origin in the sim world,
world_T_map = world_T_car * inverse(map_T_car), on /map_origin/ground_truth (transient local).
save_map writes it to map_to_world.yaml. On the real car there is no /ground_truth/pose, so
nothing is published and save_map skips the file.
"""
import math

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from geometry_msgs.msg import PoseStamped

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


def pose2d(pose):
    q = pose.orientation
    yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    return pose.position.x, pose.position.y, yaw


class MapOrigin(Node):
    def __init__(self):
        super().__init__('map_origin')
        self.gt = None
        self.pub = self.create_publisher(PoseStamped, '/map_origin/ground_truth', LATCHED)
        self.gt_sub = self.create_subscription(PoseStamped, '/ground_truth/pose', self.gt_cb, 10)
        self.tracked_sub = self.create_subscription(PoseStamped, '/tracked_pose', self.tracked_cb, 10)

    def gt_cb(self, msg):
        self.gt = msg

    def tracked_cb(self, msg):
        if self.gt is None:
            return
        wx, wy, wyaw = pose2d(self.gt.pose)   # sim vehicle frame = rear axle = base_link (AutoDRIVE bridge)
        mx, my, myaw = pose2d(msg.pose)
        yaw = wyaw - myaw
        c, s = math.cos(yaw), math.sin(yaw)
        origin = PoseStamped()
        origin.header = self.gt.header
        origin.pose.position.x = wx - (c * mx - s * my)
        origin.pose.position.y = wy - (s * mx + c * my)
        origin.pose.orientation.z = math.sin(yaw / 2.0)
        origin.pose.orientation.w = math.cos(yaw / 2.0)
        self.pub.publish(origin)
        self.destroy_subscription(self.gt_sub)
        self.destroy_subscription(self.tracked_sub)
        self.get_logger().info(f'map origin in sim world: x={origin.pose.position.x:.3f} '
                               f'y={origin.pose.position.y:.3f} yaw={yaw:.3f}')


def main():
    rclpy.init()
    node = MapOrigin()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
