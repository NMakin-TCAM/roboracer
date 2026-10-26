#!/usr/bin/env python3
"""The stack's map and pose, streamed to the AutoDRIVE app's map view (simulation only).

The app's Mapping tab (in place of its Front Camera Preview; MapView.cs in our AutoDRIVE fork's
track_select plugin) draws the occupancy grid the car localizes on, and where the car thinks it is
on it. This node sends them over TCP to the app, which listens on the host's loopback port 4581
(host.docker.internal from the container). It reconnects whenever the app (re)starts.

  /map                     OccupancyGrid: Cartographer's grid while mapping (grows as the car maps)
                           and in localization; sent as soon as it arrives
  TF map -> base_link      the newest estimate, sent at `rate` Hz

Frames, little-endian: type (1 byte), payload length (uint32), payload
  'M'  uint32 width, uint32 height, float32 resolution, origin x, origin y, origin yaw,
       then width * height int8 cells (row 0 at the origin, -1 unknown, 0..100 occupied %)
  'P'  float32 x, y, yaw, age   base_link in the map frame, and how old that estimate is (s) when
       sent, so the app can compare it with where the car really was then (its accuracy numbers)
  'H'  (empty) heartbeat, once a second, so the app can tell a dead connection from no TF
"""
import math
import socket
import struct
import time

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from nav_msgs.msg import OccupancyGrid
from tf2_ros import Buffer, TransformException, TransformListener


def frame(kind, payload=b''):
    return kind + struct.pack('<I', len(payload)) + payload


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def downsample(cells, max_cells):
    """cells (h, w) int8 -> at most max_cells per side, each block its maximum: occupied wins over
    free, free over unknown (-1). Returns (cells, factor)."""
    h, w = cells.shape
    k = max(1, math.ceil(max(h, w) / max_cells))
    if k == 1:
        return cells, 1
    padded = np.full((math.ceil(h / k) * k, math.ceil(w / k) * k), -1, dtype=np.int8)
    padded[:h, :w] = cells
    blocks = padded.reshape(padded.shape[0] // k, k, padded.shape[1] // k, k)
    return blocks.max(axis=(1, 3)), k


def map_frame(msg, max_cells):
    info = msg.info
    cells = np.asarray(msg.data, dtype=np.int8).reshape(info.height, info.width)
    cells, k = downsample(cells, max_cells)
    q = info.origin.orientation
    header = struct.pack('<IIffff', cells.shape[1], cells.shape[0], info.resolution * k,
                         info.origin.position.x, info.origin.position.y, yaw_of(q))
    return frame(b'M', header + np.ascontiguousarray(cells).tobytes())


class MapView(Node):
    def __init__(self):
        super().__init__('map_view')
        p = self.declare_parameter
        self.host = p('sim_host', 'host.docker.internal').value
        self.port = p('sim_port', 4581).value
        self.map_topic = p('map_topic', '/map').value
        self.map_frame_id = p('map_frame', 'map').value
        self.base_frame = p('base_frame', 'base_link').value
        self.max_cells = p('max_cells', 2048).value   # per side; larger grids are downsampled
        rate = p('rate', 50.0).value                  # Hz, pose updates

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.sock = None
        self.next_connect = 0.0
        self.last_heartbeat = 0.0
        self.map_msg = None       # newest /map, resent after a reconnect
        self.map_dirty = False
        self.map_sub = None
        self.map_durability = None
        self.create_timer(1.0 / rate, self.tick)
        self.create_timer(2.0, self.subscribe_map)   # the publishers can come and go
        self.subscribe_map()

    def subscribe_map(self):
        """Match the /map publishers' durability: latched (map_server, TRANSIENT_LOCAL) or not
        (Cartographer's grid node republishes every period). A TRANSIENT_LOCAL subscription
        would not match a VOLATILE publisher, and a VOLATILE one misses a latched map."""
        infos = self.get_publishers_info_by_topic(self.map_topic)
        if not infos:
            return
        latched = all(i.qos_profile.durability == DurabilityPolicy.TRANSIENT_LOCAL for i in infos)
        durability = DurabilityPolicy.TRANSIENT_LOCAL if latched else DurabilityPolicy.VOLATILE
        if durability == self.map_durability:
            return
        if self.map_sub is not None:
            self.destroy_subscription(self.map_sub)
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=durability,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.map_sub = self.create_subscription(OccupancyGrid, self.map_topic, self.map_cb, qos)
        self.map_durability = durability
        self.get_logger().info(f'{self.map_topic}: subscribed ({"latched" if latched else "volatile"})')

    def map_cb(self, msg):
        self.map_msg = msg
        self.map_dirty = True

    def connect(self):
        now = time.monotonic()
        if now < self.next_connect:
            return
        self.next_connect = now + 1.0
        try:
            s = socket.create_connection((self.host, self.port), timeout=0.2)
        except OSError:
            return   # the app is not running (yet)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.settimeout(2.0)
        self.sock = s
        self.map_dirty = self.map_msg is not None
        self.get_logger().info(f'connected to the sim map view at {self.host}:{self.port}')

    def send(self, data):
        try:
            self.sock.sendall(data)
            return True
        except OSError as e:
            self.get_logger().info(f'sim map view connection lost ({e}); reconnecting')
            self.sock.close()
            self.sock = None
            return False

    def tick(self):
        if self.sock is None:
            self.connect()
            if self.sock is None:
                return
        if self.map_dirty:
            self.map_dirty = False
            if not self.send(map_frame(self.map_msg, self.max_cells)):
                return
        try:
            t = self.tf_buffer.lookup_transform(self.map_frame_id, self.base_frame, Time(),
                                                timeout=Duration(seconds=0.0))
            # the newest common time: old once localization stops (the app then shows the pose's age)
            age = (self.get_clock().now() - Time.from_msg(t.header.stamp)).nanoseconds * 1e-9
            if age > 0.5:
                raise TransformException('stale')
            tr = t.transform
            if not self.send(frame(b'P', struct.pack('<ffff', tr.translation.x, tr.translation.y,
                                                     yaw_of(tr.rotation), max(age, 0.0)))):
                return
        except TransformException:
            pass   # no localization running (yet, or any more)
        now = time.monotonic()
        if now - self.last_heartbeat >= 1.0:
            self.last_heartbeat = now
            self.send(frame(b'H'))


def main(args=None):
    rclpy.init(args=args)
    node = MapView()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node.sock is not None:
            node.sock.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
