#!/usr/bin/env python3
"""Save the running Cartographer mapping session to tracks/<name>/.

  ros2 run perception save_map <name>

Finishes the active trajectory, waits until its poses settle (Cartographer optimizes in the
background after /finish_trajectory returns), then writes (see docs/pipeline.md):
  map.pbstream       Cartographer state via /write_state, loaded by localization.launch.py
  map.pgm, map.yaml  /map in map_server format (trinary, thresholds 0.65 / 0.196)
  trajectory.csv     optimized trajectory node poses of base_link in map (/trajectory_query)
  map_to_world.yaml  map origin in the sim world, only if map_origin latched one (sim only)
"""
import os
import sys
import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from rclpy.utilities import remove_ros_args
from cartographer_ros_msgs.msg import StatusCode
from cartographer_ros_msgs.srv import FinishTrajectory, GetTrajectoryStates, TrajectoryQuery, WriteState
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid

from perception.map_origin import pose2d
from perception.tracks import track_dir

OCCUPIED_THRESH = 0.65
FREE_THRESH = 0.25       # grid percent -> free pixel (254)
YAML_FREE_THRESH = 0.196  # map.yaml: just below unknown 205's p = 0.19608, so loaders keep it unknown


class SaveMap(Node):
    def __init__(self):
        super().__init__('save_map')
        self.timeout = self.declare_parameter('timeout', 30.0).value   # s per step
        self.settle_time = self.declare_parameter('settle_time', 5.0).value   # s of unchanged poses
        self.poll_period = 0.5
        self.map = None
        self.map_rx_t = 0.0
        self.origin = None
        # volatile + reliable matches cartographer_occupancy_grid_node whatever its durability
        self.create_subscription(OccupancyGrid, '/map', self.map_cb,
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE))
        self.create_subscription(PoseStamped, '/map_origin/ground_truth', self.origin_cb,
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))

    def map_cb(self, msg):
        self.map = msg
        self.map_rx_t = time.monotonic()

    def origin_cb(self, msg):
        self.origin = msg

    def call(self, srv_type, name, request):
        client = self.create_client(srv_type, name)
        if not client.wait_for_service(timeout_sec=self.timeout):
            raise RuntimeError(f'service {name} not available (is Cartographer running?)')
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self.timeout)
        if future.result() is None:
            raise RuntimeError(f'{name} timed out')
        return future.result()

    def spin_until(self, done, timeout):
        end = time.monotonic() + timeout
        while not done() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
        return done()

    def query_poses(self, tid):
        res = self.call(TrajectoryQuery, '/trajectory_query', TrajectoryQuery.Request(trajectory_id=tid))
        self.check(res.status, 'trajectory_query')
        poses = [(p.pose.position.x, p.pose.position.y, p.pose.position.z, p.pose.orientation.x,
                  p.pose.orientation.y, p.pose.orientation.z, p.pose.orientation.w) for p in res.trajectory]
        return res, poses

    def wait_settled(self, tid, finished):
        """Poll /trajectory_query until the poses stop changing; returns the last response.

        /finish_trajectory only queues an optimization. If this run finished the trajectory, first wait
        up to timeout for that optimization to move the poses, then for settle_time without change.
        It moves nothing when an earlier optimization already covered every node (car parked before
        Enter, as map_track.sh asks): then the poses are final and unchanged for timeout.
        """
        res, poses = self.query_poses(tid)
        if finished:
            self.get_logger().info('waiting for the final optimization')
            end = time.monotonic() + self.timeout
            while True:
                time.sleep(self.poll_period)
                res, new = self.query_poses(tid)
                if new != poses:   # list compare: length and values
                    poses = new
                    break
                if time.monotonic() > end:
                    self.get_logger().warn(f'trajectory {tid} unchanged {self.timeout:.0f} s after '
                                           'finish_trajectory: taking its poses as final')
                    return res
        end = time.monotonic() + self.timeout
        quiet_since = time.monotonic()
        while time.monotonic() - quiet_since < self.settle_time:
            if time.monotonic() > end:
                raise RuntimeError(f'trajectory {tid} still changing after {self.timeout:.0f} s')
            time.sleep(self.poll_period)
            res, new = self.query_poses(tid)
            if new != poses:
                poses, quiet_since = new, time.monotonic()
        return res

    def save(self, out):
        os.makedirs(out, exist_ok=True)

        states = self.call(GetTrajectoryStates, '/get_trajectory_states', GetTrajectoryStates.Request())
        ids = list(states.trajectory_states.trajectory_id)
        active = [i for i, s in zip(ids, states.trajectory_states.trajectory_state)
                  if s == states.trajectory_states.ACTIVE]
        if not ids:
            raise RuntimeError('Cartographer has no trajectory')
        for tid in active:
            self.get_logger().info(f'finishing trajectory {tid} (final optimization)')
            res = self.call(FinishTrajectory, '/finish_trajectory', FinishTrajectory.Request(trajectory_id=tid))
            self.check(res.status, 'finish_trajectory')
        tid = active[-1] if active else ids[-1]
        traj = self.wait_settled(tid, finished=bool(active))

        pbstream = os.path.join(out, 'map.pbstream')
        res = self.call(WriteState, '/write_state',
                        WriteState.Request(filename=pbstream, include_unfinished_submaps=True))
        self.check(res.status, 'write_state')
        self.get_logger().info(f'wrote {pbstream}')

        self.write_trajectory(os.path.join(out, 'trajectory.csv'), traj.trajectory)

        # a /map drawn after the final optimization
        after = time.monotonic()
        if not self.spin_until(lambda: self.map is not None and self.map_rx_t > after, self.timeout):
            raise RuntimeError('no /map received (is cartographer_occupancy_grid_node running?)')
        self.write_map(out, self.map)

        self.spin_until(lambda: self.origin is not None, 1.0)
        if self.origin is not None:
            self.write_origin(os.path.join(out, 'map_to_world.yaml'), self.origin)
        else:
            self.get_logger().info('no /map_origin/ground_truth (real car?), skipping map_to_world.yaml')

    def check(self, status, what):
        if status.code != StatusCode.OK:
            raise RuntimeError(f'{what} failed: {status.message}')

    def write_trajectory(self, path, poses):
        with open(path, 'w') as f:
            f.write('# t,x_m,y_m,yaw_rad\n')
            for p in poses:
                x, y, yaw = pose2d(p.pose)
                t = p.header.stamp.sec + p.header.stamp.nanosec * 1e-9
                f.write(f'{t:.6f},{x:.4f},{y:.4f},{yaw:.5f}\n')
        self.get_logger().info(f'wrote {path} ({len(poses)} poses)')

    def write_map(self, out, grid):
        w, h = grid.info.width, grid.info.height
        occ = np.array(grid.data, dtype=np.int16).reshape(h, w)[::-1]   # image row 0 is max y
        pixels = np.full((h, w), 205, dtype=np.uint8)                    # unknown
        pixels[(occ >= 0) & (occ <= FREE_THRESH * 100)] = 254
        pixels[occ >= OCCUPIED_THRESH * 100] = 0
        with open(os.path.join(out, 'map.pgm'), 'wb') as f:
            f.write(f'P5\n# CREATOR: perception save_map {grid.info.resolution:.3f} m/pix\n{w} {h}\n255\n'.encode())
            f.write(pixels.tobytes())
        x, y, yaw = pose2d(grid.info.origin)
        with open(os.path.join(out, 'map.yaml'), 'w') as f:
            f.write('image: map.pgm\n'
                    'mode: trinary\n'
                    f'resolution: {grid.info.resolution:.6f}\n'
                    f'origin: [{x:.6f}, {y:.6f}, {yaw:.6f}]\n'
                    'negate: 0\n'
                    f'occupied_thresh: {OCCUPIED_THRESH}\n'
                    f'free_thresh: {YAML_FREE_THRESH}\n')
        self.get_logger().info(f'wrote map.pgm/map.yaml ({w}x{h}, {grid.info.resolution:.3f} m)')

    def write_origin(self, path, msg):
        x, y, yaw = pose2d(msg.pose)
        with open(path, 'w') as f:
            f.write('# ground-truth pose of the map origin in the sim world frame (debugging only)\n'
                    f'frame_id: {msg.header.frame_id}\n'
                    f'x_m: {x:.4f}\n'
                    f'y_m: {y:.4f}\n'
                    f'yaw_rad: {yaw:.5f}\n')
        self.get_logger().info(f'wrote {path}')


def main():
    args = remove_ros_args(sys.argv)[1:]
    if len(args) != 1:
        print('usage: ros2 run perception save_map <track name>')
        sys.exit(2)
    rclpy.init()
    node = SaveMap()
    out = track_dir(args[0])
    try:
        node.save(out)
        node.get_logger().info(f'saved track to {out}')
    except RuntimeError as e:
        node.get_logger().error(str(e))
        sys.exit(1)
    except (KeyboardInterrupt, ExternalShutdownException):
        sys.exit(130)   # not 0: map_track.sh must not take a half-saved track as saved
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
