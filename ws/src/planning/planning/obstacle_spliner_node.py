#!/usr/bin/env python3
"""ROS2 stationary-obstacle spline planner for AutoDRIVE RoboRacer.
Publishes a full closed-loop nav_msgs/Path and a Bool validity signal.
Requires a raceline.csv, map.yaml + map.pgm; /scan and map-frame PoseStamped.
"""
import math
import os
import numpy as np
from scipy.ndimage import distance_transform_edt
import yaml
from PIL import Image

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Bool


def yaw_of(q):
    return math.atan2(2 * (q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))


class ObstacleSpliner(Node):
    def __init__(self):
        super().__init__('obstacle_spliner')
        p = self.declare_parameter
        self.raceline_file = p('raceline_path', '').value
        self.map_file = p('map_yaml', '').value
        self.pose_topic = p('pose_topic', '/ground_truth/pose').value
        self.scan_topic = p('scan_topic', '/scan').value
        self.lidar_x = float(p('lidar_x', 0.0).value)
        self.lidar_y = float(p('lidar_y', 0.0).value)
        self.lidar_yaw = float(p('lidar_yaw', 0.0).value)
        self.lookahead = float(p('lookahead', 6.0).value)
        self.start_distance = float(p('start_distance', 2.2).value)
        self.end_distance = float(p('end_distance', 2.5).value)
        self.vehicle_half_width = float(p('vehicle_half_width', 0.17).value)
        self.margin = float(p('safety_margin', 0.16).value)
        self.min_cluster_points = int(p('min_cluster_points', 3).value)
        self.cluster_gap = float(p('cluster_gap', 0.22).value)
        self.min_obstacle_radius = float(p('min_obstacle_radius', 0.10).value)
        self.map_free_threshold = float(p('map_free_threshold', 0.35).value)
        self.max_scan_age = float(p('max_scan_age', 0.4).value)
        self.max_pose_age = float(p('max_pose_age', 0.4).value)
        self.rate = float(p('rate_hz', 15.0).value)
        self.pose = None
        self.scan = None
        self.pose_time = None
        self.scan_time = None
        self.path_pub = self.create_publisher(Path, '/local_path', 10)
        self.valid_pub = self.create_publisher(Bool, '/local_path/valid', 10)
        if not self.raceline_file or not self.map_file:
            raise ValueError('Set raceline_path and map_yaml parameters')
        data = np.loadtxt(self.raceline_file, delimiter=';', comments='#', ndmin=2)
        if np.linalg.norm(data[0,1:3] - data[-1,1:3]) < 1e-5:
            data = data[:-1]
        self.base_xy = data[:,1:3]
        if len(data) < 6:
            raise ValueError('Raceline needs at least 6 waypoints')
        self.n = len(self.base_xy)
        self.seg = np.linalg.norm(np.roll(self.base_xy,-1,axis=0) - self.base_xy, axis=1)
        self.s = np.r_[0., np.cumsum(self.seg[:-1])]
        self.lap = float(np.sum(self.seg))
        edges = np.roll(self.base_xy,-1,axis=0)-self.base_xy
        self.tan = edges / np.maximum(self.seg[:,None],1e-8)
        self.normal = np.column_stack((-self.tan[:,1],self.tan[:,0]))
        self._load_map()
        self.create_subscription(PoseStamped, self.pose_topic, self.pose_cb, 10)
        self.create_subscription(LaserScan, self.scan_topic, self.scan_cb, qos_profile_sensor_data)
        self.create_timer(1/self.rate, self.tick)
        self.get_logger().info(f'Loaded {self.n} racing waypoints, lap length {self.lap:.1f} m')

    def _load_map(self):
        with open(self.map_file) as f:
            cfg = yaml.safe_load(f)
        image_path = cfg['image']
        if not os.path.isabs(image_path):
            image_path = os.path.join(os.path.dirname(self.map_file), image_path)
        pixels = np.asarray(Image.open(image_path).convert('L'), dtype=np.float32)/255.0
        # ROS map_server: white is free for negate=0; black is free for negate=1.
        occ = pixels if int(cfg.get('negate',0)) == 0 else 1-pixels
        self.map_free = occ >= (1-self.map_free_threshold)
        self.res = float(cfg['resolution'])
        self.origin = np.asarray(cfg['origin'][:2],dtype=float)
        self.map_yaw = float(cfg['origin'][2]) if len(cfg['origin'])>2 else 0.0
        self.free_dist = distance_transform_edt(self.map_free)*self.res
        self.h,self.w = self.map_free.shape

    def map_rc(self, xy):
        xy = np.asarray(xy,dtype=float)
        dx = xy[...,0]-self.origin[0]
        dy = xy[...,1]-self.origin[1]
        c,s = math.cos(self.map_yaw), math.sin(self.map_yaw)
        px = (c*dx+s*dy)/self.res
        py = (-s*dx+c*dy)/self.res
        return np.floor(self.h-1-py).astype(int),np.floor(px).astype(int)

    def clearances(self, points):
        row,col = self.map_rc(points)
        inside = (row>=0)&(row<self.h)&(col>=0)&(col<self.w)
        out=np.zeros(np.shape(row),dtype=float)
        out[inside]=self.free_dist[row[inside],col[inside]]
        return out

    def pose_cb(self,msg):
        if msg.header.frame_id != 'map':
            self.get_logger().warning('Expected pose in map frame',throttle_duration_sec=2.0)
            return
        self.pose=msg
        self.pose_time=self.get_clock().now().nanoseconds*1e-9

    def scan_cb(self,msg):
        self.scan=msg
        self.scan_time=self.get_clock().now().nanoseconds*1e-9

    def project(self,point):
        # Nearest projection onto the reference polyline, not just nearest waypoint.
        delta=np.asarray(point)-self.base_xy
        t=np.clip(np.sum(delta*self.tan,axis=1)/np.maximum(self.seg,1e-8),0,self.seg)
        foot=self.base_xy+t[:,None]*self.tan
        i=int(np.argmin(np.sum((foot-point)**2,axis=1)))
        lateral=float(np.dot(point-foot[i],self.normal[i]))
        return float((self.s[i]+t[i])%self.lap),lateral

    def scan_clusters(self):
        scan=self.scan
        r=np.asarray(scan.ranges,dtype=float)
        angles=scan.angle_min+np.arange(len(r))*scan.angle_increment+self.lidar_yaw
        good=np.isfinite(r)&(r>scan.range_min)&(r<min(scan.range_max,self.lookahead+3))
        points=np.column_stack((r*np.cos(angles),r*np.sin(angles)))
        groups=[]
        group=[]
        for j in range(len(r)):
            if not good[j]:
                if len(group)>=self.min_cluster_points: groups.append(np.asarray(group))
                group=[]
                continue
            if group and np.linalg.norm(points[j]-group[-1])>self.cluster_gap:
                if len(group)>=self.min_cluster_points: groups.append(np.asarray(group))
                group=[]
            group.append(points[j])
        if len(group)>=self.min_cluster_points: groups.append(np.asarray(group))
        if not groups:return []
        pose=self.pose.pose.position
        yaw=yaw_of(self.pose.pose.orientation)
        c,s=math.cos(yaw),math.sin(yaw)
        R=np.array([[c,-s],[s,c]])
        offset=np.array([self.lidar_x,self.lidar_y])
        pos=np.array([pose.x,pose.y])
        obstacles=[]
        for group in groups:
            xyz=(group+offset)@R.T+pos
            # Exclude wall clusters: a true surprise obstacle lies in map's known free space.
            clearance=self.clearances(xyz)
            if np.mean(clearance>self.margin+0.04)<0.65:continue
            center=np.mean(xyz,axis=0)
            s_obs,d_obs=self.project(center)
            radius=max(self.min_obstacle_radius,float(np.max(np.linalg.norm(xyz-center,axis=1))))
            obstacles.append((s_obs,d_obs,radius,center))
        return obstacles

    @staticmethod
    def smoothstep(u):
        u=np.clip(u,0,1)
        return 10*u**3-15*u**4+6*u**5

    def offsets(self, s_obs, d_goal, before, after):
        rel=(self.s-s_obs+self.lap/2)%self.lap-self.lap/2
        out=np.zeros(self.n)
        rise=(rel>=-before)&(rel<0)
        fall=(rel>=0)&(rel<=after)
        out[rise]=d_goal*self.smoothstep((rel[rise]+before)/before)
        out[fall]=d_goal*(1-self.smoothstep(rel[fall]/after))
        return out

    def collision_free(self, path, obstacles):
        # Check dense interpolated samples to avoid passing between waypoint samples.
        points=np.vstack((path,path[0]))
        d=np.linalg.norm(np.diff(points,axis=0),axis=1)
        tested=[]
        for i,length in enumerate(d):
            count=max(2,int(np.ceil(length/0.04))+1)
            u=np.linspace(0,1,count)
            tested.append(points[i]+u[:,None]*(points[i+1]-points[i]))
        dense=np.vstack(tested)
        if np.min(self.clearances(dense)) < self.vehicle_half_width+self.margin:
            return False
        for center,radius in obstacles:
            if np.min(np.linalg.norm(dense-center[None,:],axis=1)) < radius+self.vehicle_half_width+self.margin:
                return False
        return True

    def publish(self,xy,valid):
        msg=Path()
        msg.header.frame_id='map'
        msg.header.stamp=self.get_clock().now().to_msg()
        for x,y in xy:
            pt=PoseStamped()
            pt.header=msg.header
            pt.pose.position.x=float(x)
            pt.pose.position.y=float(y)
            pt.pose.orientation.w=1.0
            msg.poses.append(pt)
        self.path_pub.publish(msg)
        flag=Bool();flag.data=bool(valid)
        self.valid_pub.publish(flag)

    def tick(self):
        now=self.get_clock().now().nanoseconds*1e-9
        if (self.pose is None or self.scan is None or now-self.pose_time>self.max_pose_age or
                now-self.scan_time>self.max_scan_age):
            self.publish(self.base_xy,False)
            return
        ego=np.array([self.pose.pose.position.x,self.pose.pose.position.y])
        ego_s,_=self.project(ego)
        obstacles=self.scan_clusters()
        # Only obstacles that intersect the nominal racing corridor ahead.
        blocking=[]
        for so,do,rad,center in obstacles:
            forward=(so-ego_s)%self.lap
            if 0.0<forward<self.lookahead and abs(do)<rad+self.vehicle_half_width+self.margin:
                blocking.append((forward,so,do,rad))
        if not blocking:
            self.publish(self.base_xy,True)
            return
        _,so,do,rad=min(blocking)
        # Use all observed obstacles for collision checking (not only the nearest).
        obs_xy=[]
        for ss,dd,rr,center in obstacles:
            obs_xy.append((center,rr))
        before=min(self.start_distance,max(0.65,((so-ego_s)%self.lap)-0.5))
        after=self.end_distance
        path=None
        for side in (-1,1):
            d_goal=do+side*(rad+self.vehicle_half_width+self.margin)
            offset=self.offsets(so,d_goal,before,after)
            candidate=self.base_xy+offset[:,None]*self.normal
            if self.collision_free(candidate,obs_xy):
                if path is None or abs(d_goal)<path[0]:path=(abs(d_goal),candidate)
        if path is None:
            self.publish(self.base_xy,False)
        else:
            self.publish(path[1],True)


def main():
    rclpy.init()
    node=ObstacleSpliner()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__=='__main__':main()
