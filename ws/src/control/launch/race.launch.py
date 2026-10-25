"""Race a mapped track: Cartographer localization + pure pursuit on its raceline.

    ros2 launch control race.launch.py track:=<name> [speed_scale:=0.5] [max_speed:=0.0] [pose_topic:=/localized_pose]
        [avoid:=false]

Needs tracks/<name>/map.pbstream and raceline.csv (ws/scripts/map_track.sh <name>), and the car at
the pose mapping started from (ws/scripts/race.sh resets the sim car there). /localized_pose is
/tracked_pose once Cartographer has localized (localization_gate), so the car waits for that.
avoid:=true also starts planning's obstacle_spliner_node on the track's raceline and map, and pure
pursuit follows its /local_path around obstacles the map did not have (stopping when there is none).
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from perception.tracks import track_dir


def obstacle_spliner(context):
    if LaunchConfiguration('avoid').perform(context).lower() not in ('true', '1'):
        return []
    d = track_dir(LaunchConfiguration('track').perform(context))
    return [Node(package='planning', executable='obstacle_spliner_node', output='screen', parameters=[{
        'raceline_path': os.path.join(d, 'raceline.csv'),
        'map_yaml': os.path.join(d, 'map.yaml'),
        'pose_topic': LaunchConfiguration('pose_topic'),
    }])]


def generate_launch_description():
    def include(package, launch_file, args):
        return IncludeLaunchDescription(
            PathJoinSubstitution([FindPackageShare(package), 'launch', launch_file]),
            launch_arguments={a: LaunchConfiguration(a) for a in args}.items())

    return LaunchDescription([
        DeclareLaunchArgument('track', description='track name under ws/tracks/'),
        DeclareLaunchArgument('pose_topic', default_value='/localized_pose',
                              description="a localizer's PoseStamped in the map frame"),
        DeclareLaunchArgument('speed_scale', default_value='0.5', description='drive speed = raceline vx * speed_scale'),
        DeclareLaunchArgument('max_speed', default_value='0.0', description='cap on the drive speed [m/s], 0 = no cap'),
        DeclareLaunchArgument('avoid', default_value='false',
                              description='avoid obstacles the map did not have (obstacle_spliner_node)'),
        include('perception', 'localization.launch.py', ['track']),
        include('control', 'pure_pursuit.launch.py', ['track', 'pose_topic', 'speed_scale', 'max_speed', 'avoid']),
        OpaqueFunction(function=obstacle_spliner),
    ])
