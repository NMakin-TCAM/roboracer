"""ros2 launch perception localization.launch.py track:=<name>

Cartographer pure localization against tracks/<name>/map.pbstream (frozen): TF map -> odom,
/tracked_pose, and /map for visualization.

Assumption: the car starts from the pose mapping started from (the sim reset pose), so the
localization trajectory starts at the map origin.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def tracks_dir():
    return os.environ.get('TRACKS_DIR', os.path.expanduser('~/ws/tracks'))


def nodes(context):
    track = LaunchConfiguration('track').perform(context)
    pbstream = os.path.join(tracks_dir(), track, 'map.pbstream')
    if not os.path.isfile(pbstream):
        raise RuntimeError(f'{pbstream} not found; map the track first (save_map {track})')

    cartographer = Node(package='cartographer_ros', executable='cartographer_node', name='cartographer_node',
                        output='screen',
                        arguments=['-configuration_directory',
                                   PathJoinSubstitution([FindPackageShare('perception'), 'config']),
                                   '-configuration_basename', 'localization.lua',
                                   '-load_state_filename', pbstream,
                                   '-load_frozen_state', 'true'])

    grid = Node(package='cartographer_ros', executable='cartographer_occupancy_grid_node',
                name='cartographer_occupancy_grid_node', output='screen',
                arguments=['-resolution', '0.05', '-publish_period_sec', '1.0'])
    return [cartographer, grid]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('track', description='track name, loads tracks/<track>/map.pbstream'),
        OpaqueFunction(function=nodes),
    ])
