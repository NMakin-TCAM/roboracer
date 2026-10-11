"""Pure pursuit on a generated raceline:

    ros2 launch control pure_pursuit.launch.py track:=<name> [pose_topic:=/tracked_pose] [speed_scale:=0.5]
        [max_speed:=0.0]

Follows $TRACKS_DIR/<name>/raceline.csv (default ~/ws/tracks). Needs localization publishing pose_topic.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('track', description='track name under ws/tracks/'),
        DeclareLaunchArgument('pose_topic', default_value='/tracked_pose',
                              description='geometry_msgs/PoseStamped; /ground_truth/pose for debugging'),
        DeclareLaunchArgument('speed_scale', default_value='0.5',
                              description='drive speed = raceline vx * speed_scale'),
        DeclareLaunchArgument('max_speed', default_value='0.0',
                              description='cap on the drive speed [m/s], 0 = no cap'),
        Node(
            package='control',
            executable='pure_pursuit_node',
            output='screen',
            parameters=[{
                'track': ParameterValue(LaunchConfiguration('track'), value_type=str),
                'pose_topic': LaunchConfiguration('pose_topic'),
                'speed_scale': ParameterValue(LaunchConfiguration('speed_scale'), value_type=float),
                'max_speed': ParameterValue(LaunchConfiguration('max_speed'), value_type=float),
            }],
        ),
    ])
