"""Follow the gap with disparity extension, a second no-map fallback (reactive /scan -> /drive):

    ros2 launch control disparity_gap.launch.py [speed:=1.0]

speed is a cap: the node slows below it when the road ahead is short or the turn is tight.
Needs only the bridge (or the car's drivers). Run it instead of follow_the_gap.launch.py or race.sh,
never alongside them: they all publish /drive.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('speed', default_value='1.0', description='speed cap [m/s]'),
        Node(
            package='control',
            executable='disparity_gap_node',
            output='screen',
            parameters=[PathJoinSubstitution([FindPackageShare('control'), 'config', 'vehicle.yaml']), {
                'max_speed': ParameterValue(LaunchConfiguration('speed'), value_type=float),
            }],
        ),
    ])
