"""Race a mapped track: localization + pure pursuit on its raceline.

    ros2 launch control race.launch.py track:=<name> [speed_scale:=0.5] [max_speed:=0.0] [pose_topic:=/localized_pose]
        [localizer:=cartographer|mcl]

Needs tracks/<name>/map.pbstream (map.yaml for localizer:=mcl) and raceline.csv (ws/scripts/map_track.sh
<name>), and the car at the pose mapping started from (ws/scripts/race.sh resets the sim car there).
/localized_pose is published once the localizer has localized (Cartographer: localization_gate; mcl: the
particle cloud has converged), so the car waits for that. localizer:=mcl is the particle filter
(perception mcl.launch.py); the default is Cartographer pure localization.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    def include(package, launch_file, args, condition=None):
        return IncludeLaunchDescription(
            PathJoinSubstitution([FindPackageShare(package), 'launch', launch_file]),
            launch_arguments={a: LaunchConfiguration(a) for a in args}.items(), condition=condition)

    use_mcl = PythonExpression(["'", LaunchConfiguration('localizer'), "' == 'mcl'"])

    return LaunchDescription([
        DeclareLaunchArgument('track', description='track name under ws/tracks/'),
        DeclareLaunchArgument('pose_topic', default_value='/localized_pose',
                              description="a localizer's PoseStamped in the map frame"),
        DeclareLaunchArgument('speed_scale', default_value='0.5', description='drive speed = raceline vx * speed_scale'),
        DeclareLaunchArgument('max_speed', default_value='0.0', description='cap on the drive speed [m/s], 0 = no cap'),
        DeclareLaunchArgument('localizer', default_value='cartographer',
                              description='cartographer: pure localization on map.pbstream; mcl: particle filter on map.yaml'),
        include('perception', 'localization.launch.py', ['track'], UnlessCondition(use_mcl)),
        include('perception', 'mcl.launch.py', ['track'], IfCondition(use_mcl)),
        include('control', 'pure_pursuit.launch.py', ['track', 'pose_topic', 'speed_scale', 'max_speed']),
    ])
