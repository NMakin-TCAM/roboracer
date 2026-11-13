"""Drive the raceline round one fake stationary obstacle: our spliner -> pure pursuit.

    ros2 launch planning avoid_graph.launch.py track:=<name>
        [localizer:=cartographer|mcl|external] [obstacle_s:=8.0] [obstacle_d:=0.0]
        [obstacle_width:=0.30] [obstacle_length:=0.50] [speed_scale:=0.5] [max_speed:=0.0]
        [use_sim_time:=true]

Starts:

    localizer                       perception localization.launch.py or mcl.launch.py
                                    -> /localized_pose (see launch_helpers.localizer)
    global_waypoints_publisher_node -> /global_waypoints(_scaled), latched
    frenet_state_node               -> /car_state/odom_frenet
    fake_opponent_node (static)     -> /perception/obstacles: one obstacle at
                                    (obstacle_s, obstacle_d) on the raceline, Frenet
    spliner_node                    -> /planner/avoidance/otwpnts + /planner/avoidance/status
    local_path_node                 -> /local_path + /local_path/valid
    control pure_pursuit.launch.py  avoid:=true: follows /local_path, stops on invalid

Their obstacle_spliner_node is deliberately NOT started (race.launch.py avoid:=true
would start it): it publishes /local_path too. local_path_node refuses to mark a
path valid while another node publishes /local_path.

The obstacle exists only in software. The car swerves round a point the
simulator does not have, so a run shows whether the planner and controller
path works, not that anything was avoided physically.

Not started here, as for race.launch.py: the AutoDRIVE bridge (autodrive_bridge
autodrive_launch.py, in its own shell) and the simulator itself. Unlike
scripts/race.sh this does not reset the simulator, so put the car where mapping
started from first. Speeds are the raceline's times speed_scale: pure pursuit
ignores the spliner's own speed through the manoeuvre.

planning does not declare an exec_depend on control: control already depends on
planning (obstacle_spliner_node), and colcon refuses the cycle. control must be
built and sourced.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

from perception.tracks import track_dir, tracks_dir
from planning.launch_helpers import localizer


def graph(context):
    track = LaunchConfiguration('track').perform(context)
    d = track_dir(track)
    for f in ('raceline.csv', 'centerline.csv'):
        if not os.path.isfile(os.path.join(d, f)):
            raise RuntimeError(f'{os.path.join(d, f)} not found; map and plan the track first '
                               f'(scripts/map_track.sh {track})')

    track_params = {'tracks_dir': tracks_dir(), 'track_name': track, 'reference': 'raceline'}
    flt = lambda name: ParameterValue(LaunchConfiguration(name), value_type=float)  # noqa: E731

    def node(executable, extra=None):
        return Node(package='planning', executable=executable, output='screen',
                    parameters=[{**track_params, **(extra or {})}])

    local_path_params = {k: v for k, v in track_params.items() if k != 'reference'}
    return [
        node('global_waypoints_publisher_node'),
        node('frenet_state_node'),
        node('fake_opponent_node', {
            'mode': 'static', 's0': flt('obstacle_s'), 'd': flt('obstacle_d'),
            'width': flt('obstacle_width'), 'length': flt('obstacle_length'),
            'marker_topic': '/fake_opponent/marker',
        }),
        node('spliner_node'),
        Node(package='planning', executable='local_path_node', output='screen',
             parameters=[local_path_params]),
        IncludeLaunchDescription(
            PathJoinSubstitution([FindPackageShare('control'), 'launch', 'pure_pursuit.launch.py']),
            launch_arguments={
                'track': track, 'avoid': 'true', 'pose_topic': '/localized_pose',
                'speed_scale': LaunchConfiguration('speed_scale'),
                'max_speed': LaunchConfiguration('max_speed'),
                'use_sim_time': LaunchConfiguration('use_sim_time'),
            }.items()),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('track', description='track name under ws/tracks/'),
        DeclareLaunchArgument('localizer', default_value='cartographer',
                              description='cartographer | mcl | external (one already running)'),
        DeclareLaunchArgument('obstacle_s', default_value='8.0',
                              description='fake obstacle centre, m along the raceline'),
        DeclareLaunchArgument('obstacle_d', default_value='0.0',
                              description='fake obstacle centre, m left of the raceline'),
        DeclareLaunchArgument('obstacle_width', default_value='0.30', description='m, across'),
        DeclareLaunchArgument('obstacle_length', default_value='0.50', description='m, along'),
        DeclareLaunchArgument('speed_scale', default_value='0.5',
                              description='pure pursuit: drive speed = raceline vx * speed_scale'),
        DeclareLaunchArgument('max_speed', default_value='0.0',
                              description='pure pursuit: speed cap, m/s, 0 = none'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        SetParameter(name='use_sim_time',
                     value=ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)),
        OpaqueFunction(function=localizer),
        OpaqueFunction(function=graph),
    ])
