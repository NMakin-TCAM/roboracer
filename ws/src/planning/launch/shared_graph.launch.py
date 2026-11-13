"""Bring up the shared overtaking graph on a mapped track.

    ros2 launch planning shared_graph.launch.py track:=<name>
        [localizer:=cartographer|mcl|external] [state:=GB_TRACK|TRAILING|OVERTAKE]
        [use_sim_time:=true] [opponent_mode:=constant_speed] [opponent_speed:=2.0]
        [opponent_s0:=5.0] [opponent_d:=0.0] [opponent_seed:=0] [opponent_max_range:=10.0]

Starts, in one launch:

    localizer                       perception localization.launch.py (Cartographer +
                                    localization_gate) or mcl.launch.py -> /localized_pose
    global_waypoints_publisher_node tracks/<track>/raceline.csv -> /global_waypoints,
                                    /global_waypoints_scaled (latched)
    state_machine_stub_node         fixed `state` -> /state_machine (a stand-in)
    frenet_state_node               /localized_pose + /odom -> /car_state/odom_frenet
    fake_opponent_node              -> /perception/obstacles (its Marker moved to
                                    /fake_opponent/marker: opponent_trajectory_node
                                    publishes a MarkerArray on /opponent_marker, and two
                                    types on one topic breaks every subscriber)
    opponent_trajectory_node        -> /proj_opponent_trajectory
    predictor_opponent_trajectory_node -> /opponent_trajectory
    update_waypoints_node           -> /global_waypoints_updated
    collision_prediction_node       -> /collision_prediction/obstacles, .../force_trailing

THE LOCALIZER IS REQUIRED. frenet_state_node consumes /localized_pose only, and
every planner here waits on its output. There is no 'none' option and no
fallback to /tracked_pose (AGENTS.md: racing consumes the gated pose).
cartographer needs tracks/<track>/map.pbstream, mcl needs map.yaml, and both
need sensors: the AutoDRIVE bridge (autodrive_bridge autodrive_launch.py) or the
car. localizer:=external starts none, for when one is already running, e.g. under
race.launch.py, or when a test publishes /localized_pose itself. Never run two.
With no pose arriving, frenet_state_node logs an ERROR every 5 s saying so.

NOTHING HERE DRIVES. No node in this graph publishes /drive; the car stays where
it is unless a controller is started separately (race.launch.py, with
localizer:=external here so the localizer is not started twice).
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

from perception.tracks import track_dir, tracks_dir

LOCALIZERS = {
    'cartographer': 'localization.launch.py',
    'mcl': 'mcl.launch.py',
}


def localizer(context):
    which = LaunchConfiguration('localizer').perform(context)
    if which == 'external':
        return [LogInfo(msg='localizer:=external: this launch starts NO localizer. Something '
                            'else must publish /localized_pose (map frame), or nothing '
                            'downstream of frenet_state_node will ever run.')]
    if which not in LOCALIZERS:
        raise RuntimeError(f'localizer must be cartographer, mcl or external, got {which!r}. '
                           f'A localizer is required: frenet_state_node consumes '
                           f'/localized_pose and does not fall back to /tracked_pose.')
    return [IncludeLaunchDescription(
        PathJoinSubstitution([FindPackageShare('perception'), 'launch', LOCALIZERS[which]]),
        launch_arguments={'track': LaunchConfiguration('track'),
                          'use_sim_time': LaunchConfiguration('use_sim_time')}.items())]


def graph(context):
    track = LaunchConfiguration('track').perform(context)
    d = track_dir(track)
    for f in ('raceline.csv', 'centerline.csv'):
        if not os.path.isfile(os.path.join(d, f)):
            raise RuntimeError(f'{os.path.join(d, f)} not found; map and plan the track first '
                               f'(scripts/map_track.sh {track})')

    track_params = {'tracks_dir': tracks_dir(), 'track_name': track, 'reference': 'raceline'}

    def node(executable, extra=None):
        return Node(package='planning', executable=executable, output='screen',
                    parameters=[{**track_params, **(extra or {})}])

    def plain(executable, params):
        return Node(package='planning', executable=executable, output='screen',
                    parameters=[params])

    opponent = {
        'mode': LaunchConfiguration('opponent_mode'),
        'speed': ParameterValue(LaunchConfiguration('opponent_speed'), value_type=float),
        's0': ParameterValue(LaunchConfiguration('opponent_s0'), value_type=float),
        'd': ParameterValue(LaunchConfiguration('opponent_d'), value_type=float),
        'seed': ParameterValue(LaunchConfiguration('opponent_seed'), value_type=int),
        'max_range': ParameterValue(LaunchConfiguration('opponent_max_range'), value_type=float),
        'marker_topic': '/fake_opponent/marker',
    }
    return [
        node('global_waypoints_publisher_node'),
        plain('state_machine_stub_node', {'state': LaunchConfiguration('state')}),
        node('frenet_state_node'),
        node('fake_opponent_node', opponent),
        node('opponent_trajectory_node'),
        node('predictor_opponent_trajectory_node'),
        # update_waypoints takes no track: it works on the waypoints it receives.
        plain('update_waypoints_node', {}),
        node('collision_prediction_node'),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('track', description='track name under ws/tracks/'),
        DeclareLaunchArgument('localizer', default_value='cartographer',
                              description='cartographer | mcl | external (one already running)'),
        DeclareLaunchArgument('state', default_value='GB_TRACK',
                              description='the stand-in state machine\'s fixed state: '
                                          'GB_TRACK | TRAILING | OVERTAKE'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('opponent_mode', default_value='constant_speed',
                              description='fake_opponent_node mode: static | constant_speed | '
                                          'weaving | replay'),
        DeclareLaunchArgument('opponent_speed', default_value='2.0'),
        DeclareLaunchArgument('opponent_s0', default_value='5.0',
                              description='opponent start, m along the raceline'),
        DeclareLaunchArgument('opponent_d', default_value='0.0',
                              description='opponent offset, m left of the raceline'),
        DeclareLaunchArgument('opponent_seed', default_value='0'),
        DeclareLaunchArgument('opponent_max_range', default_value='10.0',
                              description='fake sensor range ahead of the ego, m'),
        SetParameter(name='use_sim_time',
                     value=ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)),
        OpaqueFunction(function=localizer),
        OpaqueFunction(function=graph),
    ])
