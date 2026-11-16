"""Drive the predictive chain round a moving opponent: the dense GP's collision regions ->
our spliner -> pure pursuit.

    ros2 launch planning predictive_race.launch.py track:=<name>
        [localizer:=cartographer|mcl|external] [state:=TRAILING|GB_TRACK|OVERTAKE]
        [evasion_dist:=0.25] [trail_gap:=2.0] [speed_scale:=0.5] [max_speed:=0.0]
        [opponent_mode:=weaving] [opponent_speed:=2.0] [opponent_s0:=5.0] [opponent_d:=0.0]
        [opponent_seed:=0] [opponent_max_range:=10.0] [opponent_truth_topic:=]
        [use_sim_time:=true]

Starts:

    shared_graph.launch.py gp:=dense  the localizer, global waypoints, the state machine
                                      stand-in, frenet_state_node, fake_opponent_node,
                                      obstacle_tracker_node, opponent_trajectory_node,
                                      dense_gp_opp_traj_node, predictor_opponent_trajectory,
                                      update_waypoints_node, collision_prediction_node
    spliner_node                      on /collision_prediction/obstacles, trailing on the
                                      tracked /perception/obstacles until the regions can be
                                      trusted (spliner TRAILING), at evasion_dist
    local_path_node                   -> /local_path + /local_path/valid
    control pure_pursuit.launch.py    avoid:=true: follows /local_path, stops when invalid

A SIBLING OF avoid_graph.launch.py, NOT AN OPTION ON IT. avoid_graph is the validated
stationary graph: a static fake obstacle straight into the spliner on
/perception/obstacles, no GP and no collision_prediction, with its own launch test. An
option there would swap half its nodes and the meaning of its arguments behind one flag.

state defaults to TRAILING, not shared_graph's GB_TRACK. With the stand-in state machine
fixed at GB_TRACK, update_waypoints teaches collision_prediction the ego's own past speed,
so after trailing it predicts no catch while pure pursuit drives the raceline speed
(docs/forzaeth-quirks.md section 8). That needs a real state machine; this only avoids it.

The opponent exists only in software (fake_opponent_node), as the obstacle does in
avoid_graph: a run shows whether the chain drives round a moving car the simulator does
not have, not that anything was avoided physically. opponent_truth_topic is for
evaluation only; nothing here subscribes to it.

Not started, as for race.launch.py: the AutoDRIVE bridge (autodrive_bridge
autodrive_launch.py, in its own shell) and the simulator. This does not reset the
simulator: reset it first (scripts/common.sh reset_sim_car) so the car starts where
mapping did. planning does not exec_depend on control (control depends on planning);
control must be built and sourced.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

from perception.tracks import tracks_dir

PASSED_TO_SHARED_GRAPH = ('track', 'localizer', 'state', 'use_sim_time', 'opponent_mode',
                          'opponent_speed', 'opponent_s0', 'opponent_d', 'opponent_seed',
                          'opponent_max_range', 'opponent_truth_topic')


def chain(context):
    track = LaunchConfiguration('track').perform(context)
    flt = lambda name: ParameterValue(LaunchConfiguration(name), value_type=float)  # noqa: E731
    params = {'tracks_dir': tracks_dir(), 'track_name': track}
    return [
        IncludeLaunchDescription(
            PathJoinSubstitution([FindPackageShare('planning'), 'launch',
                                  'shared_graph.launch.py']),
            launch_arguments={**{k: LaunchConfiguration(k) for k in PASSED_TO_SHARED_GRAPH},
                              'gp': 'dense', 'tracker': 'true'}.items()),
        Node(package='planning', executable='spliner_node', output='screen', parameters=[{
            **params, 'reference': 'raceline',
            'obs_topic': '/collision_prediction/obstacles',
            'trail_obs_topic': '/perception/obstacles',
            'evasion_dist': flt('evasion_dist'), 'trail_gap': flt('trail_gap'),
        }]),
        Node(package='planning', executable='local_path_node', output='screen',
             parameters=[params]),
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
        DeclareLaunchArgument('state', default_value='TRAILING',
                              description='the stand-in state machine\'s fixed state (see above)'),
        DeclareLaunchArgument('evasion_dist', default_value='0.25',
                              description='spliner clearance, m (docs/reports/clearance_sweep)'),
        DeclareLaunchArgument('trail_gap', default_value='2.0',
                              description='m, nose to a moving obstacle before trailing stops'),
        DeclareLaunchArgument('speed_scale', default_value='0.5',
                              description='pure pursuit: drive speed = raceline vx * speed_scale'),
        DeclareLaunchArgument('max_speed', default_value='0.0',
                              description='pure pursuit: speed cap, m/s, 0 = none'),
        DeclareLaunchArgument('opponent_mode', default_value='weaving'),
        DeclareLaunchArgument('opponent_speed', default_value='2.0'),
        DeclareLaunchArgument('opponent_s0', default_value='5.0'),
        DeclareLaunchArgument('opponent_d', default_value='0.0'),
        DeclareLaunchArgument('opponent_seed', default_value='0'),
        DeclareLaunchArgument('opponent_max_range', default_value='10.0'),
        DeclareLaunchArgument('opponent_truth_topic', default_value='',
                              description='evaluation only: the fake opponent\'s true pose'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        SetParameter(name='use_sim_time',
                     value=ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)),
        OpaqueFunction(function=chain),
    ])
