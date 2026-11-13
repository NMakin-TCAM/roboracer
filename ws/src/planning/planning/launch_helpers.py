"""Launch pieces shared by planning's graph launches (shared_graph, avoid_graph)."""

from launch.actions import IncludeLaunchDescription, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

LOCALIZERS = {
    'cartographer': 'localization.launch.py',
    'mcl': 'mcl.launch.py',
}


def localizer(context):
    """The localizer named by the `localizer` launch argument, which is required.

    cartographer | mcl start perception's launch for it; external starts none, for when one
    is already running or a test publishes /localized_pose. There is no 'none' and no
    fallback to /tracked_pose (AGENTS.md: racing consumes the gated pose).
    """
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
