"""ros2 launch autodrive_bridge autodrive_launch.py

fast_bridge: AutoDRIVE app <-> ROS 2 over Socket.IO on :4567 (/autodrive/roboracer_1/*)
adapter:     sim sensors -> /scan, /sensors/imu/raw, /ground_truth/odom;  /drive -> throttle/steering
wheel_odom:  rear-wheel encoders -> /odom + TF odom -> base_link (sim stand-in for vesc_to_odom)

Teleop runs in its own terminal: ros2 run autodrive_bridge teleop_keyboard
"""
from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.actions import DeclareLaunchArgument

def generate_launch_description():

    publish_restricted_topics = LaunchConfiguration('publish_restricted_topics')
    publish_camera = LaunchConfiguration('publish_camera')

    bridge = Node(package='autodrive_bridge', executable='fast_bridge', name='fast_bridge', output='screen', 
                  parameters=[{'publish_restricted_topics': publish_restricted_topics,'publish_camera': publish_camera,}])
    
    vehicle = PathJoinSubstitution([FindPackageShare('control'), 'config', 'vehicle.yaml'])
    config = PathJoinSubstitution([FindPackageShare('autodrive_bridge'), 'config', 'adapter.yaml'])
    adapter = Node(package='autodrive_bridge', executable='adapter', name='adapter', output='screen',
                  parameters=[vehicle, config])

    wheel_odom = Node(package='autodrive_bridge', executable='wheel_odom', name='wheel_odom', output='screen',
                  parameters=[vehicle, config])

    foxglove = Node(package='foxglove_bridge', executable='foxglove_bridge', name='foxglove_bridge')

    return LaunchDescription([
        DeclareLaunchArgument('publish_restricted_topics',default_value='true'),
        # nothing in the stack uses the camera, and decoding it every frame slows the sim loop
        DeclareLaunchArgument('publish_camera',default_value='false'),
        bridge,
        adapter,
        wheel_odom,
        foxglove,        
    ])
