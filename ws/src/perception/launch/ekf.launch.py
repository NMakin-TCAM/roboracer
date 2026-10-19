# Copyright 2018 Open Source Robotics Foundation, Inc.
# Copyright 2019 Samsung Research America
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():

    ekf_config_file_accl = PathJoinSubstitution([FindPackageShare('perception'), 'config', 'ekf_config_accl.yaml'])

    accl_ekf_node = Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node_accl',
            output='screen',
            respawn=False,
            parameters=[ekf_config_file_accl],
            remappings=[
                ('odometry/filtered', 'odometry/filtered_accl'),
                # #('accel/filtered', 'accel/filtered_custom')
            ]
        )

    corrector = Node(
        package='perception',
        name='ekf_corrector',
        executable='corrector',
        output='screen',
    )

    return LaunchDescription([
        accl_ekf_node,
        corrector,

    ])