import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'planning'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.ini')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='SIGRobotics',
    maintainer_email='todo@todo.todo',
    description='Unseen-track planning: centerline from the Cartographer map + TUM minimum-time raceline.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'build_raceline = planning.build_raceline:main',
            'obstacle_spliner_node = planning.obstacle_spliner_node:main',
            'frenet_state_node = planning.frenet_state_node:main',
            'fake_opponent_node = planning.fake_opponent_node:main',
            'update_waypoints_node = planning.update_waypoints_node:main',
            'collision_prediction_node = planning.collision_prediction_node:main',
            'opponent_trajectory_node = planning.opponent_trajectory_node:main',
            'predictor_opponent_trajectory_node = planning.predictor_opponent_trajectory_node:main',
            'global_waypoints_publisher_node = planning.global_waypoints_publisher_node:main',
            'state_machine_stub_node = planning.state_machine_stub_node:main',
        ],
    },
)
