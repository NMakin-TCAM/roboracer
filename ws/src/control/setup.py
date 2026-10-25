import os
from glob import glob
from setuptools import setup

package_name = 'control'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zzangupenn, Hongrui Zheng',
    maintainer_email='zzang@seas.upenn.edu, billyzheng.bz@gmail.com',
    description='Pure pursuit on the generated raceline, follow the gap, and the race launch file.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pure_pursuit_node = control.pure_pursuit_node:main',
            'follow_the_gap_node = control.follow_the_gap_node:main',
            'disparity_gap_node = control.disparity_gap_node:main',
            'wall_follow_node = control.wall_follow_node:main',
        ],
    },
)
