import os

# Domain 97, so test nodes never talk to a running stack (default domain).
os.environ['ROS_DOMAIN_ID'] = '97'

import pytest
import rclpy


@pytest.fixture(scope='session', autouse=True)
def ros():
    rclpy.init(args=['--ros-args',
                     '-p', 'wheelbase:=0.33', '-p', 'max_steering_angle:=0.4', '-p', 'publish_tf:=false',
                     '-p', 'publish_restricted_topics:=false', '-p', 'publish_camera:=false'],
               domain_id=97)
    yield
    rclpy.try_shutdown()
