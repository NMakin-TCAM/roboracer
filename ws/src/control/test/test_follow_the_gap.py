import os
from types import SimpleNamespace

import numpy as np
import pytest
import rclpy
import yaml
from rclpy.parameter import Parameter
from sensor_msgs.msg import LaserScan

from control.follow_the_gap_node import ReactiveFollowGap

HERE = os.path.dirname(os.path.abspath(__file__))
VEHICLE = os.path.join(HERE, '..', 'config', 'vehicle.yaml')


@pytest.fixture
def make_node():
    rclpy.init()
    nodes = []

    def create(**parameters):
        node = ReactiveFollowGap(
            parameter_overrides=[
                Parameter(name, value=value)
                for name, value in parameters.items()
            ]
        )
        nodes.append(node)
        return node

    try:
        yield create
    finally:
        for node in nodes:
            node.destroy_node()
        rclpy.shutdown()


def test_preprocess_caps_ranges(make_node):
    node = make_node(max_range=5.0)
    ranges = np.array([0.5, 2.0, 8.0], dtype=np.float32)

    result = node.preprocess_lidar(ranges)

    np.testing.assert_allclose(result, [0.5, 2.0, 5.0])

def test_params(make_node):
    node = make_node(max_range=5.0, alpha=0.2, speed=2.0, car_width=1.)
    assert node.MAX_RANGE == 5.0
    assert node.ALPHA == 0.2
    assert node.SPEED == 2.0
    assert node.CAR_WIDTH == 1.

def test_preprocess_smooths_previous_scan(make_node):
    node = make_node(alpha=0.6)
    node.preprocess_lidar(np.array([1.0], dtype=np.float32))

    result = node.preprocess_lidar(
        np.array([3.0], dtype=np.float32)
    )

    np.testing.assert_allclose(result, [2.2])

def test_max_gap(make_node):
    node = make_node()

    tests = [
        np.array([1., 2., 3., 0, 0, 0, 1., 2., 3., 4]),
        np.array([0, 0, 0, 1., 2., 3., 4]),
        np.array([1., 2., 3., 4, 0, 0, 0]),
        np.array([1., 2., 3., 4]),
    ]

    ans = np.array([
        [6, 9],
        [3, 6],
        [0, 3],
        [0, 3],
    ])

    for i in range(len(tests)):
        ranges = tests[i]
        result = node.find_max_gap(ranges)
        np.testing.assert_allclose(result, ans[i])

def make_scan(ranges):
    scan = LaserScan()
    scan.angle_min = -2.35619
    scan.angle_max = 2.35619
    scan.angle_increment = 0.004363323
    scan.range_min = 0.06
    scan.range_max = 10.0
    scan.ranges = [float(r) for r in ranges]
    return scan


def test_callback_clips_steering(make_node):
    node = make_node()
    msgs = []
    node.drive_pub = SimpleNamespace(publish=msgs.append)
    ranges = np.full(1081, 2.5)
    ranges[0] = 2.4      # closest point, far right
    ranges[900:] = 5.0   # only free space is far left, past the steering lock

    node.lidar_callback(make_scan(ranges))

    proc = node.lidar_processor.process(make_scan(ranges))
    start, end = node.find_max_gap(proc)
    assert node.get_angle(make_scan(ranges), node.find_best_point(start, end, proc)) > 0.524
    assert msgs[0].drive.speed == node.SPEED
    assert msgs[0].drive.steering_angle == pytest.approx(0.524)


def test_callback_stops_on_all_nan_scan(make_node):
    node = make_node()
    msgs = []
    node.drive_pub = SimpleNamespace(publish=msgs.append)

    node.lidar_callback(make_scan(np.full(1081, np.nan)))

    assert msgs[0].drive.speed == 0.0
    assert msgs[0].drive.steering_angle == 0.0


def test_max_steering_angle_default_matches_vehicle_yaml(make_node):
    with open(VEHICLE) as f:
        lock = yaml.safe_load(f)['/**']['ros__parameters']['max_steering_angle']
    assert make_node().MAX_STEERING_ANGLE == pytest.approx(lock)
