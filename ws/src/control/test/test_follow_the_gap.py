import numpy as np
import pytest
import rclpy
from rclpy.parameter import Parameter

from control.follow_the_gap_node import ReactiveFollowGap


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