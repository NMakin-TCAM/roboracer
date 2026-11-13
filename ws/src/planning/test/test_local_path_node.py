"""local_path_node's own checks, in process: staleness by arrival and the single-publisher guard."""
import numpy as np
import pytest

rclpy = pytest.importorskip('rclpy')
pytest.importorskip('f1t_msgs.msg')
from nav_msgs.msg import Path  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402

from planning.local_path_node import LocalPathNode  # noqa: E402
from planning.raceline import write_raceline  # noqa: E402
from planning.spliner import CLEAR, NO_ROOM  # noqa: E402

R, N = 5.0, 300


@pytest.fixture(scope='module')
def tracks(tmp_path_factory):
    root = tmp_path_factory.mktemp('tracks')
    d = root / 'ring'
    d.mkdir()
    th = np.linspace(0, 2 * np.pi, N, endpoint=False)
    xy = np.column_stack((R * np.cos(th), R * np.sin(th)))
    np.savetxt(d / 'centerline.csv', np.column_stack((xy, np.full(N, 1.0), np.full(N, 1.0))),
               delimiter=',', header='x_m,y_m,w_tr_right_m,w_tr_left_m')
    traj = np.column_stack((th * R, xy, th, np.full(N, 1 / R), np.full(N, 3.0), np.zeros(N)))
    traj = np.vstack((traj, traj[0]))
    traj[-1, 0] = 2 * np.pi * R
    write_raceline(str(d / 'raceline.csv'), traj, {'method': 'mintime', 'clean': True})
    return root


@pytest.fixture(scope='module')
def ros():
    rclpy.init(domain_id=59)     # never a live stack's domain
    yield
    rclpy.shutdown()


@pytest.fixture
def node(ros, tracks):
    n = LocalPathNode(parameter_overrides=[Parameter('tracks_dir', value=str(tracks)),
                                           Parameter('track_name', value='ring')])
    t = [100.0]
    n.now = lambda: t[0]          # a clock the test drives
    yield n, t
    n.destroy_node()


def feed(n, now, status, ego=True):
    stamp = (int(now), int((now % 1) * 1e9))
    n.pairer.add_trajectory(stamp, [], [], now)
    n.pairer.add_status(stamp, status, status, now)
    if ego:
        n.ego_received = now


def test_publishes_one_pose_per_raceline_row(node):
    n, t = node
    assert len(n.xy) == N


def test_clear_and_fresh_is_valid(node):
    n, t = node
    feed(n, 100.0, CLEAR)
    assert n.decision().valid


def test_no_room_is_invalid(node):
    n, t = node
    feed(n, 100.0, NO_ROOM)
    assert not n.decision().valid


def test_goes_invalid_when_the_planner_stops_publishing(node):
    n, t = node
    feed(n, 100.0, CLEAR)
    n.ego_received = 100.3
    t[0] = 100.3
    dec = n.decision()
    assert not dec.valid and 'planner output' in dec.reason


def test_goes_invalid_when_the_ego_state_stops(node):
    n, t = node
    feed(n, 100.0, CLEAR, ego=False)
    n.ego_received = 99.7
    dec = n.decision()
    assert not dec.valid and 'ego state' in dec.reason


def test_refuses_to_compete_with_another_local_path_publisher(node):
    """obstacle_spliner_node also publishes /local_path; two planners must not take turns."""
    n, t = node
    feed(n, 100.0, CLEAR)
    assert n.decision().valid
    other = rclpy.create_node('pretend_obstacle_spliner')
    try:
        other.create_publisher(Path, '/local_path', 10)
        dec = n.decision()
        assert not dec.valid and 'other node' in dec.reason
    finally:
        other.destroy_node()


def test_tick_survives_going_valid_and_invalid_and_back(node):
    """rclpy rejected a severity change at one logging call site, which crashed this node on
    its first valid tick in the launch test. tick() through every transition."""
    n, t = node
    n.tick()                                   # invalid: no planner output yet
    feed(n, 100.0, CLEAR)
    n.tick()                                   # valid
    assert n.last.valid
    t[0] = 100.05                              # each cycle carries a newer stamp
    feed(n, 100.05, NO_ROOM)
    n.tick()                                   # invalid
    assert not n.last.valid
    t[0] = 100.1
    feed(n, 100.1, CLEAR)
    n.tick()
    assert n.last.valid
