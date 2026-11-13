"""obstacle_tracker_node's plumbing, in process: one output per input, stamped as the input,
fields copied from the tracker, ego s used for ordering. The tracking itself is tested ROS-free
in test_obstacle_tracker.py."""
import numpy as np
import pytest

rclpy = pytest.importorskip('rclpy')
pytest.importorskip('f1t_msgs.msg')
from f1t_msgs.msg import Obstacle, ObstacleArray  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402

from planning.obstacle_tracker_node import ObstacleTrackerNode  # noqa: E402
from planning.raceline import write_raceline  # noqa: E402

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
    rclpy.init(domain_id=61)     # never a live stack's domain
    yield
    rclpy.shutdown()


@pytest.fixture
def node(ros, tracks):
    n = ObstacleTrackerNode(parameter_overrides=[Parameter('tracks_dir', value=str(tracks)),
                                                 Parameter('track_name', value='ring'),
                                                 Parameter('confirm_frames', value=2)])
    sent = []
    n.pub.publish = sent.append
    yield n, sent
    n.destroy_node()


def detections(t, positions):
    msg = ObstacleArray()
    msg.header.stamp.sec, msg.header.stamp.nanosec = int(t), int(round((t % 1) * 1e9))
    msg.header.frame_id = 'frenet/raceline'
    for s, d in positions:
        o = Obstacle(s_center=s, d_center=d, s_start=s - 0.25, s_end=s + 0.25,
                     d_right=d - 0.15, d_left=d + 0.15, size=0.58,
                     vs=99.0, vd=99.0, is_static=True, is_visible=True, id=42)
        msg.obstacles.append(o)
    return msg


def test_parameters_reach_the_tracker(node):
    n, _ = node
    assert n.tracker.p.confirm_frames == 2
    assert n.tracker.lap == pytest.approx(2 * np.pi * R, rel=1e-3)


def test_one_output_per_input_with_the_input_header(node):
    n, sent = node
    for k in range(5):
        n.detections_cb(detections(10.0 + 0.025 * k, [(5.0 + 0.05 * k, 0.1)]))
    assert len(sent) == 5
    assert sent[0].obstacles == []                    # tentative: not published
    last = sent[-1]
    assert last.header.frame_id == 'frenet/raceline'
    assert last.header.stamp.sec == 10 and last.header.stamp.nanosec == 100_000_000
    o = last.obstacles[0]
    assert o.id == 0 and o.is_visible and not o.is_static
    assert o.vs == pytest.approx(2.0, abs=0.3)        # 0.05 m per 25 ms; the 99 sent is ignored
    assert o.s_var >= 1e-6 and o.vs_var >= 1e-6


def test_an_empty_input_still_publishes(node):
    n, sent = node
    n.detections_cb(detections(10.0, []))
    assert len(sent) == 1 and sent[0].obstacles == []


def test_ego_s_orders_the_dynamic_obstacles(node):
    n, sent = node
    ego = Odometry()
    ego.pose.pose.position.x = 20.0
    n.ego_cb(ego)
    for k in range(30):
        t = 10.0 + 0.025 * k
        n.detections_cb(detections(t, [(10.0 + 2 * 0.025 * k, 0.3), (25.0 + 2 * 0.025 * k, -0.3)]))
    out = sent[-1].obstacles
    assert len(out) == 2
    assert out[0].s_center > 20.0                     # the one just ahead of the ego comes first
