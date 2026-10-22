"""save_map: settle wait after /finish_trajectory (mocked services), map.yaml thresholds, TRACKS_DIR."""
import os

import numpy as np
import pytest
import rclpy
from cartographer_ros_msgs.msg import StatusCode
from cartographer_ros_msgs.srv import TrajectoryQuery
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid

from perception.save_map import SaveMap
from perception.tracks import tracks_dir


@pytest.fixture(scope='module')
def node():
    rclpy.init()
    n = SaveMap()
    n.poll_period = 0.01
    n.settle_time = 0.1
    n.timeout = 1.0
    yield n
    n.destroy_node()
    rclpy.try_shutdown()


def response(xs):
    res = TrajectoryQuery.Response()
    res.status.code = StatusCode.OK
    for x in xs:
        p = PoseStamped()
        p.pose.position.x = float(x)
        p.pose.orientation.w = 1.0
        res.trajectory.append(p)
    return res


def script(node, monkeypatch, answers):
    """Mock call(): returns answers in order, then repeats the last one."""
    calls = []

    def call(srv_type, name, request):
        calls.append(name)
        return response(answers[min(len(calls) - 1, len(answers) - 1)])
    monkeypatch.setattr(node, 'call', call)
    return calls


def test_waits_for_first_change_then_quiet(node, monkeypatch):
    # unchanged for a while (optimization still running), then it lands, then quiet
    answers = [[0, 1]] * 30 + [[0, 1.1]]
    calls = script(node, monkeypatch, answers)
    res = node.wait_settled(0, finished=True)
    assert [p.pose.position.x for p in res.trajectory] == [0, 1.1]
    assert len(calls) > 31


def test_length_change_counts(node, monkeypatch):
    script(node, monkeypatch, [[0, 1]] * 3 + [[0, 1, 2]])
    res = node.wait_settled(0, finished=True)
    assert len(res.trajectory) == 3


def test_no_change_after_finish_is_final(node, monkeypatch):
    # final optimization adds nothing (car parked, earlier optimization covered every node)
    calls = script(node, monkeypatch, [[0, 1]])
    res = node.wait_settled(0, finished=True)
    assert [p.pose.position.x for p in res.trajectory] == [0, 1]
    assert len(calls) > 10   # waited the timeout first


def test_rerun_without_finish_only_needs_quiet(node, monkeypatch):
    script(node, monkeypatch, [[0, 1]])
    res = node.wait_settled(0, finished=False)
    assert len(res.trajectory) == 2


def test_never_quiet_times_out(node, monkeypatch):
    script(node, monkeypatch, [[i] for i in range(10000)])
    with pytest.raises(RuntimeError, match='still changing'):
        node.wait_settled(0, finished=False)


def test_map_yaml_keeps_unknown_unknown(node, tmp_path):
    grid = OccupancyGrid()
    grid.info.width, grid.info.height, grid.info.resolution = 3, 1, 0.05
    grid.info.origin.orientation.w = 1.0
    grid.data = [-1, 0, 100]
    node.write_map(str(tmp_path), grid)
    meta = dict(line.split(': ', 1) for line in (tmp_path / 'map.yaml').read_text().splitlines())
    free_th, occ_th = float(meta['free_thresh']), float(meta['occupied_thresh'])
    pixels = np.frombuffer((tmp_path / 'map.pgm').read_bytes()[-3:], dtype=np.uint8)
    assert list(pixels) == [205, 254, 0]
    p = (255.0 - pixels) / 255.0                    # map_server's trinary rule
    assert not p[0] < free_th and not p[0] > occ_th   # unknown
    assert p[1] < free_th and p[2] > occ_th

    centerline = pytest.importorskip('planning.centerline')
    m = centerline.OccupancyMap(str(tmp_path / 'map.yaml'))
    assert list(m.free[0]) == [False, True, False]
    assert list(m.occupied[0]) == [False, False, True]


def test_tracks_dir_is_absolute(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('TRACKS_DIR', 'rel/tracks')
    assert tracks_dir() == os.path.join(str(tmp_path), 'rel', 'tracks')
    monkeypatch.setenv('TRACKS_DIR', '~/t')
    assert tracks_dir() == os.path.join(os.path.expanduser('~'), 't')
