import math
import struct

import numpy as np
from nav_msgs.msg import OccupancyGrid

from autodrive_bridge.map_view import downsample, frame, map_frame


def test_frame():
    assert frame(b'H') == b'H\x00\x00\x00\x00'
    assert frame(b'P', b'abc') == b'P\x03\x00\x00\x00abc'


def test_downsample_keeps_small_grids():
    cells = np.zeros((3, 4), dtype=np.int8)
    out, k = downsample(cells, 4)
    assert k == 1 and out is cells


def test_downsample_block_max():
    # occupied beats free beats unknown; the padding past the edge is unknown
    cells = np.array([[-1, -1, 0, -1, 0],
                      [-1, -1, 100, 0, -1]], dtype=np.int8)
    out, k = downsample(cells, 3)
    assert k == 2
    assert out.tolist() == [[-1, 100, 0]]


def test_map_frame():
    msg = OccupancyGrid()
    msg.info.width, msg.info.height, msg.info.resolution = 3, 2, 0.05
    msg.info.origin.position.x, msg.info.origin.position.y = -1.0, 2.0
    msg.info.origin.orientation.z = math.sin(0.25)
    msg.info.origin.orientation.w = math.cos(0.25)
    msg.data = [-1, 0, 100, 50, 0, -1]
    data = map_frame(msg, 2048)
    assert data[:1] == b'M'
    (length,) = struct.unpack('<I', data[1:5])
    w, h, res, ox, oy, yaw = struct.unpack('<IIffff', data[5:29])
    assert (w, h, length) == (3, 2, 24 + 6)
    assert res == np.float32(0.05) and (ox, oy) == (-1.0, 2.0)
    assert abs(yaw - 0.5) < 1e-6
    assert np.frombuffer(data[29:], dtype=np.int8).tolist() == msg.data.tolist()
