from types import SimpleNamespace

import numpy as np

from control.lidar_processing import LidarProcessor


def test_processing_preserves_input_and_unmasked_smoothing_history():
    processor = LidarProcessor(alpha=0.5, disparity_threshold=2.0)
    raw = np.array([5., 5., 1., 5., 5.], dtype=np.float32)
    scan = SimpleNamespace(ranges=raw, angle_increment=0.1)

    first = processor.process(scan)
    np.testing.assert_allclose(first, [0., 0., 0., 0., 0.])
    np.testing.assert_allclose(raw, [5., 5., 1., 5., 5.])
    np.testing.assert_allclose(processor.ranges_window, raw)

    scan.ranges = [5., 5., 3., 5., 5.]
    processor.process(scan)
    np.testing.assert_allclose(processor.ranges_window, [5., 5., 2., 5., 5.])


def test_empty_and_changed_scan_length():
    processor = LidarProcessor()
    scan = SimpleNamespace(ranges=[], angle_increment=0.1)
    assert processor.process(scan).size == 0
    np.testing.assert_allclose(processor.preprocess_lidar([2., 8.]), [2., 5.])
    np.testing.assert_allclose(processor.preprocess_lidar([3.]), [3.])
