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


def test_nan_beams_do_not_size_the_bubble():
    processor = LidarProcessor()
    ranges = np.full(1081, 5.0, dtype=np.float32)
    ranges[500:511] = 0.4        # close obstacle
    ranges[511:513] = np.nan     # two dropouts next to it
    scan = SimpleNamespace(ranges=ranges, angle_increment=0.004363323)

    result = processor.process(scan)

    np.testing.assert_allclose(result[:100], 5.0)
    np.testing.assert_allclose(result[-100:], 5.0)   # zeroed when a NaN was the bubble edge


def test_persistent_nan_beam_stays_zero_through_smoothing():
    processor = LidarProcessor(alpha=0.7)
    ranges = np.full(11, 5.0, dtype=np.float32)
    processor.preprocess_lidar(ranges)
    ranges[5] = np.nan           # a beam that was valid drops out
    for _ in range(5):
        result = processor.preprocess_lidar(ranges)
        assert result[5] == 0.
        assert processor.ranges_window[5] == 0.
