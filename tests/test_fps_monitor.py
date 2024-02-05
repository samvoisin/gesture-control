from unittest.mock import patch

import numpy as np

from gesturemote.fps_monitor import FPSMonitor


class TestFPSMonitor:
    """
    Class of tests for FPSMonitor.
    """

    def test_active_flag(self):
        """
        Test that active_flag is changed when first frame is passed in.
        """
        monitor = FPSMonitor()
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        assert not monitor._active_flag
        monitor.monitor_fps(dummy_frame)
        assert monitor._active_flag

    @patch("logging.Logger.info")
    def test_calculate_mean(self, mock_log):
        """
        Test that _calculate_mean method correctly calculates mean FPS over period.
        """
        monitor = FPSMonitor(period=2)  # Setting a small period for easier testing
        monitor.monitor_fps(np.array([0]))
        monitor.monitor_fps(np.array([0]))  # Should trigger calculation and logging

        mock_log.assert_called_once()
        assert "Mean FPS over past 2 frames:" in mock_log.call_args[0][0]

    def test_monitor_fps_single_call(self):
        """
        Test that monitor_fps method correctly increments frame count and returns unmodified frame.
        """
        frame = np.array([1])
        monitor = FPSMonitor()
        result_frame = monitor.monitor_fps(frame)
        assert np.array_equal(result_frame, frame)
        assert monitor.frame_count == 1
        assert monitor._active_flag

    def test_monitor_fps_multiple_calls(self):
        """
        Test that monitor_fps method correctly increments frame count and returns unmodified frame.
        """
        monitor = FPSMonitor(period=3)
        frame = np.array([1])

        with patch("logging.Logger.info") as mock_log:
            for _ in range(3):
                monitor.monitor_fps(frame)

            mock_log.assert_called_once()
            assert "Mean FPS over past 3 frames:" in mock_log.call_args[0][0]
