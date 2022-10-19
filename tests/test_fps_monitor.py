# standard libraries
from math import isclose
from time import sleep

# external libraries
import pytest

# gestrol library
from gestrol.fps_monitor import FPSMonitor
from gestrol.modifiers.base import Frame


class TestFPSMonitor:
    """
    Class of tests for FPSMonitor.
    """

    def test_active_flag(self, dummy_frame: Frame):
        """
        Test that active_flag is flipped when first frame is passed in.
        """
        monitor = FPSMonitor()
        assert not monitor._active_flag
        monitor.monitor_fps(dummy_frame)
        assert monitor._active_flag

    @pytest.mark.xfail(reason="This test must be improved.")
    def test_monitor_fps(self, dummy_frame: Frame):
        """
        Test fps calculation on a few iterations with a delay built in.
        """
        monitor = FPSMonitor()
        for _ in range(3):
            monitor.monitor_fps(dummy_frame)
            sleep(1)
        res = monitor._calculate_mean()
        assert isclose(res, 1, rel_tol=2e-3)  # res is approx. 0.99895729; close enough for small sample
