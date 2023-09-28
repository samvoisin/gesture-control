# external libraries
import numpy as np

# gesturemote library
# gestrol library
from gesturemote.fps_monitor import FPSMonitor


class TestFPSMonitor:
    """
    Class of tests for FPSMonitor.
    """

    def test_active_flag(self):
        """
        Test that active_flag is flipped when first frame is passed in.
        """
        monitor = FPSMonitor()
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        assert not monitor._active_flag
        monitor.monitor_fps(dummy_frame)
        assert monitor._active_flag
