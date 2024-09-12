from time import time

import numpy as np

from gesturemote.logger_config import configure_logger


class FPSMonitor:
    """
    Monitor frames per second.
    """

    def __init__(self, period: int = 30) -> None:
        """
        Initiate method.

        Args:
            period: Number of frames in period over which average FPS is calculated. Defaults to 30.
        """
        self.period = period
        self.frame_count: int = 0
        self._active_flag: bool = False
        self.logger = configure_logger()

    def _reset_monitor(self) -> None:
        self.frame_count = 0
        self.start_time = time()

    def _calculate_mean(self) -> float:
        duration = time() - self.start_time
        avg_fps = self.frame_count / duration
        return avg_fps

    def monitor_fps(self, frame: np.ndarray) -> np.ndarray:
        """
        Continuously monitor frames per second.

        ```
        fpsm = FPSMonitor()
        for frame in fs.stream_frames():
            fpsm.monitor_fps(frame)
            procd_frame = frame_pipeline.process_frame(frame)
        ```

        Args:
            frame: `Frame` type

        Returns:
            unmodified input `Frame`
        """
        # start duration timer when first frame is received; prevents inaccurate initial result.
        if not self._active_flag:
            self._active_flag = True
            self._reset_monitor()

        self.frame_count += 1
        if self.frame_count == self.period:
            avg_fps = self._calculate_mean()
            self.logger.info(f"Mean FPS over past {self.period} frames: {avg_fps:.2f} fps.")
            self._reset_monitor()

        return frame
