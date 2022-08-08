# standard libraries
import logging
import sys
from pathlib import Path
from typing import Sequence

# external libraries
import cv2
import numpy as np

# gestrol library
from gestrol.camera.base import CameraInterface

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="[%(asctime)s] (%(filename)s:%(lineno)d) %(levelname)s : %(message)s",
)

logger = logging.getLogger()


class OpenCVCameraInterface(CameraInterface):
    def __init__(self, index: int = 0):
        self.camera = cv2.VideoCapture(index=index)
        logger.info("Camera instantiated")

    def __del__(self):
        self.camera.release()
        logger.info("Camera released")

    def get_frame(self) -> np.ndarray:
        # TODO: docstring - info on channels: first (3, 120, 120) or last (120, 120, 3)
        _, frame = self.camera.read()
        return frame

    def record_video(
        self,
        save_path: Path,
        fps: int = 30,
        frame_size: Sequence[int] = (1920, 1080),
        color: bool = True,
        codec: str = "MJPG",
    ):
        # define video writers
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(save_path, fourcc, fps, frame_size, color)

        while True:
            frame = self.get_frame()
            writer.write(frame)

            if cv2.waitKey(0) == ord("q"):
                writer.release()
                break


if __name__ == "__main__":
    pass
