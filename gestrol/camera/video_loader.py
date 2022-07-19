# standard libraries
import logging
import sys
from pathlib import Path

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


class VideoLoaderInterface(CameraInterface):
    def __init__(self, video_path: Path):
        self.camera = cv2.VideoCapture(str(video_path))
        logger.info("Video loaded")

    def __del__(self):
        self.camera.release()
        logger.info("Video ended")

    def get_frame(self) -> np.ndarray:
        _, frame = self.camera.read()
        return frame
