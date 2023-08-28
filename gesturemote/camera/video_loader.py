# standard libraries
import logging
from pathlib import Path

# external libraries
import cv2
import numpy as np

# gestrol library
from gesturemote.camera.base import CameraInterface

logger = logging.getLogger(__name__)


class VideoLoaderInterface(CameraInterface):
    """
    Subclass of `CameraInterface` for loading and streaming frames of video files saved on disk.
    """

    def __init__(self, video_path: Path):
        self.camera = cv2.VideoCapture(str(video_path))
        logger.info("Video loaded")

    def __del__(self):
        self.camera.release()
        logger.info("Video ended")

    def get_frame(self) -> np.ndarray:
        _, arr = self.camera.read()
        return arr
