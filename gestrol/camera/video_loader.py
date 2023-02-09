# standard libraries
import logging
from pathlib import Path

# external libraries
import cv2
from torch import Tensor

# gestrol library
from gestrol.camera.base import CameraInterface
from gestrol.utils.logging import configure_logging

configure_logging()


class VideoLoaderInterface(CameraInterface):
    """
    Subclass of `CameraInterface` for loading and streaming frames of video files saved on disk.
    """

    def __init__(self, video_path: Path):
        self.camera = cv2.VideoCapture(str(video_path))
        logging.info("Video loaded")

    def __del__(self):
        self.camera.release()
        logging.info("Video ended")

    def get_frame(self) -> Tensor:
        _, arr = self.camera.read()
        return Tensor(arr)
