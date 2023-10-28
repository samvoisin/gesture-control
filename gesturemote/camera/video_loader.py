import logging
from pathlib import Path
from typing import Generator

import cv2
import numpy as np

from gesturemote.camera.base import CameraInterface

logger = logging.getLogger(__name__)


class VideoLoaderInterface(CameraInterface):
    """
    Subclass of `CameraInterface` for loading and streaming frames of video files saved on disk.
    """

    def __init__(self, video_path: Path):
        """
        Args:
            video_path (Path): Path to video file.
        """
        self.camera = cv2.VideoCapture(str(video_path))
        logger.info("Video loaded")

    def get_frame(self) -> np.ndarray:
        """
        Get a frame of video.
        """
        _, arr = self.camera.read()
        return arr

    def stream_frames(self) -> Generator[np.ndarray, None, None]:
        """
        Stream frames from video.
        """
        while True:
            yield self.get_frame()
