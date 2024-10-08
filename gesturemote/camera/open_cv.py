import logging
from pathlib import Path
from typing import Generator

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class OpenCVCameraInterface:
    """
    OpenCV camera interface.
    """

    def __init__(self, index: int = 0):
        """
        Initiate method.

        Args:
            index: camera resource index. Defaults to 0.
        """
        self.camera = cv2.VideoCapture(index=index)
        logger.info("Camera instantiated")

    def __del__(self):
        """
        Destructor method for releasing the camera resource.
        """
        self.camera.release()
        logger.info("Camera released")

    def get_frame(self) -> np.ndarray:
        """
        Capture frame from camera.

        Returns:
            numpy array with dimensions (width, height, number of channels)
        """
        _, arr = self.camera.read()
        return arr

    def stream_frames(self) -> Generator[np.ndarray, None, None]:
        """
        Stream frames captured by camera.

        Yields:
            Generator[Tensor, None, None]
        """
        while True:
            yield self.get_frame()


class VideoLoaderInterface(OpenCVCameraInterface):
    """
    Subclass of `OpenCVCameraInterface` for loading and streaming frames of video files saved on disk.
    This class is primarily used for testing purposes.
    """

    def __init__(self, video_path: Path):
        """
        Args:
            video_path (Path): Path to video file.
        """
        self.camera = cv2.VideoCapture(str(video_path))
        logger.info("Video loaded")
