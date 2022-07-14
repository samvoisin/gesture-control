# standard libraries
import logging
import sys

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
        _, frame = self.camera.read()
        return frame


if __name__ == "__main__":
    pass
