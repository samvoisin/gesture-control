# standard libraries
import abc
import logging
import sys

# external libraries
import cv2
import numpy as np

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="[%(asctime)s] (%(filename)s:%(lineno)d) %(levelname)s : %(message)s",
)

logger = logging.getLogger()


class CameraInterface(abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def __del__(self):
        pass

    @abc.abstractmethod
    def get_frame(self) -> np.ndarray:
        pass


class OpenCVCameraInterface(CameraInterface):
    def __init__(self, index: int = 0):
        self.camera = cv2.VideoCapture(index=index)
        logger.info("Camera instantiated")

    def __del__(self):
        self.camera.release()
        logger.info("Camera released")

    def get_frame(self) -> np.ndarray:
        _, frame = self.camera.read()
        frame = frame
        return frame

    # def capture_frames(self):
    #    retval, frame = self.get_frame()
    #    while retval:
    #        retval, frame = self.get_frame()
    #        key = cv2.waitKey(delay=20)
    #        if key == 27:  # exit on ESC
    #            break


if __name__ == "__main__":
    pass
