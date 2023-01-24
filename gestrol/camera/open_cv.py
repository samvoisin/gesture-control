# standard libraries
import logging
import sys
from pathlib import Path
from time import time
from typing import Optional

# external libraries
import cv2

# gestrol library
from gestrol import Frame
from gestrol.camera.base import CameraInterface

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="[%(asctime)s] (%(filename)s:%(lineno)d) %(levelname)s : %(message)s",
)

logger = logging.getLogger()


class OpenCVCameraInterface(CameraInterface):
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

    def get_frame(self) -> Frame:
        """
        Capture frame from camera.

        Returns:
            numpy array with dimensions (width, height, number of channels)
        """
        _, arr = self.camera.read()
        return Frame.from_numpy(arr)

    def record_video(
        self,
        save_path: Path,
        fps: int = 30,
        color: bool = True,
        codec: str = "MJPG",
        vlen: Optional[int] = None,
    ):
        """
        Record a video and write it to disk.

        Args:
            save_path: Path to save video file
            fps: Frame rate. Defaults to 30.
            color: Color boolean. Defaults to True.
            codec: Video codec. Defaults to "MJPG".
            vlen: Length of recording in seconds. Defaults to None.
        """
        _frame_size = (640, 480)  # hard coded for compatibility with OpenCV; these are the only values that will work

        # define video writers
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(save_path), fourcc, fps, _frame_size, color)
        logger.info(f"video will save to {str(save_path)}")

        stime = time()
        while True:
            frame = self.get_frame()
            writer.write(frame)

            if vlen and time() - stime > vlen:
                writer.release()
                break
            elif cv2.waitKey(0) == ord("q"):
                writer.release()
                break


if __name__ == "__main__":
    pass
