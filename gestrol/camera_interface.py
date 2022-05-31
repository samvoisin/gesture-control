# standard libraries
import logging
import sys

# external libraries
import cv2

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="[%(asctime)s] (%(filename)s:%(lineno)d) %(levelname)s : %(message)s",
)

logger = logging.getLogger()


class CameraInterface:
    def __init__(self):
        self.camera = None

    def activate(self, index: int = 0, window_name: str = "preview"):
        cv2.namedWindow(winname=window_name)
        self.camera = cv2.VideoCapture(index=index)
        logger.info("Camera is activated.")

    def deactivate(self, window_name: str = "preview"):
        if not self.camera:
            raise ValueError("Camera not actiated.")
        self.camera.release()
        cv2.destroyWindow(window_name)
        self.camera = None
        logger.info("Camera is deactivated.")

    def capture_frame(self):
        if not self.camera:
            raise ValueError("Camera not activated.")
        if not self.camera.isOpened():
            raise ValueError("Camera not opened.")
        self.camera.read()
        logger.info("Frame recorded.")

    def capture_frames(self, winname: str = "preview"):
        logger.info("Recording...")
        if not self.camera:
            raise ValueError("Camera not activated.")
        if not self.camera.isOpened():
            raise ValueError("Camera not opened.")
        retval, frame = self.camera.read()
        while retval:
            cv2.imshow(winname=winname, mat=frame)
            retval, frame = self.camera.read()
            key = cv2.waitKey(delay=20)
            if key == 27:  # exit on ESC
                break
        logger.info("Recording ended.")


if __name__ == "__main__":
    cam = CameraInterface()
    cam.activate()
    cam.capture_frames()
    cam.deactivate()
