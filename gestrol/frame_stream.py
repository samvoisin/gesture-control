# gestrol library
from gestrol.camera.base import CameraInterface


class FrameStream:
    """
    Stream frames from a generic `CameraInterface`.
    """

    def __init__(self, camera: CameraInterface):
        """
        Args:
            camera (CameraInterface): A generic `CameraInterface` instance
        """
        self.camera = camera

    def stream_frames(self):
        while True:
            yield self.camera.get_frame()
