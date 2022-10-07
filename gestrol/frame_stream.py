# standard libraries
from typing import Generator

# gestrol library
from gestrol.camera.base import CameraInterface
from gestrol.modifiers.base import Frame


class FrameStream:
    """
    Stream frames from a `CameraInterface`.
    """

    def __init__(self, camera: CameraInterface):
        """
        Initiate method.

        Args:
            camera (CameraInterface): `CameraInterface` instance
        """
        self.camera = camera

    def stream_frames(self) -> Generator[Frame, None, None]:
        """
        Stream frames from camera.

        Yields:
            Frame object
        """
        while True:
            yield self.camera.get_frame()
