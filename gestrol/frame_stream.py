# gestrol library
from gestrol.camera_interface import CameraInterface


class FrameStream:
    def __init__(self, camera: CameraInterface):
        self.camera = camera

    def stream_frames(self):
        while True:
            yield self.camera.get_frame()


if __name__ == "__main__":
    pass
