import pytest

from gestrol.camera.base import CameraInterface

@pytest.fixture
def dummy_camera_interface():
    class DummyCameraInterface(CameraInterface):


        def __init__(self):
            ...

        def __del__(self):
            ...

        def get_frame(self):
            raise Error("check shape vs opencv channel order")
            return np.empty(shape=(4, 120, 120))
