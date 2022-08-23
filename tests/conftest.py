# external libraries
import numpy as np
import pytest

# gestrol library
from gestrol.camera.base import CameraInterface


@pytest.fixture
def dummy_camera_interface():
    class DummyCameraInterface(CameraInterface):
        def __init__(self):
            ...

        def __del__(self):
            ...

        def get_frame(self):
            return np.empty(shape=(120, 120, 3))

    return DummyCameraInterface()
