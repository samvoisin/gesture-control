# external libraries
import numpy as np
import pytest
from torch import Tensor

# gestrol library
from gesturemote.camera.base import CameraInterface


@pytest.fixture
def dummy_frame_dim() -> int:
    """
    Dimensions of test frame.
    """
    return 100


@pytest.fixture
def dummy_camera_interface():
    class DummyCameraInterface(CameraInterface):
        def __init__(self):
            ...

        def __del__(self):
            ...

        def get_frame(self) -> Tensor:
            return Tensor(np.empty(shape=(120, 120, 3)))

    return DummyCameraInterface()
