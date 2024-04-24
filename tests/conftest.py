import os

import numpy as np
import pytest
from torch import Tensor

from gesturemote.camera.base import CameraInterface


@pytest.fixture(scope="session", autouse=True)
def set_env():
    os.environ["DISPLAY"] = ":0"


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
