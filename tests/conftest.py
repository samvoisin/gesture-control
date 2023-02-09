# external libraries
import numpy as np
import pytest
from torch import Tensor

# gestrol library
from gestrol.camera.base import CameraInterface


@pytest.fixture
def dummy_frame_dim() -> int:
    """
    Dimensions of test frame.
    """
    return 100


@pytest.fixture
def dummy_frame(dummy_frame_dim) -> Tensor:
    """
    Test frame with dimensions (dummy_frame_dim, dummy_frame_dim, 3). All elements of array i are i for i in {0, 1, 2}.
    """
    res = np.empty(shape=(dummy_frame_dim, dummy_frame_dim, 3))
    ones_arr = np.ones(shape=(dummy_frame_dim, dummy_frame_dim))
    for i in range(3):
        res[:, :, i] = ones_arr * i
    return Tensor(res)


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
