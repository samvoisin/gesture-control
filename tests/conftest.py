import os

import numpy as np
import pytest
from torch import Tensor

from gesturemote.camera.base import CameraInterface


@pytest.fixture(scope="function", autouse=True)
def set_display_env():
    # Check if the DISPLAY environment variable is already set
    old_display = os.environ.get("DISPLAY")

    # Set the DISPLAY environment variable to a dummy value if not present
    if old_display is None:
        os.environ["DISPLAY"] = ":99"

    # Yield to run the test
    yield

    # Cleanup: Restore the original DISPLAY environment variable
    if old_display is None:
        del os.environ["DISPLAY"]
    else:
        os.environ["DISPLAY"] = old_display


@pytest.fixture
def dummy_frame_dim() -> int:
    """
    Dimensions of test frame.
    """
    return 100


@pytest.fixture
def dummy_camera_interface(set_env):
    class DummyCameraInterface(CameraInterface):
        def __init__(self):
            ...

        def __del__(self):
            ...

        def get_frame(self) -> Tensor:
            return Tensor(np.empty(shape=(120, 120, 3)))

    return DummyCameraInterface()
