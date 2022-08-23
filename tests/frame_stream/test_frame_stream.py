# external libraries
import numpy as np
import pytest

# gestrol library
from gestrol.frame_stream import FrameStream


@pytest.fixture
def frame_stream(dummy_camera_interface) -> FrameStream:
    return FrameStream(camera=dummy_camera_interface)


def test_frame_stream(frame_stream: FrameStream):
    frame = next(frame_stream.stream_frames())
    assert isinstance(frame, np.ndarray)
