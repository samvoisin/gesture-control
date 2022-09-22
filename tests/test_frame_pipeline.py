# standard libraries
from typing import Callable, Optional, cast

# external libraries
import numpy as np
import pytest

# gestrol library
from gestrol.frame_pipeline import FramePipeline
from gestrol.modifiers.base import Frame, FrameModifier

DummyFrameModifier = Callable[[Frame], Optional[Frame]]


@pytest.fixture
def dummy_frame() -> np.ndarray:
    return np.ones(shape=(100, 100, 3))


class TestFramePipeline:
    @pytest.fixture
    def pass_through_modifier(self) -> DummyFrameModifier:
        def pt(frame: Frame) -> Frame:
            return frame

        return pt

    @pytest.fixture
    def null_modifier(self) -> DummyFrameModifier:
        def nm(_: Frame) -> None:
            return None

        return nm

    def test_frame_passthrough(self, dummy_frame: Frame, pass_through_modifier: DummyFrameModifier):

        frame_pass = [
            cast(FrameModifier, pass_through_modifier),
            cast(FrameModifier, pass_through_modifier),
        ]

        frame_pipeline = FramePipeline(modifier_pipeline=frame_pass)
        frame = frame_pipeline(dummy_frame)
        assert np.all(frame == dummy_frame)

    def test_null_frame(
        self,
        dummy_frame: Frame,
        pass_through_modifier: DummyFrameModifier,
        null_modifier: DummyFrameModifier,
    ):

        null_pipe = [
            cast(FrameModifier, pass_through_modifier),
            cast(FrameModifier, null_modifier),
            cast(FrameModifier, pass_through_modifier),
        ]

        frame_pipeline = FramePipeline(modifier_pipeline=null_pipe)
        frame = frame_pipeline(dummy_frame)
        assert frame is None


@pytest.mark.xfail(reason="FPSMonitor not yet implemented.")
def test_fps_monitor():
    assert False
