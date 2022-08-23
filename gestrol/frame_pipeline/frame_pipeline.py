# standard libraries
from typing import Sequence

# gestrol library
from gestrol.frame_pipeline.modifiers.base import FrameFormat, FrameModifier


class FramePipeline:
    """
    Modular pipeline for concatenating and applying FrameModifiers
    """

    def __init__(self, modifier_pipeline: Sequence[FrameModifier] = None):
        self.modifier_pipeline = modifier_pipeline or []

    def __call__(self, frame: FrameFormat) -> FrameFormat:
        for modifier in self.modifier_pipeline:
            frame = modifier(frame)
        return frame
