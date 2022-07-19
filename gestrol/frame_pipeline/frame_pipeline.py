# standard libraries
from typing import Sequence

# gestrol library
from gestrol.frame_pipeline.modifiers.base import FrameModifier, ImageFormat


class FramePipeline:
    """
    Modular pipeline for concatenating and applying FrameModifiers
    """

    def __init__(self, modifier_pipeline: Sequence[FrameModifier] = None):
        self.modifier_pipeline = modifier_pipeline or []

    def __call__(self, frame: ImageFormat) -> ImageFormat:
        for modifier in self.modifier_pipeline:
            frame = modifier(frame)
        return frame
