# standard libraries
from typing import Optional, Sequence

# gestrol library
from gestrol.modifiers.base import FrameFormat, FrameModifier


class FramePipeline:
    """
    Modular pipeline for concatenating and applying FrameModifiers
    """

    def __init__(self, modifier_pipeline: Sequence[FrameModifier] = None):
        self.modifier_pipeline = modifier_pipeline or []

    def __call__(self, frame: FrameFormat) -> Optional[FrameFormat]:
        for modifier in self.modifier_pipeline:
            if frame is None:
                return None
            frame = modifier(frame)
        return frame
