# standard libraries
from typing import Sequence

# external libraries
import numpy as np

# gestrol library
from gestrol.frame_stream.modifiers.base import FrameModifier


class FramePipeline:
    """
    Modular pipeline for concatenating and applying FrameModifiers
    """

    def __init__(self, modifier_pipeline: Sequence[FrameModifier]):
        self.modifier_pipeline = modifier_pipeline

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        for modifier in self.modifier_pipeline:
            frame = modifier(frame)
        return frame
