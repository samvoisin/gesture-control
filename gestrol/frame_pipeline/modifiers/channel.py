# standard libraries
from typing import Sequence

# external libraries
import numpy as np

# gestrol library
from gestrol.frame_pipeline.modifiers.base import FrameModifier


class SingleChannelModifier(FrameModifier):
    def __init__(self, channel: int = 0):
        self.channel = channel

    def modify_frame(self, frame: np.ndarray) -> np.ndarray:
        return frame[:, :, self.channel]


class ChannelSwapModifier(FrameModifier):
    def __init__(self, channel_order: Sequence[int] = (2, 1, 0)):
        self.channel_order = channel_order

    def modify_frame(self, frame: np.ndarray) -> np.ndarray:
        return frame[:, :, self.channel_order]
