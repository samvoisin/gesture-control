# standard libraries
from typing import Sequence

# external libraries
import numpy as np

# gestrol library
from gestrol.frame_stream.modifiers.base import FrameModifier


class SingleChannelModifier(FrameModifier):
    def modify_frame(self, frame: np.ndarray, channel: int = 0) -> np.ndarray:
        return frame[:, :, channel]


class ChannelSwapModifier(FrameModifier):
    def modify_frame(self, frame: np.ndarray, channel_order: Sequence[int] = (2, 1, 0)) -> np.ndarray:
        return frame[:, :, channel_order]
