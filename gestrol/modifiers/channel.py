# standard libraries
from typing import Sequence

# external libraries
import numpy as np

# gestrol library
from gestrol.modifiers.base import FrameFormat, FrameModifier


class SingleChannelModifier(FrameModifier):
    """
    Select a single color channel from an image input as a numpy array. This assumes the input frame dimensions are in
    the format (l, w, channel).
    """

    def __init__(self, channel: int = 0):
        """
        Args:
            channel: channel to select from image array. Defaults to 0.
        """
        self.channel = channel

    def modify_frame(self, frame: FrameFormat) -> np.ndarray:
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"{self.__class__.__name__}.modify_frame requires `np.ndarray` input.")
        return frame[:, :, self.channel]


class ChannelSwapModifier(FrameModifier):
    """
    Reorder color channels in an image.
    """

    def __init__(self, channel_order: Sequence[int] = (2, 1, 0)) -> None:
        """
        Args:
            channel_order: New channel order with indices relative to original order. Defaults to (2, 1, 0).
        """
        self.channel_order = channel_order

    def modify_frame(self, frame: FrameFormat) -> np.ndarray:
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"{self.__class__.__name__}.modify_frame requires `np.ndarray` input.")
        return frame[:, :, self.channel_order]
