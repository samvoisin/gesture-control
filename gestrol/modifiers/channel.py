# standard libraries
from typing import Sequence

# external libraries
import numpy as np

# gestrol library
from gestrol.modifiers.base import Frame, FrameModifier


class SingleChannelModifier(FrameModifier):
    """
    Select single color channel from 3-channel image as a numpy array. Requires (h, w, 3) input dims.
    """

    def __init__(self, channel: int = 0):
        """
        Initiate method.

        Args:
            channel: channel to select from image array. Defaults to 0.
        """
        self.channel = channel

    def modify_frame(self, frame: Frame) -> np.ndarray:
        """
        Select single channel.

        Args:
            frame: three channel numpy array

        Raises:
            TypeError: raised if `frame` is not np.ndarray

        Returns:
            single channel numpy array
        """
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"frame must have type {np.ndarray}, but has type {type(frame)}.")
        return frame[:, :, self.channel]


class ChannelSwapModifier(FrameModifier):
    """
    Reorder color channels in an image.
    """

    def __init__(self, channel_order: Sequence[int] = (2, 1, 0)):
        """
        Initiate method.

        Args:
            channel_order: New channel order with indices relative to original order. Defaults to (2, 1, 0).
        """
        self.channel_order = channel_order

    def modify_frame(self, frame: Frame) -> np.ndarray:
        """
        Reorder channels.

        Args:
            frame: three channel numpy array

        Raises:
            TypeError: raised if `frame` is not np.ndarray

        Returns:
            three channel numpy array
        """
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"frame must have type {np.ndarray}, but has type {type(frame)}.")
        return frame[:, :, self.channel_order]
