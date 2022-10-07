# standard libraries
from typing import Literal, Sequence

# external libraries
import numpy as np

# gestrol library
from gestrol.modifiers.base import Frame, FrameModifier


class SingleChannelSelectorModifier(FrameModifier):
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
            raise TypeError(f"{self.__class__.__name__} takes {np.ndarray} as input, but has type {type(frame)}.")
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
            raise TypeError(f"{self.__class__.__name__} takes {np.ndarray} as input, but has type {type(frame)}.")
        return frame[:, :, self.channel_order]


class ChannelDimOrderModifier(FrameModifier):
    """
    Change the channel dimension of an array (i.e. element of the array.shape tuple that specifies number of channels).
    """

    def __init__(self, mode: Literal["first", "last"] = "last"):
        """
        Initiate method.

        Args:
            mode: Which dimension to make channel dimension.
            Setting to "last" will make `frame.shape` (m, n, 3). Setting to "first" will make `frame.shape` (3, m, n).
            Defaults to "last".
        """
        mode = mode or "last"
        if mode == "first":
            self._dim_order_modifier = self._first_mode
        elif mode == "last":
            self._dim_order_modifier = self._last_mode
        else:
            raise ValueError("Invalid argument to `channel_order` param. Must be 'first' or 'last'.")

    def _first_mode(self, frame: np.ndarray) -> np.ndarray:
        return np.rollaxis(frame, 2, 0)

    def _last_mode(self, frame: np.ndarray) -> np.ndarray:
        return np.rollaxis(frame, 0, 3)

    def modify_frame(self, frame: Frame) -> np.ndarray:
        """
        Swap first and last axes of an array.

        Args:
            frame: three channel numpy array

        Raises:
            TypeError: raised if `frame` is not np.ndarray

        Returns:
            three channel numpy array
        """
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"{self.__class__.__name__} takes {np.ndarray} as input, but has type {type(frame)}.")
        return self._dim_order_modifier(frame)
