# standard libraries
from typing import Literal, Sequence

# external libraries
from torch import Tensor

# gestrol library
from gestrol.modifiers.base import Frame, FrameModifier


class SingleChannelSelectorModifier(FrameModifier):
    """
    Select single color channel from 3-channel Frame. Requires (h, w, 3) input dims.
    """

    def __init__(self, channel: int = 0):
        """
        Initiate method.

        Args:
            channel: Channel to select from Frame. Defaults to 0.
        """
        self.channel = channel

    def __call__(self, frame: Frame) -> Frame:
        """
        Select single channel.

        Args:
            frame: Three channel Frame

        Returns:
            Single channel Frame
        """
        return Frame(frame[:, :, self.channel])


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

    def __call__(self, frame: Frame) -> Frame:
        """
        Reorder channels.

        Args:
            frame: Three channel Frame

        Raises:
            TypeError: raised if `frame` is not np.ndarray

        Returns:
            Frame with swapped channels
        """
        return Frame(frame[:, :, self.channel_order])


class ChannelDimOrderModifier(FrameModifier):
    """
    Change the channel dimension of an Frame to be first dimension or last.
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

    def _first_mode(self, frame: Frame) -> Tensor:
        """Move channels dim to first dimension."""
        return frame.permute(2, 0, 1)

    def _last_mode(self, frame: Frame) -> Tensor:
        """Move channels dim to last dimension."""
        return frame.permute(1, 2, 0)

    def __call__(self, frame: Frame) -> Frame:
        """
        Swap first and last axes of an array.

        Args:
            frame: three channel numpy array

        Returns:
            three channel numpy array
        """
        frame = Frame(self._dim_order_modifier(frame))
        return frame
