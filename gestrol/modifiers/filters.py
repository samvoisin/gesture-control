# standard libraries
from typing import Optional

# external libraries
from torch import Tensor

# gestrol library
from gestrol.modifiers.base import FrameModifier


class FrameSizeFilter(FrameModifier):
    """Filter frames based on their size."""

    def __init__(self, min_height: int, min_width: int, n_channels: Optional[int] = None) -> None:
        """
        Initiate method.

        Args:
            min_height (int): Minimum allowable height of frame.
            min_width (int): Minimum allowable width of frame.
            n_channels (Optional[int], optional): Number of allowable channels in frame. Defaults to None.
        """
        self.min_height = min_height
        self.min_width = min_width

        if n_channels:
            self.n_channels = n_channels

            def frame_size_check(*args) -> bool:
                nc, fh, fw = args
                return fh >= self.min_height and fw >= self.min_width and nc == self.n_channels

        else:

            def frame_size_check(*args) -> bool:
                fh, fw = args
                return fh >= self.min_height and fw >= self.min_width

        self.size_check = frame_size_check

    def __call__(self, frame: Tensor) -> Optional[Tensor]:
        if self.size_check(*frame.shape):
            return frame
        return None
