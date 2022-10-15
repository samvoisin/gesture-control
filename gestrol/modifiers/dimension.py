# standard libraries
from typing import Sequence

# external libraries
import torch

# gestrol library
from gestrol.modifiers.base import Frame, FrameModifier


class TensorDimensionSwapModifier(FrameModifier):
    """
    Swap the dimensions of a Tensor frame.

    TODO: Can probably be simplified to a function.
    """

    def __init__(self, dimension_swaps: Sequence[Sequence[int]]) -> None:
        """
        Args:
            dimension_swaps: Sequence of sequences containing two ints each. ints in nested sequences are the axes to
            be swapped (e.g. [(0, 2) (2, 1)] will swap axis 0 to axis 2 and then axis 2 to axis 1).
        """
        self.dimension_swaps = dimension_swaps

    def modify_frame(self, frame: Frame) -> torch.Tensor:
        if not isinstance(frame, torch.Tensor):
            frame = torch.Tensor(frame)
        for dimension_swap in self.dimension_swaps:
            frame = torch.swapaxes(frame, dimension_swap[0], dimension_swap[1])
        return frame
