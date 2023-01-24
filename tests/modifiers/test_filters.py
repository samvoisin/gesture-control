# standard libraries
from typing import Optional

# external libraries
import pytest
import torch

# gestrol library
from gestrol.frame import Frame
from gestrol.modifiers.filters import FrameSizeFilter


class TestFrameSizeFilter:
    """Set of tests for FrameSizeFilter."""

    @pytest.mark.parametrize(
        ["frame", "exp_res"],
        [
            (torch.ones(10, 10), torch.ones(10, 10)),
            (torch.ones(9, 10), None),
            (torch.ones(10, 9), None),
            (torch.ones(9, 9), None),
        ],
    )
    def test_frame_size_filter_2d(self, frame: Frame, exp_res: Optional[Frame]):
        filter = FrameSizeFilter(10, 10)
        res = filter(frame)
        assert type(res) == type(exp_res)

    @pytest.mark.parametrize(
        ["frame", "exp_res"],
        [
            (torch.ones(3, 10, 10), torch.ones(3, 10, 10)),
            (torch.ones(3, 9, 10), None),
            (torch.ones(3, 10, 9), None),
            (torch.ones(3, 9, 9), None),
            (torch.ones(1, 10, 10), None),
        ],
    )
    def test_frame_size_filter_3d(self, frame: Frame, exp_res: Optional[Frame]):
        filter = FrameSizeFilter(10, 10, 3)
        res = filter(frame)
        assert type(res) == type(exp_res)
