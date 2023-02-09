# standard libraries
from copy import copy

# external libraries
import pytest
import torch
from torch import Tensor

# gestrol library
from gestrol.modifiers.channel import ChannelDimOrderModifier, ChannelSwapModifier, SingleChannelSelectorModifier


@pytest.mark.parametrize(
    "channel",
    [0, 1, 2],
)
def test_single_channel_modifier(channel: int, dummy_frame_dim: int, dummy_frame: Tensor):
    """
    Test for `SingleChannelModifier` class.

    Tests ensure that:
        1. result is not `None`
        2. frame height and width dims are unchanged
        3. correct channel is selected
    """
    single_channel_modifier = SingleChannelSelectorModifier(channel=channel)
    one_channel_array = single_channel_modifier(dummy_frame)
    assert one_channel_array is not None
    assert one_channel_array.shape == (dummy_frame_dim, dummy_frame_dim)
    assert torch.all(channel == one_channel_array)


def test_channel_swap_modifier(dummy_frame: Tensor):
    """
    Test for ChannelSwapModifier class.

    Tests ensure that:
        1. result is not `None`
        2. channels are swapped in intended order
    """
    co = (2, 1, 0)
    channel_swap_modifier = ChannelSwapModifier(channel_order=co)
    swapped_array = channel_swap_modifier(dummy_frame)
    assert swapped_array is not None
    for i, channel in enumerate(co):
        assert torch.all(swapped_array[:, :, i] == channel)


def test_channel_dim_order_modifier(dummy_frame: Tensor):
    """
    Test to ensure `ChannelDimOrderModifier` swaps the channel dimension to first and last dimension.
    """
    mod = ChannelDimOrderModifier(mode="first")
    frame = copy(dummy_frame)
    frame = mod(frame)

    assert frame.shape[0] == 3
    assert frame.shape[-1] != 3

    mod = ChannelDimOrderModifier()  # defaults to "last" mode
    frame = mod(frame)
    assert frame.shape[0] != 3
    assert frame.shape[-1] == 3

    assert torch.equal(frame, dummy_frame)
