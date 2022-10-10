# standard libraries
from typing import cast

# external libraries
import numpy as np
import pytest

# gestrol library
from gestrol.modifiers.channel import ChannelDimOrderModifier, ChannelSwapModifier, SingleChannelSelectorModifier


@pytest.mark.parametrize(
    "channel",
    [0, 1, 2],
)
def test_single_channel_modifier(channel: int, dummy_frame_dim: int, dummy_frame: np.ndarray):
    """
    Test for `SingleChannelModifier` class.

    Tests ensure that:
        1. result is not `None`
        2. frame height and width dims are unchanged
        3. correct channel is selected
    """
    single_channel_modifier = SingleChannelSelectorModifier(channel=channel)
    one_channel_array = cast(np.ndarray, single_channel_modifier(dummy_frame))
    assert one_channel_array is not None
    assert one_channel_array.shape == (dummy_frame_dim, dummy_frame_dim)
    assert np.all(channel == one_channel_array)


def test_channel_swap_modifier(dummy_frame: np.ndarray):
    """
    Test for ChannelSwapModifier class.

    Tests ensure that:
        1. result is not `None`
        2. channels are swapped in intended order
    """
    co = (2, 1, 0)
    channel_swap_modifier = ChannelSwapModifier(channel_order=co)
    swapped_array = cast(np.ndarray, channel_swap_modifier(dummy_frame))
    assert swapped_array is not None
    for i, channel in enumerate(co):
        assert np.all(swapped_array[:, :, i] == channel)


def test_channel_dim_order_modifier(dummy_frame: np.ndarray):
    """
    Test to ensure `ChannelDimOrderModifier` swaps the channel dimension to first and last dimension.
    """
    mod = ChannelDimOrderModifier(mode="first")
    frame = dummy_frame.copy()
    frame = cast(np.ndarray, mod(frame))

    assert frame.shape[0] == 3
    assert frame.shape[-1] != 3

    mod = ChannelDimOrderModifier()  # defaults to "last" mode
    frame = cast(np.ndarray, mod(frame))
    assert frame.shape[0] != 3
    assert frame.shape[-1] == 3

    assert np.all(frame == dummy_frame)
