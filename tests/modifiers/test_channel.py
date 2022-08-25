# standard libraries
from typing import cast

# external libraries
import numpy as np
import pytest

# gestrol library
from gestrol.modifiers.channel import ChannelSwapModifier, SingleChannelModifier


@pytest.fixture
def dummy_frame_dim():
    return 100


@pytest.fixture(scope="function")
def dummy_frame(dummy_frame_dim):
    res = np.empty(shape=(dummy_frame_dim, dummy_frame_dim, 3))
    ones_arr = np.ones(shape=(dummy_frame_dim, dummy_frame_dim))
    for i in range(3):
        res[:, :, i] = ones_arr * i
    return res


@pytest.mark.parametrize(
    "channel",
    [0, 1, 2],
)
def test_single_channel_modifier(channel: int, dummy_frame_dim: int, dummy_frame: np.ndarray):
    single_channel_modifier = SingleChannelModifier(channel=channel)
    one_channel_array = cast(np.ndarray, single_channel_modifier(dummy_frame))
    assert one_channel_array is not None
    assert one_channel_array.shape == (dummy_frame_dim, dummy_frame_dim)
    assert np.all(channel == one_channel_array)


def test_channel_swap_modifier(dummy_frame: np.ndarray):
    co = (2, 1, 0)
    channel_swap_modifier = ChannelSwapModifier(channel_order=co)
    swapped_array = cast(np.ndarray, channel_swap_modifier(dummy_frame))
    assert swapped_array is not None
    for i, channel in enumerate(co):
        assert np.all(swapped_array[:, :, i] == channel)
