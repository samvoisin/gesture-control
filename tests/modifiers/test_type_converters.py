# standard libraries
from typing import List

# external libraries
import numpy as np
import pytest
import torch
from PIL.Image import Image, fromarray

# gestrol library
from gestrol.modifiers import FrameToTensorModifier, NumpyToImageModifier, TensorToNumpyModifier
from gestrol.modifiers.base import Frame


def test_numpy_to_image_modifier():
    """
    Test to ensure that a numpy array can be returned as Image.
    """
    arr = np.empty(shape=(180, 180, 3))
    mod = NumpyToImageModifier()
    img = mod.modify_frame(arr)
    assert isinstance(img, Image)


@pytest.mark.parametrize(
    "frame",
    [
        np.empty(shape=(180, 180, 3)),
        torch.empty((180, 180, 3)),
        fromarray(np.empty(shape=(180, 180, 3), dtype="uint8")),
    ],
)
def test_frame_to_tensor_modifier(frame: Frame):
    """
    Test to ensure FrameToTensorModifier works on all three possible frame types.
    """
    mod = FrameToTensorModifier()
    assert isinstance(mod(frame), torch.Tensor)


@pytest.mark.parametrize(
    "arr_dims",
    [
        (180, 180, 3),  # three color channel
        (180, 180),  # one color channel
    ],
)
def test_tesnor_to_numpy_modifier(arr_dims: List[int]):
    """
    Test to ensure Tensor is returned as numpy array with identical dimensions.
    """
    t = torch.empty(size=arr_dims)
    mod = TensorToNumpyModifier()
    arr = mod.modify_frame(t)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == t.shape
