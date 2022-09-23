# standard libraries
from typing import List

# external libraries
import numpy as np
import pytest
import torch
from PIL.Image import Image, fromarray

# gestrol library
from gestrol.modifiers import convert_frame_to_tensor, convert_numpy_to_image, convert_tensor_to_numpy
from gestrol.modifiers.base import Frame


def test_convert_numpy_to_image():
    """
    Test to ensure that a numpy array can be returned as Image.
    """
    arr = np.empty(shape=(180, 180, 3))
    img = convert_numpy_to_image(arr)
    assert isinstance(img, Image)


@pytest.mark.parametrize(
    "frame",
    [
        np.empty(shape=(180, 180, 3)),
        torch.empty((180, 180, 3)),
        fromarray(np.empty(shape=(180, 180, 3), dtype="uint8")),
    ],
)
def test_convert_frame_to_tensor(frame: Frame):
    """
    Test to ensure FrameToTensorModifier works on all three possible frame types.
    """
    frame = convert_frame_to_tensor(frame)
    assert isinstance(frame, torch.Tensor)


@pytest.mark.parametrize(
    "arr_dims",
    [
        (180, 180, 3),  # three color channel
        (180, 180),  # one color channel
    ],
)
def test_convert_tensor_to_numpy(arr_dims: List[int]):
    """
    Test to ensure Tensor is returned as numpy array with identical dimensions.
    """
    t = torch.empty(size=arr_dims)
    arr = convert_tensor_to_numpy(t)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == t.shape
