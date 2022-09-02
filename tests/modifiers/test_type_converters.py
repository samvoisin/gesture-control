# standard libraries
from typing import List

# external libraries
import numpy as np
import pytest
import torch
from PIL.Image import Image

# gestrol library
from gestrol.modifiers import NumpyToImageModifier, NumpyToTensorModifier, TensorToNumpyModifier


def test_numpy_to_image_modifier():
    """
    Test to ensure that a numpy array can be returned as Image.
    """
    arr = np.empty(shape=(180, 180, 3))
    mod = NumpyToImageModifier()
    img = mod.modify_frame(arr)
    assert isinstance(img, Image)


def test_numpy_to_tensor_modifier():
    """
    Test to ensure numpy array returned as Tensor with identical dimensons.
    """
    arr_dims = (180, 180, 3)
    arr = np.empty(shape=arr_dims)
    mod = NumpyToTensorModifier()
    t = mod.modify_frame(arr)
    assert isinstance(t, torch.Tensor)
    assert arr.shape == t.shape


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
