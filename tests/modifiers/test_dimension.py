# standard libraries
from typing import List, Tuple, cast

# external libraries
import numpy as np
import pytest
from torch import Tensor

# gestrol library
from gestrol.modifiers import TensorDimensionSwapModifier


@pytest.fixture
def tensor_frame() -> Tensor:
    """
    Dummy Tensor frame.
    """
    arr = np.empty(shape=(10, 20, 30))
    return Tensor(arr)


test_cases = [
    ([(0, 2)], (30, 20, 10)),
    ([(0, 2), (2, 1)], (30, 10, 20)),
]


@pytest.mark.parametrize(["swap_seq", "res_dims"], test_cases)
def test_tensor_dimension_swap_modifier(swap_seq: List[Tuple], res_dims: Tuple[int, int, int], tensor_frame: Tensor):
    tdsm = TensorDimensionSwapModifier(dimension_swaps=swap_seq)
    frame = tdsm(tensor_frame)
    assert frame is not None
    assert cast(Tensor, frame).shape == res_dims
