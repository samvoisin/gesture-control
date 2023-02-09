# standard libraries
from typing import Callable

# external libraries
import numpy as np
import pytest
import torch

# gestrol library
from gestrol.modifiers import ReverseNormalizeModifier, ScalarModifier
from gestrol.modifiers.base import Tensor


def test_reverse_normalizer_modifier():
    mu = [1.0, 1.0, 1.0]
    sigma = [2.0, 2.0, 2.0]
    rnm = ReverseNormalizeModifier(mu, sigma)
    frame = torch.ones(3, 5, 5)
    res = rnm(frame)
    assert torch.all(res == 3.0)


@pytest.mark.parametrize(
    ["frame", "allfcn"],
    [
        (np.ones(shape=(10, 10)), np.all),
        (torch.ones(10, 10), torch.all),
    ],
)
def test_scalar_modifier(frame: Tensor, allfcn: Callable):
    sm = ScalarModifier(scalar=2)
    assert allfcn(sm(frame) == 2)
