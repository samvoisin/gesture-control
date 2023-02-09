# standard libraries
from typing import List, Optional, Union

# external libraries
import numpy as np
from torch import Tensor

# gestrol library
from gestrol.modifiers.base import FrameModifier


class ReverseNormalizeModifier(FrameModifier):
    """
    Reverse mean-standard deviation normalization. This class's `modify_frame` method assumes the frame dimensions are
    channel-first (i.e. (3, m, n)).
    """

    def __init__(self, mu: Optional[List[float]] = None, sigma: Optional[List[float]] = None):
        """
        Initiate method.

        Args:
            mu: mean vector. Defaults behavior uses [0.485, 0.456, 0.406].
            sigma: standard deviation vector. Defaults behavior uses [0.229, 0.224, 0.225].
        """
        self.mu = mu or [0.485, 0.456, 0.406]
        self.sigma = sigma or [0.229, 0.224, 0.225]

    def __call__(self, frame: Tensor) -> Tensor:
        for i in range(3):
            frame[i, :, :] = frame[i, :, :] * self.sigma[i] + self.mu[i]
        return frame


class ScalarModifier(FrameModifier):
    """
    Multiply all channels of an image by a scalar.
    """

    def __init__(self, scalar: int = 255) -> None:
        """
        Initiate method.

        Args:
            scalar: scalar multiple. Defaults to 255.
        """
        self.scalar = scalar

    def __call__(self, frame: Tensor) -> Union[np.ndarray, Tensor]:
        """
        Multiply frame by a scalar.

        Args:
            frame: numpy array or Tensor

        Raises:
            TypeError: raise error if frame is `PIL.Image.Image`

        Returns:
            numpy array or Tensor
        """
        return frame * self.scalar
