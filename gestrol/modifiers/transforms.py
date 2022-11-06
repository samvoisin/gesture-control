# standard libraries
from typing import List, Optional, Union

# external libraries
import numpy as np
from PIL.Image import Image
from torch import Tensor

# gestrol library
from gestrol.modifiers.base import Frame, FrameModifier


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

    def __call__(self, frame: Frame) -> Tensor:
        if not isinstance(frame, Tensor):
            raise TypeError(f"frame must have type {Tensor}, but has type {type(frame)}.")
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
        self._scalar = scalar

    def __call__(self, frame: Frame) -> Union[np.ndarray, Tensor]:
        """
        Multiply frame by a scalar.

        Args:
            frame: numpy array or Tensor

        Raises:
            TypeError: raise error if frame is `PIL.Image.Image`

        Returns:
            numpy array or Tensor
        """
        if isinstance(frame, Image):
            raise TypeError(f"frame must be {Tensor} or {np.ndarray} but is type {type(frame)}.")
        return frame * self._scalar
