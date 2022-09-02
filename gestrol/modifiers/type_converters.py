# external libraries
import numpy as np
from PIL import Image
from torch import Tensor

# gestrol library
from gestrol.modifiers.base import Frame, FrameModifier


class NumpyToTensorModifier(FrameModifier):
    """
    Convert numpy array to pytorch Tensor.
    """

    def modify_frame(self, frame: Frame) -> Tensor:
        """
        Convert frame type.

        Args:
            frame: numpy array with shape (h, w, 3)

        Raises:
            TypeError: raised if `frame` is not `np.ndarray`

        Returns:
            pytorch Tensor with same dimensions as `frame`
        """
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"frame must have type {np.ndarray}, but has type {type(frame)}.")
        return Tensor(frame)


class TensorToNumpyModifier(FrameModifier):
    """
    Convert pytorch Tensor to numpy array.
    """

    def modify_frame(self, frame: Frame) -> np.ndarray:
        """
        Convert frame type.

        Args:
            frame: pytorch Tensor

        Raises:
            TypeError: raise if `frame` is not a pytorch Tensor

        Returns:
            numpy array with same dimensions as `frame`
        """
        if not isinstance(frame, Tensor):
            raise TypeError(f"frame must have type {Tensor}, but has type {type(frame)}.")
        return frame.numpy()  # NOTE: removed frame.detach() here; may need to add back if problems with GPU


class NumpyToImageModifier(FrameModifier):
    """
    Convert numpy array to PIL.Image.Image type object in "RGB" mode.
    """

    def modify_frame(self, frame: Frame) -> Image.Image:
        """
        Convert frame type.

        Args:
            frame: numpy array with shape (h, w, 3)

        Raises:
            TypeError: raised if `frame` is not np.ndarray

        Returns:
            PIL.Image.Image in "RGB" mode
        """
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"frame must have type {np.ndarray}, but has type {type(frame)}.")
        return Image.fromarray(frame.astype("uint8"), "RGB")
