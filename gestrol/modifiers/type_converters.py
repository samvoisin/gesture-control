# external libraries
import numpy as np
from PIL.Image import Image, fromarray
from torch import Tensor
from torchvision.transforms import ToTensor

# gestrol library
from gestrol.modifiers.base import Frame, FrameModifier


class FrameToTensorModifier(FrameModifier):
    """
    Convert PIL.Image.Image or numpy array to torch.Tensor.
    """

    def __init__(self):
        """
        Initiate method.
        """
        self._convert_type = ToTensor()

    def modify_frame(self, frame: Frame) -> Tensor:
        """
        Convert frame to `torch.Tensor`.

        Args:
            frame: frame object to be converted

        Returns:
            Tensor representation of frame
        """
        if isinstance(frame, Tensor):
            return frame
        return self._convert_type(frame)


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

    def modify_frame(self, frame: Frame) -> Image:
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
        return fromarray(frame.astype("uint8"), "RGB")
