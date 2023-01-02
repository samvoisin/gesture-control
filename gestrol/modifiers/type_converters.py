# external libraries
import numpy as np
from PIL.Image import Image, fromarray
from torch import Tensor
from torchvision.transforms import ToTensor

# gestrol library
from gestrol.modifiers.base import Frame, FrameModifier


class FrameToTensorModifier(FrameModifier):
    def __call__(self, frame: Frame) -> Tensor:
        """
        Convert a Frame to torch.Tensor type.

        Args:
            frame: frame object to be converted

        Returns:
            Tensor representation of frame
        """
        converter = ToTensor()
        if isinstance(frame, Tensor):
            return frame
        return converter(frame)


class TensorToNumpyModifier(FrameModifier):
    def __call__(self, frame: Frame) -> Tensor:
        """
        Convert pytorch Tensor to numpy array.

        Args:
            frame: pytorch Tensor

        Raises:
            TypeError: raise if `frame` is not a pytorch Tensor

        Returns:
            numpy array with same dimensions as `frame`
        """
        if not isinstance(frame, Tensor):
            raise TypeError(f"frame must have type {Tensor}, but has type {type(frame)}.")
        return frame.detach().numpy()


class NumpyToImageModifier(FrameModifier):
    def __call__(self, frame: Frame) -> Image:
        """
        Convert numpy array to PIL.Image.Image type object in "RGB" mode.

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
