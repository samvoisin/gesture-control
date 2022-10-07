# standard libraries
import abc
from typing import Optional, Union

# external libraries
import numpy as np
from PIL.Image import Image
from torch import Tensor

Frame = Union[Tensor, np.ndarray, Image]


class FrameModifier(abc.ABC):
    """
    Base class for callable which modifies a frame of video data.
    """

    def __repr__(self):
        return "<%s>" % self.__class__.__name__

    def __call__(self, frame: Frame) -> Optional[Frame]:
        """
        Call method for generic `FrameModifier`.
        Any `FrameModifier` must be capable of accepting a `Frame` object as input OR a `None` object as input.
        In the event a `None` object is provided as input, it should be passed to the next `FrameModifier` which should
        pass `None` to the next `FrameModifier` and so on.

        The reason for allowing `None` objects as input/output is to allow for operations which reduce the frame rate
        passed into a `FramePipeline` or occasional failures in extraction by Extractor models.

        Args:
            frame: `Frame` type

        Returns:
            `Frame` or `None`
        """
        return self.modify_frame(frame)

    @abc.abstractmethod
    def modify_frame(self, frame: Frame) -> Optional[Frame]:
        """
        Abstract method for actually modifying a `Frame` input. This should be overwritten in subclasses.

        Args:
            frame: a `Frame` type. `None` is handled in `__call__` method and is thus not a valid input.

        Returns: a `Frame` type or `None` depending on the subclass's function

        """
        pass
