# standard libraries
import abc
import logging
import sys
from typing import Union

# external libraries
import numpy as np
from PIL.Image import Image
from torch import Tensor

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="[%(asctime)s] (%(filename)s:%(lineno)d) %(levelname)s : %(message)s",
)

logger = logging.getLogger()


ImageFormat = Union[Tensor, np.ndarray, Image]


class FrameModifier(abc.ABC):
    """
    Base class for callable which modifies a frame of video data
    """

    def __repr__(self):
        return "<%s>" % self.__class__.__name__

    def __call__(self, frame: ImageFormat) -> ImageFormat:
        return self.modify_frame(frame)

    @abc.abstractmethod
    def modify_frame(self, frame: ImageFormat, **kwargs) -> ImageFormat:
        pass
