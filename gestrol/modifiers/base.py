# standard libraries
import abc
import logging
import sys
from typing import Optional, Union

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


FrameFormat = Union[Tensor, np.ndarray, Image]


class FrameModifier(abc.ABC):
    """
    Base class for callable which modifies a frame of video data.
    """

    def __repr__(self):
        return "<%s>" % self.__class__.__name__

    def __call__(self, frame: Optional[FrameFormat]) -> Optional[FrameFormat]:
        """
        Call method for generic `FrameModifier`. Any `FrameModifier` must be capable of accepting a `Frameformat`
        object as input OR a `None` object as input. In the event a `None` object is provided as input, it should be
        passed to the next `FrameModifier` which should pass `None` to the next `FrameModifier` and so on.

        The reason for allowing `None` objects as input/output is to allow for operations which reduce the frame rate
        passed into a `FramePipeline` or occasional failures in extraction by Extractor models.

        Args:
            frame: `FrameFormat` or `None`

        Returns:
            `FrameFormat` or `None`
        """
        if frame is None:
            return None
        else:
            return self.modify_frame(frame)

    @abc.abstractmethod
    def modify_frame(self, frame: FrameFormat) -> Optional[FrameFormat]:
        pass
