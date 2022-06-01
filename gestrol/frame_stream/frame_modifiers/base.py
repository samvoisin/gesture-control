# standard libraries
import abc
import logging
import sys

# external libraries
import numpy as np

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="[%(asctime)s] (%(filename)s:%(lineno)d) %(levelname)s : %(message)s",
)

logger = logging.getLogger()


class FrameModifier(abc.ABC):
    """
    Base class for callable which modifies a frame of video data
    """

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        try:
            return self.modify_frame(frame)
        except Exception:
            logger.exception(f"Error modifying frame in {self.__class__.__name__}")
            return frame

    @abc.abstractmethod
    def modify_frame(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        pass
