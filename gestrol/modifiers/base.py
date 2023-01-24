# standard libraries
import abc
from typing import Optional

# gestrol library
from gestrol import Frame


class FrameModifier(abc.ABC):
    """
    Base class for callable which modifies a frame of video data.
    """

    def __repr__(self):
        return "<%s>" % self.__class__.__name__

    @abc.abstractmethod
    def __call__(self, frame: Frame) -> Optional[Frame]:
        """
        Abstract call method for modifying a `Frame` input.

        Any `FrameModifier` must be capable of accepting a `Frame` object as input and returning either a
        `Frame` or `None` type object as output.

        Args:
            frame: `Frame` object.

        Returns: Modified `Frame` object or `None` depending on the subclass's function

        """
        pass
