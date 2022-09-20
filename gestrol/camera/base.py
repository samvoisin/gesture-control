# standard libraries
import abc

# gestrol library
from gestrol.modifiers.base import Frame


class CameraInterface(abc.ABC):
    """
    Base class for video recorder interfaces.
    """

    @abc.abstractmethod
    def __init__(self):
        """
        Initiate to take control of camera resource.
        """
        pass

    @abc.abstractmethod
    def __del__(self):
        """
        Destructor to ensure camera resource is released on garbage collection.
        """
        pass

    @abc.abstractmethod
    def get_frame(self) -> Frame:
        """
        Method for capturing and returning a single frame.
        """
        pass
