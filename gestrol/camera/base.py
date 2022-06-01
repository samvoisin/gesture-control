# standard libraries
import abc

# external libraries
import numpy as np


class CameraInterface(abc.ABC):
    """
    Base class for video recorder interfaces.
    """

    @abc.abstractmethod
    def __init__(self):
        """
        Initiate to take control of camera resource. Must be implemented.
        """
        pass

    @abc.abstractmethod
    def __del__(self):
        """
        Destructor to ensure camera resource is released on garbage collection. Must be implemented.
        """
        pass

    @abc.abstractmethod
    def get_frame(self) -> np.ndarray:
        """
        Method for capturing a single frame and returning a numpy array. Must be implemented.
        """
        pass
