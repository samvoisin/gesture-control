# standard libraries
from abc import ABC, abstractmethod
from typing import Generator

# external libraries
from torch import Tensor


class CameraInterface(ABC):
    """
    Base video recorder interface.
    """

    @abstractmethod
    def __init__(self):
        """
        Initiate to take control of camera resource.
        """
        pass

    @abstractmethod
    def __del__(self):
        """
        Destructor to ensure camera resource is released on garbage collection.
        """
        pass

    @abstractmethod
    def get_frame(self) -> Tensor:
        """
        Method for capturing and returning a single frame.
        """
        pass

    @abstractmethod
    def stream_frames(self) -> Generator[Tensor, None, None]:
        """
        Stream frames from source.
        """
        pass
