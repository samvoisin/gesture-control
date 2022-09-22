# standard libraries
from typing import Protocol

# gestrol library
from gestrol.modifiers.base import Frame


class CameraInterface(Protocol):
    """
    Video recorder interface protocol.
    """

    def __init__(self):
        """
        Initiate to take control of camera resource.
        """
        ...

    def __del__(self):
        """
        Destructor to ensure camera resource is released on garbage collection.
        """
        ...

    def get_frame(self) -> Frame:
        """
        Method for capturing and returning a single frame.
        """
        ...
