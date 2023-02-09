# standard libraries
from abc import ABC, abstractmethod

# gestrol library
from gestrol.modifiers.base import Tensor


class GestureClassifier(ABC):
    @abstractmethod
    def infer_gesture(self, frame: Tensor) -> int:
        pass
