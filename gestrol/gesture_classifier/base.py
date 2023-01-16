# standard libraries
from abc import ABC, abstractmethod

# gestrol library
from gestrol.modifiers.base import Frame


class GestureClassifier(ABC):
    @abstractmethod
    def infer_gesture(self, frame: Frame) -> int:
        pass
