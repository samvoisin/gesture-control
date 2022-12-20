# standard libraries
from abc import ABC, abstractmethod
from typing import Any

# gestrol library
from gestrol.modifiers.base import Frame


class GestureClassifier(ABC):
    @abstractmethod
    def __init__(self, model: Any, **kwargs: Any):
        self.model = model

    @abstractmethod
    def infer_gesture(self, frame: Frame) -> int:
        pass
