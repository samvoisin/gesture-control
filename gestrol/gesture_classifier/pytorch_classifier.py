# external libraries
import torch
from torch import Tensor

# gestrol library
from gestrol.gesture_classifier.base import GestureClassifier


class PytorchGestureClassifier(GestureClassifier):
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
    ):
        self.model = model
        self.device = device
        self.model = self.model.to(self.device)
        self.model.eval()  # ensure model in eval mode

    def infer_gesture(self, frame: Tensor) -> int:
        frame = frame.to(self.device)
        with torch.inference_mode():
            pred = self.model(frame)
        gesture_label = pred.argmax().item()
        return gesture_label
