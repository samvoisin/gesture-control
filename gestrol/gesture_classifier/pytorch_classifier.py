# external libraries
import torch

# gestrol library
from gestrol.gesture_classifier.base import GestureClassifier
from gestrol.modifiers.base import Frame


class PytorchGestureClassifier(GestureClassifier):
    def __init__(self, model: torch.nn.Module, device: torch.device):
        super().__init__(model=model)
        self.device = device
        self.model = self.model.to(self.device)
        model.eval()  # ensure model in eval mode

    def infer_gesture(self, frame: Frame) -> int:
        if not isinstance(frame, torch.Tensor):
            raise TypeError(
                f"{self.__class__.__name__}.modify_frame requires `torch.Tensor` input but received {type(frame)}."
            )
        frame = frame.to(self.device)
        with torch.inference_mode():
            gesture_label = self.model(frame)
        return gesture_label
