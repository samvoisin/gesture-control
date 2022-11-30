# standard libraries
from abc import abstractmethod
from typing import Optional

# external libraries
import torch

# gestrol library
from gestrol.modifiers.base import Frame, FrameModifier


class FrameExtractor(FrameModifier):
    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.device = device
        self.model = model
        self.model = self.model.to(self.device)
        self.model.eval()  # ensure model in eval mode

    @abstractmethod
    def __call__(self, frame: Frame) -> Optional[Frame]:
        if not isinstance(frame, torch.Tensor):
            raise TypeError(
                f"{self.__class__.__name__}.modify_frame requires `torch.Tensor` input but received {type(frame)}."
            )
        prepped_frame = [frame.to(self.device)]
        with torch.no_grad():
            prediction = self.model(prepped_frame)
            boxes = prediction[0]["boxes"]  # only 1 class so take 0; then take bounding boxes
        if len(boxes) < 1:
            return None
        boxes = boxes[0]  # top confidence prediction
        boxes = boxes.to(int)
        x0, y0, x1, y1 = boxes
        frame = frame[:, y0:y1, x0:x1]
        return frame
