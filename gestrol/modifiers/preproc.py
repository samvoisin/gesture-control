# external libraries
from torch import Tensor
from torchvision import transforms

# gestrol library
from gestrol.modifiers.base import FrameModifier


class FasterRCNNPreprocModifier(FrameModifier):
    """Preprocess a frame before feeding into Faster R-CNN hand identifier."""

    def __init__(self):
        super().__init__()
        self.fcn_composition = transforms.Compose(
            [
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, frame: Tensor) -> Tensor:
        return self.fcn_composition(frame)
