# external libraries
from torch import Tensor
from torchvision import transforms

# gestrol library
from gestrol.modifiers.base import Frame, FrameModifier


class SSDPreprocModifier(FrameModifier):
    """
    preprocess a frame before feeding into SSD hand identifier.
    """

    def __init__(self):
        super().__init__()
        self.fcn_composition = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def modify_frame(self, frame: Frame) -> Tensor:
        return self.fcn_composition(frame)
