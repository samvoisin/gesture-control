# standard libraries
from pathlib import Path
from typing import Optional

# external libraries
import torch
from torch import Tensor
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN

# gestrol library
from gestrol.modifiers.base import Frame, FrameModifier

MODELS_DIR = Path(__file__).parents[2] / "models"
FRCNN_MODEL_PATH = MODELS_DIR / "frcnn_hand_detect_mbn_v3_lrg.pt"

GPU_DEVICE = torch.device("cuda")


def load_frcnn_model(model_path: Path) -> FasterRCNN:
    """
    Load a previously trained object detection model.

    Args:
        model_path: path to `.pt` model file.

    Returns:
        a `torch.nn.Module` instance
    """
    model = fasterrcnn_mobilenet_v3_large_fpn()
    # model = fasterrcnn_mobilenet_v3_large_fpn(num_classes=2) This is for the updated model which needs more training
    model.load_state_dict(torch.load(model_path))
    return model


class SingleHandMobileNetExtractor(FrameModifier):
    """
    Use Faster R-CNN model to identify bounding box around highest confidence hand prediction.
    """

    def __init__(self, model: Optional[torch.nn.Module] = None, device: Optional[torch.device] = None):
        """
        Instantiate method.

        Args:
            model: a pytorch `nn.Module` instance. Default behavior loads internal `frcnn_hand_detect.pt` model.
            device: device on which model will run. Defaults to machine GPU instance.
        """
        self.device = device or GPU_DEVICE
        self.model = model or load_frcnn_model(model_path=FRCNN_MODEL_PATH)
        self.model = self.model.to(self.device)
        self.model.eval()

    def modify_frame(self, frame: Frame) -> Optional[Tensor]:
        if not isinstance(frame, Tensor):
            raise TypeError(
                f"{self.__class__.__name__}.modify_frame requires `torch.Tensor` input but received {type(frame)}."
            )
        prepped_frame = [frame.to(self.device)]
        boxes = self.model(prepped_frame)[0]["boxes"]
        if len(boxes) < 1:
            return None
        boxes = boxes[0]  # top confidence prediction
        boxes = boxes.to(int)
        x0, y0, x1, y1 = boxes
        frame = frame[:, y0:y1, x0:x1]
        return frame
