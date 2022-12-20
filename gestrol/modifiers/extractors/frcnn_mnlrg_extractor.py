# standard libraries
from pathlib import Path
from typing import Optional

# external libraries
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN

# gestrol library
from gestrol.modifiers.base import Frame
from gestrol.modifiers.extractors.base import FrameExtractor

MODELS_DIR = Path(__file__).parents[3] / "models"
FRCNN_MODEL_PATH = (  # MODELS_DIR / "frcnn_hand_detect_mnlrg.pt"
    "/home/svoisin/Projects/PythonProjects/mod/gesture-control/models/frcnn_hand_detect_mnlrg_best.pt"
)

GPU_DEVICE = torch.device("cuda")


def load_frcnn_model(model_path: Path) -> FasterRCNN:
    """
    Load a previously trained object detection model.

    Args:
        model_path: path to `.pt` model file.

    Returns:
        a `torch.nn.Module` instance
    """
    model = fasterrcnn_mobilenet_v3_large_fpn(num_classes=2)
    model.load_state_dict(torch.load(model_path))
    return model


class SingleHandMobileNetExtractor(FrameExtractor):
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
        device = device or GPU_DEVICE
        model = model or load_frcnn_model(model_path=FRCNN_MODEL_PATH)
        super().__init__(model, device)

    def __call__(self, frame: Frame) -> Optional[Frame]:
        return super().__call__(frame)
