# standard libraries
from pathlib import Path
from typing import Optional

# external libraries
import torch
import torchvision
from torch import Tensor
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor

# gestrol library
from gestrol.modifiers.base import Frame, FrameModifier

MODELS_DIR = Path(__file__).parents[2] / "models"
FRCNN_MODEL_PATH = MODELS_DIR / "frcnn_hand_detect.pt"

GPU_DEVICE = torch.device("cuda")


def load_frcnn_model(model_path: Path) -> FasterRCNN:
    """
    Load a previously trained single-shot detector model.

    Args:
        model_path: path to `.pt` model file.
        device: device to run model on. Defaults to GPU.

    Returns:
        a `torch.nn.Module` instance
    """
    # construct the model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True,
    )
    num_classes = 2  # 1 class (hand) + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # load the saved model and send to device
    model.load_state_dict(torch.load(model_path))
    # model = model.to(device)
    model.eval()
    return model


class SingleHandFRCNNExtractor(FrameModifier):
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
