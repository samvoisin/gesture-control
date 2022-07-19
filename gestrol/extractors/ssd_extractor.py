# standard libraries
from pathlib import Path
from typing import List

# external libraries
import torch
import torchvision
from torch import Tensor
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor

# gestrol library
from gestrol.frame_pipeline.modifiers.base import ImageFormat

MODELS_DIR = Path(__file__).parents[2] / "models"
SSD_MODEL_PATH = MODELS_DIR / "ssd_hand_detect.pt"

GPU_DEVICE = torch.device("cuda")


def load_ssd_model(model_path: Path, device: torch.device) -> FasterRCNN:
    # construct the model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True,
    )
    num_classes = 2  # 1 class (hand) + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # load the saved model and send to device
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model


class SSDExtractor:
    """
    Use SSD model to identify bounding box around hand(s) in a frame.
    """

    def __init__(self, model: torch.nn.Module = None, device: str = None, top_n: int = 1):
        self.model = model or load_ssd_model(model_path=SSD_MODEL_PATH, device=GPU_DEVICE)
        self.device = device or GPU_DEVICE
        self.top_n = top_n

    def __call__(self, frame: Tensor) -> List[Tensor]:
        prepped_frame = [frame.to(self.device)]
        bbox_preds = [bbox for bbox in self.model(prepped_frame)[0]["boxes"][: self.top_n]]
        return bbox_preds


class SingleHandSSDExtractor:
    """
    Use SSD model to identify bounding box around highest confidence hand prediction.
    """

    def __init__(self, model: torch.nn.Module = None, device: str = None):
        self.model = model or load_ssd_model(model_path=SSD_MODEL_PATH, device=GPU_DEVICE)
        self.device = device or GPU_DEVICE

    def __call__(self, frame: ImageFormat) -> Tensor:
        if not isinstance(frame, Tensor):
            frame = Tensor(frame)
        prepped_frame = [frame.to(self.device)]
        boxes = self.model(prepped_frame)[0]["boxes"]
        if len(boxes) < 1:
            return frame[:, :2, :2]
        boxes = boxes[0]  # top confidence prediction
        boxes = boxes.to(int)
        x0, y0, x1, y1 = boxes
        frame = frame[:, y0:y1, x0:x1]
        return frame
