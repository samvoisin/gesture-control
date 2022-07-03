# standard libraries
from pathlib import Path
from typing import List

# external libraries
import torch
import torchvision
from torch import Tensor
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor

MODELS_DIR = Path(__file__).parents[2] / "models"
SSD_MODEL_PATH = MODELS_DIR / "ssd_hand_detect_no_pretrain.pt"

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
    def __init__(self, model: torch.nn.Module = None, device: str = None, top_n: int = 2):
        self.model = model or load_ssd_model(model_path=SSD_MODEL_PATH, device=GPU_DEVICE)
        self.device = device or GPU_DEVICE
        self.top_n = top_n

    def __call__(self, frame: Tensor) -> List[Tensor]:
        prepped_frame = [frame.to(self.device)]
        bbox_preds = [bbox for bbox in self.model(prepped_frame)[0]["boxes"][: self.top_n]]
        return bbox_preds
