# standard libraries
from pathlib import Path
from typing import Optional

# external libraries
import torch
from torch import Tensor
from torchvision.models.detection.ssd import SSD, ssd300_vgg16

# gestrol library
from gestrol.modifiers.base import Frame, FrameModifier

MODELS_DIR = Path(__file__).parents[2] / "models"
SSD_MODEL_PATH = MODELS_DIR / "ssd_hand_detect.pt"

GPU_DEVICE = torch.device("cuda")


def load_ssd_model(model_path: Path = SSD_MODEL_PATH) -> SSD:
    """
    Load a previously trained single-shot detector model.

    Args:
        model_path: path to `.pt` model file.
        device: device to run model on. Defaults to GPU.

    Returns:
        a `torch.nn.Module` instance
    """
    # construct the model (2 classes; background and hand)
    model = ssd300_vgg16(progress=False, num_classes=2)
    # load the saved model
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


class SingleHandSSDExtractor(FrameModifier):
    """
    Use SSD model to identify bounding box around highest confidence hand prediction.
    """

    def __init__(self, model: Optional[torch.nn.Module] = None, device: Optional[torch.device] = None):
        """
        Instantiate SingleHandSSDExtractor.

        Args:
            model: a pytorch `nn.Module` instance. Default behavior loads internal `ssd_hand_detect.pt` model.
            device: device on which model will run. Defaults to machine GPU instance.
        """
        self.device = device or GPU_DEVICE
        self.model = model or load_ssd_model(model_path=SSD_MODEL_PATH)
        self.model = self.model.to(self.device)

    def modify_frame(self, frame: Frame) -> Optional[Tensor]:
        if not isinstance(frame, Tensor):
            raise TypeError(
                f"{self.__class__.__name__}.modify_frame requires `torch.Tensor` input but received {type(frame)}."
            )
        prepped_frame = [frame.to(self.device)]
        with torch.no_grad():
            boxes = self.model(prepped_frame)[0]["boxes"]
        if len(boxes) < 1:
            return None
        boxes = boxes[0]  # top confidence prediction
        boxes = boxes.to(int)
        x0, y0, x1, y1 = boxes
        frame = frame[:, y0:y1, x0:x1]
        return frame
