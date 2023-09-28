# standard libraries
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import List, Tuple

# external libraries
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image, ImageOps
from torch import nn
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torchvision.transforms import functional as f

SM_MODEL_WEIGHTS_PATH = Path("models/SSDLite_MobilenetV3_small.pth")
LG_MODEL_WEIGHTS_PATH = Path("models/SSDLite_MobilenetV3_large.pth")


TARGETS = {
    1: "call",
    2: "dislike",
    3: "fist",
    4: "four",
    5: "like",
    6: "mute",
    7: "ok",
    8: "one",
    9: "palm",
    10: "peace",
    11: "rock",
    12: "stop",
    13: "stop inverted",
    14: "three",
    15: "two up",
    16: "two up inverted",
    17: "three2",
    18: "peace inverted",
    19: "no gesture",
}


def retrieve_out_channels(model: nn.Module, size: Tuple[int, int]) -> List[int]:
    """
    This method retrieves the number of output channels of a specific model.

    Args:
        model (nn.Module): The model for which we estimate the out_channels.
            It should return a single Tensor or an OrderedDict[Tensor].
        size (Tuple[int, int]): The size (wxh) of the input.

    Returns:
        out_channels (List[int]): A list of the output channels of the model.
    """
    model.eval()

    with torch.no_grad():
        # Use dummy data to retrieve the feature map sizes to avoid hard-coding their values
        device = next(model.parameters()).device
        tmp_img = torch.zeros((1, 3, size[1], size[0]), device=device)
        features = model(tmp_img)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        out_channels = [x.size(1) for x in features.values()]

    return out_channels


def preprocess_mobilenet(img: np.ndarray) -> torch.Tensor:
    """
    Preproc image for model input
    Parameters
    ----------
    img: np.ndarray
        input image
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(img)
    width, height = image.size

    image = ImageOps.pad(image, (max(width, height), max(width, height)))
    image = image.resize((320, 320))

    img_tensor = f.pil_to_tensor(image)
    img_tensor = f.convert_image_dtype(img_tensor)
    img_tensor = img_tensor[None, :, :, :]
    return img_tensor


def build_mobilenet_large(checkpoint_fp: Path = LG_MODEL_WEIGHTS_PATH) -> nn.Module:
    num_classes = 20
    device = "cpu"

    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)

    in_channels = retrieve_out_channels(model.backbone, (320, 320))
    num_anchors = model.anchor_generator.num_anchors_per_location()
    norm_layer = partial(torch.nn.BatchNorm2d, eps=0.001, momentum=0.03)

    model.head.classification_head = SSDLiteClassificationHead(in_channels, num_anchors, num_classes, norm_layer)

    # load weights from checkpoint
    checkpoint_fp = checkpoint_fp.expanduser()
    if checkpoint_fp.exists():
        checkpoint = torch.load(checkpoint_fp, map_location=torch.device(device))
        if "state_dict" in checkpoint.keys():
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

    return model.eval()


def build_mobilenet_small(checkpoint_fp: Path = SM_MODEL_WEIGHTS_PATH) -> nn.Module:
    num_classes = 20
    device = "cpu"

    norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)

    backbone = torchvision.models.mobilenet_v3_small(pretrained=False, norm_layer=norm_layer, reduced_tail=True)
    backbone = torchvision.models.detection.ssdlite._mobilenet_extractor(
        backbone,
        0,
        norm_layer,
    )

    size = (320, 320)
    anchor_generator = torchvision.models.detection.ssdlite.DefaultBoxGenerator(
        [[2, 3] for _ in range(6)], min_ratio=0.2, max_ratio=0.95
    )

    out_channels = retrieve_out_channels(backbone, size)
    num_anchors = anchor_generator.num_anchors_per_location()

    assert len(out_channels) == len(anchor_generator.aspect_ratios)

    kwargs = {
        "score_thresh": 0.001,
        "nms_thresh": 0.55,
        "detections_per_img": 300,
        "topk_candidates": 300,
        # Rescale the input in a way compatible to the backbone:
        # The following mean/std rescale the data from [0, 1] to [-1, 1]
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
    }
    model = torchvision.models.detection.ssd.SSD(
        backbone,
        anchor_generator,
        size,
        num_classes,
        head=torchvision.models.detection.ssdlite.SSDLiteHead(out_channels, num_anchors, num_classes, norm_layer),
        **kwargs,
    )

    # load weights from checkpoint
    checkpoint_fp = checkpoint_fp.expanduser()
    if checkpoint_fp.exists():
        checkpoint = torch.load(checkpoint_fp, map_location=torch.device(device))
        if "state_dict" in checkpoint.keys():
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

    return model.eval()


class GestureRecognizerMobilenetSmall:
    def __init__(self, device: str = "cpu"):
        self.model = build_mobilenet_small()
        self.device = torch.device(device)

    def preprocess(self, frame: np.ndarray):
        return preprocess_mobilenet(frame)

    def predict(self, frame: np.ndarray) -> int:
        with torch.no_grad():
            tensor_frame = torch.Tensor(frame)
            model_output = self.model(tensor_frame.to(self.device))[0]
            return model_output
