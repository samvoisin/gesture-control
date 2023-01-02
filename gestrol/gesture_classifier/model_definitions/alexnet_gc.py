# standard libraries
from pathlib import Path
from typing import List

# external libraries
from torch import Tensor, flatten, load, nn

ALEXNET_GC_PATH = Path("models/gc_cnn_best.pt")
GESTURE_CLASSES = ["G2", "G4", "G5", "G9"]  # G2 - peace sign; G4 - open hand; G5 - closed fist; G9 - american 3


class GestureClassifierCNN(nn.Module):
    classes: List[str] = GESTURE_CLASSES
    num_classes: int = len(classes)
    dropout: float = 0.5

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, self.num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = flatten(x)
        x = self.classifier(x)
        return x


def load_alexnet_classifier(model_path: Path = ALEXNET_GC_PATH) -> nn.Module:
    model = GestureClassifierCNN()

    if model_path:
        model.load_state_dict(load(model_path))

    return model
