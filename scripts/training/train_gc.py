# standard libraries
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# external libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.folder import DatasetFolder
from tqdm import tqdm

GESTURE_CLASSES = ["G1", "G5", "G9"]  # G1 - fist; G5 - open palm; G9 - german 3
SENZ3D_PATH = Path("data/senz3d_dataset")
EXTRACTED_PATH = SENZ3D_PATH / "extracted"
BEST_MODEL_PATH = Path("models/gc_cnn_best.pt")
LAST_MODEL_PATH = Path("models/gc_cnn_last.pt")


def _split_samples_list(samples: List[Any], split_frac: float, rng=np.random.Generator) -> Tuple[List[Any], List[Any]]:
    n = int(len(samples) * split_frac)
    sampled = rng.choice(samples, replace=False, size=n)
    remaining = list(set(samples) - set(sampled))
    return sampled, remaining


# def load_pt_file(path: str):
#     return torch.load(path)


def _copy_bulk(from_paths: List[Path], to_dir: Path):
    for from_path in from_paths:
        to_path = to_dir / from_path.name
        shutil.copy(from_path, to_path)


class TrainTestDataDirBuilder:
    def __init__(
        self,
        extracted_path: Optional[Path] = None,
        gesture_classes: Optional[List[str]] = None,
        split: Dict[str, float] = {"train": 0.8, "test": 0.2},
    ):
        self.split = split
        assert sum(v for v in split.values()) == 1, "Split values do not sum to 1."
        datasets = [k for k in split.keys()]

        self.extracted_path = extracted_path or EXTRACTED_PATH
        self.gesture_classes = gesture_classes or GESTURE_CLASSES
        self.extracted_gdirs = [self.extracted_path / gc for gc in self.gesture_classes]
        self.rng = np.random.default_rng(seed=1)

        # make train test dirs
        self.gc_training_dir = SENZ3D_PATH / "gc_training_data"
        self.gc_training_dir.mkdir()
        self.tt_dirs = {}
        for ds in datasets:
            self.tt_dirs[ds] = self.gc_training_dir / ds
            self.tt_dirs[ds].mkdir()

        # sort samples
        for gc in self.gesture_classes:
            gc_extr_paths = list((self.extracted_path / gc).glob("*.pt"))
            train_paths, test_paths = _split_samples_list(
                gc_extr_paths,
                split_frac=self.split["train"],
                rng=self.rng,
            )
            split_samples = {"train": train_paths, "test": test_paths}
            for t in datasets:
                t_dir = self.gc_training_dir / t / gc
                t_dir.mkdir()
                _copy_bulk(split_samples[t], t_dir)

    def __del__(self):
        shutil.rmtree(self.gc_training_dir)


ttddb = TrainTestDataDirBuilder()

data_transforms = {
    "train": transforms.Compose(
        [
            transforms.Resize(size=(300, 300)),  # smaller than many images. may need to increase.
        ]
    ),
    "test": transforms.Compose(
        [
            transforms.Resize(size=(300, 300)),
        ]
    ),
}

dataset_folders = {
    "train": DatasetFolder(
        root=ttddb.tt_dirs["train"], loader=torch.load, extensions=(".pt",), transform=data_transforms["train"]
    ),
    "test": DatasetFolder(
        root=ttddb.tt_dirs["test"], loader=torch.load, extensions=(".pt",), transform=data_transforms["train"]
    ),
}


data_loaders = {
    "train": DataLoader(dataset_folders["train"], shuffle=True),
    "test": DataLoader(dataset_folders["test"]),
}


class GestureClassifierCNN(nn.Module):
    num_classes: int = 3
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x)
        x = self.classifier(x)
        return x


def construct_model(model_path: Optional[Path] = None) -> torch.nn.Module:
    model = GestureClassifierCNN()

    if model_path:
        model.load_state_dict(torch.load(model_path))

    return model


def make_one_hot_vector(label: torch.Tensor, n_classes: int = 3) -> torch.Tensor:
    one_hot = torch.zeros(n_classes)
    one_hot[label] += 1
    return one_hot


def train_one_epoch(model, optimizer, loss_function, data_loader, device, log_n_iters=100):
    model.train()
    running_loss = 0.0

    for i, (imgs, labels) in enumerate(tqdm(data_loader)):
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        pred = model(imgs).reshape(1, -1)
        loss = loss_function(pred, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % log_n_iters == 0:
            logging.info(f"iteration {i}; Avg. loss over last {log_n_iters}: {running_loss / log_n_iters:.4f}")
            running_loss = 0.0


def evaluate_model(model, data_loader, loss_function, device):
    model.eval()

    tot_loss = 0
    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            pred = model(imgs).reshape(1, -1)
            loss = loss_function(pred, labels).item()
            tot_loss += loss
    mean_loss = tot_loss / len(data_loader)
    return mean_loss


def main():
    device = torch.device("cuda")
    model = construct_model()
    model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    epochs = 25

    best_loss = np.inf
    for epoch in range(epochs):
        train_one_epoch(model, optimizer, loss_fcn, data_loaders["train"], device)
        val_loss = evaluate_model(model, data_loaders["test"], loss_fcn, device)
        if best_loss > val_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
        torch.save(model.state_dict(), LAST_MODEL_PATH)
        logging.info(f"Epoch {epoch} validation loss: {val_loss:.4f}; best loss is {best_loss:.4f}.")


if __name__ == "__main__":
    main()
