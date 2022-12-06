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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.folder import DatasetFolder
from tqdm import tqdm

GESTURE_CLASSES = ["G2", "G4", "G5", "G9"]  # G2 - peace sign; G4 - open hand; G5 - closed fist; G9 - american 3
SENZ3D_PATH = Path("data/senz3d_dataset")
EXTRACTED_PATH = SENZ3D_PATH / "extracted"
BEST_MODEL_PATH = Path("models/gc_cnn_best.pt")
LAST_MODEL_PATH = Path("models/gc_cnn_last.pt")


logging.basicConfig(level=logging.INFO)


def _split_samples_list(samples: List[Any], split_frac: float, rng=np.random.Generator) -> Tuple[List[Any], List[Any]]:
    n = int(len(samples) * split_frac)
    sampled = rng.choice(samples, replace=False, size=n)
    remaining = list(set(samples) - set(sampled))
    return sampled, remaining


def _copy_bulk(from_paths: List[Path], to_dir: Path):
    for from_path in from_paths:
        to_path = to_dir / from_path.name
        shutil.copy(from_path, to_path)


class TrainTestDataDirBuilder:
    # TODO stop copying files. make 1 dataframe and do split there

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
        for gesture_class in self.gesture_classes:
            gc_extr_paths = list((self.extracted_path / gesture_class).glob("*.pt"))
            train_paths, test_paths = _split_samples_list(
                gc_extr_paths,
                split_frac=self.split["train"],
                rng=self.rng,
            )
            split_samples = {"train": train_paths, "test": test_paths}
            for dataset in datasets:
                dataset_dir = self.gc_training_dir / dataset / gesture_class
                dataset_dir.mkdir()
                _copy_bulk(split_samples[dataset], dataset_dir)

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


def train_one_epoch(model, optimizer, loss_function, data_loader, device):
    model.train()
    tot_trn_loss = 0.0

    for imgs, labels in tqdm(data_loader):
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        pred = model(imgs).reshape(1, -1)
        loss = loss_function(pred, labels)

        loss.backward()
        optimizer.step()

        tot_trn_loss += loss.item()

    logging.info(f"Avg. training loss: {tot_trn_loss / len(data_loader):.4f}")


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
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.4)
    scheduler = ReduceLROnPlateau(optimizer, patience=5)
    epochs = 100
    overall_patience = 25  # end training if we hit this number of epochs w/out improvement

    op_ctr = 0
    best_loss = np.inf
    for epoch in range(epochs):
        logging.info(f"Current learning rate: {optimizer.state_dict()['param_groups'][0]['lr']}")
        train_one_epoch(model, optimizer, loss_fcn, data_loaders["train"], device)
        val_loss = evaluate_model(model, data_loaders["test"], loss_fcn, device)
        scheduler.step(val_loss)
        if best_loss > val_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            op_ctr = 0
        op_ctr += 1
        if op_ctr >= overall_patience:
            logging.info(f"Overall patience threshold ({overall_patience}) hit. Ending Training.")
            break
        logging.info(f"Epoch {epoch} validation loss: {val_loss:.4f}; best loss: {best_loss:.4f}")
    torch.save(model.state_dict(), LAST_MODEL_PATH)


if __name__ == "__main__":
    main()
