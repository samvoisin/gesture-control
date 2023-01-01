# standard libraries
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# external libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_fscore_support
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


logging.basicConfig(filename="gc_cnn_train.log", level=logging.INFO)


def _split_samples_list(samples: List[Any], split_frac: float, rng=np.random.Generator) -> Tuple[List[Any], List[Any]]:
    n = int(len(samples) * split_frac)
    sampled = rng.choice(samples, replace=False, size=n)
    remaining = list(set(samples) - set(sampled))
    return sampled, remaining


def _copy_bulk(from_paths: List[Path], to_dir: Path):
    for from_path in from_paths:
        to_path = to_dir / from_path.name
        shutil.copy(from_path, to_path)


class DatasetDirBuilder:
    def __init__(
        self,
        extracted_path: Optional[Path] = None,
        gesture_classes: Optional[List[str]] = None,
        split: Dict[str, float] = {"train": 0.6, "val": 0.2, "test": 0.2},
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
        self.tvt_dirs = {}
        for ds in datasets:
            self.tvt_dirs[ds] = self.gc_training_dir / ds
            self.tvt_dirs[ds].mkdir()

        # sort samples
        for gesture_class in self.gesture_classes:
            gc_extr_paths = list((self.extracted_path / gesture_class).glob("*.pt"))
            train_paths, remaining = _split_samples_list(
                gc_extr_paths,
                split_frac=self.split["train"],
                rng=self.rng,
            )
            val_paths, test_paths = _split_samples_list(
                remaining,
                split_frac=0.5,
                rng=self.rng,
            )
            split_samples = {
                "train": train_paths,
                "val": val_paths,
                "test": test_paths,
            }
            for dataset in datasets:
                dataset_dir = self.gc_training_dir / dataset / gesture_class
                dataset_dir.mkdir()
                _copy_bulk(split_samples[dataset], dataset_dir)

    def __del__(self):
        shutil.rmtree(self.gc_training_dir)


ttddb = DatasetDirBuilder()

img_dim = 300  # smaller than many images. may need to increase.
data_transforms = {
    "train": transforms.Compose(
        [
            transforms.Resize(size=(img_dim, img_dim)),
            transforms.GaussianBlur(kernel_size=3),
            transforms.RandomRotation(degrees=10),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize(size=(img_dim, img_dim)),
        ]
    ),
}

dataset_folders = {
    "train": DatasetFolder(
        root=ttddb.tvt_dirs["train"], loader=torch.load, extensions=(".pt",), transform=data_transforms["train"]
    ),
    "val": DatasetFolder(
        root=ttddb.tvt_dirs["val"], loader=torch.load, extensions=(".pt",), transform=data_transforms["val"]
    ),
    "test": DatasetFolder(
        root=ttddb.tvt_dirs["test"], loader=torch.load, extensions=(".pt",), transform=data_transforms["val"]
    ),
}


data_loaders = {
    "train": DataLoader(dataset_folders["train"], shuffle=True),
    "val": DataLoader(dataset_folders["val"]),
    "test": DataLoader(dataset_folders["val"]),
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


def evaluate_model(model, data_loader, loss_function, device, test_set: bool = False):
    model.eval()

    y_true = []
    y_pred = []

    tot_loss = 0
    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            pred = model(imgs).reshape(1, -1)
            loss = loss_function(pred, labels).item()
            tot_loss += loss
            y_true.append(labels.item())
            y_pred.append(pred.argmax().item())
    mean_loss = tot_loss / len(data_loader)
    prec, recall, f_score, _ = precision_recall_fscore_support(y_true, y_pred)

    if test_set:
        _ = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)  # don't use this other than to save img
        plt.savefig("conf_matrix.png")
        pd.DataFrame(
            {"mean_loss": mean_loss, "precision": prec.mean(), "recall": recall.mean(), "fscore": f_score.mean()}
        ).to_csv("test_results.csv")

    return mean_loss, prec, recall, f_score


def _update_training_history(history: Dict, loss: float, p: np.ndarray, r: np.ndarray, f: np.ndarray) -> Dict:
    history["val_loss"].append(loss)
    history["precision"].append(p.mean())
    history["recall"].append(r.mean())
    history["fscore"].append(f.mean())
    return history


def main():
    device = torch.device("cuda")
    model = construct_model()
    model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    scheduler = ReduceLROnPlateau(optimizer, patience=10)
    epochs = 250
    overall_patience = 40  # end training if we hit this number of epochs w/out improvement
    training_history = {"val_loss": [], "precision": [], "recall": [], "fscore": []}

    op_ctr = 0
    best_loss = np.inf
    for epoch in range(epochs):
        logging.info(f"Current learning rate: {optimizer.state_dict()['param_groups'][0]['lr']}")
        train_one_epoch(model, optimizer, loss_fcn, data_loaders["train"], device)
        val_loss, p, r, f = evaluate_model(model, data_loaders["val"], loss_fcn, device)
        training_history = _update_training_history(training_history, val_loss, p, r, f)
        scheduler.step(val_loss)
        if best_loss > val_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            op_ctr = 0
        op_ctr += 1

        if op_ctr >= overall_patience:
            logging.info(f"Overall patience threshold ({overall_patience}) reached. Ending Training.")
            break

        logging.info(f"Epoch {epoch} validation loss: {val_loss:.4f}; best loss: {best_loss:.4f}")

    logging.info(f"Training complete. Final model saved at {LAST_MODEL_PATH}.")
    torch.save(model.state_dict(), LAST_MODEL_PATH)

    # save history plots
    fig, ax = plt.subplots(2, 2, figsize=(20, 20))
    r, c = 0, 0
    for k, v in training_history.items():
        ax[r, c].plot(range(len(v)), v)
        ax[r, c].set_title(k)
        r += 1
        if r == 2:
            c += 1
            r = 0
    fig.savefig("gc_cnn_training_hist.png")

    # run best model on test set
    best_model = construct_model(model_path=BEST_MODEL_PATH)
    best_model.to(device)
    evaluate_model(best_model, data_loaders["test"], loss_fcn, device, test_set=True)


if __name__ == "__main__":
    main()
