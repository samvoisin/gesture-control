# standard libraries
import logging
import math
import sys
from pathlib import Path
from typing import Dict, Optional

# external libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import utils
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.rpn import AnchorGenerator

HAND_DETECT_DIR = Path("data/hand_detect_model").resolve()
DATA_DIR = HAND_DETECT_DIR / "images"

train_data_image_dir = DATA_DIR / Path("./train/")
test_data_image_dir = DATA_DIR / Path("./test/")

train_labels_csv = DATA_DIR / Path("./train_labels.csv")
test_labels_csv = DATA_DIR / Path("./test_labels.csv")

BEST_MODEL_PATH = Path("models/frcnn_hand_detect_mnlrg_best.pt")
LAST_MODEL_PATH = Path("models/frcnn_hand_detect_mnlrg_last.pt")


class HandDetectDataset(Dataset):
    def __init__(self, csv_file: Path, image_dir: Path, transform: transforms.Compose = None):
        self.labels_frame = pd.read_csv(csv_file)
        self.labels_frame.drop("class", axis=1, inplace=True)
        self.image_dir = image_dir
        self.transform = transform
        self.unq_files = self.labels_frame.filename.unique().tolist()

    def __len__(self) -> int:
        return len(self.unq_files)

    def get_bboxes_from_filename(self, filename: str):
        bboxes = self.labels_frame.loc[self.labels_frame["filename"] == filename, ["xmin", "ymin", "xmax", "ymax"]]
        return bboxes.values

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = self.unq_files[idx]
        img_path = self.image_dir / filename
        img = Image.open(img_path).convert("RGB")

        bboxes = self.get_bboxes_from_filename(filename)

        if self.transform:
            img = self.transform(img)

        # convert everything to torch tensor
        img = torch.as_tensor(img, dtype=torch.float32)
        bboxes = torch.as_tensor(bboxes.astype(np.float32), dtype=torch.float32)
        num_objs = bboxes.shape[0]
        labels = torch.ones((num_objs,), dtype=torch.int64)  # only one class type

        target = {
            "boxes": bboxes,
            "labels": labels,
        }
        return img, target


# Data augmentation and normalization for training
data_transforms = {
    "train": transforms.Compose(
        [
            transforms.RandomSolarize(threshold=192, p=0.5),  # can be PIL.Image
            transforms.RandomEqualize(p=0.5),
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.5, hue=0.3),
            transforms.GaussianBlur(kernel_size=(5, 9)),
            transforms.RandomInvert(),
        ]
    ),
    "test": transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    ),
}


image_datasets = {
    "train": HandDetectDataset(
        csv_file=train_labels_csv, image_dir=train_data_image_dir, transform=data_transforms["train"]
    ),
    "test": HandDetectDataset(
        csv_file=test_labels_csv, image_dir=test_data_image_dir, transform=data_transforms["test"]
    ),
}

dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "test"]}

data_loaders = {
    "train": DataLoader(image_datasets["train"], shuffle=True, batch_size=1),
    "test": DataLoader(image_datasets["test"]),
}


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = [image.to(device) for image in images]
        targets = [{t: v[i].to(device) for t, v in targets.items()} for i, _ in enumerate(images)]

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def evaluate_model(model, data_loader, device) -> Dict[str, float]:
    ctr = 0
    avg_loss_dict: Dict[str, float] = {
        "loss_classifier": 0.0,
        "loss_box_reg": 0.0,
        "loss_objectness": 0.0,
        "loss_rpn_box_reg": 0.0,
    }

    for images, targets in data_loader:
        images = [image.to(device) for image in images]
        targets = [{t: v[i].to(device) for t, v in targets.items()} for i, _ in enumerate(images)]

        loss_dict = model(images, targets)
        for loss_type, loss_val in loss_dict.items():
            avg_loss_dict[loss_type] += loss_val.item()

        ctr += 1

    # calc and log mean losses for each loss type
    print("~" * 20)
    logging.warning("Test losses for current epoch:")
    for loss_type in avg_loss_dict.keys():
        avg_loss_dict[loss_type] /= ctr
        logging.warning(f"{loss_type} loss: {avg_loss_dict[loss_type]:.4f}")
    print("~" * 20)

    return avg_loss_dict


def construct_model(model_path: Optional[Path] = None) -> torch.nn.Module:
    model = fasterrcnn_mobilenet_v3_large_fpn(num_classes=2)
    ag = AnchorGenerator(sizes=((128), (256), (512)), aspect_ratios=((0.5, 1.0, 1.5), (0.5, 1.0, 1.5), (0.5, 1.0, 1.5)))
    model.rpn.anchor_generator = ag

    if model_path:
        model.load_state_dict(torch.load(model_path))

    return model


def main():
    torch.manual_seed(1)
    model = construct_model()
    gpu_device = torch.device("cuda")
    model = model.to(gpu_device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

    # training loop
    num_epochs = 100
    overall_patience = 25  # end training if we hit this number of epochs w/out improvement
    test_losses = {
        "loss_classifier": [],
        "loss_box_reg": [],
        "loss_objectness": [],
        "loss_rpn_box_reg": [],
    }

    op_ctr = 0
    best_loss = np.inf
    for epoch in range(num_epochs):
        logging.warning(f"Current learning rate: {optimizer.state_dict()['param_groups'][0]['lr']}")
        train_one_epoch(model, optimizer, data_loaders["train"], gpu_device, epoch, print_freq=1000)

        # evaluate on the test dataset
        test_loss_dict = evaluate_model(model, data_loaders["test"], device=gpu_device)
        val_loss = sum(test_loss_dict.values()) / len(test_loss_dict.values())
        if val_loss < best_loss:
            best_loss = val_loss
            print(f"Model saving to {BEST_MODEL_PATH}")
            print("-" * 20)
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            op_ctr = 0

        lr_scheduler.step(val_loss)

        for loss_type, loss_val in test_loss_dict.items():
            test_losses[loss_type].append(loss_val)

        op_ctr += 1
        if op_ctr >= overall_patience:
            logging.warning(f"Overall patience threshold ({overall_patience}) hit. Ending Training.")
            break

        logging.warning(f"Epoch {epoch} validation loss: {val_loss:.4f}; best loss: {best_loss:.4f}")

    torch.save(model.state_dict(), LAST_MODEL_PATH)

    fig, ax = plt.subplots(nrows=4, ncols=1)
    fig.set_size_inches(18.5, 10.5)
    for i, loss_type in enumerate(test_losses):
        ax[i].plot(range(len(test_losses[loss_type])), test_losses[loss_type])
        ax[i].set_title(loss_type)
    fig.savefig("data/hand_detect_model/frcnn_mnlrg_trng.png")


if __name__ == "__main__":
    main()
