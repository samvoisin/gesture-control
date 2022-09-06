# standard libraries
import logging
import os
from pathlib import Path

# external libraries
import cv2
import torch

# gestrol library
from gestrol import FramePipeline
from gestrol.modifiers import (
    ChannelDimOrderModifier,
    ChannelSwapModifier,
    NumpyToImageModifier,
    ReverseNormalizeModifier,
    ScalarModifier,
    SSDPreprocModifier,
    TensorToNumpyModifier,
)
from gestrol.modifiers.extractors import SingleHandSSDExtractor
from gestrol.modifiers.extractors.ssd_extractor import load_ssd_model

data_dir = Path("./data/senz3d_dataset").resolve()
restructured_dir = data_dir / "restructured"
extracted_dir = data_dir / "extracted"
model_path = Path("./models/ssd_hand_detect.pt")


ssd_model = load_ssd_model(model_path)


# the actual frame pipeline to be used in controller
fp = FramePipeline(
    modifier_pipeline=[
        ChannelSwapModifier(),
        NumpyToImageModifier(),
        SSDPreprocModifier(),  # NOTE: ssd model has normalize step built in consider removing this step and retraining
        SingleHandSSDExtractor(model=ssd_model),
    ]
)


# revert image to be saved as `.png` for visual inspection
viz_fp = FramePipeline(
    modifier_pipeline=[
        ReverseNormalizeModifier(),
        TensorToNumpyModifier(),
        ChannelDimOrderModifier(mode="last"),
        ScalarModifier(),
        ChannelSwapModifier(),
    ]
)


os.mkdir(extracted_dir)
for src_gdir in restructured_dir.iterdir():
    if not src_gdir.is_dir():
        continue

    dst_gdir = extracted_dir / src_gdir.stem
    os.mkdir(dst_gdir)

    for img_path in src_gdir.iterdir():
        img = cv2.imread(str(img_path))
        img = fp(img)
        if img is None:
            logging.warn(f"Extraction failure on {img_path}.")
            continue
        img_write_path = dst_gdir / (img_path.stem + ".pt")
        torch.save(img, img_write_path)  # save the tensor as `.pt` file

        viz_img = viz_fp(img)  # create an image file to visually inspect
        viz_img_write_path = dst_gdir / (img_path.stem + ".png")
        cv2.imwrite(str(viz_img_write_path), viz_img)
