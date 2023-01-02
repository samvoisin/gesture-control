# standard libraries
import logging
from pathlib import Path

# external libraries
import cv2
import torch
from tqdm import tqdm

# gestrol library
from gestrol import FramePipeline
from gestrol.modifiers import (
    ChannelDimOrderModifier,
    ChannelSwapModifier,
    FrameToTensorModifier,
    ScalarModifier,
    TensorToNumpyModifier,
)
from gestrol.modifiers.extractors.frcnn_mnlrg_extractor import SingleHandMobileNetExtractor

data_dir = Path("./data/senz3d_dataset").resolve()
restructured_dir = data_dir / "restructured"
extracted_dir = data_dir / "extracted"
model_path = Path("./models/frcnn_hand_detect_mnlrg_best.pt")


# model = load_frcnn_model(model_path=model_path)


# the actual frame pipeline to be used in controller
mod_pipe = [
    ChannelSwapModifier(),
    FrameToTensorModifier(),
    SingleHandMobileNetExtractor(),
]

fp = FramePipeline(modifier_pipeline=mod_pipe)


# revert image to be saved as `.png` for visual inspection
viz_fp = FramePipeline(
    modifier_pipeline=[
        TensorToNumpyModifier(),
        ChannelDimOrderModifier(mode="last"),
        ScalarModifier(),
        ChannelSwapModifier(),
    ]
)


extracted_dir.mkdir(exist_ok=True)
for src_gdir in tqdm(restructured_dir.iterdir()):
    if not src_gdir.is_dir():
        continue

    dst_gdir = extracted_dir / src_gdir.stem
    dst_gdir.mkdir(exist_ok=True)

    for img_path in src_gdir.iterdir():
        img = cv2.imread(str(img_path))
        img = fp.process_frame(img)
        if img is None:
            logging.warning(f"Extraction failure on {img_path}.")
            continue
        img_write_path = dst_gdir / (img_path.stem + ".pt")
        torch.save(img, img_write_path)  # save the tensor as `.pt` file

        viz_img = viz_fp.process_frame(img)  # create an image file to visually inspect
        viz_img_write_path = dst_gdir / (img_path.stem + ".png")
        cv2.imwrite(str(viz_img_write_path), viz_img)
