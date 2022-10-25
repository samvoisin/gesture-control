# standard libraries
from pathlib import Path

# external libraries
import cv2

# gestrol library
from gestrol.camera import VideoLoaderInterface
from gestrol.fps_monitor import FPSMonitor
from gestrol.frame_pipeline import FramePipeline
from gestrol.frame_stream import FrameStream
from gestrol.modifiers import ChannelSwapModifier
from gestrol.modifiers.extractors.frcnn_extractor import SingleHandFRCNNExtractor, load_frcnn_model
from gestrol.modifiers.type_converters import convert_frame_to_tensor
from gestrol.utils.logging import configure_logging

configure_logging()


def bbox_xywh_coords(bbox):
    x, y = int(bbox[0]), int(bbox[1])
    w = int(bbox[2]) - x
    h = int(bbox[3]) - y
    return x, y, w, h


def make_img_w_bboxes(img, bboxes):
    for bbox in bboxes:
        x, y, w, h = bbox_xywh_coords(bbox)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return img


vp = Path("./").resolve() / "data" / "videos" / "hand_test_vid.webm"
vli = VideoLoaderInterface(video_path=vp)
fs = FrameStream(camera=vli)


mp = Path("./").resolve() / "models" / "frcnn_hand_detect.pt"
model = load_frcnn_model(mp)
mod_pipe = [
    ChannelSwapModifier(),
    convert_frame_to_tensor,
    SingleHandFRCNNExtractor(model=model),
]

frame_pipeline = FramePipeline(modifier_pipeline=mod_pipe)
fpsm = FPSMonitor()


cv2.namedWindow("preview")

for frame in fs.stream_frames():
    fpsm.monitor_fps(frame)
    procd_frame = frame_pipeline.process_frame(frame)
    if procd_frame is None:
        continue
    extr_frame = procd_frame.numpy()[0, :, :]  # TODO: need to solve this problem
    cv2.imshow("preview", extr_frame)
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break

cv2.destroyWindow("preview")
