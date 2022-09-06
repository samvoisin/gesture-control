# standard libraries
from pathlib import Path

# external libraries
import cv2

# gestrol library
from gestrol.camera import VideoLoaderInterface
from gestrol.frame_pipeline import FramePipeline
from gestrol.frame_stream import FrameStream
from gestrol.modifiers import ChannelSwapModifier
from gestrol.modifiers.extractors.ssd_extractor import SingleHandSSDExtractor, load_ssd_model
from gestrol.modifiers.type_converters import FrameToTensorModifier


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


mp = Path("./").resolve() / "models" / "ssd_hand_detect.pt"
vp = Path("./").resolve() / "data" / "videos" / "hand_test_vid.webm"
# vp = Path("./").resolve() / "data" / "gesture_training_vids" / "h_f_down" / "0.mp4"
vli = VideoLoaderInterface(video_path=vp)
fs = FrameStream(camera=vli)

model = load_ssd_model(model_path=mp)

mod_pipe = [
    ChannelSwapModifier(),
    FrameToTensorModifier(),
    SingleHandSSDExtractor(model=model),
]
frame_pipeline = FramePipeline(modifier_pipeline=mod_pipe)


cv2.namedWindow("preview")

for frame in fs.stream_frames():
    procd_frame = frame_pipeline(frame)
    if procd_frame is None:
        continue
    extr_frame = procd_frame.numpy()[0, :, :]  # TODO: need to solve this problem
    cv2.imshow("preview", extr_frame)
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break

cv2.destroyWindow("preview")
