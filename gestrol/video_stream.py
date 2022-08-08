# standard libraries
from pathlib import Path

# external libraries
import cv2

# gestrol library
from gestrol.camera import VideoLoaderInterface
from gestrol.extractors.ssd_extractor import SingleHandSSDExtractor
from gestrol.frame_pipeline import FramePipeline
from gestrol.frame_pipeline.modifiers import ChannelSwapModifier, NumpyToImageModifier, SSDPreprocModifier
from gestrol.frame_stream import FrameStream


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


vp = Path("/home/samvoisin/Videos/Webcam/hand_test_vid.webm")
vli = VideoLoaderInterface(video_path=vp)
fs = FrameStream(camera=vli)


mod_pipe = [
    ChannelSwapModifier(),
    NumpyToImageModifier(),
    SSDPreprocModifier(),
]
frame_pipeline = FramePipeline(modifier_pipeline=mod_pipe)

ssd_extractor = SingleHandSSDExtractor()


cv2.namedWindow("preview")

for frame in fs.stream_frames():
    procd_frame = frame_pipeline(frame)
    extr_frame = ssd_extractor(procd_frame)
    extr_frame = extr_frame.numpy()[0, :, :]  # need to solve this problem
    cv2.imshow("preview", extr_frame)
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break

cv2.destroyWindow("preview")
