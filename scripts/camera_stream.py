# external libraries
import cv2

# gestrol library
from gestrol.camera import OpenCVCameraInterface
from gestrol.extractors.ssd_extractor import SSDExtractor
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


cam = OpenCVCameraInterface()  # this returns numpy arrays by default
fs = FrameStream(camera=cam)

mod_pipe = [ChannelSwapModifier(), NumpyToImageModifier(), SSDPreprocModifier()]
fp = FramePipeline(modifier_pipeline=mod_pipe)

ssd_extractor = SSDExtractor()


cv2.namedWindow("preview")

for frame in fs.stream_frames():
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # adds a lot of latency
    procd_frame = fp(frame)
    bboxes = ssd_extractor(procd_frame)
    frame = make_img_w_bboxes(frame, bboxes)
    cv2.imshow("preview", frame)
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break

cv2.destroyWindow("preview")
