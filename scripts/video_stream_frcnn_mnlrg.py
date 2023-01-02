# standard libraries
from pathlib import Path

# external libraries
import cv2

# gestrol library
from gestrol import FramePipeline
from gestrol.camera import VideoLoaderInterface
from gestrol.fps_monitor import FPSMonitor
from gestrol.frame_stream import FrameStream
from gestrol.modifiers import ChannelSwapModifier, FrameToTensorModifier
from gestrol.modifiers.extractors.frcnn_mnlrg_extractor import SingleHandMobileNetExtractor, load_frcnn_model

vp = Path("./").resolve() / "data" / "videos" / "hand_test_vid.webm"
vli = VideoLoaderInterface(video_path=vp)
fs = FrameStream(camera=vli)


mp = Path("./").resolve() / "models" / "frcnn_hand_detect_mnlrg.pt"
model = load_frcnn_model(mp)

mod_pipe = [
    ChannelSwapModifier(),
    FrameToTensorModifier(),
    SingleHandMobileNetExtractor(model=model),
]

frame_pipeline = FramePipeline(modifier_pipeline=mod_pipe)
fpsm = FPSMonitor()


cv2.namedWindow("preview")

null_ctr = 0
for frame in fs.stream_frames():
    procd_frame = frame_pipeline.process_frame(frame)
    fpsm.monitor_fps(frame)
    if procd_frame is None:
        null_ctr += 1
        if null_ctr >= 20:
            break
        continue
    extr_frame = procd_frame.numpy()[0, :, :]  # TODO: need to solve this problem
    cv2.imshow("preview", extr_frame)
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break

cv2.destroyWindow("preview")
