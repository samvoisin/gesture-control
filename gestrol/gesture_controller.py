# standard libraries
from typing import Any, Optional, Protocol

# gestrol library
from gestrol.fps_monitor import FPSMonitor
from gestrol.frame_pipeline import FramePipeline
from gestrol.frame_stream import FrameStream
from gestrol.gesture_classifier.base import GestureClassifier


class ClassificationRegularizerProtocol(Protocol):
    def full(self) -> bool:
        ...

    def put(self, item: Any):
        ...

    def vote(self):
        ...


class CommandControllerProtocol(Protocol):
    def __init__(self) -> None:
        ...

    def execute_command(self, control_signal: int) -> None:
        ...


class GestureController:
    def __init__(
        self,
        frame_stream: FrameStream,
        frame_pipeline: FramePipeline,
        gesture_classifier: GestureClassifier,
        command_controller: CommandControllerProtocol,
        classification_regularizer: ClassificationRegularizerProtocol,
        fps_monitor: Optional[FPSMonitor] = None,
    ):
        self.frame_stream = frame_stream
        self.fp = frame_pipeline
        self.gc = gesture_classifier
        self.class_reg = classification_regularizer
        self.cc = command_controller
        self.fps_monitor = fps_monitor  # TODO: implement this

    def _coord_class_reg(self, inferred_label: int) -> Optional[int]:
        self.class_reg.put(inferred_label)
        if self.class_reg.full():
            return self.class_reg.vote()
        return None

    def activate(self):
        for frame in self.frame_stream.stream_frames():
            frame = self.fp.process_frame(frame)
            if frame is None:
                continue
            label = self.gc.infer_gesture(frame)
            ctrl_signal = self._coord_class_reg(label)
            if ctrl_signal is None:
                continue
            self.cc.execute_command(ctrl_signal)
