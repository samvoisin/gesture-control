# standard libraries
from queue import Queue
from typing import Optional, Protocol

# gestrol library
from gestrol.fps_monitor import FPSMonitor
from gestrol.frame_pipeline import FramePipeline
from gestrol.frame_stream import FrameStream
from gestrol.modifiers.base import Frame
from gestrol.utils.logging import configure_logging

configure_logging()


class GestureClassifierProtocol(Protocol):
    def infer_gesture(self, frame: Frame) -> Optional[int]:
        ...


class ClassificationRegularizerProtocol(Protocol):
    queue: Queue

    def put(self, item: int):
        ...

    def vote(self) -> int:
        ...


class CommandControllerProtocol(Protocol):
    def __init__(self) -> None:
        ...

    def execute_command(self, control_signal: int) -> None:
        ...


class GestureController:
    """
    Master class for gestrol library. Contains all components necessary to control machine with gestures.
    """

    def __init__(
        self,
        frame_stream: FrameStream,
        frame_pipeline: FramePipeline,
        gesture_classifier: GestureClassifierProtocol,
        command_controller: CommandControllerProtocol,
        classification_regularizer: ClassificationRegularizerProtocol,
        fps_monitor: Optional[FPSMonitor] = None,
    ):
        """
        Initiate method.

        Args:
            frame_stream: frame stream object with active camera interface
            frame_pipeline: pipeline for pre-processing frames including extractor
            gesture_classifier: gesture classifier model
            command_controller: command controller object
            classification_regularizer: regularizer to prevent volatility in commands sent to command controller
            fps_monitor: frames per second monitor. Defaults to None.
        """
        self.frame_stream = frame_stream
        self.fp = frame_pipeline
        self.gc = gesture_classifier
        self.class_reg = classification_regularizer
        self.cc = command_controller

        if fps_monitor:
            self.fps_monitor = fps_monitor
            self.monitor_fps = self.fps_monitor.monitor_fps
        else:
            self.monitor_fps = lambda frame: frame

    def _coord_class_reg(self, inferred_label: int) -> Optional[int]:
        """
        Private method for coordinating classification regularizer.

        Args:
            inferred_label: labeled inferred by gesture classifier

        Returns:
            None if regularizer does not vote; highest voted label if it does
        """
        self.class_reg.put(inferred_label)
        if self.class_reg.queue.full():
            return self.class_reg.vote()
        return None

    def activate(self):
        """
        Activate gesture control interface.
        """
        for frame in self.frame_stream.stream_frames():
            frame = self.monitor_fps(frame)
            frame = self.fp.process_frame(frame)
            if frame is None:
                continue
            label = self.gc.infer_gesture(frame)
            ctrl_signal = self._coord_class_reg(label)
            if ctrl_signal is None:
                continue
            self.cc.execute_command(ctrl_signal)

    def build_from_config(self):
        raise NotImplementedError()
