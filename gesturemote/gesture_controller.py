# standard libraries
import logging
from typing import Callable, Dict

# external libraries
import cv2
import numpy as np
import torch
from PIL import Image

# gestrol library
from gesturemote.camera import OpenCVCameraInterface
from gesturemote.detector import TARGETS, build_mobilenet_small, preprocess_mobilenet_small
from gesturemote.fps_monitor import FPSMonitor
from gesturemote.state_controller import StateController
from gesturemote.voting_queue import PopularVoteQueue


class GestureController:
    """
    Master class for gestrol library. Contains all components necessary to control machine with gestures.
    """

    def __init__(
        self,
        routines: Dict[str, Callable[[], None]],
        device: str = "cpu",
        monitor_fps: bool = False,
        verbose: bool = False,
    ):
        """
        Initiate method.

        Args:
            fps_monitor: frames per second monitor. Defaults to None.
        """
        self.monitor_fps = monitor_fps
        if self.monitor_fps:
            self.fps_monitor = FPSMonitor()
        self._device = torch.device(device)
        self.verbose = verbose
        if self.verbose:
            logging.basicConfig(level=logging.INFO)

        self.model = build_mobilenet_small()
        self.model = self.model.to(self._device)
        self.camera = OpenCVCameraInterface(0)
        self.voting_queue = PopularVoteQueue(maxsize=3, verbose=self.verbose)
        self.state_controller = StateController(verbose=self.verbose)

        state_routine = {
            "0303": self.state_controller,  # fist, fist
        }

        self._routines: Dict[str, Callable[[], None]] = {**routines, **state_routine}
        self._max_routine_length = max(len(routine) for routine in self._routines.keys())

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def activate(self, video_preview: bool = False):
        """
        Activate the gesture controller.
        """

        gesture_stream = ""
        num_hands = 1
        prvw_img_size = 320
        threshold = 0.8
        GREEN = (0, 255, 0)
        RED = (0, 0, 255)

        for frame in self.camera.stream_frames():
            if self.monitor_fps:
                frame = self.fps_monitor.monitor_fps(frame)

            procd_img = preprocess_mobilenet_small(frame)

            with torch.no_grad():
                model_output = self.model(procd_img.to(self._device))[0]

            scores = model_output["scores"][:num_hands]
            boxes = model_output["boxes"][:num_hands]
            labels = model_output["labels"][:num_hands]

            if video_preview:
                prvw_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                prvw_img = Image.fromarray(prvw_img)
                prvw_img = prvw_img.resize((prvw_img_size, prvw_img_size))
                prvw_img = np.array(prvw_img)

            for i in range(min(num_hands, len(boxes))):
                if scores[i] > threshold:
                    self.voting_queue.put(int(labels[i]))

                    if self.voting_queue.is_full():
                        gesture_class_label = self.voting_queue.vote()
                        gesture: str = (
                            str(gesture_class_label)
                            if len(str(gesture_class_label)) == 2
                            else f"0{gesture_class_label}"
                        )

                        if self.verbose:
                            self.logger.info(f"gesture class is {gesture}")

                        gesture_stream += gesture
                        if len(gesture_stream) > self._max_routine_length:
                            gesture_stream = ""
                            continue

                        if self.verbose:
                            self.logger.info(f"gesture stream is {gesture_stream}")

                        if gesture_stream in self._routines.keys():
                            routine = self._routines[gesture_stream]

                            if self.verbose:
                                self.logger.info(f"Routine is {routine}")

                            if routine == self.state_controller:
                                if self.verbose:
                                    self.logger.info("Toggling controller state")

                                routine()

                            elif self.state_controller.active:
                                if self.verbose:
                                    self.logger.info(f"Executing routine {routine}")

                                routine()

                                gesture_stream = ""

                    if video_preview:
                        x1 = int((boxes[i][0]))
                        y1 = int((boxes[i][1]))
                        x2 = int((boxes[i][2]))
                        y2 = int((boxes[i][3]))
                        cv2.rectangle(prvw_img, (x1, y1), (x2, y2), GREEN, thickness=3)

                        cv2.putText(
                            prvw_img,
                            TARGETS[int(labels[i])],
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2,
                            RED,
                            thickness=3,
                        )

            if self.state_controller.active and video_preview:
                cv2.rectangle(prvw_img, (0, 0), (prvw_img_size, prvw_img_size), RED, thickness=3)
                cv2.putText(
                    prvw_img,
                    "CONTROL MODE",
                    (10, 310),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    RED,
                    thickness=3,
                )
            elif not self.state_controller.active and video_preview:
                cv2.rectangle(prvw_img, (0, 0), (prvw_img_size, prvw_img_size), GREEN, thickness=3)

            if video_preview:
                cv2.imshow("Frame", prvw_img)

            key = cv2.waitKey(1)
            if key == ord("q"):
                return
