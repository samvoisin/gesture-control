# standard libraries
import logging
from time import sleep
from typing import Dict, List

# external libraries
import cv2
import numpy as np
import pandas as pd
import pyautogui as pag
from PIL import Image

# gesturemote library
from gesturemote.camera import OpenCVCameraInterface
from gesturemote.detector.mp_detector import LandmarkGestureDetector
from gesturemote.fps_monitor import FPSMonitor
from gesturemote.voting_queue import VoteQueue


class GestureController:
    """
    Master class for gestrol library. Contains all components necessary to control machine with gestures.
    """

    def __init__(
        self,
        monitor_fps: bool = False,
        verbose: bool = False,
    ):
        """
        Initiate method.

        Args:
            fps_monitor: frames per second monitor. Defaults to None.
        """
        self.recognizer = LandmarkGestureDetector()

        self.monitor_fps = monitor_fps
        if self.monitor_fps:
            self.fps_monitor = FPSMonitor()

        self.logger = logging.getLogger(__name__)
        if verbose:
            logging.basicConfig(level=logging.INFO)

        self.camera = OpenCVCameraInterface()
        self.voting_queue = VoteQueue(maxsize=4, verbose=verbose)
        self.is_active = False

        self.screen_width, self.screen_height = pag.size()

        self.df: Dict[str, List[float]] = {"x": [], "y": [], "z": []}

    def activate(self, video_preview: bool = False):
        """
        Activate the gesture controller.
        """
        prvw_img_size = 320
        RED = (0, 0, 255)

        sleep(3)  # wait for camera to initialize
        self.logger.info("Gesture controller initialized.")
        for frame in self.camera.stream_frames():
            if self.monitor_fps:
                frame = self.fps_monitor.monitor_fps(frame)

            prediction = self.recognizer.predict(frame)
            if prediction is None:
                continue

            gesture_class, index_finger_coords = prediction
            self.voting_queue.put(gesture_class)
            finger1_x, finger1_y, finger1_z = index_finger_coords

            self.logger.info(f"x={finger1_x:.2f}, y={finger1_y:.2f}, z={finger1_z:.2f}")

            if self.voting_queue.is_full():
                gesture_class_label = self.voting_queue.vote()

                if gesture_class_label == "Closed_Fist":
                    self.is_active = not self.is_active
                    self.logger.info(f"control mode is {self.is_active}")
                    continue

            if self.is_active:
                pag.moveTo(
                    (1 - finger1_x) * self.screen_width,
                    finger1_y * self.screen_height,
                )

            # collect diagnostics
            self.df["x"].append(finger1_x)
            self.df["y"].append(finger1_y)
            self.df["z"].append(finger1_z)

            if video_preview:
                prvw_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                prvw_img = Image.fromarray(prvw_img).resize((prvw_img_size, prvw_img_size))
                prvw_img = np.array(prvw_img)

                cv2.putText(
                    prvw_img,
                    f"x={finger1_x * self.screen_width:.2f}, y={finger1_y * self.screen_height:.2f}, z={finger1_z:.2f}",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    RED,
                    thickness=1,
                )

                cv2.imshow("Frame", prvw_img)

                key = cv2.waitKey(1)
                if key == ord("q"):
                    df = pd.DataFrame(self.df)
                    df.to_csv("data.csv", index=False)
                    return
