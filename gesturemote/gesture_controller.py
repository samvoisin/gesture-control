# standard libraries
import logging
from typing import Dict, List

# external libraries
import cv2
import numpy as np
import pyautogui as pag
from numpy.linalg import norm
from PIL import Image

# gesturemote library
from gesturemote.camera import OpenCVCameraInterface
from gesturemote.detector.mp_detector import LandmarkGestureDetector
from gesturemote.fps_monitor import FPSMonitor


class GestureController:
    """
    Gesture controller class.
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
        self.is_active = False
        self.click_down = False

        self.screen_width, self.screen_height = pag.size()

        self.df: Dict[str, List[float]] = {"x": [], "y": [], "z": []}

    def detect_click(self, finger_coords: np.ndarray):
        """
        Detect if the user is clicking.

        Args:
            finger_coords: coordinates of the finger landmarks.

        Returns:
            True if the user is clicking, False otherwise.
        """
        thumb_tip_vector = finger_coords[:, 0, 0]
        index_finger_tip_vector = finger_coords[:, 0, 1]
        middle_finger_tip_vector = finger_coords[:, 0, 2]

        middle_finger_to_index_finger_tip = norm(index_finger_tip_vector - middle_finger_tip_vector)
        middle_finger_to_thumb_tip = norm(thumb_tip_vector - middle_finger_tip_vector)

        self.logger.info(f"index to middle finger: {middle_finger_to_index_finger_tip}")
        self.logger.info(f"thumb to middle finger: {middle_finger_to_thumb_tip}")

        if middle_finger_to_index_finger_tip > middle_finger_to_thumb_tip and not self.click_down:  # click down
            self.click_down = True
            pag.mouseDown()
            self.logger.info("click down")
        elif middle_finger_to_index_finger_tip < middle_finger_to_thumb_tip and self.click_down:  # release click
            self.click_down = False
            pag.mouseUp()
            self.logger.info("click released")

    def activate(self, video_preview: bool = False):
        """
        Activate the gesture controller.
        """
        prvw_img_size = 320
        RED = (0, 0, 255)
        activate_gesture_threshold = 5
        activate_gesture_counter = 0

        self.logger.info("Gesture controller initialized.")
        for frame in self.camera.stream_frames():
            if self.monitor_fps:
                frame = self.fps_monitor.monitor_fps(frame)

            prediction = self.recognizer.predict(frame)
            if prediction is None:
                continue

            gesture_label, finger_coords = prediction
            cursor_pt = finger_coords[:, 0, 1]  # index finger tip
            self.logger.info(f"Gesture: {gesture_label}")
            self.logger.info(f"x={cursor_pt[0]:.2f}, y={cursor_pt[1]:.2f}, z={cursor_pt[2]:.2f}")

            if gesture_label == "Closed_Fist":
                activate_gesture_counter += 1
                if activate_gesture_counter > activate_gesture_threshold:
                    self.is_active = not self.is_active
                    self.logger.info(f"control mode is {self.is_active}")
                    activate_gesture_counter = 0
            else:
                activate_gesture_counter = 0

            if self.is_active:
                pag.moveTo(
                    (1 - cursor_pt[0]) * self.screen_width,
                    cursor_pt[1] * self.screen_height,
                )
                self.detect_click(finger_coords)

            if video_preview:
                prvw_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                prvw_img = Image.fromarray(prvw_img).resize((prvw_img_size, prvw_img_size))
                prvw_img = np.array(prvw_img)

                tc_str = (
                    f"x={cursor_pt[0] * self.screen_width:.2f}, y={cursor_pt[1] * self.screen_height:.2f},"
                    f" z={cursor_pt[2]:.2f}"
                )

                cv2.putText(
                    prvw_img,
                    tc_str,
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    RED,
                    thickness=1,
                )

                if self.is_active:
                    cv2.rectangle(prvw_img, (0, 0), (prvw_img_size, prvw_img_size), RED, 2)

                cv2.imshow("Frame", prvw_img)

                key = cv2.waitKey(1)
                if key == ord("q"):
                    return
