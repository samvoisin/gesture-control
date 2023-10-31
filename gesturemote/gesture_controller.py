import logging
from typing import Optional, Protocol

import cv2
import numpy as np
import pyautogui as pag
from numpy.linalg import norm
from PIL import Image

from gesturemote.camera import OpenCVCameraInterface
from gesturemote.camera.base import CameraInterface
from gesturemote.detector.mp_detector import LandmarkGestureDetector
from gesturemote.fps_monitor import FPSMonitor
from gesturemote.gesture_handler import Gesture, GestureHandler

DEFAULT_GESTURES = [
    Gesture("Thumb_Down", 3, lambda: pag.press("pagedown")),
    Gesture("Thumb_Up", 3, lambda: pag.press("pageup")),
]


class DetectorProtocol(Protocol):
    def predict(self, frame: np.ndarray) -> Optional[tuple[str, np.ndarray]]:
        ...


class GestureController:
    """
    Gesture controller class.
    """

    def __init__(
        self,
        cursor_smoothing_param: int = 5,
        activate_gesture_threshold: int = 7,
        gestures: Optional[list[Gesture]] = None,
        detector: Optional[DetectorProtocol] = None,
        camera: Optional[CameraInterface] = None,
        monitor_fps: bool = False,
        verbose: bool = False,
    ):
        """
        Initiate method.

        Args:
            fps_monitor: frames per second monitor. Defaults to None.
        """
        self.lagged_cursor_position = np.empty(shape=(cursor_smoothing_param, 3))

        gestures = gestures or DEFAULT_GESTURES
        gestures.append(Gesture("Closed_Fist", activate_gesture_threshold, self._toggle_active))  # control gesture
        self.gesture_handler = GestureHandler(gestures)

        self.detector = detector or LandmarkGestureDetector()
        self.camera = camera or OpenCVCameraInterface()

        self.monitor_fps = monitor_fps
        if self.monitor_fps:
            self.fps_monitor = FPSMonitor()

        self.logger = logging.getLogger(__name__)
        if verbose:
            logging.basicConfig(level=logging.INFO)

        self.is_active = False
        self.click_down = False

        self.screen_width, self.screen_height = pag.size()

    def _toggle_active(self):
        self.is_active = not self.is_active
        self.logger.info(f"Gesture controller is active: {self.is_active}")

    def detect_primary_click(self, finger_coordinates: np.ndarray):
        """
        Detect if the user is clicking.

        Args:
            finger_coordinates: coordinates of the finger landmarks.

        Returns:
            True if the user is clicking, False otherwise.
        """
        thumb_tip_vector = finger_coordinates[:, 0, 0]
        index_finger_tip_vector = finger_coordinates[:, 0, 1]
        middle_finger_tip_vector = finger_coordinates[:, 0, 2]

        middle_finger_to_index_finger_tip = norm(index_finger_tip_vector - middle_finger_tip_vector)
        middle_finger_to_thumb_tip = norm(thumb_tip_vector - middle_finger_tip_vector)

        self.logger.info(f"index to middle finger: {middle_finger_to_index_finger_tip}")
        self.logger.info(f"thumb to middle finger: {middle_finger_to_thumb_tip}")

        if middle_finger_to_index_finger_tip > middle_finger_to_thumb_tip and not self.click_down:  # primary click down
            pag.mouseDown()
            self.click_down = True
            self.logger.info("primary click down")
        elif (
            middle_finger_to_index_finger_tip < middle_finger_to_thumb_tip and self.click_down
        ):  # release primary click
            pag.mouseUp()
            print("y")
            self.click_down = False
            self.logger.info("primary click released")

    def detect_secondary_click(self, finger_coordinates: np.ndarray):
        """
        Detect if the user is performing a secondary click.

        Args:
            finger_coordinates: coordinates of the finger landmarks.
        """
        thumb_tip_vector = finger_coordinates[:, 0, 0]
        ring_finger_tip_vector = finger_coordinates[:, 0, 3]

        ring_finger_to_thumb_tip = norm(thumb_tip_vector - ring_finger_tip_vector)
        self.logger.info(f"secondary click distance: {ring_finger_to_thumb_tip:.3f}")

        if ring_finger_to_thumb_tip < 0.1 and not self.click_down:  # secondary click TODO: make this dynamic or param?
            pag.click(button="right")

    def get_cursor_position(self, landmarks: np.ndarray):
        """
        Smooth the cursor position.

        Args:
            landmarks: coordinates of the finger landmarks.

        Returns:
            Smoothed cursor position.
        """
        self.lagged_cursor_position = np.roll(self.lagged_cursor_position, 1, axis=0)
        self.lagged_cursor_position[0, :] = landmarks[:, 0, 1]  # (x,y,z), tip, finger one (index finger)

        return np.mean(self.lagged_cursor_position, axis=0)

    def activate(self, video_preview: bool = False):
        """
        Activate the gesture controller.
        """
        prvw_img_size = 320
        RED = (0, 0, 255)

        self.logger.info("Gesture controller initialized.")
        for frame in self.camera.stream_frames():
            if self.monitor_fps:
                frame = self.fps_monitor.monitor_fps(frame)

            if video_preview:
                prvw_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                prvw_img = Image.fromarray(prvw_img).resize((prvw_img_size, prvw_img_size))
                prvw_img = np.array(prvw_img)
            else:
                prvw_img = None

            prediction = self.detector.predict(frame)
            if prediction is None:
                self.click_down = False  # ensure click is released when cursor moves off screen.

                if video_preview and prvw_img is not None:
                    cv2.imshow("Frame", prvw_img)

                    key = cv2.waitKey(1)
                    if key == ord("q"):
                        return

                continue

            gesture_label, finger_landmarks = prediction
            cursor_pt = self.get_cursor_position(finger_landmarks)
            self.logger.info(f"Gesture: {gesture_label}")
            self.logger.info(f"x={cursor_pt[0]:.2f}, y={cursor_pt[1]:.2f}, z={cursor_pt[2]:.2f}")

            self.gesture_handler.handle(gesture_label)

            if self.is_active:
                pag.moveTo(
                    (1 - cursor_pt[0]) * self.screen_width,
                    cursor_pt[1] * self.screen_height,
                )
                self.detect_primary_click(finger_landmarks)
                self.detect_secondary_click(finger_landmarks)

            if video_preview and prvw_img is not None:
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
