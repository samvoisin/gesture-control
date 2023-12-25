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
        cursor_sensitivity: int = 5,
        activate_gesture_threshold: int = 7,
        click_threshold: float = 0.1,
        frame_margin: float = 0.1,
        gestures: Optional[list[Gesture]] = None,
        detector: Optional[DetectorProtocol] = None,
        camera: Optional[CameraInterface] = None,
        monitor_fps: bool = False,
        verbose: bool = False,
    ):
        """
        Args:
            cursor_sensitivity: Number of frames to lag the cursor position.
            activate_gesture_threshold: Number of frames to hold the activate gesture to toggle the controller.
            click_threshold: Distance between finger digits to register a primary or secondary click.
            frame_margin: percentage of the frame to pad. Ensures the cursor can access elements on edge of screen.
            gestures: List of gestures to handle.
            detector: Detector to use for gesture recognition.
            camera: Camera interface to use.
            monitor_fps: Whether to monitor the FPS of the camera.
            verbose: Send log output to terminal.
        """
        self.lagged_index_finger_landmark = np.empty(shape=(cursor_sensitivity, 2))
        self.click_threshold = click_threshold
        self._frame_margin_min = frame_margin
        self._frame_margin_max = 1 - frame_margin

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
        middle_finger_tip_vector = finger_coordinates[:, 0, 2]

        middle_finger_to_thumb_tip = norm(thumb_tip_vector - middle_finger_tip_vector)

        self.logger.info(f"thumb to middle finger: {middle_finger_to_thumb_tip}")

        if middle_finger_to_thumb_tip < self.click_threshold and not self.click_down:  # primary click down
            pag.mouseDown()
            self.click_down = True
            self.logger.info("primary click down")
        elif middle_finger_to_thumb_tip > self.click_threshold and self.click_down:  # release primary click
            pag.mouseUp()
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

        if ring_finger_to_thumb_tip < self.click_threshold and not self.click_down:
            pag.click(button="right")

    def get_cursor_position(self, landmarks: np.ndarray):
        """
        Smooth the cursor position.

        Args:
            landmarks: Coordinates of the finger landmarks.

        Returns:
            On screen cursor position.
        """
        self.lagged_index_finger_landmark = np.roll(self.lagged_index_finger_landmark, 1, axis=0)
        self.lagged_index_finger_landmark[0, :] = landmarks[:2, 0, 1]  # (x,y), tip, finger one (index finger)
        smoothed_index_finger_landmark = self.lagged_index_finger_landmark.mean(axis=0)

        smoothed_index_finger_landmark = np.clip(
            smoothed_index_finger_landmark, self._frame_margin_min, self._frame_margin_max
        )

        cursor_position_x = self.screen_width - np.interp(
            smoothed_index_finger_landmark[0],
            [self._frame_margin_min, self._frame_margin_max],
            [0, self.screen_width],
        )
        cursor_position_y = np.interp(
            smoothed_index_finger_landmark[1],
            [self._frame_margin_min, self._frame_margin_max],
            [0, self.screen_height],
        )

        self.logger.info(f"Landmark coords: ({smoothed_index_finger_landmark[0]}, {smoothed_index_finger_landmark[1]})")
        self.logger.info(f"Cursor coords: ({cursor_position_x}, {cursor_position_y})")
        return cursor_position_x, cursor_position_y

    def activate(self, video: bool = False):
        """
        Activate the gesture controller.
        """
        prvw_img_size = 320
        RED = (0, 0, 255)

        self.logger.info("Gesture controller initialized.")
        for frame in self.camera.stream_frames():
            if self.monitor_fps:
                frame = self.fps_monitor.monitor_fps(frame)

            if video:
                prvw_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                prvw_img = Image.fromarray(prvw_img).resize((prvw_img_size, prvw_img_size))
                prvw_img = np.array(prvw_img)
            else:
                prvw_img = None

            prediction = self.detector.predict(frame)
            if prediction is None:
                self.click_down = False  # ensure click is released when cursor moves off screen.

                if video and prvw_img is not None:
                    cv2.imshow("Frame", prvw_img)

                    key = cv2.waitKey(1)
                    if key == ord("q"):
                        return

                continue

            gesture_label, finger_landmarks = prediction
            self.logger.info(f"Gesture: {gesture_label}")
            self.gesture_handler.handle(gesture_label)

            cursor_pos_x, cursor_pos_y = self.get_cursor_position(finger_landmarks)

            if self.is_active:
                pag.moveTo(cursor_pos_x, cursor_pos_y)
                self.detect_primary_click(finger_landmarks)
                self.detect_secondary_click(finger_landmarks)

            if video and prvw_img is not None:
                cursor_pos_str = f"x={cursor_pos_x:.2f}, y={cursor_pos_y:.2f}"

                cv2.putText(
                    prvw_img,
                    cursor_pos_str,
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
