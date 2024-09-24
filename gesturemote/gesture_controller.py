import logging
from typing import Generator, Optional, Protocol, Tuple

import cv2
import numpy as np
import pyautogui as pag
from PIL import Image

from gesturemote.camera import OpenCVCameraInterface
from gesturemote.cursor_handler import CursorHandler
from gesturemote.detector.mp_detector import LandmarkGestureDetector
from gesturemote.fps_monitor import FPSMonitor
from gesturemote.gesture_handler import Gesture, GestureHandler
from gesturemote.logger_config import configure_logger

DEFAULT_GESTURES = [
    Gesture("Thumb_Down", 3, lambda: pag.press("pagedown")),
    Gesture("Thumb_Up", 3, lambda: pag.press("pageup")),
]

RED = (0, 0, 255)


class CameraProtocol(Protocol):
    def stream_frames(self) -> Generator[np.ndarray, None, None]: ...


class DetectorProtocol(Protocol):
    def predict(self, frame: np.ndarray) -> Optional[tuple[str, np.ndarray]]: ...


class GestureController:
    """
    Gesture controller class.
    """

    def __init__(
        self,
        cursor_sensitivity: int = 8,
        scroll_sensitivity: float = 0.075,
        inverse_scroll: bool = False,
        activate_gesture_threshold: int = 7,
        click_threshold: float = 0.1,
        frame_margin: float = 0.1,
        gestures: Optional[list[Gesture]] = None,
        detector: Optional[DetectorProtocol] = None,
        camera: Optional[CameraProtocol] = None,
        monitor_fps: bool = False,
        verbose: bool = False,
        logfile: Optional[str] = None,
    ):
        """
        Args:
            cursor_sensitivity: Number of frames to lag the cursor position.
            scroll_sensitivity: Distance between landmarks to register scrolling.
            inverse_scroll: Whether to invert the scroll direction.
            activate_gesture_threshold: Number of frames to hold the activate gesture to toggle the controller.
            click_threshold: Distance between finger digits to register a primary or secondary click.
            frame_margin: percentage of the frame to pad. Ensures the cursor can access elements on edge of screen.
            gestures: List of gestures to handle.
            detector: Detector to use for gesture recognition.
            camera: Camera interface to use.
            monitor_fps: Whether to monitor the FPS of the camera.
            verbose: Set log level to DEBUG.
            logfile: Path to log file.
        """
        self.logger = configure_logger(logfile=logfile)

        self.click_threshold = click_threshold
        self._frame_margin_min = frame_margin
        self._frame_margin_max = 1 - frame_margin

        gestures = gestures or DEFAULT_GESTURES
        gestures.append(Gesture("Closed_Fist", activate_gesture_threshold, self.toggle_control_mode))  # control gesture
        self.gesture_handler = GestureHandler(gestures)
        self.cursor_handler = CursorHandler(
            cursor_sensitivity,
            scroll_sensitivity,
            inverse_scroll,
            click_threshold,
            frame_margin,
        )

        self.detector = detector or LandmarkGestureDetector()
        self.camera = camera or OpenCVCameraInterface()

        self.monitor_fps = monitor_fps
        if self.monitor_fps:
            self.fps_monitor = FPSMonitor()

        self.control_mode = False

        if verbose:
            self.logger.setLevel(logging.DEBUG)

    def toggle_control_mode(self):
        """
        Activate/deactivate control mode on the gesture controller.
        """
        self.control_mode = not self.control_mode
        self.logger.info("Gesture controller is active: %s", self.control_mode)

    def activate(self) -> Generator[Tuple[np.ndarray, bool, Optional[np.ndarray]], None, None]:
        """
        Activate the gesture controller.
        """
        self.logger.info("Gesture controller initialized.")

        for frame in self.camera.stream_frames():
            if self.monitor_fps:
                frame = self.fps_monitor.monitor_fps(frame)

            prediction = self.detector.predict(frame)

            if prediction is None:
                yield frame, self.control_mode, None
                continue

            gesture_label, finger_landmarks = prediction

            self.logger.info("Gesture: %s", gesture_label)
            self.gesture_handler.handle(gesture_label)

            if self.control_mode and gesture_label not in self.gesture_handler.recognized_gestures:
                self.cursor_handler.process_finger_coordinates(finger_landmarks)

            yield frame, self.control_mode, finger_landmarks


def display_video(frame: np.ndarray, control_mode: bool, finger_landmarks: Optional[np.ndarray], preview_img_size):
    """
    Display frames from the camera with overlayed detection information.
    This allows for visual debugging of the gesture detection.

    Args:
        frame (np.ndarray): The frame from the camera.
        control_mode (bool): Whether the gesture controller is in control mode.
        finger_landmarks (Optional[np.ndarray]): The detected finger landmarks.
        preview_img_size (int): The size of the preview image.
    """

    prvw_img = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    prvw_img = Image.fromarray(prvw_img).resize((preview_img_size, preview_img_size))
    prvw_img_arr = np.array(prvw_img)

    cv2.putText(
        img=prvw_img_arr,
        text=f"Control mode: {control_mode}",
        org=(0, 20),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=RED,
        thickness=1,
    )

    if finger_landmarks is not None:
        _, n_marks_per_finger, n_fingers = finger_landmarks.shape
        for finger in range(n_fingers):
            for mark in range(n_marks_per_finger):
                x, y = finger_landmarks[:2, mark, finger]
                cv2.circle(
                    img=prvw_img_arr,
                    center=(int(x * preview_img_size), int(y * preview_img_size)),
                    radius=3,
                    color=RED,
                    thickness=-1,
                )

    cv2.imshow("Frame", prvw_img_arr)
