from pathlib import Path
from time import time
from typing import List, Optional, Tuple

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

from gesturemote.cursor_handler import Fingers, Knuckles

GESTURE_RECOGNIZER_TASK_PATH = Path("models/gesture_recognizer.task").resolve()

THUMB_IDXS = [4, 3, 2, 1]
INDEX_FINGER_IDXS = [8, 7, 6, 5]
MIDDLE_FINGER_IDXS = [12, 11, 10, 9]
RING_FINGER_IDXS = [16, 15, 14, 13]
PINKY_IDXS = [20, 19, 18, 17]
FINGER_IDXS = [THUMB_IDXS, INDEX_FINGER_IDXS, MIDDLE_FINGER_IDXS, RING_FINGER_IDXS, PINKY_IDXS]


def _build_coordinate_array(hand_landmarks: List[NormalizedLandmark]) -> np.ndarray:
    """
    Build a 3D coordinate array from hand landmarks.

    Args:
        hand_landmarks (List[NormalizedLandmark]): gesture recognizer task landmarks.

    Returns:
        np.ndarray: Array of shape (3, 4, 5) containing x, y, z coordinates of finger landmarks.
    """
    coordinates = np.empty(shape=(3, 4, 5))

    for finger in Fingers:
        for knuckle in Knuckles:
            landmark = hand_landmarks[FINGER_IDXS[finger.value][knuckle.value]]
            coordinates[:, knuckle.value, finger.value] = (landmark.x, landmark.y, landmark.z)

    return coordinates


def parse_gesture_recognition_result(
    result: Optional[mp.tasks.vision.GestureRecognizerResult],
) -> Optional[Tuple[str, np.ndarray]]:
    """
    Parse output from mediapipe gesture detector.

    Args:
        result: Either `GestureRecognizerResult` or None if no hand detected.

    Returns:
        None if result is None. Otherwise returns a tuple of gesture label and finger landmarks.
    """
    if result is None or len(result.gestures) == 0 or len(result.hand_landmarks) == 0:
        return None

    gesture_category = result.gestures[0][0]
    gesture_label = gesture_category.category_name

    hand_landmarks = result.hand_landmarks[0]  # list of 21 NormalizedLandmarks
    coordinates = _build_coordinate_array(hand_landmarks)

    return gesture_label, coordinates


class LandmarkGestureDetector:
    """
    Gesture Classes:
    0 - Unrecognized gesture, label: Unknown
    1 - Closed fist, label: Closed_Fist
    2 - Open palm, label: Open_Palm
    3 - Pointing up, label: Pointing_Up
    4 - Thumbs down, label: Thumb_Down
    5 - Thumbs up, label: Thumb_Up
    6 - Victory, label: Victory
    7 - Love, label: ILoveYou
    """

    def __init__(self):
        options = mp.tasks.vision.GestureRecognizerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=GESTURE_RECOGNIZER_TASK_PATH),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            result_callback=self._result_callback,
        )

        self.recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(options)
        self._latest_result = None

    def _result_callback(
        self, result: mp.tasks.vision.GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int
    ):
        """
        Callback for handling asynchronous detection results. Results are available only in this callback.
        Therefore we need to store the latest result in a class variable.
        """
        self._latest_result = result

    def predict(self, frame: np.ndarray) -> Optional[Tuple[str, np.ndarray]]:
        """
        Predict gesture and landmarks from a frame.

        Args:
            frame (np.ndarray): Frame to predict.

        Returns:
            Tuple[str, np.ndarray]: Gesture label and finger landmarks.
        """
        frame_timestamp_ms = int(time() * 1000)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        self.recognizer.recognize_async(image, frame_timestamp_ms)
        return parse_gesture_recognition_result(self._latest_result)
