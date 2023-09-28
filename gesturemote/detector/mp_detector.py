# standard libraries
from pathlib import Path
from time import time
from typing import Optional, Tuple

# external libraries
import mediapipe as mp
import numpy as np

GESTURE_RECOGNIZER_TASK_PATH = Path("models/gesture_recognizer.task").resolve()


def parse_gesture_recognition_result(
    result: mp.tasks.vision.GestureRecognizerResult,
) -> Optional[Tuple[str, Tuple[int, int, int]]]:
    if result is None or len(result.gestures) == 0 or len(result.hand_landmarks) == 0:
        return None

    gesture_category = result.gestures[0][0]
    gesture_label = gesture_category.category_name

    hand_landmarks = result.hand_landmarks[0]
    index_finger_landmark = hand_landmarks[8]
    index_finger_coords = (index_finger_landmark.x, index_finger_landmark.y, index_finger_landmark.z)

    return gesture_label, index_finger_coords


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

    def __init__(self) -> None:
        BaseOptions = mp.tasks.BaseOptions
        GestureRecognizer = mp.tasks.vision.GestureRecognizer
        GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=GESTURE_RECOGNIZER_TASK_PATH),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self._result_callback,
        )

        self.recognizer = GestureRecognizer.create_from_options(options)
        self._latest_result = None

    def _result_callback(
        self, result: mp.tasks.vision.GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int
    ):
        self._latest_result = result

    def predict(self, frame: np.ndarray):
        frame_timestamp_ms = int(time() * 1000)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        self.recognizer.recognize_async(image, frame_timestamp_ms)
        return parse_gesture_recognition_result(self._latest_result)
