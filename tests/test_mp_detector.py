from unittest.mock import create_autospec

import mediapipe as mp
from mediapipe.tasks.python.components.containers.category import Category
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

from gesturemote.detector.mp_detector import parse_gesture_recognition_result


def test_parse_gesture_recognition_result():
    assert parse_gesture_recognition_result(None) is None

    grr = create_autospec(mp.tasks.vision.GestureRecognizerResult)
    grr.gestures = [[Category(category_name="Mock_Gesture")]]
    grr.hand_landmarks = [[NormalizedLandmark(x=1, y=2, z=3)] * 21]

    gesture_label, coordinates = parse_gesture_recognition_result(grr)
    assert gesture_label == "Mock_Gesture"
    assert coordinates.shape == (3, 4, 5)
