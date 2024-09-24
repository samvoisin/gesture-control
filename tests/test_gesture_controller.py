from typing import Generator
from unittest.mock import create_autospec, patch

import numpy as np
import pytest

from gesturemote.gesture_controller import DetectorProtocol, GestureController, display_video
from gesturemote.gesture_handler import Gesture


class MockCameraInterface:
    def stream_frames(self) -> Generator[np.ndarray, None, None]:
        while True:
            yield np.empty(shape=(1, 1, 1))
            break


@pytest.fixture
def gesture_controller() -> GestureController:
    g1 = Gesture("test_gesture_1", 3, lambda: print("test1"))
    g2 = Gesture("test_gesture_2", 3, lambda: print("test2"))

    mock_camera = MockCameraInterface()
    mock_detector = create_autospec(DetectorProtocol)

    gesture_controller = GestureController(
        cursor_sensitivity=3,
        frame_margin=0.0,
        gestures=[g1, g2],
        detector=mock_detector,
        camera=mock_camera,
    )
    return gesture_controller


class TestGestureController:
    def test_toggle_control_mode(self, gesture_controller: GestureController):
        gesture_controller.toggle_control_mode()
        assert gesture_controller.control_mode
        gesture_controller.toggle_control_mode()
        assert not gesture_controller.control_mode

    def test_control_gesture_present(self, gesture_controller: GestureController):
        """
        Ensure that the control gesture is present in the list of gestures to prevent accidental removal.
        """
        assert not gesture_controller.control_mode
        gesture_controller.gesture_handler.gestures["Closed_Fist"].callback()
        assert gesture_controller.control_mode

    def test_activate_w_gest_label(self, gesture_controller: GestureController):
        gesture_controller.detector.predict.return_value = ("test_gesture_1", None)

        next(gesture_controller.activate())
        assert gesture_controller.gesture_handler.label_queue.pop() == "test_gesture_1"

    def test_no_prediction(self, gesture_controller: GestureController):
        gesture_controller.detector.predict.return_value = None

        frame, control_mode, landmarks = next(gesture_controller.activate())
        assert isinstance(frame, np.ndarray)
        assert isinstance(control_mode, bool)
        assert landmarks is None


@pytest.fixture
def sample_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_landmarks():
    return np.zeros((3, 4, 5))


def test_display_video_no_landmarks(sample_frame):
    with patch("cv2.imshow") as mock_imshow, patch("cv2.putText") as mock_putText:
        display_video(sample_frame, True, None, 100)
        mock_imshow.assert_called_once()
        mock_putText.assert_called_once()


def test_display_video_with_landmarks(sample_frame, sample_landmarks):
    with patch("cv2.imshow") as mock_imshow, patch("cv2.putText") as mock_putText, patch("cv2.circle") as mock_circle:
        display_video(sample_frame, False, sample_landmarks, 100)
        mock_imshow.assert_called_once()
        mock_putText.assert_called_once()

        assert mock_circle.call_count == 20  # 5 fingers with 4 landmarks each
