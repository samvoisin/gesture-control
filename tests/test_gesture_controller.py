from unittest.mock import create_autospec

import numpy as np
import pytest

from gesturemote.camera.base import CameraInterface
from gesturemote.gesture_controller import DetectorProtocol, GestureController
from gesturemote.gesture_handler import Gesture


class MockCameraInterface(CameraInterface):
    def __init__(self):
        pass

    def __del__(self):
        pass

    def get_frame(self):
        pass

    def stream_frames(self):
        return np.empty(shape=(1, 1, 1))


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

        gesture_controller.activate()
        assert gesture_controller.gesture_handler.label_queue.pop() == "test_gesture_1"
