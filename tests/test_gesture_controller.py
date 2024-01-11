from unittest.mock import create_autospec

import pytest

from gesturemote.camera.base import CameraInterface
from gesturemote.gesture_controller import DetectorProtocol, GestureController
from gesturemote.gesture_handler import Gesture


@pytest.fixture
def gesture_controller() -> GestureController:
    g1 = Gesture("test_gesture_1", 3, lambda: print("test1"))
    g2 = Gesture("test_gesture_2", 3, lambda: print("test2"))

    mock_camera = create_autospec(CameraInterface)
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
    def test_control_gesture_present(self, gesture_controller: GestureController):
        """
        Ensure that the control gesture is present in the list of gestures to prevent accidental removal.
        """
        assert not gesture_controller.is_active
        gesture_controller.gesture_handler.gestures["Closed_Fist"].callback()
        assert gesture_controller.is_active
