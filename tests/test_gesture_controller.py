from unittest.mock import create_autospec

import numpy as np
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

    gesture_controller = GestureController(gestures=[g1, g2], detector=mock_detector, camera=mock_camera)
    return gesture_controller


class TestGestureController:
    def test_control_gesture_present(self, gesture_controller: GestureController):
        assert not gesture_controller.is_active
        gesture_controller.gesture_handler.gestures["Closed_Fist"].callback()
        assert gesture_controller.is_active

    def test_get_cursor_position(self, gesture_controller: GestureController):
        landmark_coords = np.zeros(shape=(3, 4, 5))
        landmark_coords[:, 0, 1] = np.array([1, 2, 3])
        for _ in range(gesture_controller.lagged_cursor_position.shape[0]):
            gesture_controller.get_cursor_position(landmark_coords)
        assert np.all(gesture_controller.get_cursor_position(landmark_coords) == np.array([1, 2, 3]))
