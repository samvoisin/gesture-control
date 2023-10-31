from unittest.mock import create_autospec, patch

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

    @patch("pyautogui.mouseUp")
    @patch("pyautogui.mouseDown")
    def test_detect_primary_click(self, mock_mouseDown, mock_mouseUp, gesture_controller: GestureController):
        assert not gesture_controller.click_down

        finger_coords = np.zeros(shape=(3, 4, 5))
        finger_coords[:, 0, 0] = np.zeros(3)  # thumb tip
        finger_coords[:, 0, 1] = np.ones(3)  # index finger tip

        # engage primary click
        finger_coords[:, 0, 2] = np.zeros(3)  # middle finger tip
        gesture_controller.detect_primary_click(finger_coords)
        mock_mouseDown.assert_called_once()
        assert gesture_controller.click_down

        # release primary click
        finger_coords[:, 0, 2] = np.ones(3)  # middle finger tip
        gesture_controller.detect_primary_click(finger_coords)
        mock_mouseUp.assert_called_once()
        assert not gesture_controller.click_down

    def test_detect_secondary_click(self, gesture_controller: GestureController):
        finger_coords = np.zeros(shape=(3, 4, 5))
        finger_coords[:, 0, 0] = np.ones(3)  # thumb tip
        finger_coords[:, 0, 3] = np.ones(3)  # ring finger tip

        with patch("pyautogui.click") as mock_click:
            gesture_controller.detect_secondary_click(finger_coords)
            mock_click.assert_called_once()
            mock_click.assert_called_with(button="right")

            # ensure no click is registered when click_down is True
            gesture_controller.click_down = True
            gesture_controller.detect_secondary_click(finger_coords)
            mock_click.assert_called_once()
