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

    def test_get_cursor_position(self, gesture_controller: GestureController):
        # modify screen size to simplify test calculations
        gesture_controller.screen_height, gesture_controller.screen_width = 10, 10
        landmark_coords = np.zeros(shape=(3, 4, 5))
        landmark_coords[:-1, 0, 1] = np.array([0.0, 0.3])
        cp_x, cp_y = 0, 0
        for _ in range(gesture_controller.lagged_index_finger_landmark.shape[0]):
            landmark_coords[:-1, 0, 1] += 0.1
            cp_x, cp_y = gesture_controller.get_cursor_position(landmark_coords)
        assert cp_x == 8
        assert cp_y == 5

    @patch("pyautogui.mouseUp")
    @patch("pyautogui.mouseDown")
    def test_detect_primary_click(self, mock_mouseDown, mock_mouseUp, gesture_controller: GestureController):
        assert not gesture_controller.click_down

        finger_coords = np.ones(shape=(3, 4, 5))
        finger_coords[:, 0, 0] = np.zeros(3)  # thumb tip

        # engage primary click
        finger_coords[:, 0, 2] = np.zeros(3)  # middle finger tip at same position as thumb tip
        gesture_controller.detect_primary_click(finger_coords)
        mock_mouseDown.assert_called_once()
        assert gesture_controller.click_down

        # release primary click
        finger_coords[:, 0, 2] = np.ones(3)  # middle finger tip at different position from thumb tip
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
