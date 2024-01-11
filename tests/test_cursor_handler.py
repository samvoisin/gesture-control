from unittest.mock import patch

import numpy as np

from gesturemote.cursor_handler import CursorHandler


class TestCursorHandler:
    def test_get_cursor_position(self):
        cursor_handler = CursorHandler(cursor_sensitivity=3, frame_margin=0.0)

        # modify screen size to simplify test calculations
        cursor_handler.screen_height, cursor_handler.screen_width = 10, 10

        landmark_coords = np.zeros(shape=(3, 4, 5))
        landmark_coords[:-1, 0, 1] = np.array([0.0, 0.3])
        cp_x, cp_y = 0, 0
        for _ in range(cursor_handler.lagged_index_finger_landmark.shape[0]):
            landmark_coords[:-1, 0, 1] += 0.1
            cp_x, cp_y = cursor_handler.get_cursor_position(landmark_coords)

        assert cp_x == 8
        assert cp_y == 5

    @patch("pyautogui.mouseUp")
    @patch("pyautogui.mouseDown")
    def test_detect_primary_click(self, mock_mouseDown, mock_mouseUp):
        cursor_handler = CursorHandler(cursor_sensitivity=3, click_threshold=0.1)

        assert not cursor_handler.click_down

        finger_coords = np.ones(shape=(3, 4, 5))
        finger_coords[:, 0, 0] = np.zeros(3)  # thumb tip

        # engage primary click
        finger_coords[:, 0, 2] = np.zeros(3)  # middle finger tip at same position as thumb tip
        cursor_handler.detect_primary_click(finger_coords)
        mock_mouseDown.assert_called_once()
        assert cursor_handler.click_down

        # release primary click
        finger_coords[:, 0, 2] = np.ones(3)  # middle finger tip at different position from thumb tip
        cursor_handler.detect_primary_click(finger_coords)
        mock_mouseUp.assert_called_once()
        assert not cursor_handler.click_down

    def test_detect_secondary_click(self):
        finger_coords = np.zeros(shape=(3, 4, 5))
        finger_coords[:, 0, 0] = np.ones(3)  # thumb tip
        finger_coords[:, 0, 3] = np.ones(3)  # ring finger tip

        cursor_handler = CursorHandler(cursor_sensitivity=3, click_threshold=0.1)

        with patch("pyautogui.click") as mock_click:
            cursor_handler.detect_secondary_click(finger_coords)
            mock_click.assert_called_once()
            mock_click.assert_called_with(button="right")

            # ensure no click is registered when click_down is True
            cursor_handler.click_down = True
            cursor_handler.detect_secondary_click(finger_coords)
            mock_click.assert_called_once()
