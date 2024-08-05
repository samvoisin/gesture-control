from unittest.mock import patch

import numpy as np
import pytest

from gesturemote.cursor_handler import CursorHandler, Fingers


class TestCursorHandler:
    def test_get_cursor_position(self):
        cursor_handler = CursorHandler(cursor_sensitivity=3, frame_margin=0.0)

        # modify screen size to simplify test calculations
        cursor_handler.screen_height, cursor_handler.screen_width = 10, 10

        landmark_coords = np.zeros(shape=(3, 4, 5))
        landmark_coords[:-1, 0, Fingers.INDEX.value] = np.array([0.0, 0.3])
        cp_x, cp_y = 0, 0
        for _ in range(cursor_handler.lagged_index_finger_landmark.shape[0]):
            landmark_coords[:-1, 0, Fingers.INDEX.value] += 0.1
            cp_x, cp_y = cursor_handler.get_cursor_position(landmark_coords)

        assert cp_x == 8
        assert cp_y == 5

    @pytest.mark.parametrize(
        ["primary_click_status", "middle_finger_arr", "mock_fcn"],
        [
            pytest.param(False, np.zeros(3), "_mouse_down", id="down"),
            pytest.param(True, np.ones(3), "_mouse_up", id="up"),
        ],
    )
    def test_detect_primary_click_down(self, primary_click_status, middle_finger_arr, mock_fcn):
        cursor_handler = CursorHandler(cursor_sensitivity=3, click_threshold=0.1)
        cursor_handler.primary_click_down = primary_click_status

        finger_coords = np.ones(shape=(3, 4, 5))
        finger_coords[:, 0, Fingers.THUMB.value] = np.zeros(3)  # thumb tip

        # engage primary click
        finger_coords[:, 0, Fingers.MIDDLE.value] = middle_finger_arr  # middle finger tip at same position as thumb tip
        with patch("gesturemote.cursor_handler.CursorHandler." + mock_fcn) as mock_click:
            cursor_handler.detect_primary_click(finger_coords, 0, 0)
            mock_click.assert_called_once()

    def test_detect_secondary_click(self):
        finger_coords = np.zeros(shape=(3, 4, 5))
        finger_coords[:, 0, Fingers.THUMB.value] = np.ones(3)  # thumb tip
        finger_coords[:, 0, Fingers.RING.value] = np.ones(3)  # ring finger tip

        cursor_handler = CursorHandler(cursor_sensitivity=3, click_threshold=0.1)

        with patch("gesturemote.cursor_handler.CursorHandler._mouse_click") as mock_click:
            cursor_handler.detect_secondary_click(finger_coords, 0, 0)
            mock_click.assert_called_once()

    @pytest.mark.parametrize(
        ["index_finger", "middle_finger", "scroll_amount", "inverse_scroll"],
        [
            pytest.param(np.ones(shape=(3, 4)), np.ones(shape=(3, 4)), 12, False, id="scroll up"),
            pytest.param(np.zeros(shape=(3, 4)), np.zeros(shape=(3, 4)), -12, False, id="scroll down"),
            pytest.param(np.ones(shape=(3, 4)), np.ones(shape=(3, 4)), -12, True, id="scroll up inv"),
            pytest.param(np.zeros(shape=(3, 4)), np.zeros(shape=(3, 4)), 12, True, id="scroll down inv"),
        ],
    )
    def test_detect_scroll(self, index_finger, middle_finger, scroll_amount, inverse_scroll):
        finger_coords = np.zeros(shape=(3, 4, 5))
        finger_coords[:, :, Fingers.INDEX.value] = index_finger
        finger_coords[:, :, Fingers.MIDDLE.value] = middle_finger

        cursor_handler = CursorHandler(scroll_sensitivity=0.0, inverse_scroll=inverse_scroll)

        with patch("gesturemote.cursor_handler.CursorHandler._mouse_scroll") as mock_scroll:
            scroll_detected = cursor_handler.detect_scroll(finger_coords)
            assert scroll_detected
            mock_scroll.assert_called_once()
            mock_scroll.assert_called_with(scroll_amount)

    def test_detect_scroll_no_scroll(self):
        finger_coords = np.zeros(shape=(3, 4, 5))
        finger_coords[:, :, Fingers.INDEX.value] = np.ones(shape=(3, 4))
        finger_coords[:, :, Fingers.MIDDLE.value] = np.zeros(shape=(3, 4))

        cursor_handler = CursorHandler(scroll_sensitivity=0.1)

        with patch("pyautogui.scroll") as mock_scroll:
            scroll_detected = cursor_handler.detect_scroll(finger_coords)
            assert not scroll_detected
            mock_scroll.assert_not_called()
