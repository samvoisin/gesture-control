import logging
from enum import Enum

import numpy as np
import pyautogui as pag
from numpy.linalg import norm


class Fingers(Enum):
    """
    Enum for finger names/coordinate indices.
    """

    THUMB = 0
    INDEX = 1
    MIDDLE = 2
    RING = 3
    PINKY = 4


class CursorHandler:
    """
    Class for converting hand landmarks into cursor controls.
    """

    def __init__(
        self,
        cursor_sensitivity: int = 5,
        scroll_sensitivity: float = 0.1,
        click_threshold: float = 0.1,
        frame_margin: float = 0.1,
        verbose: bool = False,
    ):
        """
        Args:
            cursor_sensitivity (int, optional): Number of frames to lag the cursor position. Defaults to 5.
            scroll_sensitivity (float, optional): Distance between landmarks to register scrolling.
            click_threshold (float, optional): Distance between finger digits to register a primary or secondary click.
            Defaults to 0.1.
            frame_margin (float, optional): percentage of the frame to pad. Ensures the cursor can access elements on
            edge of screen. Defaults to 0.1.
            verbose (bool, optional): Send log output to terminal.. Defaults to False.
        """
        self.lagged_index_finger_landmark = np.zeros(shape=(cursor_sensitivity, 2))

        self._frame_margin_min = frame_margin
        self._frame_margin_max = 1 - frame_margin

        self.scroll_sensitivity = scroll_sensitivity
        self.click_threshold = click_threshold
        self.click_down = False

        self.screen_width, self.screen_height = pag.size()

        self.logger = logging.getLogger(__name__)
        if verbose:
            logging.basicConfig(level=logging.INFO)

    def process_finger_coordinates(self, finger_coordinates: np.ndarray):
        """
        Consume hand landmark coordinates to change cursor state.

        Args:
            finger_coordinates (np.ndarray): finger coordinates must be 3 x 4 x 5 array of coordinates corresponding to
            3 spatial dimensions (x,y,z), 4 landmarks, 5 fingers
        """
        scroll_detected = self.detect_scroll(finger_coordinates)
        if scroll_detected:  # do not move cursor if scrolling
            return

        cursor_pos_x, cursor_pos_y = self.get_cursor_position(finger_coordinates)
        pag.moveTo(cursor_pos_x, cursor_pos_y)

        self.detect_primary_click(finger_coordinates)
        self.detect_secondary_click(finger_coordinates)

    def detect_primary_click(self, finger_coordinates: np.ndarray):
        """
        Detect if the user is clicking.

        Args:
            finger_coordinates (np.ndarray): coordinates of the finger landmarks.

        Returns:
            True if the user is clicking, False otherwise.
        """
        thumb_tip_vector = finger_coordinates[:, 0, Fingers.THUMB.value]
        middle_finger_tip_vector = finger_coordinates[:, 0, Fingers.MIDDLE.value]

        middle_finger_to_thumb_tip = norm(thumb_tip_vector - middle_finger_tip_vector)

        self.logger.info(f"thumb to middle finger: {middle_finger_to_thumb_tip}")

        if middle_finger_to_thumb_tip < self.click_threshold and not self.click_down:  # primary click down
            pag.mouseDown()
            self.click_down = True
            self.logger.info("primary click down")
        elif middle_finger_to_thumb_tip > self.click_threshold and self.click_down:  # release primary click
            pag.mouseUp()
            self.click_down = False
            self.logger.info("primary click released")

    def detect_secondary_click(self, finger_coordinates: np.ndarray):
        """
        Detect if the user is performing a secondary click.

        Args:
            finger_coordinates (np.ndarray): coordinates of the finger landmarks.
        """
        thumb_tip_vector = finger_coordinates[:, 0, Fingers.THUMB.value]
        ring_finger_tip_vector = finger_coordinates[:, 0, Fingers.RING.value]

        ring_finger_to_thumb_tip = norm(thumb_tip_vector - ring_finger_tip_vector)
        self.logger.info(f"secondary click distance: {ring_finger_to_thumb_tip:.3f}")

        if ring_finger_to_thumb_tip < self.click_threshold and not self.click_down:
            pag.click(button="right")

    def detect_scroll(self, finger_coordinates: np.ndarray) -> bool:
        """
        Detect if the user is scrolling (vertical only).

        Args:
            finger_coordinates (np.ndarray): coordinates of the finger landmarks.

        Returns:
            bool: True if user is scrolling, False otherwise.
        """
        max_scroll = 24
        index_finger_array = finger_coordinates[:, :, 1]
        middle_finger_array = finger_coordinates[:, :, 2]

        # first two landmarks
        index_middle_finger_distance = norm(index_finger_array[:, :2] - middle_finger_array[:, :2])

        if index_middle_finger_distance > self.scroll_sensitivity:
            return False

        index_finger_tip = index_finger_array[:, 0]
        scroll_amount = max_scroll * (index_finger_tip[1] - 0.5)
        pag.scroll(scroll_amount)
        return True

    def get_cursor_position(self, landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the position of the cursor on screen.

        Args:
            landmarks: Coordinates of the finger landmarks.

        Returns:
            On screen cursor position.
        """
        self.lagged_index_finger_landmark = np.roll(self.lagged_index_finger_landmark, 1, axis=0)
        self.lagged_index_finger_landmark[0, :] = landmarks[:2, 0, Fingers.INDEX.value]  # (x,y), tip
        self.logger.info(self.lagged_index_finger_landmark)
        smoothed_index_finger_landmark = self.lagged_index_finger_landmark.mean(axis=0)

        smoothed_index_finger_landmark = np.clip(
            smoothed_index_finger_landmark, self._frame_margin_min, self._frame_margin_max
        )

        cursor_position_x = self.screen_width - np.interp(
            smoothed_index_finger_landmark[0],
            [self._frame_margin_min, self._frame_margin_max],
            [0, self.screen_width],
        )
        cursor_position_y = np.interp(
            smoothed_index_finger_landmark[1],
            [self._frame_margin_min, self._frame_margin_max],
            [0, self.screen_height],
        )

        self.logger.info(f"Landmark coords: ({smoothed_index_finger_landmark[0]}, {smoothed_index_finger_landmark[1]})")
        self.logger.info(f"Cursor coords: ({cursor_position_x}, {cursor_position_y})")
        return cursor_position_x, cursor_position_y
