import logging

import numpy as np
import pyautogui as pag
from numpy.linalg import norm


class CursorHandler:
    def __init__(
        self,
        cursor_sensitivity: int = 5,
        click_threshold: float = 0.1,
        frame_margin: float = 0.1,
        verbose: bool = False,
    ):
        self.lagged_index_finger_landmark = np.zeros(shape=(cursor_sensitivity, 2))

        self._frame_margin_min = frame_margin
        self._frame_margin_max = 1 - frame_margin

        self.click_threshold = click_threshold
        self.click_down = False

        self.screen_width, self.screen_height = pag.size()

        self.logger = logging.getLogger(__name__)
        if verbose:
            logging.basicConfig(level=logging.INFO)

    def process_finger_coordinates(self, finger_coordinates: np.ndarray):
        cursor_pos_x, cursor_pos_y = self.get_cursor_position(finger_coordinates)
        pag.moveTo(cursor_pos_x, cursor_pos_y)
        self.detect_primary_click(finger_coordinates)

        self.detect_secondary_click(finger_coordinates)

    def detect_primary_click(self, finger_coordinates: np.ndarray):
        """
        Detect if the user is clicking.

        Args:
            finger_coordinates: coordinates of the finger landmarks.

        Returns:
            True if the user is clicking, False otherwise.
        """
        thumb_tip_vector = finger_coordinates[:, 0, 0]
        middle_finger_tip_vector = finger_coordinates[:, 0, 2]

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
            finger_coordinates: coordinates of the finger landmarks.
        """
        thumb_tip_vector = finger_coordinates[:, 0, 0]
        ring_finger_tip_vector = finger_coordinates[:, 0, 3]

        ring_finger_to_thumb_tip = norm(thumb_tip_vector - ring_finger_tip_vector)
        self.logger.info(f"secondary click distance: {ring_finger_to_thumb_tip:.3f}")

        if ring_finger_to_thumb_tip < self.click_threshold and not self.click_down:
            pag.click(button="right")

    def get_cursor_position(self, landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Smooth the cursor position.

        Args:
            landmarks: Coordinates of the finger landmarks.

        Returns:
            On screen cursor position.
        """
        self.lagged_index_finger_landmark = np.roll(self.lagged_index_finger_landmark, 1, axis=0)
        self.lagged_index_finger_landmark[0, :] = landmarks[:2, 0, 1]  # (x,y), tip, finger one (index finger)
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
