import logging
from enum import Enum

import numpy as np
from numpy.linalg import norm
from Quartz.CoreGraphics import (
    CGDisplayBounds,
    CGEventCreateMouseEvent,
    CGEventCreateScrollWheelEvent,
    CGEventPost,
    CGMainDisplayID,
    kCGEventLeftMouseDown,
    kCGEventLeftMouseUp,
    kCGEventMouseMoved,
    kCGEventRightMouseDown,
    kCGEventRightMouseUp,
    kCGEventSourceStateHIDSystemState,
    kCGMouseButtonLeft,
    kCGMouseButtonRight,
    kCGScrollEventUnitPixel,
)


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
        inverse_scroll: bool = False,
        click_threshold: float = 0.1,
        frame_margin: float = 0.1,
        verbose: bool = False,
    ):
        """
        Args:
            cursor_sensitivity (int, optional): Number of frames to lag the cursor position. Defaults to 5.
            scroll_sensitivity (float, optional): Distance between landmarks to register scrolling.
            inverse_scroll (bool, optional): Whether to invert the scroll direction.
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
        self.inverse_scroll = inverse_scroll
        self.click_threshold = click_threshold

        # get screen dimensions; NOTE: currently only supports main display
        display_bounds = CGDisplayBounds(CGMainDisplayID())
        self.screen_width, self.screen_height = display_bounds.size.width, display_bounds.size.height

        self.logger = logging.getLogger(__name__)
        if verbose:
            logging.basicConfig(level=logging.INFO)

    def process_finger_coordinates(self, finger_coordinates: np.ndarray):
        """
        Consume hand landmark coordinates to change cursor state.

        Args:
            finger_coordinates (np.ndarray): 3 x 4 x 5 array of coordinates corresponding to
            3 spatial dimensions (x,y,z), 4 landmarks, 5 fingers
        """
        scroll_detected = self.detect_scroll(finger_coordinates)
        if scroll_detected:  # do not move cursor if scrolling
            return

        cursor_pos_x, cursor_pos_y = self.get_cursor_position(finger_coordinates)

        self.detect_click_and_drag(finger_coordinates, cursor_pos_x, cursor_pos_y)

        self._move_mouse(cursor_pos_x, cursor_pos_y)

        self.detect_primary_click(finger_coordinates, cursor_pos_x, cursor_pos_y)
        self.detect_secondary_click(finger_coordinates, cursor_pos_x, cursor_pos_y)

    def detect_primary_click(self, finger_coordinates: np.ndarray, cursor_pos_x: float, cursor_pos_y: float):
        """
        Detect if the user is clicking.

        Args:
            finger_coordinates (np.ndarray): coordinates of the finger landmarks.
            cursor_pos_x (float): x-coordinate of the cursor.
            cursor_pos_y (float): y-coordinate of the cursor.

        Returns:
            True if the user is clicking, False otherwise.
        """
        thumb_tip_vector = finger_coordinates[:, 0, Fingers.THUMB.value]
        middle_finger_tip_vector = finger_coordinates[:, 0, Fingers.MIDDLE.value]

        middle_finger_to_thumb_tip = norm(thumb_tip_vector - middle_finger_tip_vector)

        self.logger.info("thumb to middle finger: %f", middle_finger_to_thumb_tip)

        if middle_finger_to_thumb_tip < self.click_threshold:
            self.logger.info("primary click")
            self._mouse_click(kCGMouseButtonLeft, cursor_pos_x, cursor_pos_y)

    def detect_secondary_click(self, finger_coordinates: np.ndarray, cursor_pos_x: float, cursor_pos_y: float):
        """
        Detect if the user is performing a secondary click.

        Args:
            finger_coordinates (np.ndarray): coordinates of the finger landmarks.
            cursor_pos_x (float): x-coordinate of the cursor.
            cursor_pos_y (float): y-coordinate of the cursor.
        """
        thumb_tip_vector = finger_coordinates[:, 0, Fingers.THUMB.value]
        ring_finger_tip_vector = finger_coordinates[:, 0, Fingers.RING.value]

        ring_finger_to_thumb_tip = norm(thumb_tip_vector - ring_finger_tip_vector)
        self.logger.info("secondary click distance: %f", ring_finger_to_thumb_tip)

        if ring_finger_to_thumb_tip < self.click_threshold:
            self.logger.info("secondary click")
            self._mouse_click(kCGMouseButtonRight, cursor_pos_x, cursor_pos_y)

    def detect_click_and_drag(self, finger_coordinates: np.ndarray, cursor_pos_x: float, cursor_pos_y: float):
        pass

    def detect_scroll(self, finger_coordinates: np.ndarray) -> bool:
        """
        Detect if the user is scrolling (vertical only).

        Args:
            finger_coordinates (np.ndarray): coordinates of the finger landmarks.

        Returns:
            bool: True if user is scrolling, False otherwise.
        """
        max_scroll = 24
        index_finger_position = finger_coordinates[:, :, Fingers.INDEX.value]
        middle_finger_position = finger_coordinates[:, :, Fingers.MIDDLE.value]

        # first two landmarks
        index_middle_finger_distance = norm(index_finger_position[:, :2] - middle_finger_position[:, :2])
        self.logger.info("index to middle finger (scroll) distance: %f", index_middle_finger_distance)

        if index_middle_finger_distance > self.scroll_sensitivity:
            return False

        self.logger.info("scrolling detected")
        index_finger_tip = index_finger_position[:, 0]
        scroll_amount = max_scroll * (index_finger_tip[1] - 0.5)
        scroll_amount = int(scroll_amount) if not self.inverse_scroll else -int(scroll_amount)
        self.logger.info("scroll amount: %d", scroll_amount)
        self._mouse_scroll(scroll_amount)
        return True

    def get_cursor_position(self, landmarks: np.ndarray) -> tuple[float, float]:
        """
        Calculate the position of the cursor on screen.

        Args:
            landmarks: Coordinates of the finger landmarks.

        Returns:
            On screen cursor position.
        """
        self.lagged_index_finger_landmark = np.roll(self.lagged_index_finger_landmark, 1, axis=0)
        self.lagged_index_finger_landmark[0, :] = landmarks[:2, 0, Fingers.INDEX.value]  # (x,y), tip
        smoothed_index_finger_landmark = self.lagged_index_finger_landmark.mean(axis=0)
        self.logger.info(
            "Smoothed index finger landmark: %f, %f",
            smoothed_index_finger_landmark[0],
            smoothed_index_finger_landmark[1],
        )

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

        self.logger.info(
            "Landmark coords: (%f, %f)",
            smoothed_index_finger_landmark[0],
            smoothed_index_finger_landmark[1],
        )
        self.logger.info("Cursor coords: (%f, %f)", cursor_position_x, cursor_position_y)
        return float(cursor_position_x), float(cursor_position_y)

    def _move_mouse(self, x: float, y: float):
        event = CGEventCreateMouseEvent(None, kCGEventMouseMoved, (x, y), kCGMouseButtonLeft)
        CGEventPost(kCGEventSourceStateHIDSystemState, event)

    def _mouse_down(self, button: object, x: float, y: float):
        event = CGEventCreateMouseEvent(
            None, kCGEventLeftMouseDown if button == kCGMouseButtonLeft else kCGEventRightMouseDown, (x, y), button
        )
        self.logger.info("Mouse down: %s", button)
        CGEventPost(kCGEventSourceStateHIDSystemState, event)

    def _mouse_up(self, button: object, x: float, y: float):
        event = CGEventCreateMouseEvent(
            None, kCGEventLeftMouseUp if button == kCGMouseButtonLeft else kCGEventRightMouseUp, (x, y), button
        )
        CGEventPost(kCGEventSourceStateHIDSystemState, event)

    def _mouse_click(self, button: object, x: float, y: float):
        self._mouse_down(button, x, y)
        self._mouse_up(button, x, y)

    def _mouse_scroll(self, amount):
        event = CGEventCreateScrollWheelEvent(None, kCGScrollEventUnitPixel, 1, amount)
        CGEventPost(kCGEventSourceStateHIDSystemState, event)
