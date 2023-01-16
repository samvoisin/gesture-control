# standard libraries
import logging
from typing import Optional

# gestrol library
from gestrol.utils.logging import configure_logging

configure_logging()


class ControlModeSwitch:
    """
    Switch to toggle gesture classifier with control interface.
    """

    def __init__(self, control_mode_signal: int):
        self.control_mode_signal = control_mode_signal
        self.control_mode = False

    def toggle_control_mode(self):
        """
        Toggle control mode on or off.
        """
        self.control_mode = not self.control_mode
        if self.control_mode:
            logging.info("Control mode activated.")
        else:
            logging.info("Control mode deactivated.")

    def assess_gesture_signal(self, gesture_signal: int) -> Optional[int]:
        """
        Assess incoming signal from gesture classifier to determine if it should be passed on to control interface.

        Args:
            gesture_signal (int): Signal from gesture classifier

        Returns:
            Optional[int]: Signal passed on to control interface. May be `None`.
        """
        if gesture_signal == self.control_mode_signal:  # check for control mode signal
            self.toggle_control_mode()
            return None
        elif self.control_mode:  # not control mode signal; check for control mode
            return gesture_signal
        else:  # not control mode signal and control mode is off
            return None
