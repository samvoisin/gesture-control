import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class StateController:
    """
    Controls whether or not the gesture controller is active.
    """

    def __init__(self, verbose: bool = False):
        """
        Args:
            verbose (bool, optional): Log output. Defaults to False.
        """
        self.active = False
        self.verbose = verbose

    def toggle_state(self):
        """
        Toggle the state of the controller.
        """
        self.active = not self.active
        if self.verbose:
            logger.info(f"control mode is {self.active}")
