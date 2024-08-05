import logging
from dataclasses import dataclass
from typing import Callable


@dataclass
class Gesture:
    """
    Gesture class. Used to store gesture information including the label, delay, and callback.

    label: label provided by the classifier
    delay: number of classified gestures before calling the callback
    callback: control function corresponding to the gesture label
    """

    label: str
    delay: int
    callback: Callable[[], None]


class GestureHandler:
    """
    Gesture handler class. Used to handle gesture recognition and apply corresponding callback action.
    """

    def __init__(
        self,
        gestures: list[Gesture],
        verbose: bool = False,
    ):
        """
        Args:
            gestures (list[Gesture]): List of gestures to handle. See Gesture dataclass for more information.
            verbose (bool, optional): Send log output to terminal. Defaults to False.
        """
        self.gestures = {gesture.label: gesture for gesture in gestures}
        self.recognized_gestures = list(self.gestures.keys())
        self.label_queue: list[str] = []

        self.logger = logging.getLogger(__name__)
        if verbose:
            logging.basicConfig(level=logging.INFO)

    def handle(self, label: str):
        """
        Take gesture label and call the corresponding callback if the gesture is recognized.

        Args:
            label (str): label provided by the classifier
        """
        if label not in self.gestures:
            return

        if all(gesture == label for gesture in self.label_queue):
            self.label_queue.append(label)
        else:
            self.label_queue = [label]
        self.logger.info("Label queue: %s", self.label_queue)

        gesture = self.gestures[label]
        if len(self.label_queue) == gesture.delay:
            self.logger.info("Calling callback for gesture: %s", gesture.label)
            gesture.callback()
            self.label_queue = []
