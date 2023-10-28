# standard libraries
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
    def __init__(
        self,
        gestures: list[Gesture],
        verbose: bool = False,
    ):
        self.gestures = {gesture.label: gesture for gesture in gestures}
        self.logger = logging.getLogger(__name__)
        if verbose:
            logging.basicConfig(level=logging.INFO)
        self.label_queue: list[str] = []

    def handle(self, label: str):
        if label not in self.gestures:
            return

        if all(gesture == label for gesture in self.label_queue):
            self.label_queue.append(label)
        else:
            self.label_queue = [label]
        self.logger.info(f"Label queue: {self.label_queue}")

        gesture = self.gestures[label]
        if len(self.label_queue) == gesture.delay:
            self.logger.info(f"Calling callback for gesture: {gesture.label}")
            gesture.callback()
            self.label_queue = []
