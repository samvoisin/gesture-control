# standard libraries
from typing import Callable, Optional, Sequence

# gestrol library
from gestrol.modifiers.base import Frame

FrameModifierCallable = Callable[[Frame], Optional[Frame]]


class FramePipeline:
    """
    Modular pipeline for concatenating and applying FrameModifiers.
    """

    def __init__(self, modifier_pipeline: Optional[Sequence[FrameModifierCallable]] = None):
        """
        Initiate method.

        Args:
            modifier_pipeline: A sequence of callables which can modify a Frame. Defaults to an empty list.
        """
        self.modifier_pipeline = modifier_pipeline or []

    def process_frame(self, frame: Frame) -> Optional[Frame]:
        """
        Modify frame by sequentially calling modifiers.

        Args:
            frame: Frame type

        Returns:
            Frame type or None if a modifier has null result
        """
        for modifier in self.modifier_pipeline:
            frame = modifier(frame)
            if frame is None:
                return None
        return frame
