# standard libraries
from typing import Optional, Sequence

# gestrol library
from gestrol.modifiers.base import Frame, FrameModifier


class FramePipeline:
    """
    Modular pipeline for concatenating and applying FrameModifiers.
    """

    def __init__(self, modifier_pipeline: Optional[Sequence[FrameModifier]] = None):
        """
        Initiate method.

        Args:
            modifier_pipeline: A sequence of callables which can modify a Frame. Defaults to an empty list.
        """
        self.modifier_pipeline = modifier_pipeline or []

    def process_frame(self, frame: Optional[Frame]) -> Optional[Frame]:
        """
        Modify frame by sequentially calling modifiers.

        Args:
            frame: Frame type

        Returns:
            Frame type or None if a modifier has null result
        """
        for modifier in self.modifier_pipeline:
            if frame is None:
                return None
            frame = modifier(frame)
        return frame
