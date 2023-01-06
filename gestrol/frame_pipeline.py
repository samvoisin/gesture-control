# standard libraries
from typing import Optional, Protocol, Sequence

# gestrol library
from gestrol.modifiers.base import Frame


class FrameModifierProtocol(Protocol):
    """Protocol for objects that modify frames. To be used in frame pipeline."""

    def __call__(self, frame: Frame) -> Optional[Frame]:
        ...


class FramePipeline:
    """
    Modular pipeline for concatenating and applying FrameModifiers.
    """

    def __init__(self, modifier_pipeline: Optional[Sequence[FrameModifierProtocol]] = None):
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
