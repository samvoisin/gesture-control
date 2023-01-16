# standard libraries
from typing import Optional

# gestrol library
from gestrol.modifiers.base import Frame, FrameModifier


class FrameSizeFilter(FrameModifier):
    def __init__(self, min_height: int, min_width: int, n_channels: Optional[int] = None) -> None:
        self.min_height = min_height
        self.min_width = min_width

        if n_channels:
            self.n_channels = n_channels

            def frame_size_check(*args) -> bool:
                nc, fh, fw = args
                return fh >= self.min_height and fw >= self.min_width and nc == self.n_channels

        else:

            def frame_size_check(*args) -> bool:
                fh, fw = args
                return fh >= self.min_height and fw >= self.min_width

        self.size_check = frame_size_check

    def __call__(self, frame: Frame) -> Optional[Frame]:
        if not hasattr(frame, "shape"):
            raise ValueError(f"{type(frame)} has no shape attribute.")

        if self.size_check(*frame.shape):
            return frame
        return None
