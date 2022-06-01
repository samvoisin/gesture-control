# external libraries
import numpy as np

# gestrol library
from gestrol.frame_stream.frame_modifiers.base import FrameModifier


class SingleChannelModifier(FrameModifier):
    def modify_frame(self, frame: np.ndarray, channel: int = 0) -> np.ndarray:
        return frame[:, :, channel]
